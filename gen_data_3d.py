# ==========================================
# 3D Electromagnetic Scattering Data Generator
# ==========================================
import argparse
import sys


class CustomArgumentParser(argparse.ArgumentParser):
    """Custom parser with ASCII art header and formatted help."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, formatter_class=argparse.RawTextHelpFormatter)

    def format_help(self):
        header = (
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║      3D Electromagnetic Scattering Data Generator            ║\n"
            "║      Generates 3D training and test datasets                 ║\n"
            "║      Fibonacci spherical antenna array                       ║\n"
            "╚══════════════════════════════════════════════════════════════╝\n\n"
        )
        return header + super().format_help()


def main():
    parser = CustomArgumentParser(
        description="3D EM inverse scattering data generator with Fibonacci spherical array"
    )
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Use CPU instead of CUDA')
    parser.add_argument('--eps_bg', type=float, default=1.5,
                        help='Background relative permittivity (default: 1.5)')
    parser.add_argument('-o', '--out_dir', type=str, default='train_data',
                        help='Output directory for datasets (default: train_data)')
    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help='Forward solve batch size (default: 1)')
    parser.add_argument('--num_train', type=int, default=2000,
                        help='Number of training samples (default: 2000)')
    parser.add_argument('--num_test', type=int, default=500,
                        help='Number of test samples (default: 500)')
    parser.add_argument('--train_prefix', type=str, default='sphere_train',
                        help='Output prefix for training set (default: sphere_train)')
    parser.add_argument('--test_prefix', type=str, default='sphere_test',
                        help='Output prefix for test set (default: sphere_test)')
    args = parser.parse_args()

    # ── Heavy imports deferred until after --help is handled ──
    import os
    import time
    import numpy as np
    import torch
    import scipy.spatial

    # ==========================================
    # 0. HPC Environment Configuration
    # ==========================================
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')

    # ==========================================
    # 1. 3D Coupling Liquid Physical Parameters
    # ==========================================
    FREQ = 2.45e9
    C0 = 299792458.0
    K0 = 2 * np.pi * FREQ / C0

    EPS_BG = args.eps_bg
    K_BG = K0 * np.sqrt(EPS_BG)

    L = 0.25
    N = 32
    N_CELLS = N**3
    DX = L / N

    N_ANT = 64
    R_ANT = 0.35

    DTYPE = torch.complex64

    # ==========================================
    # 2. 3D Spatial Grid and Fibonacci Sphere Antenna Initialization
    # ==========================================
    x_arr = np.linspace(-L/2 + DX/2, L/2 - DX/2, N)
    y_arr = np.linspace(-L/2 + DX/2, L/2 - DX/2, N)
    z_arr = np.linspace(-L/2 + DX/2, L/2 - DX/2, N)
    X, Y, Z = np.meshgrid(x_arr, y_arr, z_arr, indexing='ij')
    r_domain_np = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T

    # Fibonacci uniformly distributed spherical antenna array
    indices = np.arange(0, N_ANT, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / N_ANT)
    theta = np.pi * (1 + 5**0.5) * indices
    ant_x = R_ANT * np.cos(theta) * np.sin(phi)
    ant_y = R_ANT * np.sin(theta) * np.sin(phi)
    ant_z = R_ANT * np.cos(phi)
    r_ant_np = np.vstack((ant_x, ant_y, ant_z)).T
    r_ant_gpu = torch.from_numpy(r_ant_np).to(device=device, dtype=torch.float32)

    # ==========================================
    # 3. 3D Scalar Wave Green's Function Precomputation
    # ==========================================
    print("Precomputing 3D domain-to-domain operator on CPU (approx. 9 GB memory)...")
    dist_DD = scipy.spatial.distance_matrix(r_domain_np, r_domain_np).astype(np.float32)

    G_DD_base_np = np.empty_like(dist_DD, dtype=np.complex64)
    with np.errstate(divide='ignore', invalid='ignore'):
        np.divide(np.exp(-1j * K_BG * dist_DD), (4 * np.pi * dist_DD), out=G_DD_base_np)
        G_DD_base_np *= (DX**3)

    # Equivalent sphere self-term integral for 3D singularity
    a_eq = (3 * (DX**3) / (4 * np.pi))**(1/3)
    self_term = (1 / K_BG**2) * (1 - (1 + 1j * K_BG * a_eq) * np.exp(-1j * K_BG * a_eq))
    np.fill_diagonal(G_DD_base_np, self_term)

    del dist_DD

    print("Loading 3D physics core onto GPU...")
    G_DD_base_gpu = torch.from_numpy(G_DD_base_np).to(device=device)
    r_domain_gpu = torch.from_numpy(r_domain_np).to(device=device, dtype=torch.float32)

    # ==========================================
    # 4. 3D Forward Scattering Solver
    # ==========================================
    def solve_batch_3d_gpu(eps_r_batch, add_noise=False):
        batch_size = eps_r_batch.shape[0]
        eps_r_gpu = torch.from_numpy(eps_r_batch).to(device=device, dtype=DTYPE)

        chi = eps_r_gpu.view(batch_size, N_CELLS) - EPS_BG

        dist_SD = torch.cdist(r_ant_gpu.unsqueeze(0), r_domain_gpu.unsqueeze(0)).squeeze(0)

        G_SD = torch.exp(-1j * K_BG * dist_SD) / (4 * np.pi * dist_SD) * (DX**3)
        E_inc = torch.exp(-1j * K_BG * dist_SD.T) / (4 * np.pi * dist_SD.T)

        E_tot = torch.zeros((batch_size, N_CELLS, N_ANT), dtype=DTYPE, device=device)
        I = torch.eye(N_CELLS, device=device, dtype=DTYPE)

        for i in range(batch_size):
            A_i = I - (K0**2) * G_DD_base_gpu * chi[i].unsqueeze(0)
            E_tot[i] = torch.linalg.solve(A_i, E_inc)

        E_sca = (K0**2) * torch.bmm(G_SD.unsqueeze(0).expand(batch_size, -1, -1),
                                     chi.unsqueeze(2) * E_tot)

        if add_noise:
            snr_db = torch.rand(batch_size, 1, 1, device=device) * 10 + 20
            snr_linear = 10 ** (snr_db / 10)
            signal_power = torch.mean(torch.abs(E_sca)**2, dim=(1, 2), keepdim=True)
            noise_power = signal_power / snr_linear
            noise = torch.sqrt(noise_power / 2) * (torch.randn_like(E_sca) + 1j * torch.randn_like(E_sca))
            E_sca += noise

        E_sca_cpu = E_sca.cpu().numpy()
        E_sca_tensor = np.stack((np.real(E_sca_cpu), np.imag(E_sca_cpu)), axis=1)
        return E_sca_tensor, eps_r_batch

    # ==========================================
    # 5. 3D Random Phantom Generator
    # ==========================================
    def generate_phantom_3d_batch(batch_size):
        eps_r_batch = np.ones((batch_size, N, N, N), dtype=np.float32) * EPS_BG
        for i in range(batch_size):
            num_objects = np.random.randint(1, 4)
            for _ in range(num_objects):
                cx, cy, cz = np.random.uniform(-L/3, L/3, 3)
                rx, ry, rz = np.random.uniform(0.02, 0.045, 3)
                val = np.random.uniform(3.0, 5.0)

                mask = ((X - cx)/rx)**2 + ((Y - cy)/ry)**2 + ((Z - cz)/rz)**2 <= 1.0
                eps_r_batch[i, mask] = val
        return eps_r_batch

    # ==========================================
    # 6. Batch Scheduler and Progress Control
    # ==========================================
    def format_time(seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def generate_dataset_3d(num_samples, save_prefix, out_dir, batch_size=4, add_noise=False):
        print(f"\n{'='*50}")
        print(f"Generating: [3D {save_prefix}]")
        print(f"Volume: {N}x{N}x{N} | Matrix order: {N_CELLS} | VRAM batch: {batch_size}")
        print(f"{'='*50}")

        X_all = np.zeros((num_samples, 2, N_ANT, N_ANT), dtype=np.float32)
        Y_all = np.zeros((num_samples, N, N, N), dtype=np.float32)

        num_batches = int(np.ceil(num_samples / batch_size))
        start_time = time.time()

        for b in range(num_batches):
            b_start = b * batch_size
            b_end = min((b + 1) * batch_size, num_samples)
            current_bs = b_end - b_start

            eps_r_batch = generate_phantom_3d_batch(current_bs)
            E_sca_tensor, eps_r_cpu = solve_batch_3d_gpu(eps_r_batch, add_noise)

            X_all[b_start:b_end] = E_sca_tensor
            Y_all[b_start:b_end] = eps_r_cpu

            torch.cuda.empty_cache()

            completed = b_end
            elapsed = time.time() - start_time
            speed = completed / elapsed
            eta = (num_samples - completed) / speed if speed > 0 else 0

            print(f"Progress: {completed/num_samples*100:5.1f}% | Done: {completed:4d}/{num_samples} | "
                  f"Speed: {speed:5.2f} samples/s | Elapsed: {format_time(elapsed)} | "
                  f"ETA: {format_time(eta)}")

        print()
        os.makedirs(out_dir, exist_ok=True)
        np.save(f'{out_dir}/{save_prefix}_X.npy', X_all)
        np.save(f'{out_dir}/{save_prefix}_Y.npy', Y_all)
        print(f"3D dataset saved: {out_dir}/{save_prefix}_X.npy, {out_dir}/{save_prefix}_Y.npy")
        return X_all, Y_all

    # ==========================================
    # 7. Task Entry Point
    # ==========================================
    print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    t_start = time.time()

    generate_dataset_3d(args.num_train, args.train_prefix, args.out_dir,
                        args.batch_size, add_noise=True)
    generate_dataset_3d(args.num_test,  args.test_prefix,  args.out_dir,
                        args.batch_size, add_noise=True)

    print(f"\n3D dataset generation completed. Total time: {format_time(time.time() - t_start)}")


if __name__ == "__main__":
    main()