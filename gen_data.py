# ==========================================
# 2D Electromagnetic Scattering Data Generator
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
            "║      2D Electromagnetic Scattering Data Generator            ║\n"
            "║      Generates clean/noisy training and test datasets        ║\n"
            "╚══════════════════════════════════════════════════════════════╝\n\n"
        )
        return header + super().format_help()


def main():
    parser = CustomArgumentParser(
        description="2D EM inverse scattering data generator"
    )
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Use CPU instead of CUDA')
    parser.add_argument('--eps_bg', type=float, default=1.5,
                        help='Background relative permittivity (default: 1.5)')
    parser.add_argument('-a', '--array_type', type=str, required=True,
                        choices=['full_circle', 'half_circle'],
                        help='Antenna array configuration (required)')
    parser.add_argument('-o', '--out_dir', type=str, default='train_data',
                        help='Output directory for datasets (default: train_data)')
    parser.add_argument('-b', '--batch_size', type=int, default=16,
                        help='Forward solve batch size (default: 16)')
    parser.add_argument('--num_train_clean', type=int, default=10000,
                        help='Number of clean training samples (default: 10000)')
    parser.add_argument('--num_train_noisy', type=int, default=3000,
                        help='Number of noisy training samples (default: 3000)')
    parser.add_argument('--num_test', type=int, default=2000,
                        help='Number of test samples (default: 2000)')
    parser.add_argument('--train_clean_prefix', type=str, default=None,
                        help='Filename prefix for clean training set '
                             '(default: {type}_train_clean)')
    parser.add_argument('--train_noisy_prefix', type=str, default=None,
                        help='Filename prefix for noisy training set '
                             '(default: {type}_train_noisy)')
    parser.add_argument('--test_prefix', type=str, default=None,
                        help='Filename prefix for test set (default: {type}_test)')
    args = parser.parse_args()

    # ── Heavy imports deferred until after --help is handled ──
    import os
    import time
    import numpy as np
    import torch
    from scipy.special import hankel2 as scipy_hankel2
    from scipy.spatial import distance_matrix

    # ==========================================
    # 0. HPC Environment Configuration
    # ==========================================
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')

    # ==========================================
    # 1. Coupling Liquid Physical Parameters
    # ==========================================
    FREQ = 2.45e9
    C0 = 299792458.0
    K0 = 2 * np.pi * FREQ / C0

    EPS_BG = args.eps_bg
    K_BG = K0 * np.sqrt(EPS_BG)

    L = 0.25
    N = 128
    N_CELLS = N * N
    DX = L / N

    N_ANT = 64
    R_ANT = 0.35

    DTYPE = torch.complex64

    # ==========================================
    # 2. Spatial Grid and Green's Function Precomputation (CPU)
    # ==========================================
    x_arr = np.linspace(-L/2 + DX/2, L/2 - DX/2, N)
    y_arr = np.linspace(-L/2 + DX/2, L/2 - DX/2, N)
    X, Y = np.meshgrid(x_arr, y_arr)
    r_domain_np = np.vstack((X.flatten(), Y.flatten())).T
    r_domain_gpu = torch.from_numpy(r_domain_np).to(device=device, dtype=torch.float32)
    dist_DD = torch.cdist(r_domain_np, r_domain_np)
    dist_DD.fill_diagonal_(1e-8)
    J0 = torch.special.bessel_j0(K_BG * dist_DD)
    Y0 = torch.special.bessel_y0(K_BG * dist_DD)
    G_DD_base_gpu = -1j / 4 * (J0 - 1j * Y0) * (DX**2)
    a_eq = DX / np.sqrt(np.pi)
    self_term = (1 / K_BG**2) * (-1j * np.pi * K_BG * a_eq / 2 * scipy_hankel2(1, K_BG * a_eq) + 1)
    G_DD_base_gpu.fill_diagonal_(complex(self_term))
    del dist_DD, J0, Y0
    torch.cuda.empty_cache()

    # ==========================================
    # 3. Forward Scattering Batch Solver (GPU)
    # ==========================================
    def torch_hankel2_0(x):
        return torch.special.bessel_j0(x) - 1j * torch.special.bessel_y0(x)

    def solve_batch_gpu(eps_r_batch, array_type, add_jitter=False, add_noise=False):
        batch_size = eps_r_batch.shape[0]

        eps_r_gpu = torch.from_numpy(eps_r_batch).to(device=device, dtype=DTYPE)
        chi = eps_r_gpu.view(batch_size, N_CELLS) - EPS_BG

        if array_type == 'full_circle':
            angles = torch.arange(0, N_ANT, device=device, dtype=torch.float32) * (2 * np.pi / N_ANT)
            ant_x = R_ANT * torch.cos(angles)
            ant_y = R_ANT * torch.sin(angles)
        elif array_type == 'half_circle':
            angles_half = torch.linspace(-np.pi/2, np.pi/2, N_ANT, device=device)
            ant_x = R_ANT * torch.cos(angles_half)
            ant_y = R_ANT * torch.sin(angles_half)

        r_ant = torch.stack((ant_x, ant_y), dim=-1).unsqueeze(0).repeat(batch_size, 1, 1)

        if add_jitter:
            r_ant = r_ant + torch.normal(0, 0.003, size=r_ant.shape, device=device)

        dist_SD = torch.cdist(r_ant, r_domain_gpu.unsqueeze(0).expand(batch_size, -1, -1))

        G_SD = -1j / 4 * torch_hankel2_0(K_BG * dist_SD) * (DX**2)
        E_inc = -1j / 4 * torch_hankel2_0(K_BG * dist_SD.transpose(1, 2))

        E_tot = torch.zeros((batch_size, N_CELLS, N_ANT), dtype=DTYPE, device=device)
        I = torch.eye(N_CELLS, device=device, dtype=DTYPE)

        for i in range(batch_size):
            A_i = I - (K0**2) * G_DD_base_gpu * chi[i].unsqueeze(0)
            E_tot[i] = torch.linalg.solve(A_i, E_inc[i])

        E_sca = (K0**2) * torch.bmm(G_SD, chi.unsqueeze(2) * E_tot)

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
    # 4. Random Phantom Batch Generator
    # ==========================================
    def generate_complex_phantom_batch(batch_size):
        eps_r_batch = np.ones((batch_size, N, N), dtype=np.float32) * EPS_BG
        for i in range(batch_size):
            num_objects = np.random.randint(1, 4)
            for _ in range(num_objects):
                cx = np.random.uniform(-L/3, L/3)
                cy = np.random.uniform(-L/3, L/3)
                rx = np.random.uniform(0.015, 0.04)
                ry = np.random.uniform(0.015, 0.04)
                rotation = np.random.uniform(0, np.pi)
                val = np.random.uniform(3.0, 5.0)

                x_rot = (X - cx) * np.cos(rotation) + (Y - cy) * np.sin(rotation)
                y_rot = -(X - cx) * np.sin(rotation) + (Y - cy) * np.cos(rotation)
                mask = (x_rot / rx)**2 + (y_rot / ry)**2 <= 1.0
                eps_r_batch[i, mask] = val
        return eps_r_batch

    # ==========================================
    # 5. Batch Scheduler and Progress Output
    # ==========================================
    def format_time(seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def generate_dataset_gpu_pipeline(num_samples, array_type, save_prefix, out_dir, batch_size=16,
                                       add_jitter=False, add_noise=False):
        print(f"\n{'='*50}")
        print(f"Generating: [{save_prefix}]")
        print(f"Background wavenumber K_BG: {K_BG:.4f} | Resolution: {N}x{N}")
        print(f"{'='*50}")

        X_all = np.zeros((num_samples, 2, N_ANT, N_ANT), dtype=np.float32)
        Y_all = np.zeros((num_samples, N, N), dtype=np.float32)

        num_batches = int(np.ceil(num_samples / batch_size))
        start_time = time.time()

        for b in range(num_batches):
            b_start_idx = b * batch_size
            b_end_idx = min((b + 1) * batch_size, num_samples)
            current_batch_size = b_end_idx - b_start_idx

            eps_r_batch = generate_complex_phantom_batch(current_batch_size)
            E_sca_tensor, eps_r_cpu = solve_batch_gpu(eps_r_batch, array_type, add_jitter, add_noise)

            X_all[b_start_idx:b_end_idx] = E_sca_tensor
            Y_all[b_start_idx:b_end_idx] = eps_r_cpu

            torch.cuda.empty_cache()

            completed = b_end_idx
            elapsed_time = time.time() - start_time
            speed = completed / elapsed_time
            remaining = num_samples - completed
            eta_seconds = remaining / speed if speed > 0 else 0

            progress_pct = (completed / num_samples) * 100
            print(f"Progress: {progress_pct:5.1f}% | Done: {completed:5d}/{num_samples} | "
                  f"Speed: {speed:5.2f} samples/s | Elapsed: {format_time(elapsed_time)} | "
                  f"ETA: {format_time(eta_seconds)}", end='\r')

        print()

        os.makedirs(out_dir, exist_ok=True)
        np.save(f'{out_dir}/{save_prefix}_X.npy', X_all)
        np.save(f'{out_dir}/{save_prefix}_Y.npy', Y_all)
        print(f"Dataset saved: {out_dir}/{save_prefix}_X.npy, {out_dir}/{save_prefix}_Y.npy")
        return X_all, Y_all

    # ==========================================
    # 6. Data Generation Entry Point
    # ==========================================
    array_type = args.array_type
    save_suffix = "full" if array_type == "full_circle" else "half"

    train_clean_prefix = args.train_clean_prefix or f"{save_suffix}_train_clean"
    train_noisy_prefix = args.train_noisy_prefix or f"{save_suffix}_train_noisy"
    test_prefix = args.test_prefix or f"{save_suffix}_test"

    print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    t_start = time.time()
    generate_dataset_gpu_pipeline(args.num_train_clean, array_type, train_clean_prefix,
                                  args.out_dir, args.batch_size, False, False)
    generate_dataset_gpu_pipeline(args.num_train_noisy, array_type, train_noisy_prefix,
                                  args.out_dir, args.batch_size, True, True)
    generate_dataset_gpu_pipeline(args.num_test, array_type, test_prefix,
                                  args.out_dir, args.batch_size, True, True)

    print(f"\nDataset [{array_type.upper()}] generation completed. Total time: {format_time(time.time() - t_start)}")


if __name__ == "__main__":
    main()