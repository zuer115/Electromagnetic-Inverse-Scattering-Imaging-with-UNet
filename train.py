# ==========================================
# 2D U-Net Training Pipeline
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
            "║      2D U-Net Training Pipeline                              ║\n"
            "║      Trains Bilinear U-Net on Born BP reconstructions        ║\n"
            "╚══════════════════════════════════════════════════════════════╝\n\n"
        )
        return header + super().format_help()


def main():
    parser = CustomArgumentParser(
        description="2D U-Net training pipeline with Born BP physical prior"
    )
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Use CPU instead of CUDA')
    parser.add_argument('--eps_bg', type=float, default=1.5,
                        help='Background relative permittivity (default: 1.5)')
    parser.add_argument('-a', '--array_type', type=str, required=True,
                        choices=['full_circle', 'half_circle'],
                        help='Antenna array configuration (required)')
    parser.add_argument('-o', '--out_dir', type=str, default='models',
                        help='Output directory for model weights (default: models)')
    parser.add_argument('-b', '--batch_size', type=int, default=128,
                        help='Training batch size (default: 128)')
    parser.add_argument('-e', '--epochs', type=int, default=400,
                        help='Number of training epochs (default: 400)')
    parser.add_argument('--train_clean_path', type=str, default=None,
                        help='Path stem for clean training data '
                             '(default: train_data/{type}_train_clean)')
    parser.add_argument('--train_noisy_path', type=str, default=None,
                        help='Path stem for noisy training data '
                             '(default: train_data/{type}_train_noisy)')
    parser.add_argument('--test_path', type=str, default=None,
                        help='Path stem for test data (default: train_data/{type}_test)')
    parser.add_argument('--no_noisy', action='store_true', default=False,
                        help='Skip loading noisy training data')
    args = parser.parse_args()

    # ── Heavy imports deferred until after --help is handled ──
    import os
    import time
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from scipy.special import hankel2 as scipy_hankel2
    from scipy.spatial import distance_matrix

    # ==========================================
    # 0. Environment and Device Configuration
    # ==========================================
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')

    # ==========================================
    # 1. Global Physical Parameters
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
    # 2. Physical Prior Adjoint Operator (Born BP)
    # ==========================================
    def precompute_physical_bp(X_np, array_type):
        print("Computing physical prior (Born BP)...")
        start = time.time()
        num_samples = X_np.shape[0]

        if array_type == 'full_circle':
            angles = torch.arange(0, N_ANT, device=device, dtype=torch.float32) * (2 * np.pi / N_ANT)
        else:
            angles = torch.linspace(-np.pi/2, np.pi/2, N_ANT, device=device)
        r_ant = torch.stack((R_ANT * torch.cos(angles), R_ANT * torch.sin(angles)), dim=-1).unsqueeze(0)

        dist_SD = torch.cdist(r_ant, r_domain_gpu.unsqueeze(0))
        G_SD = -1j / 4 * (torch.special.bessel_j0(K_BG * dist_SD) - 1j * torch.special.bessel_y0(K_BG * dist_SD)) * (DX**2)
        G_SD = G_SD.squeeze(0)
        E_inc = -1j / 4 * (torch.special.bessel_j0(K_BG * dist_SD.transpose(1, 2)) - 1j * torch.special.bessel_y0(K_BG * dist_SD.transpose(1, 2)))
        E_inc = E_inc.squeeze(0)

        K = (K0**2 * DX**2) * G_SD.T.unsqueeze(2) * E_inc.unsqueeze(1)
        K_conj = torch.conj(K)

        eps_bg = torch.ones(N_CELLS, device=device, dtype=DTYPE) * EPS_BG
        chi_bg = eps_bg - 1.0
        A_bg = torch.eye(N_CELLS, device=device, dtype=DTYPE) - (K0**2) * G_DD_base_gpu * chi_bg.unsqueeze(0)
        E_tot_bg = torch.linalg.solve(A_bg, E_inc)
        E_sca_bg = (K0**2) * (G_SD @ (chi_bg.unsqueeze(1) * E_tot_bg))

        S = torch.sum(torch.abs(K)**2, dim=(1, 2))

        BP_all = np.zeros((num_samples, 1, N, N), dtype=np.float32)
        bp_inner_batch = 256
        num_batches = int(np.ceil(num_samples / bp_inner_batch))
        for b in range(num_batches):
            b_start = b * bp_inner_batch
            b_end = min((b + 1) * bp_inner_batch, num_samples)

            X_batch_gpu = torch.complex(
                torch.from_numpy(X_np[b_start:b_end, 0]).to(device),
                torch.from_numpy(X_np[b_start:b_end, 1]).to(device)
            )

            X_diff = X_batch_gpu - E_sca_bg.unsqueeze(0)

            bp_gpu = torch.einsum('b r t, d r t -> b d', X_diff, K_conj).abs()
            bp_gpu = bp_gpu / (S.unsqueeze(0) + 1e-6)

            # Global amplitude scaling for network-friendly dynamic range
            bp_gpu = bp_gpu * 500.0

            BP_all[b_start:b_end] = bp_gpu.view(-1, 1, N, N).cpu().numpy()

        print(f"Physical prior computed. Time: {time.time() - start:.2f}s")
        return BP_all

    # ==========================================
    # 3. Bilinear U-Net Architecture
    # ==========================================
    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        def forward(self, x):
            return self.double_conv(x)

    class UpBlock(nn.Module):
        """Bilinear interpolation upsampling block — avoids checkerboard artifacts."""
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(inplace=True)
            )
            self.conv = DoubleConv(in_channels, out_channels)

        def forward(self, x1, x2):
            x1 = self.up(x1)
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)

    class SimpleUNet(nn.Module):
        def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
            super().__init__()
            self.downs = nn.ModuleList()
            self.ups = nn.ModuleList()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            in_ch = in_channels
            for feature in features:
                self.downs.append(DoubleConv(in_ch, feature))
                in_ch = feature

            self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

            for feature in reversed(features):
                self.ups.append(UpBlock(feature * 2, feature))

            self.final_conv = nn.Sequential(
                nn.Conv2d(features[0], out_channels, kernel_size=1),
                nn.Sigmoid()
            )

        def forward(self, x):
            skip_connections = []
            for down in self.downs:
                x = down(x)
                skip_connections.append(x)
                x = self.pool(x)

            x = self.bottleneck(x)
            skip_connections = skip_connections[::-1]

            for i in range(len(self.ups)):
                x = self.ups[i](x, skip_connections[i])

            return self.final_conv(x)

    # ==========================================
    # 4. Main Training Loop
    # ==========================================
    array_type = args.array_type
    prefix = "full" if array_type == "full_circle" else "half"

    # Resolve path defaults
    train_clean_path = args.train_clean_path or f"train_data/{prefix}_train_clean"
    train_noisy_path = args.train_noisy_path or f"train_data/{prefix}_train_noisy"
    test_path = args.test_path or f"train_data/{prefix}_test"

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR_MAX = 5e-4

    print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    print(f"Loading [{prefix.upper()}] dataset...")
    X_clean = np.load(f'{train_clean_path}_X.npy')
    Y_clean = np.load(f'{train_clean_path}_Y.npy')

    if args.no_noisy:
        print("Skipping noisy training data (--no_noisy set).")
        X_train_raw = X_clean
        Y_train_raw = Y_clean
    else:
        X_noisy = np.load(f'{train_noisy_path}_X.npy')
        Y_noisy = np.load(f'{train_noisy_path}_Y.npy')
        X_train_raw = np.concatenate((X_clean, X_noisy), axis=0)
        Y_train_raw = np.concatenate((Y_clean, Y_noisy), axis=0)

    X_test = np.load(f'{test_path}_X.npy')
    Y_test = np.load(f'{test_path}_Y.npy')

    # Normalize labels to [0, 1]
    Y_train_raw = (Y_train_raw - EPS_BG) / (5.0 - EPS_BG)
    Y_test = (Y_test - EPS_BG) / (5.0 - EPS_BG)

    X_train_bp = precompute_physical_bp(X_train_raw, array_type)
    X_test_bp  = precompute_physical_bp(X_test, array_type)

    train_dataset = TensorDataset(torch.from_numpy(X_train_bp), torch.from_numpy(Y_train_raw).unsqueeze(1))
    test_dataset  = TensorDataset(torch.from_numpy(X_test_bp), torch.from_numpy(Y_test).unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = SimpleUNet().to(device)

    l1_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-6)

    best_val_loss = float('inf')
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n" + "="*50)
    print(f"Training [{prefix.upper()}] network (Bilinear U-Net)")
    print("="*50)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_list = []
        t0 = time.time()

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = 0.8 * mse_criterion(outputs, targets) + 0.2 * l1_criterion(outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_list.append(loss.item())

        scheduler.step()

        model.eval()
        val_loss_list = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(inputs)
                    loss = 0.8 * mse_criterion(outputs, targets) + 0.2 * l1_criterion(outputs, targets)
                val_loss_list.append(loss.item())

        avg_train = np.mean(train_loss_list)
        avg_val = np.mean(val_loss_list)
        epoch_time = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        status_str = (f"Epoch [{epoch:03d}/{EPOCHS}] | Time: {epoch_time:.1f}s | "
                      f"LR: {current_lr:.1e} | Train Loss: {avg_train:.5f} | Val Loss: {avg_val:.5f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), f"{args.out_dir}/{prefix}_best_model.pth")
            print(status_str + " [BEST]")
        elif epoch % 5 == 0 or epoch == 1:
            print(status_str)

    print(f"\nTraining completed. Model saved to {args.out_dir}/{prefix}_best_model.pth")


if __name__ == "__main__":
    main()