# ==========================================
# 3D U-Net Training Pipeline
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
            "║      3D U-Net Training Pipeline                              ║\n"
            "║      Trains Trilinear 3D U-Net on 3D Born BP volumes         ║\n"
            "╚══════════════════════════════════════════════════════════════╝\n\n"
        )
        return header + super().format_help()


def main():
    parser = CustomArgumentParser(
        description="3D U-Net training pipeline with Born BP physical prior and TV loss"
    )
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Use CPU instead of CUDA')
    parser.add_argument('--eps_bg', type=float, default=1.5,
                        help='Background relative permittivity (default: 1.5)')
    parser.add_argument('-o', '--out_dir', type=str, default='models',
                        help='Output directory for model weights (default: models)')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Training batch size (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=250,
                        help='Number of training epochs (default: 250)')
    parser.add_argument('--train_path', type=str, default='train_data/sphere_train',
                        help='Path stem for training data (default: train_data/sphere_train)')
    parser.add_argument('--test_path', type=str, default='train_data/sphere_test',
                        help='Path stem for test data (default: train_data/sphere_test)')
    args = parser.parse_args()

    # ── Heavy imports deferred until after --help is handled ──
    import os
    import time
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader

    # ==========================================
    # 0. Environment and Device Configuration
    # ==========================================
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')

    # ==========================================
    # 1. 3D Global Physical Parameters
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

    r_domain_gpu = torch.empty(0)

    # ==========================================
    # 2. Custom 3D Smooth Value Loss (with TV regularization)
    # ==========================================
    class TVLoss3D(nn.Module):
        """3D Total Variation regularization for smoothing voxel artifacts."""
        def __init__(self):
            super().__init__()

        def forward(self, x):
            d_tv = torch.mean(torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :]))
            h_tv = torch.mean(torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]))
            w_tv = torch.mean(torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]))
            return d_tv + h_tv + w_tv

    class SmoothValueLoss3D(nn.Module):
        def __init__(self, tumor_weight=5.0):
            super().__init__()
            self.tumor_weight = tumor_weight
            self.mse = nn.MSELoss(reduction='none')
            self.l1 = nn.L1Loss(reduction='none')
            self.tv = TVLoss3D()

        def forward(self, pred, target):
            weight = torch.where(target > 0.05, self.tumor_weight, 1.0)

            mse_loss = torch.mean(weight * self.mse(pred, target))
            l1_loss = torch.mean(weight * self.l1(pred, target))
            tv_loss = self.tv(pred)

            return 1.0 * mse_loss + 0.5 * l1_loss + 0.2 * tv_loss

    # ==========================================
    # 3. 3D Physical Prior Adjoint Operator (3D Born BP)
    # ==========================================
    def precompute_physical_bp_3d(X_np):
        print("Computing 3D physical prior (3D Distorted Born BP)...")
        start = time.time()
        num_samples = X_np.shape[0]

        x_arr = np.linspace(-L/2 + DX/2, L/2 - DX/2, N)
        X, Y, Z = np.meshgrid(x_arr, x_arr, x_arr, indexing='ij')
        r_domain_np = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
        global r_domain_gpu
        r_domain_gpu = torch.from_numpy(r_domain_np).to(device=device, dtype=torch.float32)

        # Fibonacci spherical antenna array
        indices = torch.arange(0, N_ANT, dtype=torch.float32, device=device) + 0.5
        phi = torch.acos(1 - 2 * indices / N_ANT)
        theta = torch.pi * (1 + 5**0.5) * indices
        ant_x = R_ANT * torch.cos(theta) * torch.sin(phi)
        ant_y = R_ANT * torch.sin(theta) * torch.sin(phi)
        ant_z = R_ANT * torch.cos(phi)
        r_ant_gpu = torch.stack((ant_x, ant_y, ant_z), dim=-1)

        dist_SD = torch.cdist(r_ant_gpu.unsqueeze(0), r_domain_gpu.unsqueeze(0)).squeeze(0)

        G_SD = torch.exp(-1j * K_BG * dist_SD) / (4 * np.pi * dist_SD) * (DX**3)
        E_inc = torch.exp(-1j * K_BG * dist_SD.T) / (4 * np.pi * dist_SD.T)

        K = (K0**2) * G_SD.T.unsqueeze(2) * E_inc.unsqueeze(1)
        K_conj = torch.conj(K)
        S = torch.sum(torch.abs(K)**2, dim=(1, 2))

        BP_all = np.zeros((num_samples, 1, N, N, N), dtype=np.float32)
        bp_inner_batch = 128
        num_batches = int(np.ceil(num_samples / bp_inner_batch))
        for b in range(num_batches):
            b_start = b * bp_inner_batch
            b_end = min((b + 1) * bp_inner_batch, num_samples)

            X_batch_gpu = torch.complex(
                torch.from_numpy(X_np[b_start:b_end, 0]).to(device),
                torch.from_numpy(X_np[b_start:b_end, 1]).to(device)
            )

            bp_gpu = torch.einsum('b r t, d r t -> b d', X_batch_gpu, K_conj).abs()
            bp_gpu = bp_gpu / (S.unsqueeze(0) + 1e-6)
            bp_gpu = bp_gpu * 1000.0

            BP_all[b_start:b_end] = bp_gpu.view(-1, 1, N, N, N).cpu().numpy()

        print(f"3D physical prior computed. Time: {time.time() - start:.2f}s")
        return BP_all

    # ==========================================
    # 4. Trilinear 3D U-Net Architecture
    # ==========================================
    class DoubleConv3D(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.double_conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
        def forward(self, x):
            return self.double_conv(x)

    class UpBlock3D(nn.Module):
        """Trilinear interpolation upsampling block."""
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(in_channels // 2),
                nn.ReLU(inplace=True)
            )
            self.conv = DoubleConv3D(in_channels, out_channels)

        def forward(self, x1, x2):
            x1 = self.up(x1)
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)

    class SimpleUNet3D(nn.Module):
        def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
            super().__init__()
            self.downs = nn.ModuleList()
            self.ups = nn.ModuleList()
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

            in_ch = in_channels
            for feature in features:
                self.downs.append(DoubleConv3D(in_ch, feature))
                in_ch = feature

            self.bottleneck = DoubleConv3D(features[-1], features[-1] * 2)

            for feature in reversed(features):
                self.ups.append(UpBlock3D(feature * 2, feature))

            self.final_conv = nn.Sequential(
                nn.Conv3d(features[0], out_channels, kernel_size=1),
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
    # 5. Main Training Loop
    # ==========================================
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR_MAX = 8e-4

    print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    print("Loading [3D SPHERE] dataset...")
    X_train_raw = np.load(f'{args.train_path}_X.npy')
    Y_train_raw = np.load(f'{args.train_path}_Y.npy')
    X_test = np.load(f'{args.test_path}_X.npy')
    Y_test = np.load(f'{args.test_path}_Y.npy')

    Y_train_raw = (Y_train_raw - EPS_BG) / (5.0 - EPS_BG)
    Y_test = (Y_test - EPS_BG) / (5.0 - EPS_BG)

    X_train_bp = precompute_physical_bp_3d(X_train_raw)
    X_test_bp  = precompute_physical_bp_3d(X_test)

    train_dataset = TensorDataset(torch.from_numpy(X_train_bp), torch.from_numpy(Y_train_raw).unsqueeze(1))
    test_dataset  = TensorDataset(torch.from_numpy(X_test_bp), torch.from_numpy(Y_test).unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = SimpleUNet3D().to(device)

    criterion = SmoothValueLoss3D(tumor_weight=5.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)

    best_val_loss = float('inf')
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n" + "="*50)
    print("Training [3D SPHERE] network (Trilinear U-Net with TV Loss)")
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
                loss = criterion(outputs, targets)

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
                    loss = criterion(outputs, targets)
                val_loss_list.append(loss.item())

        avg_train = np.mean(train_loss_list)
        avg_val = np.mean(val_loss_list)
        epoch_time = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        status_str = (f"Epoch [{epoch:03d}/{EPOCHS}] | Time: {epoch_time:.1f}s | "
                      f"LR: {current_lr:.1e} | Train Loss: {avg_train:.5f} | Val Loss: {avg_val:.5f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), f"{args.out_dir}/sphere_best_model.pth")
            print(status_str + " [BEST]")
        elif epoch % 5 == 0 or epoch == 1:
            print(status_str)

    print(f"\nTraining completed. Model saved to {args.out_dir}/sphere_best_model.pth")


if __name__ == "__main__":
    main()