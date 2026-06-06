# ==========================================
# 2D U-Net Evaluation: GT vs BP vs Prediction
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
            "║      2D U-Net Evaluation                                     ║\n"
            "║      Generates GT vs BP vs Prediction comparison plots       ║\n"
            "╚══════════════════════════════════════════════════════════════╝\n\n"
        )
        return header + super().format_help()


def main():
    parser = CustomArgumentParser(
        description="2D U-Net evaluation: GT vs Born BP vs Prediction comparison plots"
    )
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Use CPU instead of CUDA')
    parser.add_argument('--eps_bg', type=float, default=1.5,
                        help='Background relative permittivity (default: 1.5)')
    parser.add_argument('-o', '--out_dir', type=str, default='output',
                        help='Output directory for plots (default: output)')
    parser.add_argument('--test_full_path', type=str, default='train_data/full_test',
                        help='Path stem for full-circle test data (default: train_data/full_test)')
    parser.add_argument('--test_half_path', type=str, default='train_data/half_test',
                        help='Path stem for half-circle test data (default: train_data/half_test)')
    parser.add_argument('--full_model', type=str, default='models/full_best_model.pth',
                        help='Path to full-circle model weights (default: models/full_best_model.pth)')
    parser.add_argument('--half_model', type=str, default='models/half_best_model.pth',
                        help='Path to half-circle model weights (default: models/half_best_model.pth)')
    parser.add_argument('-n', '--num_images', type=int, default=3,
                        help='Number of sample images per array type (default: 3)')
    parser.add_argument('--no_full', action='store_true', default=False,
                        help='Skip full-circle evaluation')
    parser.add_argument('--no_half', action='store_true', default=False,
                        help='Skip half-circle evaluation')
    args = parser.parse_args()

    # ── Heavy imports deferred until after --help is handled ──
    import os
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    import torch
    import torch.nn as nn
    from scipy.special import hankel2 as scipy_hankel2
    from scipy.spatial import distance_matrix

    device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')

    L, N, N_ANT, R_ANT, FREQ = 0.25, 128, 64, 0.35, 2.45e9
    DX = L / N
    DTYPE = torch.complex64
    C0 = 299792458.0
    K0 = 2 * np.pi * FREQ / C0
    EPS_BG = args.eps_bg
    K_BG = K0 * np.sqrt(EPS_BG)
    
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

    def self_contained_bp(X_np, array_type):
        num_samples = X_np.shape[0]
        N_CELLS = N * N

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
        bp_batch = 256
        num_batches = int(np.ceil(num_samples / bp_batch))
        for b in range(num_batches):
            b_start = b * bp_batch
            b_end = min((b + 1) * bp_batch, num_samples)

            X_batch_gpu = torch.complex(
                torch.from_numpy(X_np[b_start:b_end, 0]).to(device),
                torch.from_numpy(X_np[b_start:b_end, 1]).to(device)
            )

            X_diff = X_batch_gpu - E_sca_bg.unsqueeze(0)
            bp_gpu = torch.einsum('b r t, d r t -> b d', X_diff, K_conj).abs()
            bp_gpu = bp_gpu / (S.unsqueeze(0) + 1e-6)

            # Matching training-time scaling
            bp_gpu = bp_gpu * 500.0

            BP_all[b_start:b_end] = bp_gpu.view(-1, 1, N, N).cpu().numpy()

        return BP_all

    # ==========================================
    # Bilinear U-Net (matches train.py architecture)
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

    def denormalize(img):
        return img * (5.0 - EPS_BG) + EPS_BG

    # ==========================================
    # Evaluation entry point
    # ==========================================
    model_full = SimpleUNet().to(device)
    model_half = SimpleUNet().to(device)
    has_full = False
    has_half = False

    if not args.no_full:
        try:
            model_full.load_state_dict(torch.load(args.full_model, map_location=device))
            has_full = True
            print(f"Full-circle model loaded: {args.full_model}")
        except Exception as e:
            print(f"Failed to load full-circle model from '{args.full_model}': {e}")
    else:
        print("Full-circle evaluation skipped (--no_full).")

    if not args.no_half:
        try:
            model_half.load_state_dict(torch.load(args.half_model, map_location=device))
            has_half = True
            print(f"Half-circle model loaded: {args.half_model}")
        except Exception as e:
            print(f"Failed to load half-circle model from '{args.half_model}': {e}")
    else:
        print("Half-circle evaluation skipped (--no_half).")

    if not has_full and not has_half:
        print("No models loaded. Exiting.")
        return

    model_full.eval()
    model_half.eval()

    print("Loading test data...")
    sample_indices = list(range(min(args.num_images, 10000)))  # safeguard upper bound

    n_rows = 0
    if has_full:
        n_rows += len(sample_indices)
    if has_half:
        n_rows += len(sample_indices)
    n_cols = 4  # GT, BP, Pred, Error

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    current_row = 0

    if has_full:
        X_test_full = np.load(f'{args.test_full_path}_X.npy')
        Y_test_full = np.load(f'{args.test_full_path}_Y.npy')
        X_bp_full = self_contained_bp(X_test_full[sample_indices], 'full_circle')

        with torch.no_grad():
            pred_full_norm = model_full(torch.from_numpy(X_bp_full).to(device)).cpu().numpy().squeeze(1)

        Y_gt_full = Y_test_full[sample_indices]
        pred_full = denormalize(pred_full_norm)

        for row_idx, s_idx in enumerate(sample_indices):
            gt, bp, pred = Y_gt_full[row_idx], X_bp_full[row_idx, 0], pred_full[row_idx]
            err = np.abs(gt - pred)

            im0 = axes[current_row, 0].imshow(gt, extent=[-L/2, L/2, -L/2, L/2],
                                               origin='lower', cmap='jet', vmin=EPS_BG, vmax=5.0)
            axes[current_row, 0].set_title(f"Full-Circle GT #{s_idx}")
            fig.colorbar(im0, ax=axes[current_row, 0], fraction=0.046, pad=0.04)

            im1 = axes[current_row, 1].imshow(bp, extent=[-L/2, L/2, -L/2, L/2],
                                               origin='lower', cmap='jet')
            axes[current_row, 1].set_title("Absolute Born BP")
            fig.colorbar(im1, ax=axes[current_row, 1], fraction=0.046, pad=0.04)

            im2 = axes[current_row, 2].imshow(pred, extent=[-L/2, L/2, -L/2, L/2],
                                               origin='lower', cmap='jet', vmin=EPS_BG, vmax=5.0)
            axes[current_row, 2].set_title("Bilinear U-Net")
            fig.colorbar(im2, ax=axes[current_row, 2], fraction=0.046, pad=0.04)

            im3 = axes[current_row, 3].imshow(err, extent=[-L/2, L/2, -L/2, L/2],
                                               origin='lower', cmap='hot', vmin=0, vmax=1.0)
            axes[current_row, 3].set_title("Absolute Error")
            fig.colorbar(im3, ax=axes[current_row, 3], fraction=0.046, pad=0.04)

            current_row += 1

    if has_half:
        X_test_half = np.load(f'{args.test_half_path}_X.npy')
        Y_test_half = np.load(f'{args.test_half_path}_Y.npy')
        X_bp_half = self_contained_bp(X_test_half[sample_indices], 'half_circle')

        with torch.no_grad():
            pred_half_norm = model_half(torch.from_numpy(X_bp_half).to(device)).cpu().numpy().squeeze(1)

        Y_gt_half = Y_test_half[sample_indices]
        pred_half = denormalize(pred_half_norm)

        for row_idx, s_idx in enumerate(sample_indices):
            gt, bp, pred = Y_gt_half[row_idx], X_bp_half[row_idx, 0], pred_half[row_idx]
            err = np.abs(gt - pred)

            im0 = axes[current_row, 0].imshow(gt, extent=[-L/2, L/2, -L/2, L/2],
                                               origin='lower', cmap='jet', vmin=EPS_BG, vmax=5.0)
            axes[current_row, 0].set_title(f"Half-Circle GT #{s_idx}")
            fig.colorbar(im0, ax=axes[current_row, 0], fraction=0.046, pad=0.04)

            im1 = axes[current_row, 1].imshow(bp, extent=[-L/2, L/2, -L/2, L/2],
                                               origin='lower', cmap='jet')
            axes[current_row, 1].set_title("Absolute Born BP")
            fig.colorbar(im1, ax=axes[current_row, 1], fraction=0.046, pad=0.04)

            im2 = axes[current_row, 2].imshow(pred, extent=[-L/2, L/2, -L/2, L/2],
                                               origin='lower', cmap='jet', vmin=EPS_BG, vmax=5.0)
            axes[current_row, 2].set_title("Bilinear U-Net")
            fig.colorbar(im2, ax=axes[current_row, 2], fraction=0.046, pad=0.04)

            im3 = axes[current_row, 3].imshow(err, extent=[-L/2, L/2, -L/2, L/2],
                                               origin='lower', cmap='hot', vmin=0, vmax=1.0)
            axes[current_row, 3].set_title("Absolute Error")
            fig.colorbar(im3, ax=axes[current_row, 3], fraction=0.046, pad=0.04)

            current_row += 1

    for ax in axes.flat:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    plt.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    output_path = os.path.join(args.out_dir, 'evaluation_comparison.png')
    plt.savefig(output_path, dpi=600)
    print(f"Evaluation plot saved to: {output_path}")


if __name__ == "__main__":
    main()