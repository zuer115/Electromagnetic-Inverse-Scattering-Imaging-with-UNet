# ==========================================
# 3D U-Net Evaluation: Slice and Voxel Plots
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
            "║      3D U-Net Evaluation                                     ║\n"
            "║      Generates 2D slice plots and 3D voxel renderings        ║\n"
            "╚══════════════════════════════════════════════════════════════╝\n\n"
        )
        return header + super().format_help()


def main():
    parser = CustomArgumentParser(
        description="3D U-Net evaluation: 2D slice plots and 3D voxel renderings"
    )
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Use CPU instead of CUDA')
    parser.add_argument('--eps_bg', type=float, default=1.5,
                        help='Background relative permittivity (default: 1.5)')
    parser.add_argument('-o', '--out_dir', type=str, default='output',
                        help='Output directory for plots (default: output)')
    parser.add_argument('--test_path', type=str, default='train_data/sphere_test',
                        help='Path stem for test data (default: train_data/sphere_test)')
    parser.add_argument('-w', '--model_path', type=str, default='models/sphere_best_model.pth',
                        help='Path to model weights (default: models/sphere_best_model.pth)')
    parser.add_argument('-n', '--num_images', type=int, default=3,
                        help='Number of sample images (default: 3)')
    parser.add_argument('--no_slices', action='store_true', default=False,
                        help='Skip 2D slice plots')
    parser.add_argument('--no_voxels', action='store_true', default=False,
                        help='Skip 3D voxel plots')
    args = parser.parse_args()

    # ── Heavy imports deferred until after --help is handled ──
    import os
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    import torch
    import torch.nn as nn

    device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')

    # ==========================================
    # 1. 3D Physics and Grid Parameters
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

    x_arr = np.linspace(-L/2 + DX/2, L/2 - DX/2, N)
    X, Y, Z = np.meshgrid(x_arr, x_arr, x_arr, indexing='ij')
    r_domain_np = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T

    # ==========================================
    # 2. Self-Contained 3D Physical Prior (Born BP)
    # ==========================================
    def self_contained_bp_3d(X_np):
        print("Computing 3D physical prior (Born BP)...")
        num_samples = X_np.shape[0]

        indices = torch.arange(0, N_ANT, dtype=torch.float32, device=device) + 0.5
        phi = torch.acos(1 - 2 * indices / N_ANT)
        theta = torch.pi * (1 + 5**0.5) * indices
        ant_x = R_ANT * torch.cos(theta) * torch.sin(phi)
        ant_y = R_ANT * torch.sin(theta) * torch.sin(phi)
        ant_z = R_ANT * torch.cos(phi)
        r_ant_gpu = torch.stack((ant_x, ant_y, ant_z), dim=-1)

        r_domain_gpu = torch.from_numpy(r_domain_np).to(device=device, dtype=torch.float32)
        dist_SD = torch.cdist(r_ant_gpu.unsqueeze(0), r_domain_gpu.unsqueeze(0)).squeeze(0)

        G_SD = torch.exp(-1j * K_BG * dist_SD) / (4 * np.pi * dist_SD) * (DX**3)
        E_inc = torch.exp(-1j * K_BG * dist_SD.T) / (4 * np.pi * dist_SD.T)

        K = (K0**2) * G_SD.T.unsqueeze(2) * E_inc.unsqueeze(1)
        K_conj = torch.conj(K)
        S = torch.sum(torch.abs(K)**2, dim=(1, 2))

        BP_all = np.zeros((num_samples, 1, N, N, N), dtype=np.float32)
        for b in range(num_samples):
            X_batch_gpu = torch.complex(
                torch.from_numpy(X_np[b:b+1, 0]).to(device),
                torch.from_numpy(X_np[b:b+1, 1]).to(device)
            )
            bp_gpu = torch.einsum('b r t, d r t -> b d', X_batch_gpu, K_conj).abs()
            bp_gpu = bp_gpu / (S.unsqueeze(0) + 1e-6)

            # Amplify physical features
            bp_gpu = bp_gpu * 1000.0

            BP_all[b:b+1] = bp_gpu.view(-1, 1, N, N, N).cpu().numpy()

        return BP_all

    # ==========================================
    # 3. Trilinear 3D U-Net Architecture
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

    def denormalize(img):
        return img * (5.0 - EPS_BG) + EPS_BG

    # ==========================================
    # 4. 3D Voxel Rendering Function
    # ==========================================
    def plot_3d_voxels(ax, vol, title, threshold=1.7):
        """
        Render a 3D matrix as transparent voxels in physical space.
        """
        edge_x = np.linspace(-L/2, L/2, N+1)
        X_edge, Y_edge, Z_edge = np.meshgrid(edge_x, edge_x, edge_x, indexing='ij')

        mask = vol > threshold

        norm = mcolors.Normalize(vmin=EPS_BG, vmax=5.0)
        cmap = plt.get_cmap('jet')

        colors = cmap(norm(vol))
        colors[..., 3] = 0.7

        ax.voxels(X_edge, Y_edge, Z_edge, mask, facecolors=colors, edgecolors=None)

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("X (m)", fontsize=9)
        ax.set_ylabel("Y (m)", fontsize=9)
        ax.set_zlabel("Z (m)", fontsize=9)

        ax.set_box_aspect([1, 1, 1])

    # ==========================================
    # 5. 3D Evaluation Entry Point
    # ==========================================
    model = SimpleUNet3D().to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"3D model weights loaded: {args.model_path}")
    except Exception as e:
        print(f"Failed to load 3D model from '{args.model_path}': {e}")
        return

    model.eval()
    print("Loading 3D test data...")
    X_test = np.load(f'{args.test_path}_X.npy')
    Y_test = np.load(f'{args.test_path}_Y.npy')

    sample_indices = list(range(min(args.num_images, X_test.shape[0])))

    X_bp_3d = self_contained_bp_3d(X_test[sample_indices])

    with torch.no_grad():
        inputs = torch.from_numpy(X_bp_3d).to(device)
        pred_norm = model(inputs)
        pred_norm = torch.clamp(pred_norm, 0.0, 1.0).cpu().numpy().squeeze(1)

    Y_gt_3d = Y_test[sample_indices]
    pred_3d = denormalize(pred_norm)
    bp_3d   = denormalize(X_bp_3d.squeeze(1))

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # Task A: 2D Slice Plots
    # ---------------------------------------------------------------
    if not args.no_slices:
        print("Generating 3D slice plots...")
        n_samples = len(sample_indices)
        fig_slice, axes_slice = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
        if n_samples == 1:
            axes_slice = axes_slice.reshape(1, -1)

        for row_idx, s_idx in enumerate(sample_indices):
            gt_vol = Y_gt_3d[row_idx]
            bp_vol = bp_3d[row_idx]
            pred_vol = pred_3d[row_idx]

            # Find tumor center slice
            max_idx = np.unravel_index(np.argmax(gt_vol), gt_vol.shape)
            z_slice_idx = max_idx[2]

            gt_slice = gt_vol[:, :, z_slice_idx]
            bp_slice = bp_vol[:, :, z_slice_idx]
            pred_slice = pred_vol[:, :, z_slice_idx]
            err_slice = np.abs(gt_slice - pred_slice)

            im0 = axes_slice[row_idx, 0].imshow(gt_slice, extent=[-L/2, L/2, -L/2, L/2],
                                                 origin='lower', cmap='jet', vmin=EPS_BG, vmax=5.0)
            axes_slice[row_idx, 0].set_title(f"3D GT (Z-Slice: {z_slice_idx}/{N})")
            fig_slice.colorbar(im0, ax=axes_slice[row_idx, 0], fraction=0.046, pad=0.04)

            im1 = axes_slice[row_idx, 1].imshow(bp_slice, extent=[-L/2, L/2, -L/2, L/2],
                                                 origin='lower', cmap='jet')
            axes_slice[row_idx, 1].set_title("3D Born BP")
            fig_slice.colorbar(im1, ax=axes_slice[row_idx, 1], fraction=0.046, pad=0.04)

            im2 = axes_slice[row_idx, 2].imshow(pred_slice, extent=[-L/2, L/2, -L/2, L/2],
                                                 origin='lower', cmap='jet', vmin=EPS_BG, vmax=5.0)
            axes_slice[row_idx, 2].set_title("Trilinear 3D U-Net")
            fig_slice.colorbar(im2, ax=axes_slice[row_idx, 2], fraction=0.046, pad=0.04)

            im3 = axes_slice[row_idx, 3].imshow(err_slice, extent=[-L/2, L/2, -L/2, L/2],
                                                 origin='lower', cmap='hot', vmin=0, vmax=1.0)
            axes_slice[row_idx, 3].set_title("Absolute Error")
            fig_slice.colorbar(im3, ax=axes_slice[row_idx, 3], fraction=0.046, pad=0.04)

        for ax in axes_slice.flat:
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")

        fig_slice.tight_layout()
        slice_path = os.path.join(args.out_dir, 'evaluation_3d_slices.png')
        fig_slice.savefig(slice_path, dpi=600)
        plt.close(fig_slice)
        print(f"Slice plots saved to: {slice_path}")
    else:
        print("Skipping slice plots (--no_slices).")

    # ---------------------------------------------------------------
    # Task B: True 3D Voxel Rendering
    # ---------------------------------------------------------------
    if not args.no_voxels:
        print("Generating 3D voxel renderings...")
        n_samples = len(sample_indices)
        fig_vox, axes_vox = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples),
                                         subplot_kw={'projection': '3d'})
        if n_samples == 1:
            axes_vox = axes_vox.reshape(1, -1)

        for row_idx, s_idx in enumerate(sample_indices):
            gt_vol = Y_gt_3d[row_idx]
            bp_vol = bp_3d[row_idx]
            pred_vol = pred_3d[row_idx]

            plot_3d_voxels(axes_vox[row_idx, 0], gt_vol, f"3D Ground Truth #{s_idx}", threshold=1.7)

            bp_thresh = bp_vol.min() + 0.5 * (bp_vol.max() - bp_vol.min())
            plot_3d_voxels(axes_vox[row_idx, 1], bp_vol, "3D Born BP", threshold=bp_thresh)

            plot_3d_voxels(axes_vox[row_idx, 2], pred_vol, "Trilinear 3D U-Net", threshold=1.7)

        fig_vox.tight_layout()
        voxel_path = os.path.join(args.out_dir, 'evaluation_3d_voxels.png')
        fig_vox.savefig(voxel_path, dpi=600)
        plt.close(fig_vox)
        print(f"Voxel plots saved to: {voxel_path}")
    else:
        print("Skipping voxel plots (--no_voxels).")

    print(f"All 3D plots saved to {args.out_dir}/.")


if __name__ == "__main__":
    main()