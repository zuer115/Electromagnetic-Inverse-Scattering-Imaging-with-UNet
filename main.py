#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Electromagnetic Inverse Scattering Inference Engine
Supports 2D/3D full-aperture and limited-view reconstructions using
Distorted Born Approximation (DBA) and a Bilinear U-Net.
"""
import argparse
import sys


class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f'Error: {message}\n\n')
        self.print_help()
        sys.exit(2)


def main():
    parser = CustomArgumentParser(
        description="Electromagnetic Inverse Scattering Inference Engine"
    )
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Use CPU instead of CUDA')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to the input scattered field .npy file (required)')
    parser.add_argument('-w', '--weights', type=str, default='models/full_best_model.pth',
                        help='Path to the trained model .pth file (default: models/full_best_model.pth)')
    parser.add_argument('-o', '--out_dir', type=str, default='output',
                        help='Directory to save the results (default: output)')
    parser.add_argument('-d', '--dim', type=str, choices=['2d', '3d'], default='2d',
                        help='Data dimension: 2d or 3d (default: 2d)')
    parser.add_argument('-m', '--mode', type=str, choices=['full_circle', 'half_circle'], default='full_circle',
                        help='Antenna array configuration (default: full_circle)')
    parser.add_argument('-f', '--formats', nargs='+', default=['npy', 'png'],
                        help='List of output formats, e.g., npy png pdf svg (default: npy png)')
    parser.add_argument('-b', '--batch_size', type=int, default=16,
                        help='Inference batch size (default: 16)')
    parser.add_argument('--eps_bg', type=float, default=1.5,
                        help='Background relative permittivity (default: 1.5)')
    parser.add_argument('--bp', action='store_true', default=False,
                        help='Only use Physical Prior Extraction and do not use UNet')

    args = parser.parse_args()

    # ── Heavy imports deferred until after --help is handled ──
    import os
    import time
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    import torch
    import torch.nn as nn
    from scipy.special import hankel2 as scipy_hankel2
    import scipy.spatial

    # ==========================================
    # 0. Environment and Device Configuration
    # ==========================================
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')
    DTYPE = torch.complex64

    # ==========================================
    # 1. Physics Engine
    # ==========================================
    class ElectromagneticsEngine:
        def __init__(self, dim, mode, eps_bg=1.5):
            self.dim = dim
            self.mode = mode
            self.FREQ = 2.45e9
            self.C0 = 299792458.0
            self.K0 = 2 * np.pi * self.FREQ / self.C0
            self.EPS_BG = eps_bg
            self.K_BG = self.K0 * np.sqrt(self.EPS_BG)

            self.L = 0.25
            self.N = 128 if dim == '2d' else 32
            self.N_CELLS = self.N ** (2 if dim == '2d' else 3)
            self.DX = self.L / self.N
            self.N_ANT = 64
            self.R_ANT = 0.35

            self._init_geometry()

        def _init_geometry(self):
            print(f"[Physics Engine] Initializing {self.dim.upper()} spatial grid and Green's functions...")
            x_arr = np.linspace(-self.L/2 + self.DX/2, self.L/2 - self.DX/2, self.N)

            if self.dim == '2d':
                X, Y = np.meshgrid(x_arr, x_arr)
                r_domain_np = np.vstack((X.flatten(), Y.flatten())).T
                r_domain_gpu = torch.from_numpy(r_domain_np).to(device=device, dtype=torch.float32)
                dist_DD = torch.cdist(r_domain_gpu, r_domain_gpu)
                dist_DD.fill_diagonal_(1e-8)
                J0 = torch.special.bessel_j0(self.K_BG * dist_DD)
                Y0 = torch.special.bessel_y0(self.K_BG * dist_DD)
                G_DD_base_gpu = -1j / 4 * (J0 - 1j * Y0) * (self.DX**2)
                a_eq = self.DX / np.sqrt(np.pi)
                self_term = (1 / self.K_BG**2) * (-1j * np.pi * self.K_BG * a_eq / 2 * scipy_hankel2(1, self.K_BG * a_eq) + 1)
                G_DD_base_gpu.fill_diagonal_(complex(self_term))
                del dist_DD, J0, Y0
                torch.cuda.empty_cache()

                if self.mode == 'full_circle':
                    angles = torch.arange(0, self.N_ANT, dtype=torch.float32, device=device) * (2 * np.pi / self.N_ANT)
                else:
                    angles = torch.linspace(-np.pi/2, np.pi/2, self.N_ANT, device=device)
                self.r_ant = torch.stack((self.R_ANT * torch.cos(angles), self.R_ANT * torch.sin(angles)), dim=-1)
                self.G_DD = G_DD_base_gpu

            else:  # 3D
                if self.mode == 'half_circle':
                    raise ValueError("3D half_circle is not supported. Use full_circle for 3D data.")
                X, Y, Z = np.meshgrid(x_arr, x_arr, x_arr, indexing='ij')
                r_domain_np = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
                dist_DD = scipy.spatial.distance_matrix(r_domain_np, r_domain_np).astype(np.float32)
                G_DD_np = np.empty_like(dist_DD, dtype=np.complex64)
                with np.errstate(divide='ignore', invalid='ignore'):
                    np.divide(np.exp(-1j * self.K_BG * dist_DD), (4 * np.pi * dist_DD), out=G_DD_np)
                    G_DD_np *= (self.DX**3)

                a_eq = (3 * (self.DX**3) / (4 * np.pi))**(1/3)
                self_term = (1 / self.K_BG**2) * (1 - (1 + 1j * self.K_BG * a_eq) * np.exp(-1j * self.K_BG * a_eq))
                np.fill_diagonal(G_DD_np, self_term)

                # Fibonacci spherical lattice for 3D antenna array
                indices = torch.arange(0, self.N_ANT, dtype=torch.float32, device=device) + 0.5
                phi = torch.acos(1 - 2 * indices / self.N_ANT)
                theta = torch.pi * (1 + 5**0.5) * indices
                self.r_ant = torch.stack((self.R_ANT * torch.cos(theta) * torch.sin(phi),
                                          self.R_ANT * torch.sin(theta) * torch.sin(phi),
                                          self.R_ANT * torch.cos(phi)), dim=-1)
                self.G_DD = torch.from_numpy(G_DD_np).to(device=device, dtype=DTYPE)
            
            self.r_domain = torch.from_numpy(r_domain_np).to(device=device, dtype=torch.float32)

            # Precompute G_SD and E_inc
            dist_SD = torch.cdist(self.r_ant.unsqueeze(0), self.r_domain.unsqueeze(0)).squeeze(0)
            if self.dim == '2d':
                self.G_SD = -1j / 4 * (torch.special.bessel_j0(self.K_BG * dist_SD) - 1j * torch.special.bessel_y0(self.K_BG * dist_SD)) * (self.DX**2)
                self.E_inc = -1j / 4 * (torch.special.bessel_j0(self.K_BG * dist_SD.transpose(0, 1)) - 1j * torch.special.bessel_y0(self.K_BG * dist_SD.transpose(0, 1)))
            else:
                self.G_SD = torch.exp(-1j * self.K_BG * dist_SD) / (4 * np.pi * dist_SD) * (self.DX**3)
                self.E_inc = torch.exp(-1j * self.K_BG * dist_SD.transpose(0, 1)) / (4 * np.pi * dist_SD.transpose(0, 1))

            self.K_kernel = (self.K0**2 * (self.DX**2 if self.dim == '2d' else 1.0)) * self.G_SD.T.unsqueeze(2) * self.E_inc.unsqueeze(1)
            self.K_conj = torch.conj(self.K_kernel)

            # Background subtraction and sensitivity map computation
            eps_bg = torch.ones(self.N_CELLS, device=device, dtype=DTYPE) * self.EPS_BG
            chi_bg = eps_bg - 1.0
            A_bg = torch.eye(self.N_CELLS, device=device, dtype=DTYPE) - (self.K0**2) * self.G_DD * chi_bg.unsqueeze(0)
            E_tot_bg = torch.linalg.solve(A_bg, self.E_inc)
            self.E_sca_bg = (self.K0**2) * (self.G_SD @ (chi_bg.unsqueeze(1) * E_tot_bg))
            self.S_map = torch.sum(torch.abs(self.K_kernel)**2, dim=(1, 2))

        def run_db_bp(self, X_batch_np):
            with torch.no_grad():
                X_batch_gpu = torch.complex(
                    torch.from_numpy(X_batch_np[:, 0]).to(device),
                    torch.from_numpy(X_batch_np[:, 1]).to(device)
                )
                X_diff = X_batch_gpu - self.E_sca_bg.unsqueeze(0)
                bp_gpu = torch.einsum('b r t, d r t -> b d', X_diff, self.K_conj).abs()
                bp_gpu = bp_gpu / (self.S_map.unsqueeze(0) + 1e-6)

                # Amplitude scaling (must match training-time scaling)
                bp_gpu = bp_gpu * (500.0 if self.dim == '2d' else 1000.0)

                shape = (-1, 1, self.N, self.N) if self.dim == '2d' else (-1, 1, self.N, self.N, self.N)
                return bp_gpu.view(shape)

    # ==========================================
    # 2. Neural Network Architecture (U-Net)
    # ==========================================
    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels, is_3d=False):
            super().__init__()
            conv = nn.Conv3d if is_3d else nn.Conv2d
            bn = nn.BatchNorm3d if is_3d else nn.BatchNorm2d
            
            # [Bug Fixed]: Corrected variable name back to `self.double_conv` to strictly match state_dict keys.
            self.double_conv = nn.Sequential(
                conv(in_channels, out_channels, 3, padding=1, bias=False), bn(out_channels), nn.ReLU(inplace=True),
                conv(out_channels, out_channels, 3, padding=1, bias=False), bn(out_channels), nn.ReLU(inplace=True)
            )
        def forward(self, x): 
            return self.double_conv(x)

    class UpBlock(nn.Module):
        def __init__(self, in_channels, out_channels, is_3d=False):
            super().__init__()
            mode = 'trilinear' if is_3d else 'bilinear'
            conv = nn.Conv3d if is_3d else nn.Conv2d
            bn = nn.BatchNorm3d if is_3d else nn.BatchNorm3d if is_3d else nn.BatchNorm2d
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=mode, align_corners=True),
                conv(in_channels, in_channels // 2, 3, padding=1, bias=False),
                bn(in_channels // 2), nn.ReLU(inplace=True)
            )
            self.conv = DoubleConv(in_channels, out_channels, is_3d)
        def forward(self, x1, x2):
            return self.conv(torch.cat([x2, self.up(x1)], dim=1))

    class SimpleUNet(nn.Module):
        def __init__(self, dim='2d'):
            super().__init__()
            is_3d = (dim == '3d')
            features = [32, 64, 128, 256] if is_3d else [64, 128, 256, 512]
            pool = nn.MaxPool3d if is_3d else nn.MaxPool2d
            conv = nn.Conv3d if is_3d else nn.Conv2d

            self.downs = nn.ModuleList()
            self.ups = nn.ModuleList()
            self.pool = pool(2, 2)

            in_ch = 1
            for f in features:
                self.downs.append(DoubleConv(in_ch, f, is_3d))
                in_ch = f

            self.bottleneck = DoubleConv(features[-1], features[-1] * 2, is_3d)

            for f in reversed(features):
                self.ups.append(UpBlock(f * 2, f, is_3d))

            self.final_conv = nn.Sequential(conv(features[0], 1, 1), nn.Sigmoid())

        def forward(self, x):
            skips = []
            for down in self.downs:
                x = down(x)
                skips.append(x)
                x = self.pool(x)
            x = self.bottleneck(x)
            skips = skips[::-1]
            for i in range(len(self.ups)):
                x = self.ups[i](x, skips[i])
            return self.final_conv(x)

    def denormalize(img):
        return img * (5.0 - 1.5) + 1.5

    # ==========================================
    # 3. Main Inference Logic
    # ==========================================
    print(f"\n[INFO] Starting inference engine (Mode: {args.dim.upper()} {args.mode.upper()})")
    print(f"[INFO] Device: {device}")

    # Load data
    if not os.path.exists(args.input):
        sys.stderr.write(f"Error: Input file '{args.input}' not found.\n\n")
        parser.print_help()
        sys.exit(1)

    X_input = np.load(args.input)
    num_samples = X_input.shape[0]
    print(f"[INFO] Data loaded successfully. Total samples: {num_samples}.")

    # Initialize Engine and Model
    engine = ElectromagneticsEngine(args.dim, args.mode, args.eps_bg)
    
    if not args.bp:
        model = SimpleUNet(args.dim).to(device)

        if not os.path.exists(args.weights):
            sys.stderr.write(f"Error: Weights file '{args.weights}' not found.\n\n")
            parser.print_help()
            sys.exit(1)

        model.load_state_dict(torch.load(args.weights, map_location=device))
        model.eval()
        print(f"[INFO] Model weights loaded: {args.weights}")

    # Inference Loop
    os.makedirs(args.out_dir, exist_ok=True)
    predictions = []

    print("[INFO] Executing physical adjoint mapping and neural network reconstruction...")
    start_time = time.time()
    num_batches = int(np.ceil(num_samples / args.batch_size))

    with torch.no_grad():
        for b in range(num_batches):
            b_start = b * args.batch_size
            b_end = min((b + 1) * args.batch_size, num_samples)

            bp_tensor = engine.run_db_bp(X_input[b_start:b_end])
            pred_norm = bp_tensor.cpu().numpy().squeeze(1) if args.bp else model(bp_tensor).cpu().numpy().squeeze(1) 
            pred_phys = denormalize(pred_norm)

            predictions.append(pred_phys)
            print(f"Progress: {b_end}/{num_samples} [{(b_end/num_samples)*100:.1f}%]", end='\r')

    predictions = np.concatenate(predictions, axis=0)
    print(f"\n[INFO] Inference completed. Elapsed time: {time.time() - start_time:.2f} s")

    # Export results
    print("[INFO] Exporting results...")

    if 'npy' in args.formats:
        npy_path = os.path.join(args.out_dir, f"predictions_{args.dim}_{args.mode}.npy")
        np.save(npy_path, predictions)
        print(f"  -> Raw matrix saved to: {npy_path}")

    img_formats = [fmt for fmt in args.formats if fmt.lower() in ['png', 'pdf', 'svg', 'jpg']]

    if img_formats:
        print(f"  -> Generating visualization plots...")
        img_out_dir = os.path.join(args.out_dir, "images")
        os.makedirs(img_out_dir, exist_ok=True)

        for i in range(num_samples):
            fig, ax = plt.subplots(figsize=(5, 5))

            if args.dim == '2d':
                img_data = predictions[i]
                title = f"2D Reconstruction #{i}"
            else:  # 3D: Find max response slice along Z-axis
                vol = predictions[i]
                z_slice_idx = np.unravel_index(np.argmax(vol), vol.shape)[2]
                img_data = vol[:, :, z_slice_idx]
                title = f"3D Tomographic Slice #{i} (Z={z_slice_idx}/{engine.N})"

            im = ax.imshow(img_data, extent=[-engine.L/2, engine.L/2, -engine.L/2, engine.L/2],
                           origin='lower', cmap='jet', vmin=1.5, vmax=5.0)
            ax.set_title(title)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            for fmt in img_formats:
                save_path = os.path.join(img_out_dir, f"sample_{i:04d}.{fmt}")
                fig.savefig(save_path, dpi=200 if fmt in ['png', 'jpg'] else None, bbox_inches='tight')

            plt.close(fig)
            if (i+1) % 10 == 0 or (i+1) == num_samples:
                print(f"    Export progress: {i+1}/{num_samples}", end='\r')

        print(f"\n  -> All images saved to: {img_out_dir}")

    print("[INFO] Processing finished.")


if __name__ == "__main__":
    main()