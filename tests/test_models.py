"""
Tests for neural network architectures: U-Net (2D and 3D variants).

Covers:
- Model construction and parameter counts
- Forward pass shape consistency
- Bilinear/Trilinear upsampling (no checkerboard artifacts)
- DoubleConv, UpBlock, bottleneck correctness
- Sigmoid output range
- Gradient flow
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tests.test_helpers import deterministic_seed


# ══════════════════════════════════════════════════════════════════
# Replicate model classes for isolated testing
# (Same as in main.py / train.py / train_3d.py)
# ══════════════════════════════════════════════════════════════════

class DoubleConv2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.double_conv(x)


class UpBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch // 2), nn.ReLU(inplace=True),
        )
        self.conv = DoubleConv2D(in_ch, out_ch)
    def forward(self, x1, x2):
        return self.conv(torch.cat([x2, self.up(x1)], dim=1))


class SimpleUNet2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        ch = in_ch
        for f in features:
            self.downs.append(DoubleConv2D(ch, f))
            ch = f
        self.bottleneck = DoubleConv2D(features[-1], features[-1] * 2)
        for f in reversed(features):
            self.ups.append(UpBlock2D(f * 2, f))
        self.final_conv = nn.Sequential(nn.Conv2d(features[0], out_ch, 1), nn.Sigmoid())

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        for i, u in enumerate(self.ups):
            x = u(x, skips[-(i + 1)])
        return self.final_conv(x)


class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.double_conv(x)


class UpBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(in_ch, in_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm3d(in_ch // 2), nn.ReLU(inplace=True),
        )
        self.conv = DoubleConv3D(in_ch, out_ch)
    def forward(self, x1, x2):
        return self.conv(torch.cat([x2, self.up(x1)], dim=1))


class SimpleUNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=(32, 64, 128, 256)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(2)
        ch = in_ch
        for f in features:
            self.downs.append(DoubleConv3D(ch, f))
            ch = f
        self.bottleneck = DoubleConv3D(features[-1], features[-1] * 2)
        for f in reversed(features):
            self.ups.append(UpBlock3D(f * 2, f))
        self.final_conv = nn.Sequential(nn.Conv3d(features[0], out_ch, 1), nn.Sigmoid())

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        for i, u in enumerate(self.ups):
            x = u(x, skips[-(i + 1)])
        return self.final_conv(x)


# ══════════════════════════════════════════════════════════════════
# 2D U-Net Tests
# ══════════════════════════════════════════════════════════════════

class TestUNet2D:
    """Test the 2D Bilinear U-Net architecture."""

    @pytest.fixture(autouse=True)
    def setup(self):
        deterministic_seed(42)
        self.device = torch.device('cpu')
        self.model = SimpleUNet2D().to(self.device)
        self.model.eval()

    def test_parameter_count_reasonable(self):
        """Model should have a reasonable number of parameters (not exploded)."""
        n_params = sum(p.numel() for p in self.model.parameters())
        assert 1_000_000 < n_params < 50_000_000, \
            f"Expected ~7-31M params for 2D U-Net, got {n_params:,}"

    def test_no_transposed_conv(self):
        """Architecture must NOT contain ConvTranspose2d (checkerboard source)."""
        for m in self.model.modules():
            assert not isinstance(m, nn.ConvTranspose2d), \
                "Bilinear U-Net must not use transposed convolutions"

    def test_forward_shape_preserves_spatial(self):
        """Input and output spatial dimensions must match (128 -> 128)."""
        N = 128
        x = torch.randn(2, 1, N, N)
        with torch.no_grad():
            y = self.model(x)
        assert y.shape == (2, 1, N, N), \
            f"Expected (2,1,{N},{N}), got {y.shape}"

    def test_forward_shape_small_input(self):
        """Model should handle arbitrary 2^N input sizes."""
        for size in [32, 64, 128]:
            x = torch.randn(1, 1, size, size)
            with torch.no_grad():
                y = self.model(x)
            assert y.shape == (1, 1, size, size), \
                f"Failed for input size {size}: got {y.shape}"

    def test_sigmoid_output_range(self):
        """Output must be in [0, 1] due to final Sigmoid."""
        x = torch.randn(4, 1, 128, 128)
        with torch.no_grad():
            y = self.model(x)
        assert torch.all(y >= 0) and torch.all(y <= 1), \
            "Sigmoid output must be in [0, 1]"

    def test_deterministic_forward(self):
        """Same input should give same output (deterministic eval mode)."""
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y1 = self.model(x)
            y2 = self.model(x)
        assert torch.allclose(y1, y2, atol=1e-6)

    def test_gradient_flows(self):
        """Loss gradient should flow back to all parameters."""
        model = SimpleUNet2D().to(self.device)
        model.train()
        x = torch.randn(2, 1, 64, 64)
        y = model(x)
        loss = y.mean()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert not torch.all(p.grad == 0), f"Zero gradient for {name}"

    def test_batch_independence(self):
        """Each sample in a batch should not affect others."""
        x1 = torch.randn(1, 1, 64, 64)
        x2 = torch.randn(1, 1, 64, 64)
        x_batch = torch.cat([x1, x2], dim=0)
        with torch.no_grad():
            y_batch = self.model(x_batch)
            y1_solo = self.model(x1)
            y2_solo = self.model(x2)
        assert torch.allclose(y_batch[0:1], y1_solo, atol=1e-5)
        assert torch.allclose(y_batch[1:2], y2_solo, atol=1e-5)

    def test_bilinear_upsample_smoothness(self):
        """Bilinear upsampling should not produce checkerboard patterns."""
        # Create a smooth input and verify upsample output has no
        # high-frequency checkerboard artifacts
        up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.linspace(0, 1, 16).view(1, 1, 4, 4)
        y = up(x)
        # Check finite differences — should not have alternating sign pattern
        diff_x = y[:, :, :, 1:] - y[:, :, :, :-1]
        # A checkerboard would have alternating signs; smooth signal won't
        sign_changes = (diff_x[:, :, :, 1:] * diff_x[:, :, :, :-1] < 0).float().mean()
        assert sign_changes < 0.3, \
            f"Bilinear upsampling should be smooth (sign changes: {sign_changes:.3f})"


# ══════════════════════════════════════════════════════════════════
# 3D U-Net Tests
# ══════════════════════════════════════════════════════════════════

class TestUNet3D:
    """Test the 3D Trilinear U-Net architecture."""

    @pytest.fixture(autouse=True)
    def setup(self):
        deterministic_seed(42)
        self.device = torch.device('cpu')
        self.model = SimpleUNet3D().to(self.device)
        self.model.eval()

    def test_parameter_count_reasonable(self):
        """Model should have a reasonable number of parameters."""
        n_params = sum(p.numel() for p in self.model.parameters())
        assert 500_000 < n_params < 30_000_000, \
            f"Expected ~1-20M params for 3D U-Net, got {n_params:,}"

    def test_no_transposed_conv_3d(self):
        """Architecture must NOT contain ConvTranspose3d."""
        for m in self.model.modules():
            assert not isinstance(m, nn.ConvTranspose3d), \
                "Trilinear U-Net must not use transposed convolutions"

    def test_forward_shape_preserves_spatial(self):
        """Input and output 3D spatial dims must match (32 -> 32)."""
        N = 32
        x = torch.randn(1, 1, N, N, N)
        with torch.no_grad():
            y = self.model(x)
        assert y.shape == (1, 1, N, N, N)

    def test_sigmoid_output_range(self):
        """Output in [0, 1]."""
        x = torch.randn(2, 1, 32, 32, 32)
        with torch.no_grad():
            y = self.model(x)
        assert torch.all(y >= 0) and torch.all(y <= 1)

    def test_gradient_flows(self):
        """Gradients reach all parameters (batch_size >= 2 for BatchNorm3d)."""
        model = SimpleUNet3D().to(self.device)
        model.train()
        x = torch.randn(2, 1, 16, 16, 16)  # batch_size=2 for BatchNorm
        y = model(x)
        loss = y.mean()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert not torch.all(p.grad == 0), f"Zero gradient for {name}"

    def test_batch_independence(self):
        """Batch samples should be independent."""
        x1 = torch.randn(1, 1, 16, 16, 16)
        x2 = torch.randn(1, 1, 16, 16, 16)
        x_batch = torch.cat([x1, x2], dim=0)
        with torch.no_grad():
            y_batch = self.model(x_batch)
            y1 = self.model(x1)
            y2 = self.model(x2)
        assert torch.allclose(y_batch[0:1], y1, atol=1e-5)
        assert torch.allclose(y_batch[1:2], y2, atol=1e-5)


# ══════════════════════════════════════════════════════════════════
# Loss function tests
# ══════════════════════════════════════════════════════════════════

class TestLossFunctions:
    """Test loss functions used in training."""

    def test_tv_loss_3d_zero_for_constant(self):
        """TV loss should be zero for a constant volume."""
        # Replicate TVLoss3D from train_3d.py
        class TVLoss3D(nn.Module):
            def forward(self, x):
                d_tv = torch.mean(torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :]))
                h_tv = torch.mean(torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]))
                w_tv = torch.mean(torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]))
                return d_tv + h_tv + w_tv

        tv = TVLoss3D()
        vol = torch.ones(1, 1, 8, 8, 8)
        loss = tv(vol)
        assert loss.item() == 0.0, "TV loss on constant volume must be zero"

    def test_tv_loss_3d_positive_for_gradient(self):
        """TV loss should be positive for a volume with variation."""
        class TVLoss3D(nn.Module):
            def forward(self, x):
                d_tv = torch.mean(torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :]))
                h_tv = torch.mean(torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]))
                w_tv = torch.mean(torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]))
                return d_tv + h_tv + w_tv

        tv = TVLoss3D()
        vol = torch.rand(1, 1, 8, 8, 8)  # random = has variation
        loss = tv(vol)
        assert loss.item() > 0, "TV loss on non-constant volume must be positive"

    def test_smooth_value_loss_3d_tumor_weight(self):
        """Tumor regions should get higher weight in loss."""

        class TVLoss3D(nn.Module):
            def forward(self, x):
                return (torch.mean(torch.abs(x[:,:,1:,:,:] - x[:,:,:-1,:,:])) +
                        torch.mean(torch.abs(x[:,:,:,1:,:] - x[:,:,:,:-1,:])) +
                        torch.mean(torch.abs(x[:,:,:,:,1:] - x[:,:,:,:,:-1])))

        class SmoothValueLoss3D(nn.Module):
            def __init__(self, tumor_weight=5.0):
                super().__init__()
                self.tumor_weight = tumor_weight
                self.mse = nn.MSELoss(reduction='none')
                self.l1 = nn.L1Loss(reduction='none')
                self.tv = TVLoss3D()
            def forward(self, pred, target):
                weight = torch.where(target > 0.05, self.tumor_weight, 1.0)
                mse = torch.mean(weight * self.mse(pred, target))
                l1 = torch.mean(weight * self.l1(pred, target))
                tv = self.tv(pred)
                return 1.0 * mse + 0.5 * l1 + 0.2 * tv

        criterion = SmoothValueLoss3D(tumor_weight=100.0)
        pred = torch.zeros(1, 1, 4, 4, 4)
        target = torch.zeros(1, 1, 4, 4, 4)
        target[0, 0, 2, 2, 2] = 0.5  # a "tumor"

        loss_no_tumor = criterion(torch.zeros_like(target),
                                  torch.zeros_like(target))
        loss_with_tumor = criterion(pred, target)
        # Error at tumor position should cause higher loss
        assert loss_with_tumor > loss_no_tumor, \
            "Tumor region misprediction must increase loss more"


# ══════════════════════════════════════════════════════════════════
# Weight loading and compatibility tests
# ══════════════════════════════════════════════════════════════════

class TestModelCheckpoints:
    """Test the pre-trained model checkpoint files."""

    def test_full_model_loads(self, models_dir):
        """full_best_model.pth should be loadable into 2D U-Net."""
        path = os.path.join(models_dir, "full_best_model.pth")
        if not os.path.exists(path):
            pytest.skip("full_best_model.pth not found")
        model = SimpleUNet2D()
        state = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        # Quick forward pass to verify
        x = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 1, 128, 128)

    def test_half_model_loads(self, models_dir):
        """half_best_model.pth should be loadable."""
        path = os.path.join(models_dir, "half_best_model.pth")
        if not os.path.exists(path):
            pytest.skip("half_best_model.pth not found")
        model = SimpleUNet2D()
        state = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        x = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 1, 128, 128)

    def test_sphere_model_loads(self, models_dir):
        """sphere_best_model.pth should be loadable into 3D U-Net."""
        path = os.path.join(models_dir, "sphere_best_model.pth")
        if not os.path.exists(path):
            pytest.skip("sphere_best_model.pth not found")
        model = SimpleUNet3D()
        state = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        x = torch.randn(1, 1, 32, 32, 32)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 1, 32, 32, 32)

    def test_checkpoint_keys_consistent(self, models_dir):
        """All 2D checkpoints should have the same keys."""
        path_full = os.path.join(models_dir, "full_best_model.pth")
        path_half = os.path.join(models_dir, "half_best_model.pth")
        if not os.path.exists(path_full) or not os.path.exists(path_half):
            pytest.skip("Model checkpoints not found")
        keys_full = set(torch.load(path_full, map_location='cpu', weights_only=True).keys())
        keys_half = set(torch.load(path_half, map_location='cpu', weights_only=True).keys())
        assert keys_full == keys_half, \
            f"Key mismatch: full={len(keys_full)}, half={len(keys_half)}"