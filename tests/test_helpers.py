"""
Test helper utilities for the EM inverse scattering project.

Provides deterministic test data generators for unit testing without
requiring full GPU-accelerated MoM forward solves.
"""
import numpy as np
import torch
import os
import sys

# Ensure repo root is on the path so we can import from scripts if needed
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ── Physical constants (matching project defaults) ──
FREQ = 2.45e9
C0 = 299792458.0
K0 = 2 * np.pi * FREQ / C0
EPS_BG_DEFAULT = 1.5
L = 0.25
N_2D = 128
N_3D = 32
N_ANT = 64
R_ANT = 0.35


def deterministic_seed(seed=42):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_fake_2d_scattered(n_samples=4, n_ant=N_ANT, dtype=np.float32):
    """
    Generate synthetic 2D scattered field data with the same shape as real data.
    Shape: (n_samples, 2, n_ant, n_ant) — channel 0=real, channel 1=imag.
    """
    rng = np.random.RandomState(42)
    real = rng.randn(n_samples, n_ant, n_ant).astype(dtype) * 0.01
    imag = rng.randn(n_samples, n_ant, n_ant).astype(dtype) * 0.01
    return np.stack([real, imag], axis=1)


def make_fake_2d_permittivity(n_samples=4, n=N_2D, eps_bg=EPS_BG_DEFAULT, dtype=np.float32):
    """
    Generate synthetic 2D permittivity maps with simple geometric inclusions.
    Shape: (n_samples, n, n)
    Values: ellipses with eps_r between 3.0-5.0 on a background of eps_bg.
    """
    rng = np.random.RandomState(42)
    x_arr = np.linspace(-L/2, L/2, n)
    X, Y = np.meshgrid(x_arr, x_arr)
    eps = np.full((n_samples, n, n), eps_bg, dtype=dtype)
    for i in range(n_samples):
        cx = rng.uniform(-0.06, 0.06)
        cy = rng.uniform(-0.06, 0.06)
        rx = rng.uniform(0.02, 0.035)
        ry = rng.uniform(0.02, 0.035)
        val = rng.uniform(3.5, 4.5)
        mask = ((X - cx) / rx)**2 + ((Y - cy) / ry)**2 <= 1.0
        eps[i, mask] = val
    return eps


def make_fake_3d_scattered(n_samples=4, n_ant=N_ANT, dtype=np.float32):
    """Generate synthetic 3D scattered field data."""
    rng = np.random.RandomState(42)
    real = rng.randn(n_samples, n_ant, n_ant).astype(dtype) * 0.01
    imag = rng.randn(n_samples, n_ant, n_ant).astype(dtype) * 0.01
    return np.stack([real, imag], axis=1)


def make_fake_3d_permittivity(n_samples=4, n=N_3D, eps_bg=EPS_BG_DEFAULT, dtype=np.float32):
    """
    Generate synthetic 3D permittivity volumes.
    Shape: (n_samples, n, n, n)
    """
    rng = np.random.RandomState(42)
    x_arr = np.linspace(-L/2, L/2, n)
    X, Y, Z = np.meshgrid(x_arr, x_arr, x_arr, indexing='ij')
    eps = np.full((n_samples, n, n, n), eps_bg, dtype=dtype)
    for i in range(n_samples):
        cx, cy, cz = rng.uniform(-0.05, 0.05, 3)
        r = rng.uniform(0.025, 0.04)
        val = rng.uniform(3.5, 4.5)
        mask = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2 <= r**2
        eps[i, mask] = val
    return eps


def save_fake_test_dataset_2d(out_dir, prefix, n_samples=4):
    """Save fake 2D X/Y pairs to disk as .npy files."""
    os.makedirs(out_dir, exist_ok=True)
    X = make_fake_2d_scattered(n_samples)
    Y = make_fake_2d_permittivity(n_samples)
    np.save(os.path.join(out_dir, f"{prefix}_X.npy"), X)
    np.save(os.path.join(out_dir, f"{prefix}_Y.npy"), Y)


def save_fake_test_dataset_3d(out_dir, prefix, n_samples=4):
    """Save fake 3D X/Y pairs to disk as .npy files."""
    os.makedirs(out_dir, exist_ok=True)
    X = make_fake_3d_scattered(n_samples)
    Y = make_fake_3d_permittivity(n_samples)
    np.save(os.path.join(out_dir, f"{prefix}_X.npy"), X)
    np.save(os.path.join(out_dir, f"{prefix}_Y.npy"), Y)


def cleanup_test_files(*paths):
    """Remove test files/directories."""
    import shutil
    for p in paths:
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        elif os.path.isfile(p):
            os.remove(p)