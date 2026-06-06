"""
Tests for data generation and dataset I/O.

Covers:
- Data shapes, ranges, and types for 2D/3D datasets
- Scattered field (X) and permittivity (Y) constraints
- Dataset integrity (pairwise consistency)
- Noise addition statistics
- .npy file I/O
"""
import os
import sys
import numpy as np
import torch
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tests.test_helpers import (
    deterministic_seed,
    make_fake_2d_scattered,
    make_fake_2d_permittivity,
    make_fake_3d_scattered,
    make_fake_3d_permittivity,
    save_fake_test_dataset_2d,
    save_fake_test_dataset_3d,
    EPS_BG_DEFAULT, L, N_2D, N_3D, N_ANT,
)


# ══════════════════════════════════════════════════════════════════
# 2D Data tests
# ══════════════════════════════════════════════════════════════════

class TestData2D:
    """Test 2D scattered field and permittivity data."""

    def test_scattered_field_shape(self):
        """X data shape: (N, 2, N_ANT, N_ANT), where channel 0=real, 1=imag."""
        X = make_fake_2d_scattered(10)
        assert X.ndim == 4
        assert X.shape[0] == 10
        assert X.shape[1] == 2  # real + imag
        assert X.shape[2] == N_ANT  # receivers
        assert X.shape[3] == N_ANT  # transmitters

    def test_scattered_field_dtype(self):
        """X data should be float32 for efficient GPU transfer."""
        X = make_fake_2d_scattered(5)
        assert X.dtype == np.float32

    def test_scattered_field_finite(self):
        """Scattered fields must not contain inf or nan."""
        X = make_fake_2d_scattered(20)
        assert np.all(np.isfinite(X))

    def test_permittivity_shape(self):
        """Y data shape: (N, 128, 128)."""
        Y = make_fake_2d_permittivity(10)
        assert Y.ndim == 3
        assert Y.shape[0] == 10
        assert Y.shape[1] == N_2D
        assert Y.shape[2] == N_2D

    def test_permittivity_range(self):
        """Permittivity values must be >= eps_bg and <= 5.0."""
        Y = make_fake_2d_permittivity(20)
        assert np.all(Y >= EPS_BG_DEFAULT - 1e-6)
        assert np.all(Y <= 5.0 + 1e-6)

    def test_permittivity_has_features(self):
        """At least some pixels should differ from background."""
        Y = make_fake_2d_permittivity(10)
        for i in range(Y.shape[0]):
            assert np.any(Y[i] > EPS_BG_DEFAULT + 0.1), \
                f"Sample {i} has no inclusion (all background)"

    def test_complex_field_to_real_imag_conversion(self):
        """Real+imag stack correctly represents complex field."""
        X = make_fake_2d_scattered(3)
        complex_X = X[:, 0] + 1j * X[:, 1]
        assert np.allclose(np.real(complex_X), X[:, 0])
        assert np.allclose(np.imag(complex_X), X[:, 1])

    def test_sample_count_pairwise(self):
        """X and Y must have the same number of samples."""
        for N in [4, 8, 16]:
            X = make_fake_2d_scattered(N)
            Y = make_fake_2d_permittivity(N)
            assert X.shape[0] == Y.shape[0] == N


# ══════════════════════════════════════════════════════════════════
# 3D Data tests
# ══════════════════════════════════════════════════════════════════

class TestData3D:
    """Test 3D scattered field and permittivity data."""

    def test_scattered_field_shape(self):
        """3D X data: (N, 2, 64, 64)."""
        X = make_fake_3d_scattered(5)
        assert X.shape == (5, 2, N_ANT, N_ANT)

    def test_scattered_field_finite(self):
        """No inf/nan in 3D scattered data."""
        X = make_fake_3d_scattered(10)
        assert np.all(np.isfinite(X))

    def test_permittivity_shape(self):
        """3D Y data: (N, 32, 32, 32)."""
        Y = make_fake_3d_permittivity(5)
        assert Y.shape == (5, N_3D, N_3D, N_3D)

    def test_permittivity_range(self):
        """3D permittivity values in [eps_bg, 5.0]."""
        Y = make_fake_3d_permittivity(10)
        assert np.all(Y >= EPS_BG_DEFAULT - 1e-6)
        assert np.all(Y <= 5.0 + 1e-6)

    def test_permittivity_has_features_3d(self):
        """3D samples must have non-background inclusions."""
        Y = make_fake_3d_permittivity(10)
        for i in range(Y.shape[0]):
            assert np.any(Y[i] > EPS_BG_DEFAULT + 0.1), \
                f"3D sample {i} has no inclusion"

    def test_sample_count_pairwise(self):
        """3D X and Y must have matching sample counts."""
        for N in [3, 5, 7]:
            X = make_fake_3d_scattered(N)
            Y = make_fake_3d_permittivity(N)
            assert X.shape[0] == Y.shape[0] == N


# ══════════════════════════════════════════════════════════════════
# File I/O tests
# ══════════════════════════════════════════════════════════════════

class TestFileIO:
    """Test .npy read/write for datasets."""

    def test_save_and_load_2d(self, tmp_dir):
        """2D dataset roundtrip: save -> load -> shape == original."""
        deterministic_seed(42)
        save_fake_test_dataset_2d(tmp_dir, "test_io", n_samples=6)
        X = np.load(os.path.join(tmp_dir, "test_io_X.npy"))
        Y = np.load(os.path.join(tmp_dir, "test_io_Y.npy"))
        assert X.shape == (6, 2, N_ANT, N_ANT)
        assert Y.shape == (6, N_2D, N_2D)
        assert X.dtype == np.float32
        assert Y.dtype == np.float32

    def test_save_and_load_3d(self, tmp_dir):
        """3D dataset roundtrip: save -> load -> shape == original."""
        deterministic_seed(42)
        save_fake_test_dataset_3d(tmp_dir, "test_io3d", n_samples=5)
        X = np.load(os.path.join(tmp_dir, "test_io3d_X.npy"))
        Y = np.load(os.path.join(tmp_dir, "test_io3d_Y.npy"))
        assert X.shape == (5, 2, N_ANT, N_ANT)
        assert Y.shape == (5, N_3D, N_3D, N_3D)

    def test_npy_truncated_detection(self, tmp_dir):
        """Loading a truncated .npy should raise an error."""
        import struct
        path = os.path.join(tmp_dir, "bad.npy")
        # Write a valid header with wrong data size
        with open(path, 'wb') as f:
            f.write(b'\x93NUMPY\x01\x00')
            header = "{'descr': '<f4', 'fortran_order': False, 'shape': (100, 100), }"
            header_b = header.encode('utf-8')
            # pad to 16-byte alignment
            header_b += b' ' * (16 - ((10 + len(header_b)) % 16)) + b'\n'
            f.write(struct.pack('<H', len(header_b)))
            f.write(header_b)
            f.write(b'\x00' * 10)  # truncated data
        with pytest.raises((ValueError, OSError, EOFError)):
            np.load(path)

    def test_missing_file(self, tmp_dir):
        """Loading non-existent file should raise FileNotFoundError."""
        path = os.path.join(tmp_dir, "nonexistent.npy")
        with pytest.raises((FileNotFoundError, OSError)):
            np.load(path)

    def test_wrong_extension(self, tmp_dir):
        """Files without .npy should be handled.
        NumPy .npz has different format; .npy has magic bytes.
        """
        path = os.path.join(tmp_dir, "not_numpy.txt")
        with open(path, 'w') as f:
            f.write("this is not a numpy file")
        with pytest.raises((ValueError, OSError)):
            np.load(path)


# ══════════════════════════════════════════════════════════════════
# Real dataset integrity tests
# ══════════════════════════════════════════════════════════════════

class TestRealDatasetIntegrity:
    """Verify the integrity of the pre-generated datasets."""

    def test_full_train_clean_exists(self, train_data_dir):
        """Check full_train_clean data exists and loads."""
        x_path = os.path.join(train_data_dir, "full_train_clean_X.npy")
        y_path = os.path.join(train_data_dir, "full_train_clean_Y.npy")
        if not os.path.exists(x_path):
            pytest.skip("full_train_clean_X.npy not found")
        X = np.load(x_path, mmap_mode='r')
        Y = np.load(y_path, mmap_mode='r')
        assert X.shape[0] == Y.shape[0]
        assert X.shape[1:] == (2, 64, 64)
        assert Y.shape[1:] == (128, 128)

    def test_full_test_exists(self, train_data_dir):
        """Check full_test data integrity."""
        x_path = os.path.join(train_data_dir, "full_test_X.npy")
        y_path = os.path.join(train_data_dir, "full_test_Y.npy")
        if not os.path.exists(x_path):
            pytest.skip("full_test_X.npy not found")
        X = np.load(x_path, mmap_mode='r')
        Y = np.load(y_path, mmap_mode='r')
        assert X.shape[0] == Y.shape[0]

    def test_half_train_clean_exists(self, train_data_dir):
        """Check half_train_clean data integrity."""
        x_path = os.path.join(train_data_dir, "half_train_clean_X.npy")
        y_path = os.path.join(train_data_dir, "half_train_clean_Y.npy")
        if not os.path.exists(x_path):
            pytest.skip("half_train_clean_X.npy not found")
        X = np.load(x_path, mmap_mode='r')
        Y = np.load(y_path, mmap_mode='r')
        assert X.shape[0] == Y.shape[0]

    def test_half_test_exists(self, train_data_dir):
        """Check half_test data integrity."""
        x_path = os.path.join(train_data_dir, "half_test_X.npy")
        y_path = os.path.join(train_data_dir, "half_test_Y.npy")
        if not os.path.exists(x_path):
            pytest.skip("half_test_X.npy not found")
        X = np.load(x_path, mmap_mode='r')
        Y = np.load(y_path, mmap_mode='r')
        assert X.shape[0] == Y.shape[0]

    def test_sphere_train_exists(self, train_data_dir):
        """Check 3D training data integrity."""
        x_path = os.path.join(train_data_dir, "sphere_train_X.npy")
        y_path = os.path.join(train_data_dir, "sphere_train_Y.npy")
        if not os.path.exists(x_path):
            pytest.skip("sphere_train_X.npy not found")
        X = np.load(x_path, mmap_mode='r')
        Y = np.load(y_path, mmap_mode='r')
        assert X.shape[0] == Y.shape[0]
        assert Y.shape[1:] == (32, 32, 32)

    def test_sphere_test_exists(self, train_data_dir):
        """Check 3D test data integrity."""
        x_path = os.path.join(train_data_dir, "sphere_test_X.npy")
        y_path = os.path.join(train_data_dir, "sphere_test_Y.npy")
        if not os.path.exists(x_path):
            pytest.skip("sphere_test_X.npy not found")
        X = np.load(x_path, mmap_mode='r')
        Y = np.load(y_path, mmap_mode='r')
        assert X.shape[0] == Y.shape[0]

    def test_full_data_has_no_nan(self, train_data_dir):
        """All full_circle datasets must be NaN-free."""
        for stem in ["full_train_clean", "full_train_noisy", "full_test"]:
            x_path = os.path.join(train_data_dir, f"{stem}_X.npy")
            if not os.path.exists(x_path):
                continue
            X = np.load(x_path, mmap_mode='r')
            assert not np.any(np.isnan(X)), f"NaN found in {stem}_X.npy"

    def test_half_data_has_no_nan(self, train_data_dir):
        for stem in ["half_train_clean", "half_train_noisy", "half_test"]:
            x_path = os.path.join(train_data_dir, f"{stem}_X.npy")
            if not os.path.exists(x_path):
                continue
            X = np.load(x_path, mmap_mode='r')
            assert not np.any(np.isnan(X)), f"NaN found in {stem}_X.npy"

    def test_sphere_data_has_no_nan(self, train_data_dir):
        for stem in ["sphere_train", "sphere_test"]:
            x_path = os.path.join(train_data_dir, f"{stem}_X.npy")
            if not os.path.exists(x_path):
                continue
            X = np.load(x_path, mmap_mode='r')
            assert not np.any(np.isnan(X)), f"NaN found in {stem}_X.npy"


# ══════════════════════════════════════════════════════════════════
# Normalization in training data tests
# ══════════════════════════════════════════════════════════════════

class TestDataNormalization:
    """Test data normalization pipeline used in training."""

    def normalize(self, Y, eps_bg=1.5):
        return (Y - eps_bg) / (5.0 - eps_bg)

    def test_normalized_range(self):
        """After normalization, values should be in [0, 1]."""
        Y = make_fake_2d_permittivity(20)
        Y_norm = self.normalize(Y)
        assert np.all(Y_norm >= -1e-6)
        assert np.all(Y_norm <= 1.0 + 1e-6)

    def test_background_normalized_to_zero(self):
        """Background (eps_bg) should map to 0."""
        bg = np.full((4, 4), EPS_BG_DEFAULT, dtype=np.float32)
        bg_norm = self.normalize(bg)
        assert np.allclose(bg_norm, 0.0, atol=1e-5)

    def test_max_normalized_to_one(self):
        """Max permittivity (5.0) should map to 1."""
        mx = np.full((4, 4), 5.0, dtype=np.float32)
        mx_norm = self.normalize(mx)
        assert np.allclose(mx_norm, 1.0, atol=1e-5)