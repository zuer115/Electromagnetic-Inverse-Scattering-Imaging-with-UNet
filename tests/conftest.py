"""
Shared pytest fixtures for EM Inverse Scattering test suite.
"""
import os
import sys
import tempfile
import shutil
import pytest
import numpy as np
import torch

# Ensure repo root is on the path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tests.test_helpers import (
    make_fake_2d_scattered,
    make_fake_2d_permittivity,
    make_fake_3d_scattered,
    make_fake_3d_permittivity,
    save_fake_test_dataset_2d,
    save_fake_test_dataset_3d,
    deterministic_seed,
    EPS_BG_DEFAULT, L, N_2D, N_3D, N_ANT, K0,
)


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope="session")
def train_data_dir():
    return os.path.join(REPO_ROOT, "train_data")


@pytest.fixture(scope="session")
def models_dir():
    return os.path.join(REPO_ROOT, "models")


@pytest.fixture(autouse=True)
def seed_everything():
    """Ensure deterministic behavior in every test."""
    deterministic_seed(42)
    torch.use_deterministic_algorithms(False)  # conv ops may not support it


@pytest.fixture
def device():
    """Return the available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def tmp_dir():
    """Create a temporary directory that is cleaned up after the test."""
    d = tempfile.mkdtemp(prefix="em_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def fake_2d_data():
    """Create fake 2D scattered fields + permittivity pairs in memory."""
    deterministic_seed(42)
    X = make_fake_2d_scattered(4)
    Y = make_fake_2d_permittivity(4)
    return X, Y


@pytest.fixture
def fake_3d_data():
    """Create fake 3D scattered fields + permittivity pairs in memory."""
    deterministic_seed(42)
    X = make_fake_3d_scattered(4)
    Y = make_fake_3d_permittivity(4)
    return X, Y


@pytest.fixture
def fake_2d_dataset_on_disk(tmp_dir):
    """Create fake 2D train_clean / train_noisy / test data on disk."""
    deterministic_seed(42)
    for prefix in ("full_train_clean", "full_train_noisy", "full_test",
                   "half_train_clean", "half_train_noisy", "half_test"):
        save_fake_test_dataset_2d(tmp_dir, prefix, n_samples=4)
    return tmp_dir


@pytest.fixture
def fake_3d_dataset_on_disk(tmp_dir):
    """Create fake 3D train / test data on disk."""
    deterministic_seed(42)
    for prefix in ("sphere_train", "sphere_test"):
        save_fake_test_dataset_3d(tmp_dir, prefix, n_samples=4)
    return tmp_dir


@pytest.fixture
def real_train_data_exists(train_data_dir):
    """Check whether real pre-generated training data exists."""
    needed = ["full_train_clean_X.npy", "full_test_X.npy",
              "half_train_clean_X.npy", "half_test_X.npy"]
    return all(os.path.exists(os.path.join(train_data_dir, f)) for f in needed)


@pytest.fixture
def real_models_exist(models_dir):
    """Check whether real pre-trained models exist."""
    return os.path.exists(os.path.join(models_dir, "full_best_model.pth"))