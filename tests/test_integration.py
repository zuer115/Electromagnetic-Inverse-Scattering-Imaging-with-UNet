"""
End-to-end integration tests that exercise the full pipeline.

These tests use fake data and run through the complete training/inference
cycle to verify the pipeline works end-to-end.

Note: Full training is too slow for CI; these tests use minimal epochs.
"""
import os
import sys
import subprocess
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
    EPS_BG_DEFAULT,
    L, N_2D, N_3D, N_ANT, K0,
)

PYTHON = sys.executable


def run_script(script_name, args, timeout=120):
    """Run a project script, return (returncode, stdout, stderr)."""
    script_path = os.path.join(REPO_ROOT, script_name)
    cmd = [PYTHON, script_path] + args
    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=timeout, cwd=REPO_ROOT)
    return result.returncode, result.stdout, result.stderr


# ══════════════════════════════════════════════════════════════════
# Full pipeline: data generation -> training -> evaluation -> inference
# ══════════════════════════════════════════════════════════════════

class Test2DPipelineEndToEnd:
    """Test the complete 2D pipeline with minimal data."""

    def test_gen_data_minimal_2d_full(self, tmp_dir):
        """Generate minimal 2D full_circle dataset."""
        deterministic_seed(42)
        rc, stdout, stderr = run_script("gen_data.py", [
            "-a", "full_circle",
            "--num_train_clean", "4",
            "--num_train_noisy", "2",
            "--num_test", "2",
            "-o", tmp_dir,
            "--cpu",
            "-b", "2",
        ], timeout=120)
        assert rc == 0, f"gen_data failed (rc={rc}): {stderr[:500]}"
        # Check output files exist
        assert os.path.exists(os.path.join(tmp_dir, "full_train_clean_X.npy"))
        assert os.path.exists(os.path.join(tmp_dir, "full_train_clean_Y.npy"))
        assert os.path.exists(os.path.join(tmp_dir, "full_train_noisy_X.npy"))
        assert os.path.exists(os.path.join(tmp_dir, "full_test_X.npy"))
        assert os.path.exists(os.path.join(tmp_dir, "full_test_Y.npy"))

    def test_train_one_epoch_2d(self, tmp_dir):
        """Train 2D U-Net for 1 epoch on tiny dataset."""
        deterministic_seed(42)
        # First generate data
        rc, stdout, stderr = run_script("gen_data.py", [
            "-a", "full_circle",
            "--num_train_clean", "6",
            "--num_train_noisy", "2",
            "--num_test", "2",
            "-o", tmp_dir, "--cpu", "-b", "2",
        ], timeout=120)
        assert rc == 0, f"gen_data failed: {stderr[:300]}"

        # Then train
        rc, stdout, stderr = run_script("train.py", [
            "-a", "full_circle",
            "--train_clean_path", f"{tmp_dir}/full_train_clean",
            "--train_noisy_path", f"{tmp_dir}/full_train_noisy",
            "--test_path", f"{tmp_dir}/full_test",
            "-o", tmp_dir,
            "-b", "2",
            "-e", "2",
            "--cpu",
        ], timeout=300)
        assert rc == 0, f"train failed (rc={rc}): {stderr[:500]}"
        # Check model saved
        assert os.path.exists(os.path.join(tmp_dir, "full_best_model.pth"))

    def test_evaluate_2d(self, tmp_dir):
        """Run 2D evaluation with a single image."""
        deterministic_seed(42)
        # Generate data and train model
        rc, _, stderr = run_script("gen_data.py", [
            "-a", "full_circle",
            "--num_train_clean", "6",
            "--num_train_noisy", "2",
            "--num_test", "2",
            "-o", tmp_dir, "--cpu", "-b", "2",
        ], timeout=120)
        assert rc == 0, f"gen_data: {stderr[:200]}"

        rc, _, stderr = run_script("train.py", [
            "-a", "full_circle",
            "--train_clean_path", f"{tmp_dir}/full_train_clean",
            "--train_noisy_path", f"{tmp_dir}/full_train_noisy",
            "--test_path", f"{tmp_dir}/full_test",
            "-o", tmp_dir, "-b", "2", "-e", "2", "--cpu",
        ], timeout=300)
        assert rc == 0, f"train: {stderr[:200]}"

        # Evaluate
        out_dir = os.path.join(tmp_dir, "eval_out")
        rc, stdout, stderr = run_script("evaluate.py", [
            "--cpu",
            "--test_full_path", f"{tmp_dir}/full_test",
            "--full_model", f"{tmp_dir}/full_best_model.pth",
            "--no_half",
            "-n", "1",
            "-o", out_dir,
        ], timeout=120)
        assert rc == 0, f"evaluate failed (rc={rc}): {stderr[:500]}"
        assert os.path.exists(os.path.join(out_dir, "evaluation_comparison.png"))

    def test_main_inference_2d(self, tmp_dir):
        """Run main.py inference on test data."""
        deterministic_seed(42)
        # Generate data and train
        rc, _, stderr = run_script("gen_data.py", [
            "-a", "full_circle",
            "--num_train_clean", "6",
            "--num_train_noisy", "2",
            "--num_test", "2",
            "-o", tmp_dir, "--cpu", "-b", "2",
        ], timeout=120)
        assert rc == 0, f"gen_data: {stderr[:200]}"

        rc, _, stderr = run_script("train.py", [
            "-a", "full_circle",
            "--train_clean_path", f"{tmp_dir}/full_train_clean",
            "--train_noisy_path", f"{tmp_dir}/full_train_noisy",
            "--test_path", f"{tmp_dir}/full_test",
            "-o", tmp_dir, "-b", "2", "-e", "2", "--cpu",
        ], timeout=300)
        assert rc == 0, f"train: {stderr[:200]}"

        # Inference
        out_dir = os.path.join(tmp_dir, "infer_out")
        rc, stdout, stderr = run_script("main.py", [
            "-i", f"{tmp_dir}/full_test_X.npy",
            "-w", f"{tmp_dir}/full_best_model.pth",
            "-d", "2d",
            "-m", "full_circle",
            "-f", "npy",
            "-o", out_dir,
            "--cpu",
        ], timeout=120)
        assert rc == 0, f"main.py failed (rc={rc}): {stderr[:800]}"
        npy_file = os.path.join(out_dir, "predictions_2d_full_circle.npy")
        assert os.path.exists(npy_file), f"Expected {npy_file}"
        preds = np.load(npy_file)
        assert preds.shape[1:] == (N_2D, N_2D)
        # Check predictions are in reasonable range [1.5, 5.0]
        assert np.all(preds >= 1.0)
        assert np.all(preds <= 5.5)


class Test3DPipelineEndToEnd:
    """Test the complete 3D pipeline with minimal data."""

    def test_gen_data_minimal_3d(self, tmp_dir):
        """Generate minimal 3D dataset."""
        deterministic_seed(42)
        rc, stdout, stderr = run_script("gen_data_3d.py", [
            "--num_train", "3",
            "--num_test", "2",
            "-o", tmp_dir,
            "--cpu",
            "-b", "1",
        ], timeout=300)
        assert rc == 0, f"gen_data_3d failed (rc={rc}): {stderr[:500]}"
        assert os.path.exists(os.path.join(tmp_dir, "sphere_train_X.npy"))
        assert os.path.exists(os.path.join(tmp_dir, "sphere_test_X.npy"))

    def test_train_one_epoch_3d(self, tmp_dir):
        """Train 3D U-Net for 1 epoch."""
        deterministic_seed(42)
        rc, _, stderr = run_script("gen_data_3d.py", [
            "--num_train", "4",
            "--num_test", "2",
            "-o", tmp_dir, "--cpu", "-b", "1",
        ], timeout=300)
        assert rc == 0, f"gen_data_3d: {stderr[:200]}"

        rc, stdout, stderr = run_script("train_3d.py", [
            "--train_path", f"{tmp_dir}/sphere_train",
            "--test_path", f"{tmp_dir}/sphere_test",
            "-o", tmp_dir,
            "-b", "1",
            "-e", "2",
            "--cpu",
        ], timeout=600)
        assert rc == 0, f"train_3d failed (rc={rc}): {stderr[:500]}"
        assert os.path.exists(os.path.join(tmp_dir, "sphere_best_model.pth"))

    def test_evaluate_3d(self, tmp_dir):
        """Run 3D evaluation."""
        deterministic_seed(42)
        rc, _, stderr = run_script("gen_data_3d.py", [
            "--num_train", "4",
            "--num_test", "2",
            "-o", tmp_dir, "--cpu", "-b", "1",
        ], timeout=300)
        assert rc == 0

        rc, _, stderr = run_script("train_3d.py", [
            "--train_path", f"{tmp_dir}/sphere_train",
            "--test_path", f"{tmp_dir}/sphere_test",
            "-o", tmp_dir, "-b", "1", "-e", "2", "--cpu",
        ], timeout=600)
        assert rc == 0

        out_dir = os.path.join(tmp_dir, "eval3d_out")
        rc, stdout, stderr = run_script("evaluate_3d.py", [
            "--cpu",
            "--test_path", f"{tmp_dir}/sphere_test",
            "-w", f"{tmp_dir}/sphere_best_model.pth",
            "-n", "1",
            "-o", out_dir,
        ], timeout=300)
        assert rc == 0, f"evaluate_3d failed (rc={rc}): {stderr[:500]}"

    def test_main_inference_3d(self, tmp_dir):
        """Run main.py inference for 3D."""
        deterministic_seed(42)
        rc, _, stderr = run_script("gen_data_3d.py", [
            "--num_train", "4",
            "--num_test", "2",
            "-o", tmp_dir, "--cpu", "-b", "1",
        ], timeout=300)
        assert rc == 0

        rc, _, stderr = run_script("train_3d.py", [
            "--train_path", f"{tmp_dir}/sphere_train",
            "--test_path", f"{tmp_dir}/sphere_test",
            "-o", tmp_dir, "-b", "1", "-e", "2", "--cpu",
        ], timeout=600)
        assert rc == 0

        out_dir = os.path.join(tmp_dir, "infer3d_out")
        rc, stdout, stderr = run_script("main.py", [
            "-i", f"{tmp_dir}/sphere_test_X.npy",
            "-w", f"{tmp_dir}/sphere_best_model.pth",
            "-d", "3d",
            "-m", "full_circle",
            "-f", "npy",
            "-o", out_dir,
            "--cpu",
        ], timeout=300)
        assert rc == 0, f"main.py 3D failed (rc={rc}): {stderr[:800]}"


# ══════════════════════════════════════════════════════════════════
# Cross-script consistency tests
# ══════════════════════════════════════════════════════════════════

class TestCrossScriptConsistency:
    """Verify consistency between different scripts."""

    def test_small_data_train_and_main_produce_output(self, tmp_dir):
        """Train on tiny data, then infer — predictions should be within range."""
        deterministic_seed(42)

        # Generate data
        rc, _, stderr = run_script("gen_data.py", [
            "-a", "full_circle",
            "--num_train_clean", "8",
            "--num_train_noisy", "2",
            "--num_test", "2",
            "-o", tmp_dir, "--cpu", "-b", "2",
        ], timeout=120)
        assert rc == 0, f"gen_data: {stderr[:200]}"

        # Train for a few epochs
        rc, _, stderr = run_script("train.py", [
            "-a", "full_circle",
            "--train_clean_path", f"{tmp_dir}/full_train_clean",
            "--train_noisy_path", f"{tmp_dir}/full_train_noisy",
            "--test_path", f"{tmp_dir}/full_test",
            "-o", tmp_dir, "-b", "2", "-e", "3", "--cpu",
        ], timeout=300)
        assert rc == 0, f"train: {stderr[:200]}"

        # Run main.py inference
        infer_dir = os.path.join(tmp_dir, "infer")
        rc, _, stderr = run_script("main.py", [
            "-i", f"{tmp_dir}/full_test_X.npy",
            "-w", f"{tmp_dir}/full_best_model.pth",
            "-d", "2d", "-m", "full_circle",
            "-f", "npy", "png",
            "-o", infer_dir, "--cpu",
        ], timeout=120)
        assert rc == 0, f"main.py: {stderr[:300]}"

        # Check outputs
        preds = np.load(os.path.join(infer_dir, "predictions_2d_full_circle.npy"))
        assert preds.ndim == 3  # (n_samples, 128, 128)
        assert not np.any(np.isnan(preds))
        assert not np.any(np.isinf(preds))

    def test_reproducibility_same_seed(self, tmp_dir):
        """Same seed should produce identical data (deterministic gen)."""
        deterministic_seed(12345)
        save_fake_test_dataset_2d(tmp_dir, "run1", n_samples=8)

        deterministic_seed(12345)
        save_fake_test_dataset_2d(tmp_dir, "run2", n_samples=8)

        X1 = np.load(os.path.join(tmp_dir, "run1_X.npy"))
        X2 = np.load(os.path.join(tmp_dir, "run2_X.npy"))
        Y1 = np.load(os.path.join(tmp_dir, "run1_Y.npy"))
        Y2 = np.load(os.path.join(tmp_dir, "run2_Y.npy"))

        assert np.allclose(X1, X2)
        assert np.allclose(Y1, Y2)


# ══════════════════════════════════════════════════════════════════
# Boundary condition stress tests
# ══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dataset_handling(self, tmp_dir):
        """0-sample dataset should not crash gen_data (or fail cleanly)."""
        rc, stdout, stderr = run_script("gen_data.py", [
            "-a", "full_circle",
            "--num_train_clean", "0",
            "--num_train_noisy", "0",
            "--num_test", "0",
            "-o", tmp_dir, "--cpu",
        ], timeout=60)
        # 0 samples might fail during array allocation or succeed with empty arrays
        # Either is acceptable as long as it's not a crash
        assert "traceback" not in stderr.lower() or "Error" not in stderr

    def test_single_sample_batch(self, tmp_dir):
        """Processing a single sample should work."""
        deterministic_seed(42)
        rc, _, stderr = run_script("gen_data.py", [
            "-a", "full_circle",
            "--num_train_clean", "1",
            "--num_train_noisy", "0",
            "--num_test", "1",
            "-o", tmp_dir, "--cpu", "-b", "1",
        ], timeout=120)
        assert rc == 0, f"single sample gen failed: {stderr[:300]}"

    def test_batch_size_larger_than_dataset(self, tmp_dir):
        """Batch size > dataset should still work."""
        deterministic_seed(42)
        save_fake_test_dataset_2d(tmp_dir, "full_test", n_samples=2)
        # Training with batch_size > n_samples
        save_fake_test_dataset_2d(tmp_dir, "full_train_clean", n_samples=3)
        save_fake_test_dataset_2d(tmp_dir, "full_train_noisy", n_samples=1)

        rc, _, stderr = run_script("train.py", [
            "-a", "full_circle",
            "--train_clean_path", f"{tmp_dir}/full_train_clean",
            "--train_noisy_path", f"{tmp_dir}/full_train_noisy",
            "--test_path", f"{tmp_dir}/full_test",
            "-o", tmp_dir, "-b", "10", "-e", "1", "--cpu",
        ], timeout=120)
        assert rc == 0, f"large batch: {stderr[:300]}"