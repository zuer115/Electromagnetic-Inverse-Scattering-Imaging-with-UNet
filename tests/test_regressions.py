"""
Regression tests that capture known bugs found in the codebase.

These tests are marked with xfail when the bug still exists,
and will start passing when the bugs are fixed.
"""
import os
import sys
import numpy as np
import torch
import pytest
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PYTHON = sys.executable


# ══════════════════════════════════════════════════════════════════
# BUG #1: torch.cdist called with numpy arrays instead of tensors
# ══════════════════════════════════════════════════════════════════
#
# In gen_data.py:99, main.py:95, evaluate.py:76, train.py:93,
# torch.cdist(r_domain_np, r_domain_np) passes a numpy array
# instead of the already-converted tensor. PyTorch 2.12+ fails
# on this with an internal assertion error.
#
# Fix: replace r_domain_np with r_domain_gpu in these calls.
# ==================================================================

class TestBugTorchCdistNumpyInput:
    """Verify that torch.cdist() with numpy inputs triggers the bug."""

    def test_cdist_accepts_tensor(self):
        """torch.cdist should accept tensor inputs (sanity check)."""
        t = torch.randn(10, 3)
        d = torch.cdist(t, t)
        assert d.shape == (10, 10)
        assert torch.all(d >= 0)

    @pytest.mark.xfail(
        reason="BUG: torch.cdist(r_domain_np, r_domain_np) in source uses numpy, "
               "which fails in PyTorch 2.12+. Fix: use r_domain_gpu instead."
    )
    def test_gen_data_cdist_uses_numpy(self):
        """Bug: gen_data.py:99 passes r_domain_np (numpy) to torch.cdist."""
        # Read gen_data.py source and check for the pattern
        src_path = os.path.join(REPO_ROOT, "gen_data.py")
        with open(src_path) as f:
            source = f.read()
        # Look for the buggy pattern: torch.cdist(r_domain_np, ...
        import re
        bug_pattern = re.search(r'torch\.cdist\(r_domain_np,\s*r_domain_np\)', source)
        assert bug_pattern is None, \
            "BUG FOUND: torch.cdist(r_domain_np, r_domain_np) in gen_data.py " \
            "should be torch.cdist(r_domain_gpu, r_domain_gpu)"

    @pytest.mark.xfail(
        reason="Same torch.cdist numpy bug as gen_data.py"
    )
    def test_main_cdist_uses_numpy(self):
        """Bug: main.py:95 passes r_domain_np (numpy) to torch.cdist."""
        src_path = os.path.join(REPO_ROOT, "main.py")
        with open(src_path) as f:
            source = f.read()
        import re
        bug_pattern = re.search(r'torch\.cdist\(r_domain_np,\s*r_domain_np\)', source)
        assert bug_pattern is None, \
            "BUG FOUND: torch.cdist(r_domain_np, r_domain_np) in main.py"

    @pytest.mark.xfail(
        reason="Same torch.cdist numpy bug as gen_data.py"
    )
    def test_evaluate_cdist_uses_numpy(self):
        """Bug: evaluate.py:76 passes r_domain_np (numpy) to torch.cdist."""
        src_path = os.path.join(REPO_ROOT, "evaluate.py")
        with open(src_path) as f:
            source = f.read()
        import re
        bug_pattern = re.search(r'torch\.cdist\(r_domain_np,\s*r_domain_np\)', source)
        assert bug_pattern is None, \
            "BUG FOUND: torch.cdist(r_domain_np, r_domain_np) in evaluate.py"

    @pytest.mark.xfail(
        reason="Same torch.cdist numpy bug as gen_data.py"
    )
    def test_train_cdist_uses_numpy(self):
        """Bug: train.py:93 passes r_domain_np (numpy) to torch.cdist."""
        src_path = os.path.join(REPO_ROOT, "train.py")
        with open(src_path) as f:
            source = f.read()
        import re
        bug_pattern = re.search(r'torch\.cdist\(r_domain_np,\s*r_domain_np\)', source)
        assert bug_pattern is None, \
            "BUG FOUND: torch.cdist(r_domain_np, r_domain_np) in train.py"


# ══════════════════════════════════════════════════════════════════
# BUG #2: Amplitude scaling inconsistency between train/inference
# ══════════════════════════════════════════════════════════════════
#
# train.py uses scaling factor 500.0 for BP (line 153)
# main.py  uses scaling factor 10.0  for BP (line 167)
# evaluate.py uses scaling factor 500.0 for BP (line 131)
#
# main.py and evaluate.py/train.py disagree. This means inference
# with main.py on the same data will produce different results
# than what was trained — the BP input dynamic range won't match
# what the network expects.
# ==================================================================

class TestBugBPScalingInconsistency:
    """Verify BP amplitude scaling consistency."""

    @pytest.mark.xfail(
        reason="BUG: main.py uses BP scaling factor 10.0 for 2D, "
               "but train.py and evaluate.py use 500.0. "
               "This mismatch will produce incorrect inference results."
    )
    def test_bp_scaling_consistent_2d(self):
        """main.py and train.py should use the same BP scaling factor for 2D."""
        import re
        # Get main.py scaling
        main_src = open(os.path.join(REPO_ROOT, "main.py")).read()
        m = re.search(r'bp_gpu = bp_gpu \* \(([\d.]+)', main_src)
        main_scale = float(m.group(1)) if m else None

        # Get train.py scaling
        train_src = open(os.path.join(REPO_ROOT, "train.py")).read()
        m = re.search(r'bp_gpu = bp_gpu \* ([\d.]+)', train_src)
        train_scale = float(m.group(1)) if m else None

        # Get evaluate.py scaling
        eval_src = open(os.path.join(REPO_ROOT, "evaluate.py")).read()
        m = re.search(r'bp_gpu = bp_gpu \* ([\d.]+)', eval_src)
        eval_scale = float(m.group(1)) if m else None

        assert main_scale == train_scale == eval_scale, \
            f"BP scaling mismatch: main={main_scale}, train={train_scale}, eval={eval_scale}"


# ══════════════════════════════════════════════════════════════════
# BUG #3: Missing error handling for 3D mode with half_circle
# ══════════════════════════════════════════════════════════════════
#
# main.py accepts -d 3d -m half_circle, but the ElectromagneticsEngine
# 3D path doesn't implement half_circle geometry setup.
# ==================================================================

class TestBug3DHalfCircle:
    """Verify 3D antenna geometry handling."""

    @pytest.mark.xfail(
        reason="BUG: 3D ElectromagneticsEngine branch doesn't validate mode "
               "(half_circle silently ignored, always uses Fibonacci sphere)"
    )
    def test_3d_mode_only_has_full_circle(self):
        """main.py ElectromagneticsEngine 3D path ignores mode/half_circle."""
        src_path = os.path.join(REPO_ROOT, "main.py")
        with open(src_path) as f:
            source = f.read()
        # In the 3D branch (else), there's no check for mode/full_circle/half_circle
        # This means 3D half_circle silently uses full-circle Fibonacci sphere
        import re
        else_block = source.split("else:  # 3D")[1].split("# Precompute")[0] if "else:  # 3D" in source else ""
        has_mode_check = "mode" in else_block and "half_circle" in else_block
        assert has_mode_check, \
            "BUG: 3D ElectromagneticsEngine branch doesn't validate mode " \
            "(half_circle silently ignored, always uses Fibonacci sphere)"


# ══════════════════════════════════════════════════════════════════
# BUG #4: Dataset naming inconsistency between gen_data_3d and train_3d
# ══════════════════════════════════════════════════════════════════
#
# gen_data_3d.py saves data as sphere_train_X.npy (3D permittivity: Nx32x32x32)
# but the X (scattered field) shape is (N, 2, 64, 64) — same 2D antenna order.
# This is fine since it's 64 transmitter x 64 receiver antenna.
# The 64 came from N_ANT = 64. shape (N, 2, N_ANT, N_ANT).
# ==================================================================


# ══════════════════════════════════════════════════════════════════
# BUG #5: Hardcoded background permittivity in main.py ElectromagneticsEngine
# ══════════════════════════════════════════════════════════════════
#
# main.py's ElectromagneticsEngine.__init__ hardcodes EPS_BG=1.5
# on line 76, and also hardcodes chi_bg = eps_bg - 1.0 (line 150)
# with eps_bg = torch.ones(...) * 1.5 (line 149).
# There is no command-line --eps_bg argument for main.py.
# This means inference always uses eps_bg=1.5 regardless of training data.
# ==================================================================

class TestBugMainEpsBgHardcoded:
    """Verify eps_bg is configurable in main.py."""

    @pytest.mark.xfail(
        reason="BUG: main.py ElectromagneticsEngine hardcodes EPS_BG=1.5 "
               "(line 76) and has no --eps_bg CLI argument. "
               "Inference cannot match training if eps_bg differs."
    )
    def test_main_has_eps_bg_argument(self):
        """main.py should accept --eps_bg command line argument."""
        src_path = os.path.join(REPO_ROOT, "main.py")
        with open(src_path) as f:
            source = f.read()
        assert "--eps_bg" in source, \
            "BUG: main.py has no --eps_bg argument, " \
            "always uses eps_bg=1.5 hardcoded in ElectromagneticsEngine"


# ══════════════════════════════════════════════════════════════════
# BUG #6: evaluate.py dies on CPU (torch.cdist npy + device issue)
# ══════════════════════════════════════════════════════════════════
#
# When evaluate.py runs with --cpu, it crashes during physics init
# because torch.cdist(r_domain_np, r_domain_np) passes a numpy array
# (same as BUG #1). Combined with the slow numpy->cpu path,
# it produces a hard-to-debug crash.
# ==================================================================