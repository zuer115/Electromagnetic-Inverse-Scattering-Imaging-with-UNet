"""
Regression tests that verify previously discovered bugs are now fixed.
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
# BUG #1 (FIXED): torch.cdist called with numpy arrays
# ==================================================================
# In gen_data.py:99, main.py:95, evaluate.py:76, train.py:93,
# torch.cdist(r_domain_np, r_domain_np) used numpy array input.
# Fixed by replacing r_domain_np with r_domain_gpu.
# ==================================================================

class TestBugTorchCdistNumpyInput:
    """Verify torch.cdist numpy-input bug is fixed across all files."""

    def test_cdist_accepts_tensor(self):
        """torch.cdist should accept tensor inputs (sanity check)."""
        t = torch.randn(10, 3)
        d = torch.cdist(t, t)
        assert d.shape == (10, 10)
        assert torch.all(d >= 0)

    def test_gen_data_cdist_fixed(self):
        """gen_data.py no longer passes numpy to torch.cdist."""
        src_path = os.path.join(REPO_ROOT, "gen_data.py")
        with open(src_path) as f:
            source = f.read()
        import re
        bug = re.search(r'torch\.cdist\(r_domain_np,\s*r_domain_np\)', source)
        assert bug is None, \
            "BUG still present: torch.cdist(r_domain_np, r_domain_np) in gen_data.py"

    def test_main_cdist_fixed(self):
        """main.py no longer passes numpy to torch.cdist."""
        src_path = os.path.join(REPO_ROOT, "main.py")
        with open(src_path) as f:
            source = f.read()
        import re
        bug = re.search(r'torch\.cdist\(r_domain_np,\s*r_domain_np\)', source)
        assert bug is None, \
            "BUG still present: torch.cdist(r_domain_np, r_domain_np) in main.py"

    def test_evaluate_cdist_fixed(self):
        """evaluate.py no longer passes numpy to torch.cdist."""
        src_path = os.path.join(REPO_ROOT, "evaluate.py")
        with open(src_path) as f:
            source = f.read()
        import re
        bug = re.search(r'torch\.cdist\(r_domain_np,\s*r_domain_np\)', source)
        assert bug is None, \
            "BUG still present: torch.cdist(r_domain_np, r_domain_np) in evaluate.py"

    def test_train_cdist_fixed(self):
        """train.py no longer passes numpy to torch.cdist."""
        src_path = os.path.join(REPO_ROOT, "train.py")
        with open(src_path) as f:
            source = f.read()
        import re
        bug = re.search(r'torch\.cdist\(r_domain_np,\s*r_domain_np\)', source)
        assert bug is None, \
            "BUG still present: torch.cdist(r_domain_np, r_domain_np) in train.py"


# ══════════════════════════════════════════════════════════════════
# BUG #2 (FIXED): BP amplitude scaling inconsistency
# ==================================================================
# main.py used 10.0 for 2D BP scaling, but train.py/evaluate.py
# use 500.0. Fixed: main.py now uses 500.0.
# ==================================================================

class TestBugBPScalingInconsistency:
    """Verify BP scaling is now consistent across all scripts."""

    def test_bp_scaling_consistent_2d(self):
        """main.py, train.py, and evaluate.py should all use 500.0 for 2D."""
        import re
        main_src = open(os.path.join(REPO_ROOT, "main.py")).read()
        m = re.search(r'bp_gpu = bp_gpu \* \(([\d.]+)', main_src)
        main_scale = float(m.group(1)) if m else None

        train_src = open(os.path.join(REPO_ROOT, "train.py")).read()
        m = re.search(r'bp_gpu = bp_gpu \* ([\d.]+)', train_src)
        train_scale = float(m.group(1)) if m else None

        eval_src = open(os.path.join(REPO_ROOT, "evaluate.py")).read()
        m = re.search(r'bp_gpu = bp_gpu \* ([\d.]+)', eval_src)
        eval_scale = float(m.group(1)) if m else None

        assert main_scale == train_scale == eval_scale == 500.0, \
            f"BP scaling mismatch: main={main_scale}, train={train_scale}, eval={eval_scale}"


# ══════════════════════════════════════════════════════════════════
# BUG #3 (FIXED): main.py missing --eps_bg argument
# ==================================================================
# Fixed: main.py now has --eps_bg argument wired into the engine.
# ==================================================================

class TestBugMainEpsBgHardcoded:
    """Verify main.py now accepts --eps_bg."""

    def test_main_has_eps_bg_argument(self):
        """main.py should accept --eps_bg command line argument."""
        src_path = os.path.join(REPO_ROOT, "main.py")
        with open(src_path) as f:
            source = f.read()
        assert "--eps_bg" in source, \
            "BUG still present: main.py is missing --eps_bg argument"

    def test_main_engine_uses_eps_bg_param(self):
        """ElectromagneticsEngine should use the eps_bg parameter."""
        src_path = os.path.join(REPO_ROOT, "main.py")
        with open(src_path) as f:
            source = f.read()
        # Should not have hardcoded eps_bg=1.5 in the background subtraction
        import re
        # Check init accepts eps_bg parameter
        assert "def __init__(self, dim, mode, eps_bg" in source, \
            "BUG: Engine init doesn't accept eps_bg parameter"
        # Check the background subtraction uses self.EPS_BG not hardcoded 1.5
        assert "self.EPS_BG" in source, \
            "BUG: Engine doesn't use self.EPS_BG"


# ══════════════════════════════════════════════════════════════════
# BUG #4 (FIXED): 3D mode silently ignores half_circle
# ==================================================================
# Fixed: ElectromagneticsEngine 3D branch now raises ValueError
# for half_circle mode.
# ==================================================================

class TestBug3DHalfCircle:
    """Verify 3D mode now rejects half_circle."""

    def test_3d_half_circle_raises_error(self):
        """3D ElectromagneticsEngine should raise ValueError for half_circle."""
        src_path = os.path.join(REPO_ROOT, "main.py")
        with open(src_path) as f:
            source = f.read()
        # Should contain a raise or warning for 3D half_circle
        import re
        else_block = source.split("else:  # 3D")[1].split("# Precompute")[0] if "else:  # 3D" in source else ""
        assert "half_circle" in else_block and "raise" in else_block, \
            "BUG still present: 3D mode doesn't reject half_circle"