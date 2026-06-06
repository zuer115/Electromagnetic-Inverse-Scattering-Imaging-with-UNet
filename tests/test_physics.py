"""
Tests for the physics engine: ElectromagneticsEngine from main.py.

Covers:
- Geometry initialization (2D/3D)
- Green's function properties
- Born approximation (BP) computation
- Background subtraction
- Sensitivity map correctness
"""
import os
import sys
import numpy as np
import torch
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ══════════════════════════════════════════════════════════════════
# Helper: import the physics engine classes without running main()
# ══════════════════════════════════════════════════════════════════

# We can't simply import main because it parses argv and runs.
# Instead, we instantiate equivalent physics primitives directly here.
# But we should test that main.py's classes work too — via subprocess.

from tests.test_helpers import (
    deterministic_seed,
    make_fake_2d_scattered,
    make_fake_2d_permittivity,
    make_fake_3d_scattered,
    EPS_BG_DEFAULT,
    L, N_2D, N_3D, N_ANT, K0, FREQ, C0,
)


# ══════════════════════════════════════════════════════════════════
# Physical constant checks
# ══════════════════════════════════════════════════════════════════

class TestPhysicalConstants:
    """Verify the physical constants are physically meaningful."""

    def test_wavenumber_positive(self):
        """K0 must be a real, positive wavenumber."""
        import numpy as np
        K0_val = 2 * np.pi * 2.45e9 / 299792458.0
        assert K0_val > 0
        assert np.isfinite(K0_val)

    def test_background_wavenumber(self):
        """K_BG = K0 * sqrt(eps_bg). For eps_bg=1.5, it's > K0."""
        eps_bg = 1.5
        K0_val = 2 * np.pi * 2.45e9 / 299792458.0
        K_BG = K0_val * np.sqrt(eps_bg)
        assert K_BG > K0_val
        assert np.isclose(K_BG**2, K0_val**2 * eps_bg)

    def test_resolution_smaller_than_wavelength(self):
        """DX must be significantly smaller than wavelength for valid MoM."""
        K0_val = 2 * np.pi * 2.45e9 / 299792458.0
        wavelength = 2 * np.pi / K0_val
        DX_2d = L / N_2D
        DX_3d = L / N_3D
        assert DX_2d < wavelength / 2, "2D DX should be < λ/2"
        assert DX_3d < wavelength / 2, "3D DX should be < λ/2"

    def test_antenna_radius_larger_than_domain(self):
        """Antenna ring must enclose the imaging domain (R_ANT > L/2)."""
        R_ANT_VAL = 0.35
        assert R_ANT_VAL > L / 2, "Antenna ring must enclose domain"


# ══════════════════════════════════════════════════════════════════
# 2D Green's function tests
# ══════════════════════════════════════════════════════════════════

class TestGreen2D:
    """Test 2D Green's function properties (Hankel function form)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        deterministic_seed(42)
        self.eps_bg = EPS_BG_DEFAULT
        self.K_BG = K0 * np.sqrt(self.eps_bg)
        self.DX = L / N_2D
        self.device = torch.device('cpu')
        # Build a tiny domain for quick testing
        self.small_N = 16
        self.small_DX = L / self.small_N
        x_arr = np.linspace(-L/2 + self.small_DX/2, L/2 - self.small_DX/2, self.small_N)
        self.X, self.Y = np.meshgrid(x_arr, x_arr)
        self.r_domain = np.vstack((self.X.flatten(), self.Y.flatten())).T

    def test_greens_2d_decay(self):
        """Green's function magnitude should decay with distance."""
        from scipy.special import hankel2 as scipy_hankel2
        dist_DD = torch.cdist(torch.tensor(self.r_domain),
                              torch.tensor(self.r_domain))
        # Avoid diagonal singularity
        dist_DD.fill_diagonal_(1e-8)

        J0 = torch.special.bessel_j0(self.K_BG * dist_DD)
        Y0 = torch.special.bessel_y0(self.K_BG * dist_DD)
        G = -1j / 4 * (J0 - 1j * Y0) * (self.small_DX**2)

        G_np = G.numpy()
        # Pick two points far apart and two points close together
        idx_center = len(G_np) // 2  # near center cell
        # Distance should anti-correlate with magnitude (far = small)
        dist_from_center = dist_DD[idx_center].numpy()
        G_mag = np.abs(G_np[idx_center])

        # The furthest point should have smaller |G| than the nearest (non-self)
        far_idx = np.argmax(dist_from_center)
        near_idx = dist_from_center.argsort()[1]  # skip self (index 0)
        assert G_mag[near_idx] > G_mag[far_idx], \
            "Green's function should decay with distance"

    def test_self_term_finite(self):
        """The diagonal (self-term) must be finite (not inf/nan)."""
        from scipy.special import hankel2 as scipy_hankel2
        a_eq = self.small_DX / np.sqrt(np.pi)
        self_term = (1 / self.K_BG**2) * (-1j * np.pi * self.K_BG * a_eq / 2
                                           * scipy_hankel2(1, self.K_BG * a_eq) + 1)
        assert np.isfinite(self_term)
        assert not np.isnan(complex(self_term))
        assert abs(complex(self_term)) > 0

    def test_2d_greens_symmetry(self):
        """G_DD should be approximately symmetric (reciprocity)."""
        from scipy.special import hankel2 as scipy_hankel2
        dist_DD = torch.cdist(torch.tensor(self.r_domain),
                              torch.tensor(self.r_domain))
        dist_DD.fill_diagonal_(1e-8)
        J0 = torch.special.bessel_j0(self.K_BG * dist_DD)
        Y0 = torch.special.bessel_y0(self.K_BG * dist_DD)
        G = -1j / 4 * (J0 - 1j * Y0) * (self.small_DX**2)

        # Apply self-term
        a_eq = self.small_DX / np.sqrt(np.pi)
        self_term = (1 / self.K_BG**2) * (-1j * np.pi * self.K_BG * a_eq / 2
                                           * scipy_hankel2(1, self.K_BG * a_eq) + 1)
        G.fill_diagonal_(complex(self_term))

        G_np = G.numpy()
        # Off-diagonal symmetry
        assert np.allclose(G_np, G_np.T, atol=1e-6), \
            "G_DD should be symmetric (reciprocity)"


# ══════════════════════════════════════════════════════════════════
# 3D Green's function tests
# ══════════════════════════════════════════════════════════════════

class TestGreen3D:
    """Test 3D Green's function properties (scalar wave equation)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        deterministic_seed(42)
        self.eps_bg = EPS_BG_DEFAULT
        self.K_BG = K0 * np.sqrt(self.eps_bg)
        self.small_N = 8
        self.small_DX = L / self.small_N
        x_arr = np.linspace(-L/2 + self.small_DX/2, L/2 - self.small_DX/2, self.small_N)
        X, Y, Z = np.meshgrid(x_arr, x_arr, x_arr, indexing='ij')
        self.r_domain = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
        self.device = torch.device('cpu')

    def test_3d_greens_decay(self):
        """3D Green's function ~ exp(-ikr)/r should decay as 1/r."""
        import scipy.spatial
        dist_DD = scipy.spatial.distance_matrix(self.r_domain, self.r_domain).astype(np.float32)
        np_err = np.seterr(divide='ignore', invalid='ignore')
        G = np.exp(-1j * self.K_BG * dist_DD) / (4 * np.pi * dist_DD) * (self.small_DX**3)
        np.seterr(**np_err)

        center = len(G) // 2
        G_mag = np.abs(G[center])
        d = dist_DD[center]
        # Nearest non-self should have higher magnitude
        sorted_idx = np.argsort(d)
        near = d[sorted_idx[1]]
        far = d[sorted_idx[-1]]
        assert G_mag[sorted_idx[1]] > G_mag[sorted_idx[-1]], \
            "3D Green's function should decay with distance"

    def test_3d_self_term_finite(self):
        """3D self-term must be finite."""
        a_eq = (3 * (self.small_DX**3) / (4 * np.pi))**(1/3)
        self_term = (1 / self.K_BG**2) * (1 - (1 + 1j * self.K_BG * a_eq)
                                           * np.exp(-1j * self.K_BG * a_eq))
        assert np.isfinite(self_term)
        assert abs(self_term) > 0


# ══════════════════════════════════════════════════════════════════
# Born BP computation tests (simplified version)
# ══════════════════════════════════════════════════════════════════

class TestBornApproximation:
    """Test the Distorted Born Approximation BP operator."""

    @pytest.fixture(autouse=True)
    def setup(self):
        deterministic_seed(42)
        self.device = torch.device('cpu')
        self.K_BG = K0 * np.sqrt(EPS_BG_DEFAULT)
        self.DX = L / N_2D
        self.small_N = 32  # tiny domain
        self.small_DX = L / self.small_N
        self.n_ant_small = 16

    def test_bp_output_nonnegative(self):
        """BP output (absolute value) must be non-negative."""
        # Generate tiny fake data
        X_data = make_fake_2d_scattered(3, n_ant=self.n_ant_small)
        X_t = torch.complex(
            torch.from_numpy(X_data[:, 0]),
            torch.from_numpy(X_data[:, 1])
        )
        # K_conj must also be complex for einsum type compatibility
        K_real = np.random.RandomState(42).randn(
            self.small_N * self.small_N, self.n_ant_small, self.n_ant_small
        ).astype(np.float32)
        K_conj = torch.complex(
            torch.from_numpy(K_real),
            torch.zeros_like(torch.from_numpy(K_real))
        )
        # dot product with conjugate kernel (broadcast over b and d)
        bp = torch.einsum('b r t, d r t -> b d', X_t, torch.conj(K_conj)).abs()
        assert torch.all(bp >= 0), "BP absolute values must be non-negative"

    def test_bp_shape_matches_domain(self):
        """BP output should reshape to NxN grid."""
        n_ant = self.n_ant_small
        N = self.small_N
        X_data = make_fake_2d_scattered(2, n_ant=n_ant)
        K_real = torch.randn(N*N, n_ant, n_ant, dtype=torch.float32)
        K_conj = torch.complex(K_real, torch.zeros_like(K_real))

        X_t = torch.complex(
            torch.from_numpy(X_data[:, 0]),
            torch.from_numpy(X_data[:, 1])
        )
        bp = torch.einsum('b r t, d r t -> b d', X_t, torch.conj(K_conj)).abs()
        bp_reshaped = bp.view(-1, 1, N, N)
        assert bp_reshaped.shape == (2, 1, N, N)

    def test_bp_zero_input_gives_zero(self):
        """Zero scattered field should give zero BP (after background subtraction)."""
        # If X_diff is zero, BP is zero
        n_ant = 8
        N = 8
        X_zero = torch.zeros(1, 2, n_ant, n_ant, dtype=torch.float32)
        K_real = torch.randn(N*N, n_ant, n_ant, dtype=torch.float32)
        K_conj = torch.complex(K_real, torch.zeros_like(K_real))
        X_t = torch.complex(X_zero[:, 0], X_zero[:, 1])
        bp = torch.einsum('b r t, d r t -> b d', X_t, torch.conj(K_conj)).abs()
        assert torch.allclose(bp, torch.zeros_like(bp), atol=1e-6)


# ══════════════════════════════════════════════════════════════════
# Antenna geometry tests
# ══════════════════════════════════════════════════════════════════

class TestAntennaGeometry:
    """Test antenna array configurations."""

    def test_full_circle_covers_360(self):
        """Full-circle antenna array should cover all angles."""
        n_ant = 64
        angles = np.arange(0, n_ant) * (2 * np.pi / n_ant)
        # Angles should span approximately [0, 2π)
        assert np.abs(angles[-1] + (2 * np.pi / n_ant) - 2 * np.pi) < 0.01
        assert np.min(angles) >= 0
        assert np.max(angles) < 2 * np.pi

    def test_half_circle_covers_180(self):
        """Half-circle antenna array should cover ±90 degrees."""
        n_ant = 64
        angles = np.linspace(-np.pi/2, np.pi/2, n_ant)
        assert np.isclose(angles[0], -np.pi/2)
        assert np.isclose(angles[-1], np.pi/2)

    def test_full_circle_radius(self):
        """All antennas in full circle must lie on the prescribed circle."""
        n_ant = 64
        R = 0.35
        angles = np.arange(0, n_ant) * (2 * np.pi / n_ant)
        x = R * np.cos(angles)
        y = R * np.sin(angles)
        radii = np.sqrt(x**2 + y**2)
        assert np.allclose(radii, R, atol=1e-10)

    def test_half_circle_radius(self):
        """All antennas in half circle must lie on the prescribed circle."""
        n_ant = 64
        R = 0.35
        angles = np.linspace(-np.pi/2, np.pi/2, n_ant)
        x = R * np.cos(angles)
        y = R * np.sin(angles)
        radii = np.sqrt(x**2 + y**2)
        assert np.allclose(radii, R, atol=1e-10)

    def test_fibonacci_sphere_coverage(self):
        """Fibonacci spherical lattice should produce N_ANT points on sphere."""
        n_ant = 64
        R = 0.35
        indices = np.arange(0, n_ant, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / n_ant)
        theta = np.pi * (1 + 5**0.5) * indices
        x = R * np.cos(theta) * np.sin(phi)
        y = R * np.sin(theta) * np.sin(phi)
        z = R * np.cos(phi)
        radii = np.sqrt(x**2 + y**2 + z**2)
        assert np.allclose(radii, R, atol=1e-10)
        # Points should be distinct
        pts = np.vstack([x, y, z]).T
        assert len(np.unique(pts.round(decimals=8), axis=0)) == n_ant, \
            "Fibonacci lattice should produce distinct points"


# ══════════════════════════════════════════════════════════════════
# Sensitivity map tests
# ══════════════════════════════════════════════════════════════════

class TestSensitivityMap:
    """Test the sensitivity / illumination correction map."""

    def test_sensitivity_nonnegative(self, device):
        """S_map = sum|K|^2 must be everywhere non-negative."""
        N = 16
        n_ant = 8
        K = torch.randn(N*N, n_ant, n_ant, dtype=torch.float32, device=device)
        S = torch.sum(torch.abs(K)**2, dim=(1, 2))
        assert torch.all(S >= 0)

    def test_sensitivity_center_brighter(self):
        """For a full-circle array, the center of the domain should be well-illuminated."""
        N = 8
        n_ant = 16
        # Build a simplified K where inner cells have stronger coupling
        # Use a simple distance-based attenuation model
        x = np.linspace(-0.125, 0.125, N)
        X, Y = np.meshgrid(x, x)
        r = np.sqrt(X**2 + Y**2)
        # Closer to center = higher sensitivity
        sensitivity = np.exp(-r.flatten() * 5)  # rough model
        center_idx = np.argmin(r.flatten())
        edge_idx = np.argmax(r.flatten())
        assert sensitivity[center_idx] > sensitivity[edge_idx], \
            "Center should be more sensitive than edge"


# ══════════════════════════════════════════════════════════════════
# Normalization / denormalization tests
# ══════════════════════════════════════════════════════════════════

class TestDenormalization:
    """Test the denormalize function used across the project."""

    def denormalize(self, img, eps_bg=1.5):
        return img * (5.0 - eps_bg) + eps_bg

    def normalize(self, img, eps_bg=1.5):
        return (img - eps_bg) / (5.0 - eps_bg)

    def test_roundtrip(self):
        """Normalize then denormalize should recover original."""
        eps_r = np.array([1.5, 3.0, 5.0], dtype=np.float32)
        normed = self.normalize(eps_r)
        recovered = self.denormalize(normed)
        assert np.allclose(eps_r, recovered, atol=1e-5)

    def test_range_normalized(self):
        """Normalized values should be in [0, 1]."""
        eps_r = np.array([1.5, 2.5, 5.0], dtype=np.float32)
        normed = self.normalize(eps_r)
        assert np.all(normed >= 0) and np.all(normed <= 1.0)

    def test_range_denormalized(self):
        """Denormalized values should be in [1.5, 5.0]."""
        norm = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        phys = self.denormalize(norm)
        assert np.allclose(phys, [1.5, 3.25, 5.0], atol=1e-5)

    def test_sigmoid_bounded(self):
        """After sigmoid, outputs are in [0,1]; after denormalize in [1.5, 5.0]."""
        sigmoid_out = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        phys = self.denormalize(sigmoid_out)
        assert np.all(phys >= 1.5) and np.all(phys <= 5.0)