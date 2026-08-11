"""pseudo_potential_hessian 及其集成测试。

验证 Hessian 形状、对称性、有限差分一致性，
以及 L1/L2/L3 稳定性指数回归值。
"""

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System, LibrationPoint
from e2m2e.algorithm.dynamics.potential import pseudo_potential_hessian
from e2m2e.data.constants import Datum

pytestmark = pytest.mark.theory


class TestPseudoPotentialHessian:
    """Property-based tests for the Hessian function."""

    def test_shape(self):
        H = pseudo_potential_hessian(0.012, 0.5, 0.3, 0.1)
        assert H.shape == (3, 3)

    @pytest.mark.parametrize(
        "x, y, z",
        [
            (0.5, 0.0, 0.0),
            (0.8, 0.2, 0.0),
            (0.5, 0.3, 0.1),
            (1.0, 0.5, -0.3),
            (0.1, -0.4, 0.7),
        ],
    )
    def test_symmetry(self, x, y, z):
        mu = Datum.DE421.mu
        H = pseudo_potential_hessian(mu, x, y, z)
        np.testing.assert_allclose(H, H.T, atol=1e-14)

    @pytest.mark.parametrize(
        "x, y, z",
        [
            (0.5, 0.0, 0.0),
            (0.8, 0.2, 0.0),
            (0.5, 0.3, 0.1),
            (1.0, 0.5, -0.3),
            (0.1, -0.4, 0.7),
        ],
    )
    def test_finite_difference_consistency(self, x, y, z):
        """Hessian entries should agree with central finite-difference second derivatives."""
        mu = Datum.DE421.mu
        eps = 1e-5
        H = pseudo_potential_hessian(mu, x, y, z)

        def omega(x_, y_, z_):
            r1 = np.sqrt((x_ + mu) ** 2 + y_**2 + z_**2)
            r2 = np.sqrt((x_ - 1 + mu) ** 2 + y_**2 + z_**2)
            return 0.5 * (x_**2 + y_**2) + (1 - mu) / r1 + mu / r2

        # U_xx
        fd_xx = (omega(x + eps, y, z) - 2 * omega(x, y, z) + omega(x - eps, y, z)) / eps**2
        np.testing.assert_allclose(H[0, 0], fd_xx, rtol=1e-4, atol=1e-8)

        # U_yy
        fd_yy = (omega(x, y + eps, z) - 2 * omega(x, y, z) + omega(x, y - eps, z)) / eps**2
        np.testing.assert_allclose(H[1, 1], fd_yy, rtol=1e-4, atol=1e-8)

        # U_zz
        fd_zz = (omega(x, y, z + eps) - 2 * omega(x, y, z) + omega(x, y, z - eps)) / eps**2
        np.testing.assert_allclose(H[2, 2], fd_zz, rtol=1e-4, atol=1e-8)

        # U_xy
        fd_xy = (
            omega(x + eps, y + eps, z)
            - omega(x + eps, y - eps, z)
            - omega(x - eps, y + eps, z)
            + omega(x - eps, y - eps, z)
        ) / (4 * eps**2)
        np.testing.assert_allclose(H[0, 1], fd_xy, rtol=1e-5)

        # U_xz
        fd_xz = (
            omega(x + eps, y, z + eps)
            - omega(x + eps, y, z - eps)
            - omega(x - eps, y, z + eps)
            + omega(x - eps, y, z - eps)
        ) / (4 * eps**2)
        np.testing.assert_allclose(H[0, 2], fd_xz, rtol=1e-5)

        # U_yz
        fd_yz = (
            omega(x, y + eps, z + eps)
            - omega(x, y + eps, z - eps)
            - omega(x, y - eps, z + eps)
            + omega(x, y - eps, z - eps)
        ) / (4 * eps**2)
        np.testing.assert_allclose(H[1, 2], fd_yz, rtol=1e-5)

    def test_diagonal_at_origin(self):
        mu = Datum.DE421.mu
        H = pseudo_potential_hessian(mu, 0.0, 0.0, 0.0)
        assert H[0, 2] == 0.0
        assert H[1, 2] == 0.0


class TestRegressionStabilityIndex:
    """Stability indices at L1/L2/L3 with DE421 datum."""

    @pytest.fixture
    def system(self):
        sys = CR3BP_System(
            mu=Datum.DE421.mu, primary="Earth", secondary="Moon"
        )._with_default_scales()
        sys.compute_libration_points()
        return sys

    @pytest.mark.parametrize(
        "lp, expected_max_real",
        [
            (LibrationPoint.L1, 2.932055930434),
            (LibrationPoint.L2, 2.158674322705),
            (LibrationPoint.L3, 0.177875357099),
        ],
    )
    def test_max_real_part(self, system, lp, expected_max_real):
        result = system.compute_stability_index(lp)
        np.testing.assert_allclose(result["max_real_part"], expected_max_real, rtol=1e-10)

    @pytest.mark.parametrize(
        "lp, expected_max_imag",
        [
            (LibrationPoint.L1, 2.334385883065),
            (LibrationPoint.L2, 1.862645863558),
            (LibrationPoint.L3, 1.010419895129),
        ],
    )
    def test_max_imag_part(self, system, lp, expected_max_imag):
        result = system.compute_stability_index(lp)
        np.testing.assert_allclose(result["max_imag_part"], expected_max_imag, rtol=1e-10)

    def test_collinear_points_unstable(self, system):
        for lp in [LibrationPoint.L1, LibrationPoint.L2, LibrationPoint.L3]:
            result = system.compute_stability_index(lp)
            assert not result["is_stable"]

    @pytest.mark.parametrize(
        "lp, expected_uxx",
        [
            (LibrationPoint.L1, 11.295189056284),
            (LibrationPoint.L2, 7.380850436958),
            (LibrationPoint.L3, 3.021382556380),
        ],
    )
    def test_a_matrix_uxx(self, system, lp, expected_uxx):
        result = system.compute_stability_index(lp)
        A = result["linear_matrix"]
        np.testing.assert_allclose(A[3, 0], expected_uxx, rtol=1e-10)


class TestRegressionJacobianA:
    """compute_jacobian_A must produce consistent results after refactor."""

    @pytest.fixture
    def dynamics(self):
        from e2m2e.algorithm.dynamics import CR3BP_System

        sys = CR3BP_System(
            mu=Datum.DE421.mu, primary="Earth", secondary="Moon"
        )._with_default_scales()
        return CR3BP_Dynamics(system=sys)

    def test_jacobian_shape(self, dynamics):
        state = np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0])
        A = dynamics.compute_jacobian_A(state)
        assert A.shape == (6, 6)

    def test_jacobian_structure(self, dynamics):
        state = np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0])
        A = dynamics.compute_jacobian_A(state)
        np.testing.assert_array_equal(A[:3, :3], np.zeros((3, 3)))
        np.testing.assert_array_equal(A[:3, 3:], np.eye(3))
        assert A[3, 4] == pytest.approx(2.0)
        assert A[4, 3] == pytest.approx(-2.0)

    def test_jacobian_hessian_block_matches(self, dynamics):
        """The 3x3 Hessian block in A should match pseudo_potential_hessian."""
        state = np.array([0.5, 0.3, 0.1, 0.0, 0.1, 0.0])
        A = dynamics.compute_jacobian_A(state)
        mu = dynamics.system.mu
        H = pseudo_potential_hessian(mu, 0.5, 0.3, 0.1)
        np.testing.assert_allclose(A[3:, :3], H)
