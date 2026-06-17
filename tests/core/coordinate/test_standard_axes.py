"""标准坐标轴实现单元测试。

覆盖 ICRSAxes 恒等旋转与 IAU2000EqAxes 正交性。
"""

import numpy as np

from e2m2e.core.standard_axes import IAU2000EqAxes, ICRSAxes


class TestICRSAxes:
    """Tests for ICRSAxes."""

    def test_rotation_matrix_is_identity(self):
        """ICRF rotation matrix is the identity matrix."""
        axes = ICRSAxes()
        np.testing.assert_allclose(axes.rotation_matrix(et=0.0), np.eye(3), atol=1e-14)

    def test_angular_velocity_is_zero(self):
        """ICRF angular velocity is zero."""
        axes = ICRSAxes()
        np.testing.assert_allclose(axes.angular_velocity(et=0.0), np.zeros(3), atol=1e-14)

    def test_rotation_matrix_orthogonal(self):
        """ICRF rotation matrix is orthogonal."""
        axes = ICRSAxes()
        r = axes.rotation_matrix(et=123.45)
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-14)


class TestIAU2000EqAxes:
    """Tests for IAU2000EqAxes."""

    def test_rotation_matrix_orthogonal(self):
        """IAU2000Eq rotation matrix is orthogonal."""
        axes = IAU2000EqAxes()
        r = axes.rotation_matrix(et=0.0)
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)

    def test_rotation_matrix_at_j2000_is_identity(self):
        """At J2000, the IAU2000Eq rotation matrix is approximately identity."""
        axes = IAU2000EqAxes()
        r = axes.rotation_matrix(et=0.0)
        np.testing.assert_allclose(r, np.eye(3), atol=1e-10)

    def test_angular_velocity_is_small(self):
        """IAU2000Eq angular velocity is small (Earth orientation rate)."""
        axes = IAU2000EqAxes()
        omega = axes.angular_velocity(et=0.0)
        assert np.linalg.norm(omega) < 1e-10  # rad/s

    def test_precession_angle_reasonable(self):
        """Precession angle at 10 years is in the expected arcsecond range."""
        from e2m2e.core.iau_2006 import precession_angles

        t = 10.0 / 100.0  # 10 years in Julian centuries
        zeta, theta, z = precession_angles(t)
        # After 10 years, each angle should be around 200 arcseconds
        assert 100 * np.pi / (180 * 3600) < abs(zeta) < 1000 * np.pi / (180 * 3600)
        assert 100 * np.pi / (180 * 3600) < abs(theta) < 1000 * np.pi / (180 * 3600)
        assert 100 * np.pi / (180 * 3600) < abs(z) < 1000 * np.pi / (180 * 3600)
