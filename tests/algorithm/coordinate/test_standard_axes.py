"""标准坐标轴实现单元测试。

覆盖 ICRSAxes 恒等旋转与 IAU2000EqAxes 正交性。
"""

import numpy as np
import pytest

from e2m2e.algorithm.coordinate.standard_axes import IAU2000EqAxes, ICRSAxes

pytestmark = pytest.mark.data


class TestICRSAxes:
    """ICRSAxes 测试。"""

    def test_rotation_matrix_is_identity(self):
        """ICRF 旋转矩阵为单位矩阵。"""
        axes = ICRSAxes()
        np.testing.assert_allclose(axes.rotation_matrix(et=0.0), np.eye(3), atol=1e-14)

    def test_angular_velocity_is_zero(self):
        """ICRF 角速度为零。"""
        axes = ICRSAxes()
        np.testing.assert_allclose(axes.angular_velocity(et=0.0), np.zeros(3), atol=1e-14)

    def test_rotation_matrix_orthogonal(self):
        """ICRF 旋转矩阵正交。"""
        axes = ICRSAxes()
        r = axes.rotation_matrix(et=123.45)
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-14)


class TestIAU2000EqAxes:
    """IAU2000EqAxes 测试。"""

    def test_rotation_matrix_orthogonal(self):
        """IAU2000Eq 旋转矩阵正交。"""
        axes = IAU2000EqAxes()
        r = axes.rotation_matrix(et=0.0)
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)

    def test_rotation_matrix_at_j2000_is_identity(self):
        """在 J2000 时刻，IAU2000Eq 旋转矩阵近似为单位矩阵。"""
        axes = IAU2000EqAxes()
        r = axes.rotation_matrix(et=0.0)
        np.testing.assert_allclose(r, np.eye(3), atol=1e-10)

    def test_angular_velocity_is_small(self):
        """IAU2000Eq 角速度很小（地球定向速率量级）。"""
        axes = IAU2000EqAxes()
        omega = axes.angular_velocity(et=0.0)
        assert np.linalg.norm(omega) < 1e-10  # rad/s

    def test_precession_angle_reasonable(self):
        """10 年后的岁差角在预期的角秒范围内。"""
        from e2m2e.algorithm.coordinate.iau_2006 import precession_angles

        t = 10.0 / 100.0  # 10 years in Julian centuries
        zeta, theta, z = precession_angles(t)
        # 10 年后各角应约为 200 角秒
        assert 100 * np.pi / (180 * 3600) < abs(zeta) < 1000 * np.pi / (180 * 3600)
        assert 100 * np.pi / (180 * 3600) < abs(theta) < 1000 * np.pi / (180 * 3600)
        assert 100 * np.pi / (180 * 3600) < abs(z) < 1000 * np.pi / (180 * 3600)
