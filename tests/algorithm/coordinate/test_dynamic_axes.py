"""动态坐标轴单元测试。

覆盖 VNBAxes 与 LVLHAxes 的轴方向、正交性、未 update 异常及状态更新。
"""

import numpy as np
import pytest

from e2m2e.algorithm.coordinate.standard_dynamic_axes import LVLHAxes, VNBAxes

# ---------------------------------------------------------------------------
# 公共状态
# ---------------------------------------------------------------------------

# 简单圆轨道：r = [1, 0, 0], v = [0, 1, 0]（逆时针，角动量沿 +z）
STATE_CIRCULAR = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

# 椭圆轨道：r = [2, 0, 0], v = [0, 0.5, 0]（角动量仍沿 +z）
STATE_ELLIPTIC = np.array([2.0, 0.0, 0.0, 0.0, 0.5, 0.0])

# 倾斜轨道：r = [1, 0, 0], v = [0, 0.5, 0.5]（角动量有 y、z 分量）
STATE_INCLINED = np.array([1.0, 0.0, 0.0, 0.0, 0.5, 0.5])

# 任意非零状态
STATE_ARBITRARY = np.array([1.0, 2.0, 3.0, -0.5, 1.5, 0.8])


# ---------------------------------------------------------------------------
# VNB 测试
# ---------------------------------------------------------------------------


class TestVNBaxes:
    """Tests for VNBAxes."""

    def test_vnb_directions_circular(self):
        """圆轨道：V 沿速度，N 沿角动量，B = V × N。"""
        axes = VNBAxes()
        axes.update(0.0, STATE_CIRCULAR)
        r = axes.rotation_matrix(0.0)

        v_hat = r[:, 0]
        n_hat = r[:, 1]
        b_hat = r[:, 2]

        np.testing.assert_allclose(v_hat, [0.0, 1.0, 0.0], atol=1e-14)
        np.testing.assert_allclose(n_hat, [0.0, 0.0, 1.0], atol=1e-14)
        np.testing.assert_allclose(b_hat, [1.0, 0.0, 0.0], atol=1e-14)

    def test_vnb_rotation_matrix_orthogonal(self):
        """VNB 旋转矩阵正交：R @ R.T = I。"""
        axes = VNBAxes()
        axes.update(0.0, STATE_ARBITRARY)
        r = axes.rotation_matrix(0.0)
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-14)

    def test_vnb_update_changes_directions(self):
        """update 不同状态后轴方向正确更新。"""
        axes = VNBAxes()
        axes.update(0.0, STATE_CIRCULAR)
        r1 = axes.rotation_matrix(0.0)

        axes.update(0.0, STATE_ELLIPTIC)
        r2 = axes.rotation_matrix(0.0)

        # 椭圆轨道速度方向不变（仍沿 +y），但大小不同不影响 v_hat
        np.testing.assert_allclose(r1[:, 0], r2[:, 0], atol=1e-14)
        # 角动量方向不变（仍沿 +z）
        np.testing.assert_allclose(r1[:, 1], r2[:, 1], atol=1e-14)

    def test_vnb_unupdated_raises(self):
        """未 update 时调用 rotation_matrix 抛 RuntimeError。"""
        axes = VNBAxes()
        with pytest.raises(RuntimeError):
            axes.rotation_matrix(0.0)

    def test_vnb_rotation_and_rate_unupdated_raises(self):
        """未 update 时调用 rotation_and_rate 抛 RuntimeError。"""
        axes = VNBAxes()
        with pytest.raises(RuntimeError):
            axes.rotation_and_rate(0.0)

    def test_vnb_inclined_orbit(self):
        """倾斜轨道：V、N、B 仍构成右手正交系。"""
        axes = VNBAxes()
        axes.update(0.0, STATE_INCLINED)
        r = axes.rotation_matrix(0.0)
        v_hat = r[:, 0]
        n_hat = r[:, 1]
        b_hat = r[:, 2]

        # 速度归一化
        v = STATE_INCLINED[3:]
        np.testing.assert_allclose(v_hat, v / np.linalg.norm(v), atol=1e-14)
        # 角动量归一化
        h = np.cross(STATE_INCLINED[:3], STATE_INCLINED[3:])
        np.testing.assert_allclose(n_hat, h / np.linalg.norm(h), atol=1e-14)
        # B = V × N
        np.testing.assert_allclose(b_hat, np.cross(v_hat, n_hat), atol=1e-14)


# ---------------------------------------------------------------------------
# LVLH 测试
# ---------------------------------------------------------------------------


class TestLVLHaxes:
    """Tests for LVLHAxes."""

    def test_lvlh_directions_circular(self):
        """圆轨道：R 沿径向，V 沿速度，H 沿角动量。"""
        axes = LVLHAxes()
        axes.update(0.0, STATE_CIRCULAR)
        r = axes.rotation_matrix(0.0)

        r_hat = r[:, 0]
        v_hat = r[:, 1]
        h_hat = r[:, 2]

        np.testing.assert_allclose(r_hat, [1.0, 0.0, 0.0], atol=1e-14)
        np.testing.assert_allclose(v_hat, [0.0, 1.0, 0.0], atol=1e-14)
        np.testing.assert_allclose(h_hat, [0.0, 0.0, 1.0], atol=1e-14)

    def test_lvlh_rotation_matrix_orthogonal(self):
        """LVLH 旋转矩阵正交：R @ R.T = I。"""
        axes = LVLHAxes()
        axes.update(0.0, STATE_ARBITRARY)
        r = axes.rotation_matrix(0.0)
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-14)

    def test_lvlh_update_changes_directions(self):
        """update 不同状态后轴方向正确更新。"""
        axes = LVLHAxes()
        axes.update(0.0, STATE_CIRCULAR)
        r1 = axes.rotation_matrix(0.0)

        axes.update(0.0, STATE_ELLIPTIC)
        r2 = axes.rotation_matrix(0.0)

        # 椭圆轨道径向仍沿 +x，速度仍沿 +y，角动量仍沿 +z
        np.testing.assert_allclose(r1[:, 0], r2[:, 0], atol=1e-14)
        np.testing.assert_allclose(r1[:, 1], r2[:, 1], atol=1e-14)
        np.testing.assert_allclose(r1[:, 2], r2[:, 2], atol=1e-14)

    def test_lvlh_unupdated_raises(self):
        """未 update 时调用 rotation_matrix 抛 RuntimeError。"""
        axes = LVLHAxes()
        with pytest.raises(RuntimeError):
            axes.rotation_matrix(0.0)

    def test_lvlh_rotation_and_rate_unupdated_raises(self):
        """未 update 时调用 rotation_and_rate 抛 RuntimeError。"""
        axes = LVLHAxes()
        with pytest.raises(RuntimeError):
            axes.rotation_and_rate(0.0)

    def test_lvlh_inclined_orbit(self):
        """倾斜轨道：R、V、H 仍构成右手正交系。"""
        axes = LVLHAxes()
        axes.update(0.0, STATE_INCLINED)
        r = axes.rotation_matrix(0.0)
        r_hat = r[:, 0]
        v_hat = r[:, 1]
        h_hat = r[:, 2]

        # 径向归一化
        pos = STATE_INCLINED[:3]
        np.testing.assert_allclose(r_hat, pos / np.linalg.norm(pos), atol=1e-14)
        # 角动量归一化
        h = np.cross(STATE_INCLINED[:3], STATE_INCLINED[3:])
        np.testing.assert_allclose(h_hat, h / np.linalg.norm(h), atol=1e-14)
        # V = H × R
        np.testing.assert_allclose(v_hat, np.cross(h_hat, r_hat), atol=1e-14)
