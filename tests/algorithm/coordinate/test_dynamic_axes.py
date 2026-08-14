"""动态坐标轴单元测试。

覆盖 VNBAxes 与 LVLHAxes 的轴方向、正交性、未 update 异常及状态更新。
"""

import numpy as np
import pytest

from e2m2e.algorithm.coordinate.standard_dynamic_axes import (
    ANGULAR_MOMENTUM_NORM_MIN,
    POSITION_NORM_MIN,
    VELOCITY_NORM_MIN,
    LVLHAxes,
    VNBAxes,
)

pytestmark = pytest.mark.data


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
    """VNBAxes 测试。"""

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
    """LVLHAxes 测试。"""

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


# ---------------------------------------------------------------------------
# 退化态（ADR 0007 补白 / ADR 0020 决策 5）：轴向奇异时显式失败
# ---------------------------------------------------------------------------


class TestDegenerateStates:
    """零速度 / 零角动量 / 零位置时抛 ValueError，不静默产出 NaN 轴向。"""

    # 零速度：v = 0
    STATE_ZERO_VELOCITY = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # 零角动量：r ∥ v（径向运动）
    STATE_ZERO_ANGULAR_MOMENTUM = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    # 零位置：r = 0
    STATE_ZERO_POSITION = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    def test_vnb_zero_velocity_raises(self):
        """VNB：零速度时无法定义速度方向，抛 ValueError。"""
        axes = VNBAxes()
        with pytest.raises(ValueError, match="速度为零"):
            axes.update(0.0, self.STATE_ZERO_VELOCITY)

    def test_vnb_zero_angular_momentum_raises(self):
        """VNB：零角动量（r ∥ v）时无法定义法向，抛 ValueError。"""
        axes = VNBAxes()
        with pytest.raises(ValueError, match="角动量为零"):
            axes.update(0.0, self.STATE_ZERO_ANGULAR_MOMENTUM)

    def test_lvlh_zero_position_raises(self):
        """LVLH：零位置时无法定义径向，抛 ValueError。"""
        axes = LVLHAxes()
        with pytest.raises(ValueError, match="位置为零"):
            axes.update(0.0, self.STATE_ZERO_POSITION)

    def test_lvlh_zero_angular_momentum_raises(self):
        """LVLH：零角动量（r ∥ v）时无法定义法向，抛 ValueError。"""
        axes = LVLHAxes()
        with pytest.raises(ValueError, match="角动量为零"):
            axes.update(0.0, self.STATE_ZERO_ANGULAR_MOMENTUM)

    # --- 阈值可观测（ADR 0020 决策 5）：异常信息含实测范数与阈值 ---

    def test_vnb_zero_velocity_message_reports_measured_and_threshold(self):
        """VNB 零速度异常信息含实测 |v| 与阈值。"""
        axes = VNBAxes()
        with pytest.raises(ValueError) as exc_info:
            axes.update(0.0, self.STATE_ZERO_VELOCITY)
        msg = str(exc_info.value)
        assert "0.000e+00" in msg  # 实测 |v| = 0
        assert f"{VELOCITY_NORM_MIN:.1e}" in msg

    def test_vnb_small_angular_momentum_message_reports_measured(self):
        """VNB 近零角动量异常信息含实测 |r×v|（非恰零，验证实测量确实写入）。"""
        # r=[1,0,0], v=[1,1e-13,0] → r×v = [0,0,1e-13]，|h| = 1e-13 < 阈值
        state = np.array([1.0, 0.0, 0.0, 1.0, 1e-13, 0.0])
        axes = VNBAxes()
        with pytest.raises(ValueError) as exc_info:
            axes.update(0.0, state)
        msg = str(exc_info.value)
        assert "1.000e-13" in msg
        assert f"{ANGULAR_MOMENTUM_NORM_MIN:.1e}" in msg

    def test_lvlh_zero_position_message_reports_measured_and_threshold(self):
        """LVLH 零位置异常信息含实测 |r| 与阈值。"""
        axes = LVLHAxes()
        with pytest.raises(ValueError) as exc_info:
            axes.update(0.0, self.STATE_ZERO_POSITION)
        msg = str(exc_info.value)
        assert "0.000e+00" in msg  # 实测 |r| = 0
        assert f"{POSITION_NORM_MIN:.1e}" in msg

    def test_lvlh_small_angular_momentum_message_reports_measured(self):
        """LVLH 近零角动量异常信息含实测 |r×v| 与阈值。"""
        state = np.array([1.0, 0.0, 0.0, 1.0, 1e-13, 0.0])
        axes = LVLHAxes()
        with pytest.raises(ValueError) as exc_info:
            axes.update(0.0, state)
        msg = str(exc_info.value)
        assert "1.000e-13" in msg
        assert f"{ANGULAR_MOMENTUM_NORM_MIN:.1e}" in msg
