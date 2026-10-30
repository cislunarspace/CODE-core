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
    lvlh_basis,
    reference_body_position,
)
from e2m2e.data.templates.enums import LVLHReference

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


# ---------------------------------------------------------------------------
# 参考体（#834）
# ---------------------------------------------------------------------------

MU_EARTH_MOON = 0.0121505856


class TestLVLHReference:
    """LVLHAxes 与 lvlh_basis 的参考体选项。"""

    def test_reference_body_position_values(self):
        """双质心在原点，主天体与次天体按 mu 定位。"""
        np.testing.assert_allclose(reference_body_position(LVLHReference.BARYCENTER), np.zeros(3))
        np.testing.assert_allclose(
            reference_body_position(LVLHReference.PRIMARY, MU_EARTH_MOON),
            [-MU_EARTH_MOON, 0.0, 0.0],
        )
        np.testing.assert_allclose(
            reference_body_position(LVLHReference.SECONDARY, MU_EARTH_MOON),
            [1.0 - MU_EARTH_MOON, 0.0, 0.0],
        )

    def test_reference_body_requires_mu(self):
        """非双质心缺 mu 时构造与查询都报 ValueError。"""
        with pytest.raises(ValueError, match="mu"):
            LVLHAxes(reference=LVLHReference.SECONDARY)
        with pytest.raises(ValueError, match="mu"):
            reference_body_position(LVLHReference.PRIMARY)

    def test_reference_body_rejects_invalid_mu(self):
        """mu 不在 (0, 1) 内时报 ValueError。"""
        with pytest.raises(ValueError, match="mu"):
            reference_body_position(LVLHReference.SECONDARY, 0.0)
        with pytest.raises(ValueError, match="mu"):
            reference_body_position(LVLHReference.SECONDARY, 1.0)

    def test_default_matches_barycenter(self):
        """默认构造与显式双质心逐位一致。"""
        default = LVLHAxes()
        default.update(0.0, STATE_ARBITRARY)
        explicit = LVLHAxes(reference=LVLHReference.BARYCENTER, mu=MU_EARTH_MOON)
        explicit.update(0.0, STATE_ARBITRARY)
        np.testing.assert_array_equal(default.rotation_matrix(0.0), explicit.rotation_matrix(0.0))

    def test_moon_radial(self):
        """次天体参考体的径向从月球指向目标。"""
        state = np.array([0.8, 0.1, 0.0, 0.0, 1.0, 0.0])
        axes = LVLHAxes(reference=LVLHReference.SECONDARY, mu=MU_EARTH_MOON)
        axes.update(0.0, state)
        r_rel = state[:3] - np.array([1.0 - MU_EARTH_MOON, 0.0, 0.0])
        np.testing.assert_allclose(
            axes.rotation_matrix(0.0)[:, 0],
            r_rel / np.linalg.norm(r_rel),
            atol=1e-14,
        )

    def test_earth_radial(self):
        """主天体参考体的径向从地球指向目标。"""
        state = np.array([0.8, 0.1, 0.0, 0.0, 1.0, 0.0])
        axes = LVLHAxes(reference=LVLHReference.PRIMARY, mu=MU_EARTH_MOON)
        axes.update(0.0, state)
        r_rel = state[:3] - np.array([-MU_EARTH_MOON, 0.0, 0.0])
        np.testing.assert_allclose(
            axes.rotation_matrix(0.0)[:, 0],
            r_rel / np.linalg.norm(r_rel),
            atol=1e-14,
        )

    def test_reference_basis_orthonormal(self):
        """三种参考体的旋转矩阵都正交且右手。"""
        state = np.array([0.8, 0.1, 0.0, 0.0, 1.0, 0.0])
        for reference in (
            LVLHReference.BARYCENTER,
            LVLHReference.PRIMARY,
            LVLHReference.SECONDARY,
        ):
            axes = LVLHAxes(reference=reference, mu=MU_EARTH_MOON)
            axes.update(0.0, state)
            rotation = axes.rotation_matrix(0.0)
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-14)
            assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-14)

    def test_lvlh_basis_matches_axes(self):
        """共享基函数与 LVLHAxes 对同一相对状态给出相同矩阵。"""
        state = np.array([0.8, 0.1, 0.0, 0.0, 1.0, 0.0])
        body = reference_body_position(LVLHReference.SECONDARY, MU_EARTH_MOON)
        expected = lvlh_basis(state[:3] - body, state[3:])
        axes = LVLHAxes(reference=LVLHReference.SECONDARY, mu=MU_EARTH_MOON)
        axes.update(0.0, state)
        np.testing.assert_array_equal(expected, axes.rotation_matrix(0.0))

    def test_moon_reference_zero_relative_position_raises(self):
        """状态位于次天体位置时相对位置为零，报 ValueError。"""
        state = np.array([1.0 - MU_EARTH_MOON, 0.0, 0.0, 0.0, 1.0, 0.0])
        axes = LVLHAxes(reference=LVLHReference.SECONDARY, mu=MU_EARTH_MOON)
        with pytest.raises(ValueError, match="位置为零"):
            axes.update(0.0, state)

    def test_lvlh_basis_zero_position_raises(self):
        """共享基函数对零相对位置报 ValueError。"""
        with pytest.raises(ValueError, match="位置为零"):
            lvlh_basis(np.zeros(3), np.array([0.0, 1.0, 0.0]))

    def test_unknown_reference_rejected(self):
        """非枚举参考体被拒绝。"""
        with pytest.raises(ValueError, match="参考体"):
            LVLHAxes(reference="moon", mu=MU_EARTH_MOON)
