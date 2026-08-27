"""推进模型模块测试。

覆盖 ImpulsivePropulsion 具体实现。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.transfer.cost import TransferCost
from e2m2e.algorithm.transfer.propulsion import ImpulsivePropulsion

pytestmark = pytest.mark.orchestration


# =============================================================================
# ImpulsivePropulsion 初始化
# =============================================================================


class TestImpulsivePropulsionInit:
    def test_default_normal(self):
        """默认法向量为 [0, 0, 1]。"""
        p = ImpulsivePropulsion()
        np.testing.assert_array_equal(p.normal, np.array([0.0, 0.0, 1.0]))

    def test_custom_normal(self):
        """可传入自定义法向量。"""
        n = np.array([1.0, 0.0, 0.0])
        p = ImpulsivePropulsion(normal=n)
        np.testing.assert_array_equal(p.normal, n)

    def test_normal_is_float_dtype(self):
        """法向量会被转换为 float dtype。"""
        p = ImpulsivePropulsion(normal=[0, 0, 1])
        assert p.normal.dtype == np.float64


# =============================================================================
# compute_departure_velocity
# =============================================================================


class TestComputeDepartureVelocity:
    def test_alpha_1_beta_0_preserves_velocity(self):
        """alpha=1, beta=0 时注入速度等于原始速度。"""
        state = np.array([0.1, 0.0, 0.0, 1.0, 0.0, 0.0])
        p = ImpulsivePropulsion()
        v = p.compute_departure_velocity(state, alpha=1.0, beta=0.0)
        np.testing.assert_allclose(v, state[3:], atol=1e-12)

    def test_alpha_2_doubles_speed(self):
        """alpha=2 时速度大小翻倍。"""
        state = np.array([0.1, 0.0, 0.0, 1.0, 0.0, 0.0])
        p = ImpulsivePropulsion()
        v = p.compute_departure_velocity(state, alpha=2.0, beta=0.0)
        np.testing.assert_allclose(v, np.array([2.0, 0.0, 0.0]), atol=1e-12)

    def test_alpha_0_zero_speed(self):
        """alpha=0 时切向分量为零。"""
        state = np.array([0.1, 0.0, 0.0, 1.0, 0.0, 0.0])
        p = ImpulsivePropulsion()
        v = p.compute_departure_velocity(state, alpha=0.0, beta=0.0)
        np.testing.assert_allclose(v, np.zeros(3), atol=1e-12)

    def test_direction_preserved_for_various_orientations(self):
        """不同方向的速度，alpha=1 时方向保持不变。"""
        p = ImpulsivePropulsion()
        for vel in [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 1.0, 0.0]),
            np.array([1.0, -1.0, 1.0]),
        ]:
            state = np.concatenate([np.zeros(3), vel])
            v = p.compute_departure_velocity(state, alpha=1.0, beta=0.0)
            original_dir = vel / np.linalg.norm(vel)
            new_dir = v / np.linalg.norm(v)
            np.testing.assert_allclose(new_dir, original_dir, atol=1e-12)

    def test_beta_nonzero_adds_normal_component(self):
        """beta != 0 时注入速度包含法向分量。"""
        state = np.array([0.1, 0.0, 0.0, 1.0, 0.0, 0.0])
        p = ImpulsivePropulsion()
        v = p.compute_departure_velocity(state, alpha=1.0, beta=0.5)
        # 法向分量应垂直于切向（原始速度方向）
        tangential = state[3:] / np.linalg.norm(state[3:])
        dot = np.dot(v, tangential)
        # 切向分量大小应为 alpha * |v| = 1.0
        assert pytest.approx(dot, abs=1e-12) == 1.0
        # 法向分量大小应为 beta * |v| = 0.5
        normal_component = v - dot * tangential
        assert pytest.approx(np.linalg.norm(normal_component), abs=1e-12) == 0.5

    def test_zero_speed_warns(self):
        """速度接近零时发出警告并返回原速度副本。"""
        state = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        p = ImpulsivePropulsion()
        with pytest.warns(UserWarning, match="速度接近零"):
            v = p.compute_departure_velocity(state, alpha=1.0, beta=0.0)
        np.testing.assert_array_equal(v, state[3:])

    def test_ignores_extra_kwargs(self):
        """忽略额外关键字参数。"""
        state = np.array([0.1, 0.0, 0.0, 1.0, 0.0, 0.0])
        p = ImpulsivePropulsion()
        # 不应抛出异常
        v = p.compute_departure_velocity(state, alpha=1.0, beta=0.0, extra_param=42)
        np.testing.assert_allclose(v, state[3:], atol=1e-12)

    def test_returns_copy(self):
        """返回的速度向量是副本，不共享内存。"""
        state = np.array([0.1, 0.0, 0.0, 1.0, 0.0, 0.0])
        p = ImpulsivePropulsion()
        v = p.compute_departure_velocity(state, alpha=1.0, beta=0.0)
        v[0] = 999.0
        assert state[3] == 1.0  # 原始状态未被修改

    def test_normal_parallel_to_velocity_raises(self):
        """法向退化（速度与法向量平行）时抛异常，不替换为任意 [1,0,0]。

        ``normal_dir`` 退化时若静默替换为任意 ``[1,0,0]``，法向分量
        方向会与几何无关（谎报一个不存在的方向）。法向未定义就该报错，不猜。
        """
        # 速度沿 z 轴，默认法向也是 z 轴 → cross(t, n) = 0 → 退化
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        p = ImpulsivePropulsion()
        with pytest.raises(ValueError, match="法向退化"):
            p.compute_departure_velocity(state, alpha=1.0, beta=0.5)

    def test_speed_magnitude(self):
        """注入速度大小 = |v| * sqrt(alpha^2 + beta^2)。"""
        state = np.array([0.1, 0.0, 0.0, 3.0, 0.0, 0.0])
        p = ImpulsivePropulsion()
        alpha, beta = 1.5, 0.5
        v = p.compute_departure_velocity(state, alpha=alpha, beta=beta)
        expected_mag = 3.0 * np.sqrt(alpha**2 + beta**2)
        assert pytest.approx(np.linalg.norm(v), abs=1e-12) == expected_mag


# =============================================================================
# compute_cost
# =============================================================================


class TestComputeCost:
    def test_delegates_to_compute_transfer_cost(self):
        """compute_cost 应委托给 compute_transfer_cost 并返回 TransferCost。"""
        p = ImpulsivePropulsion()
        departure_state = np.array([0.1, 0.0, 0.0, 1.0, 0.0, 0.0])
        initial_velocity = np.array([1.3, 0.0, 0.0])
        final_velocity = np.array([0.0, 0.6, 0.0])
        insertion_velocity = np.array([0.0, 0.2, 0.0])

        cost = p.compute_cost(
            departure_state=departure_state,
            initial_velocity=initial_velocity,
            final_velocity=final_velocity,
            insertion_velocity=insertion_velocity,
        )

        assert isinstance(cost, TransferCost)
        assert cost.dv1 == pytest.approx(0.3, abs=1e-12)
        assert cost.dv2 == pytest.approx(0.4, abs=1e-12)
        assert cost.total == pytest.approx(0.7, abs=1e-12)

    def test_zero_cost_when_velocities_match(self):
        """所有速度匹配时成本为零。"""
        p = ImpulsivePropulsion()
        vel = np.array([1.0, 0.0, 0.0])
        departure_state = np.array([0.1, 0.0, 0.0, 1.0, 0.0, 0.0])

        cost = p.compute_cost(
            departure_state=departure_state,
            initial_velocity=vel,
            final_velocity=vel,
            insertion_velocity=vel,
        )

        assert cost.total == pytest.approx(0.0, abs=1e-12)
