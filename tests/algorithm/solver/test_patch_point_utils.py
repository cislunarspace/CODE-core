"""sample_patch_points 与 convert_to_j2000 库函数测试。

覆盖采样形状、时间对齐、J2000 转换与一致性。
"""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from e2m2e.algorithm.solver.multiple_shooting import (
    convert_to_j2000,
    sample_patch_points,
    sample_patch_points_drop_near_perilune,
)
from e2m2e.data.constants import Datum
from e2m2e.data.templates import ConvergenceState
from e2m2e.data.types.orbit import Orbit

pytestmark = [
    pytest.mark.orchestration,
    pytest.mark.spice,
]


# =============================================================================
# 物理参数（DE421 基准）
# =============================================================================
MU = Datum.DE421.mu
TU_SECONDS = Datum.DE421.char_time_s

TU_DAYS = TU_SECONDS / 86400

DRO_31_X0 = 1.1202109158830986
DRO_31_VY0 = -0.46178983697629084
DRO_31_PERIOD = 2.095


# =============================================================================
# Fixtures
# =============================================================================
# 公共 SPICE fixtures 来自 tests/conftest.py:
#   spice_manager, spice_eph_system, spice_eph_dynamics, spice_syn_j2000,
#   reference_epoch, spice_kernel_path


@pytest.fixture
def dro_orbit(cr3bp_dynamics):
    from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection

    seed_state = np.array([DRO_31_X0, 0.0, 0.0, 0.0, DRO_31_VY0, 0.0])
    seed_orbit = Orbit([seed_state], [0])
    seed_orbit.period = DRO_31_PERIOD

    corrector = DifferentialCorrection(cr3bp_dynamics)
    corrector.setup_2D_symmetric_x_fixed_x0(DRO_31_X0)
    result = corrector.iterate_correction(seed_orbit, verbose=False)

    assert result.status is ConvergenceState.CONVERGED
    assert result.orbit is not None
    return result.orbit


# =============================================================================
# Test: sample_patch_points
# =============================================================================
class TestSamplePatchPoints:
    """测试 sample_patch_points 库函数"""

    def test_returns_correct_shapes(self, dro_orbit):
        """返回值形状应正确"""
        n_points = 8
        t_patch, states = sample_patch_points(dro_orbit, n_points)

        assert t_patch.shape == (n_points,)
        assert states.shape == (n_points, 6)

    def test_time_starts_at_zero(self, dro_orbit):
        """第一个采样时间应为 0"""
        t_patch, _ = sample_patch_points(dro_orbit, 4)
        assert_allclose(t_patch[0], 0.0)

    def test_time_is_uniformly_spaced(self, dro_orbit):
        """采样时间应等间距"""
        n_points = 8
        t_patch, _ = sample_patch_points(dro_orbit, n_points)
        dt = np.diff(t_patch)
        assert_allclose(dt, dt[0], rtol=1e-12)

    def test_time_does_not_include_period(self, dro_orbit):
        """endpoint=False: 最后一个时间点应小于 period"""
        n_points = 8
        t_patch, _ = sample_patch_points(dro_orbit, n_points)
        assert t_patch[-1] < dro_orbit.period

    def test_first_state_matches_orbit(self, dro_orbit):
        """第一个 patch point 应与轨道初始状态一致"""
        t_patch, states = sample_patch_points(dro_orbit, 4)
        assert_allclose(states[0], dro_orbit.states[0], atol=1e-10)

    def test_states_are_finite(self, dro_orbit):
        """所有状态值应为有限数"""
        _, states = sample_patch_points(dro_orbit, 8)
        assert np.all(np.isfinite(states))

    def test_planar_orbit_remains_planar(self, dro_orbit):
        """平面轨道的采样点应保持 z=0, vz=0"""
        _, states = sample_patch_points(dro_orbit, 8)
        assert_allclose(states[:, 2], 0.0, atol=1e-8)
        assert_allclose(states[:, 5], 0.0, atol=1e-8)

    def test_raises_when_period_is_none(self):
        """轨道 period 为 None 时应抛出异常"""
        orbit = Orbit(
            states=np.array([[0.5, 0, 0, 0, 0.5, 0]]),
            times=np.array([0.0]),
        )
        orbit.period = None
        with pytest.raises((AssertionError, ValueError)):
            sample_patch_points(orbit, 4)

    def test_two_points(self, dro_orbit):
        """n_points=2 应返回首末两段"""
        t_patch, states = sample_patch_points(dro_orbit, 2)
        assert len(t_patch) == 2
        assert_allclose(t_patch[0], 0.0)
        assert_allclose(t_patch[1], dro_orbit.period / 2, rtol=1e-10)

    def test_different_n_points_give_consistent_states(self, dro_orbit):
        """不同采样数在相同时间点应给出一致的状态"""
        t4, s4 = sample_patch_points(dro_orbit, 4)
        t8, s8 = sample_patch_points(dro_orbit, 8)

        # 8 点的前半部分应与 4 点的前半部分时间对齐
        assert_allclose(t8[0], t4[0])
        assert_allclose(t8[2], t4[1], rtol=1e-10)
        assert_allclose(s8[2], s4[1], atol=1e-8)


# =============================================================================
# Test: sample_patch_points_drop_near_perilune
# =============================================================================
class TestSamplePatchPointsDropNearPerilune:
    """删近月点附近节点采样：节点全部落在近月点禁区外。

    用已有 DRO 周期解作几何样本（CR3BP 可积、有明确近月点），不走
    design_nrho / 星历修正——本类只验证采样几何契约。
    """

    def test_nodes_outside_perilune_window(self, dro_orbit, cr3bp_dynamics):
        """非历元节点与近月点的圆周距离 > drop_window·period。

        ``t=0`` 为历元钉点，允许落在禁区内；其余节点仍在互补弧上。
        """
        drop_window = 0.12
        t_patch, states = sample_patch_points_drop_near_perilune(
            dro_orbit, cr3bp_dynamics, n_points=8, drop_window=drop_window
        )
        period = float(dro_orbit.period)
        mu = cr3bp_dynamics.system.mu
        moon_x = 1.0 - mu
        # 与实现一致：密探定位近月点
        t_probe = np.linspace(0.0, period, 200, endpoint=False)
        probe = cr3bp_dynamics.propagate(dro_orbit.states[0], (0.0, period), t_eval=t_probe)
        st = probe["states"]
        dists = np.sqrt((st[:, 0] - moon_x) ** 2 + st[:, 1] ** 2 + st[:, 2] ** 2)
        t_p = float(t_probe[int(np.argmin(dists))])
        half_w = drop_window * period

        def circ_dist(t: float) -> float:
            return abs((t - t_p + period / 2.0) % period - period / 2.0)

        assert len(t_patch) == 8
        assert np.all(np.isfinite(states))
        for t in t_patch:
            if abs(float(t)) < 1e-12:
                continue  # 历元钉点
            assert circ_dist(float(t)) > half_w - 1e-12, (
                f"节点 t={t:.4f} 落入近月点禁区（t_p={t_p:.4f}, half_w={half_w:.4f}）"
            )

    def test_includes_epoch_t0(self, dro_orbit, cr3bp_dynamics):
        """首节点钉历元 t=0，节点数符合请求。"""
        t_patch, states = sample_patch_points_drop_near_perilune(
            dro_orbit, cr3bp_dynamics, n_points=8
        )
        assert len(t_patch) == 8
        assert abs(float(t_patch[0])) < 1e-12
        assert states.shape == (8, 6)

    def test_falls_back_to_uniform_when_epoch_deduplication_loses_points(self):
        """钉历元后去重使节点不足时，回退等时间且保留请求点数。"""
        period = 1e-13
        orbit = SimpleNamespace(
            period=period,
            times=np.array([0.0, period]),
            states=np.zeros((2, 6)),
        )

        class TinyPeriodDynamics:
            system = SimpleNamespace(mu=0.01215)

            @staticmethod
            def propagate(state, t_span, t_eval):
                return {"states": np.zeros((len(t_eval), 6))}

        t_patch, states = sample_patch_points_drop_near_perilune(
            orbit, TinyPeriodDynamics(), n_points=8
        )
        assert_allclose(t_patch, np.linspace(0.0, period, 8, endpoint=False))
        assert states.shape == (8, 6)

    def test_times_sorted_within_period(self, dro_orbit, cr3bp_dynamics):
        """返回时间升序且落在 [0, period)。"""
        t_patch, _ = sample_patch_points_drop_near_perilune(dro_orbit, cr3bp_dynamics, n_points=8)
        period = float(dro_orbit.period)
        assert np.all(np.diff(t_patch) > 0)
        assert abs(float(t_patch[0])) < 1e-12
        assert t_patch[-1] < period


# =============================================================================
# Test: convert_to_j2000
# =============================================================================
class TestConvertToJ2000:
    """测试 convert_to_j2000 库函数"""

    def test_returns_correct_shapes(self, dro_orbit, spice_syn_j2000, reference_et):
        """返回值形状应正确"""
        t_syn, states_syn = sample_patch_points(dro_orbit, 8)
        t_j2000, states_j2000 = convert_to_j2000(
            t_syn, states_syn, spice_syn_j2000, reference_et, TU_DAYS
        )

        assert t_j2000.shape == (8,)
        assert states_j2000.shape == (8, 6)

    def test_time_starts_at_reference_et(self, dro_orbit, spice_syn_j2000, reference_et):
        """J2000 时间起点应等于 reference_et"""
        t_syn, states_syn = sample_patch_points(dro_orbit, 4)
        t_j2000, _ = convert_to_j2000(t_syn, states_syn, spice_syn_j2000, reference_et, TU_DAYS)
        assert_allclose(t_j2000[0], reference_et)

    def test_time_is_monotonically_increasing(self, dro_orbit, spice_syn_j2000, reference_et):
        """J2000 时间应单调递增"""
        t_syn, states_syn = sample_patch_points(dro_orbit, 8)
        t_j2000, _ = convert_to_j2000(t_syn, states_syn, spice_syn_j2000, reference_et, TU_DAYS)
        assert np.all(np.diff(t_j2000) > 0)

    def test_time_span_matches_period(self, dro_orbit, spice_syn_j2000, reference_et):
        """J2000 时间跨度应等于 orbit period * TU_SECONDS"""
        t_syn, states_syn = sample_patch_points(dro_orbit, 4)
        t_j2000, _ = convert_to_j2000(t_syn, states_syn, spice_syn_j2000, reference_et, TU_DAYS)
        expected_span = dro_orbit.period * TU_SECONDS
        actual_span = t_j2000[-1] - t_j2000[0]
        # endpoint=False so last point is not a full period
        assert actual_span < expected_span

    def test_states_are_finite(self, dro_orbit, spice_syn_j2000, reference_et):
        """所有 J2000 状态值应为有限数"""
        t_syn, states_syn = sample_patch_points(dro_orbit, 8)
        _, states_j2000 = convert_to_j2000(
            t_syn, states_syn, spice_syn_j2000, reference_et, TU_DAYS
        )
        assert np.all(np.isfinite(states_j2000))

    def test_position_near_moon_distance(self, dro_orbit, spice_syn_j2000, reference_et):
        """J2000 下的 DRO 位置应在月球距离附近 (300000-500000 km)"""
        t_syn, states_syn = sample_patch_points(dro_orbit, 4)
        _, states_j2000 = convert_to_j2000(
            t_syn, states_syn, spice_syn_j2000, reference_et, TU_DAYS
        )
        for i in range(len(states_j2000)):
            r = np.linalg.norm(states_j2000[i, :3])
            assert 300000 < r < 500000, f"Patch {i} 距地球 {r:.0f} km，超出合理范围"

    def test_velocity_is_reasonable(self, dro_orbit, spice_syn_j2000, reference_et):
        """J2000 下的速度应在合理范围 (< 5 km/s)"""
        t_syn, states_syn = sample_patch_points(dro_orbit, 4)
        _, states_j2000 = convert_to_j2000(
            t_syn, states_syn, spice_syn_j2000, reference_et, TU_DAYS
        )
        for i in range(len(states_j2000)):
            v = np.linalg.norm(states_j2000[i, 3:])
            assert v < 5.0, f"Patch {i} 速度 {v:.2f} km/s 过大"

    def test_consistency_with_direct_call(self, dro_orbit, spice_syn_j2000, reference_et):
        """应与直接调用 spice_syn_j2000.batch_synodic_to_j2000 结果一致"""
        t_syn, states_syn = sample_patch_points(dro_orbit, 4)

        t_j2000, states_j2000 = convert_to_j2000(
            t_syn, states_syn, spice_syn_j2000, reference_et, TU_DAYS
        )

        expected_states = spice_syn_j2000.batch_synodic_to_j2000(
            states_syn=states_syn,
            t_syn_arr=t_syn,
            et0=reference_et,
        )
        expected_times = reference_et + t_syn * TU_SECONDS  # TU_SECONDS = TU_DAYS * 86400

        assert_allclose(states_j2000, expected_states, rtol=1e-12)
        assert_allclose(t_j2000, expected_times, rtol=1e-12)

    def test_single_point(self, dro_orbit, spice_syn_j2000, reference_et):
        """单点输入应正常工作"""
        t_syn, states_syn = sample_patch_points(dro_orbit, 1)
        assert len(t_syn) == 1

        t_j2000, states_j2000 = convert_to_j2000(
            t_syn, states_syn, spice_syn_j2000, reference_et, TU_DAYS
        )
        assert t_j2000.shape == (1,)
        assert states_j2000.shape == (1, 6)
        assert_allclose(t_j2000[0], reference_et)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
