"""不变环面构造与诊断测试。

种子为 L2 Halo（面外振幅 3000 km，单中心对形态）。断言一律按定义级
性质：频闪不变性残差、共轭性残差、旋转数与单值矩阵中心对辐角的零振幅
极限一致、长时到环面距离有界。论文对照值（第二周期 36.5 天与 58.3
天、法向振幅 0.031）属结果性数值，不进断言。可达振幅包络以 ADR 0063
的标定为准：1e-4 到约 2.9e-4（月球距离单位）收敛，超出包络的目标以
状态三元组如实报告不收敛。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.torus import (
    TorusError,
    conjugacy_residual,
    construct_invariant_torus,
    evaluate_torus_curve,
    invariance_residual,
    torus_long_time_distance,
)

pytestmark = pytest.mark.orchestration

#: 测试用目标振幅（月球距离单位），取收敛包络内（ADR 0063 标定）。
_TEST_AMPLITUDE = 1.5e-4


def _center_argument(orbit, dynamics) -> float:
    """在测试内独立复算中心乘子辐角（不经被测代码）。"""
    from e2m2e.algorithm.family.resonant_branch import monodromy

    matrix = monodromy(dynamics, np.asarray(orbit.states[0], float), float(orbit.period))
    values = np.linalg.eigvals(matrix)
    unit = [v for v in values if abs(abs(v) - 1.0) < 1e-8 and v.imag > 0.0]
    assert len(unit) == 1
    return float(np.angle(unit[0]))


class TestConstruction:
    def test_small_amplitude_converges(self, halo_seed) -> None:
        """包络内振幅构造：收敛、残差达筛选级、振幅命中目标、周期自洽。"""
        orbit, dynamics = halo_seed
        result = construct_invariant_torus(orbit, _TEST_AMPLITUDE, n_points=16, dynamics=dynamics)
        assert result.status.value == "converged"
        assert result.cause.value == "none"
        assert result.invariance_residual <= 1e-9
        assert 0.0 < result.rotation_number < 2.0 * np.pi
        assert result.amplitude == pytest.approx(_TEST_AMPLITUDE, rel=1e-2)
        # 频闪周期固定为种子周期，第二周期按 T·2π/ρ 换算物理时间
        assert result.strobe_period == pytest.approx(float(orbit.period), rel=1e-12)
        expected_days = result.second_period * dynamics.system.TU
        assert result.second_period_days == pytest.approx(expected_days, rel=1e-12)
        assert result.second_period_days > 0.0

    def test_rotation_number_hits_multiplier_argument_limit(self, halo_seed) -> None:
        """振幅趋于零时旋转数回到单值矩阵中心对辐角（定义性断言）。"""
        orbit, dynamics = halo_seed
        result = construct_invariant_torus(orbit, 5e-5, n_points=16, dynamics=dynamics)
        assert result.status.value == "converged"
        alpha = _center_argument(orbit, dynamics)
        assert result.multiplier_argument == pytest.approx(alpha, abs=1e-10)
        assert result.rotation_number == pytest.approx(alpha, abs=1e-6)

    def test_result_shapes(self, halo_seed) -> None:
        """输出契约：曲线网格、相位网格与迭代历史。"""
        orbit, dynamics = halo_seed
        result = construct_invariant_torus(orbit, _TEST_AMPLITUDE, n_points=16, dynamics=dynamics)
        assert result.curve_states.shape == (16, 6)
        np.testing.assert_allclose(result.phases, np.arange(16) * (2.0 * np.pi / 16))
        assert result.newton_iterations >= 1
        assert result.residual_history
        assert result.monodromy.shape == (6, 6)


class TestDiagnostics:
    def test_invariance_residual_recomputes(self, halo_seed) -> None:
        """不变性残差作为独立函数可复算，与结果字段同量级。"""
        orbit, dynamics = halo_seed
        result = construct_invariant_torus(orbit, _TEST_AMPLITUDE, n_points=16, dynamics=dynamics)
        residual = invariance_residual(
            result.curve_states,
            result.rotation_number,
            result.strobe_period,
            dynamics,
        )
        assert 0.0 <= residual <= 1e-9

    def test_conjugacy_residual_small(self, halo_seed) -> None:
        """共轭性残差：各采样点的推进角与 ρ 的偏离为小量。"""
        orbit, dynamics = halo_seed
        result = construct_invariant_torus(orbit, _TEST_AMPLITUDE, n_points=16, dynamics=dynamics)
        residual = conjugacy_residual(
            result.curve_states,
            result.rotation_number,
            result.strobe_period,
            dynamics,
        )
        assert residual <= 1e-5

    def test_long_time_distance_bounded(self, halo_seed) -> None:
        """短弧传播保持在环面邻域内。

        L2 Halo 的双曲乘子单周期放大约千倍，舍入误差按周期指数放大，
        长弧读数由种子双曲性主导（见诊断 docstring），故取两个周期断
        言。
        """
        orbit, dynamics = halo_seed
        result = construct_invariant_torus(orbit, _TEST_AMPLITUDE, n_points=16, dynamics=dynamics)
        distance = torus_long_time_distance(
            result.curve_states,
            result.strobe_period,
            dynamics,
            n_periods=2,
        )
        assert distance <= 1e-7

    def test_evaluate_curve_interpolates_grid(self, halo_seed) -> None:
        """带限插值在网格点上复现采样，在半格点上给出有界中间值。"""
        orbit, dynamics = halo_seed
        result = construct_invariant_torus(orbit, _TEST_AMPLITUDE, n_points=16, dynamics=dynamics)
        np.testing.assert_allclose(
            evaluate_torus_curve(result.curve_states, result.phases),
            result.curve_states,
            atol=1e-12,
        )
        mid = evaluate_torus_curve(result.curve_states, result.phases + np.pi / 16.0)
        assert np.all(np.isfinite(mid))
        assert mid.shape == (16, 6)


class TestFailurePaths:
    def test_non_convergence_reports_status_triple(self, halo_seed) -> None:
        """迭代预算耗尽时如实报告状态三元组，不抛异常。"""
        orbit, dynamics = halo_seed
        result = construct_invariant_torus(
            orbit, _TEST_AMPLITUDE, n_points=16, dynamics=dynamics, max_iterations=1
        )
        assert result.status.value == "max_iterations"
        assert result.cause.value == "max_iterations_reached"
        assert result.message

    def test_beyond_envelope_reports_status_triple(self, halo_seed) -> None:
        """超出可达振幅包络的目标如实报告不收敛，不伪造结果。"""
        orbit, dynamics = halo_seed
        result = construct_invariant_torus(orbit, 0.01, n_points=16, dynamics=dynamics)
        assert result.status.value != "converged"
        assert result.message

    def test_invalid_amplitude_raises(self, halo_seed) -> None:
        """非正振幅属确定性输入错误，抛错而非软失败。"""
        orbit, dynamics = halo_seed
        with pytest.raises(TorusError):
            construct_invariant_torus(orbit, 0.0, dynamics=dynamics)

    def test_missing_period_raises(self, halo_seed) -> None:
        """period 未设置的轨道无法定义频闪映射，确定性报错。"""
        orbit, dynamics = halo_seed
        orbit.period = None
        with pytest.raises(TorusError, match="period"):
            construct_invariant_torus(orbit, _TEST_AMPLITUDE, dynamics=dynamics)
