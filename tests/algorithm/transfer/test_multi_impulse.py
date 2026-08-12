"""MultiImpulseTransfer 多脉冲优化与主矢量检验测试。

几何均为二体圆轨道间转移（地球，km/s）：

- 霍曼转移（7000 → 42164 km，tof 取霍曼时间）：最优双脉冲，主矢量检验
  应判定满足 Lawden 必要条件（弧内 |p|≤1、无插入建议）
- 同端点但 tof 取 0.5 倍霍曼时间的双脉冲转移：明显非最优，弧内 |p|>1，
  应给出中途脉冲插入建议；以建议点为零脉冲初猜做三脉冲优化，总 ΔV 应下降
- 三脉冲 optimize 的收敛性与 TransferSolution/legs 结构校验
"""

import numpy as np
import pytest

from e2m2e.algorithm.transfer import (
    CoastArc,
    Impulse,
    MultiImpulseTransfer,
    StateTerminal,
    TransferSolution,
)
from e2m2e.data.templates import ConvergenceState

pytestmark = pytest.mark.orchestration


MU_EARTH = 398600.4418  # km³/s²

R1 = 7000.0  # km
R2 = 42164.0  # km（GEO）
TOF_HOHMANN = np.pi * np.sqrt(((R1 + R2) / 2) ** 3 / MU_EARTH)


def _circular(r: float, angle: float = 0.0) -> np.ndarray:
    """半径 r、相位角 angle 的圆轨道状态（逆时针）。"""
    v = np.sqrt(MU_EARTH / r)
    return np.array(
        [r * np.cos(angle), r * np.sin(angle), 0.0, -v * np.sin(angle), v * np.cos(angle), 0.0]
    )


def _make_transfer(tof: float) -> MultiImpulseTransfer:
    """R1 圆轨道 → R2 圆轨道（对侧相位）的双/多脉冲转移规划器。"""
    return MultiImpulseTransfer(
        StateTerminal(_circular(R1), 0.0),
        StateTerminal(_circular(R2, np.pi), tof),
        mu=MU_EARTH,
    )


class TestHohmannPrimer:
    """霍曼转移满足 Lawden 必要条件"""

    def test_hohmann_satisfies_lawden(self):
        transfer = _make_transfer(TOF_HOHMANN)
        sol = transfer.optimize(2)
        assert sol.status is ConvergenceState.CONVERGED
        # 霍曼 ΔV 基准（LEO→GEO 量级，解析值 3.7708 km/s）
        assert sol.total_delta_v == pytest.approx(3.7708, abs=1e-3)

        report = transfer.check_primer_vector(sol, n_samples=200)
        assert report.lawden_satisfied, report.message
        assert report.suggested_insertion_time is None
        assert report.suggested_insertion_position is None
        # 脉冲点 |p|=1 且方向与脉冲共线（端点由横截条件构造，检验数值一致性）
        assert np.allclose(report.impulse_magnitudes, 1.0, atol=1e-6)
        assert np.allclose(np.abs(report.impulse_alignment_cosines), 1.0, atol=1e-6)
        # 弧内 |p| 不超 1
        interior = report.primer_magnitude[1:-1]
        assert interior.max() <= 1.0 + 1e-3


class TestNonOptimalGeometry:
    """0.5 倍霍曼时间的双脉冲转移：非最优，应给出插入建议且三脉冲降本"""

    def test_primer_detects_violation_and_suggests_insertion(self):
        transfer = _make_transfer(0.5 * TOF_HOHMANN)
        sol = transfer.optimize(2)
        report = transfer.check_primer_vector(sol, n_samples=300)

        assert not report.lawden_satisfied
        assert report.primer_magnitude.max() > 1.0 + 1e-3
        assert report.suggested_insertion_time is not None
        assert 0.0 < report.suggested_insertion_time < sol.transfer_time
        assert report.suggested_insertion_position is not None
        assert report.suggested_insertion_position.shape == (3,)

    def test_three_impulse_reduces_total_dv(self):
        transfer = _make_transfer(0.5 * TOF_HOHMANN)
        sol2 = transfer.optimize(2)
        report = transfer.check_primer_vector(sol2, n_samples=300)

        # 以建议插入点为零脉冲节点做初猜：目标值恰为双脉冲成本
        x0 = np.concatenate(
            [[report.suggested_insertion_time], report.suggested_insertion_position]
        )
        sol3 = transfer.optimize(3, x0=x0)

        assert sol3.status is ConvergenceState.CONVERGED, sol3.message
        assert sol3.total_delta_v < sol2.total_delta_v - 0.05

    def test_three_impulse_retries_stalled_initial_guess(self):
        """零脉冲初猜使 SLSQP 提前收敛时，微扰重试仍应找到三脉冲降本（#384）。

        建议插入点落在双脉冲弧上的位置随主矢量检验采样密度而变；n_samples=200
        的建议点处于目标函数平坦走廊，SLSQP 一步即停（单次优化改善 ≈ 0）。
        optimize 检测到改善不足后从微扰初猜重试，仍应显著降本。
        """
        transfer = _make_transfer(0.5 * TOF_HOHMANN)
        sol2 = transfer.optimize(2)
        report = transfer.check_primer_vector(sol2, n_samples=200)

        x0 = np.concatenate(
            [[report.suggested_insertion_time], report.suggested_insertion_position]
        )
        sol3 = transfer.optimize(3, x0=x0)

        assert sol3.status is ConvergenceState.CONVERGED, sol3.message
        assert sol3.total_delta_v < sol2.total_delta_v - 0.05


class TestThreeImpulseStructure:
    """三脉冲 optimize 的收敛性与结果结构"""

    def test_solution_structure(self):
        transfer = _make_transfer(TOF_HOHMANN)
        sol = transfer.optimize(3)

        assert isinstance(sol, TransferSolution)
        assert sol.status is ConvergenceState.CONVERGED, sol.message
        assert sol.n_iter > 0
        # 三脉冲 = 两段滑行弧 + 到达脉冲
        assert len(sol.arcs) == 2
        assert sol.total_delta_v == pytest.approx(
            sum(arc.delta_v for arc in sol.arcs) + sol.arrival_delta_v
        )
        assert sol.transfer_time == pytest.approx(TOF_HOHMANN)
        # 弧段结构：时间单调、状态形状正确
        for arc in sol.arcs:
            assert arc.states.shape == (50, 6)
            assert np.all(np.diff(arc.times) > 0)
        assert sol.arcs[0].times[-1] == pytest.approx(sol.arcs[1].times[0])

        # legs 刷新为 Impulse/CoastArc 交替（首末端脉冲均记录）
        kinds = [type(leg) for leg in transfer.legs]
        assert kinds == [Impulse, CoastArc, Impulse, CoastArc, Impulse]

    def test_optimal_three_impulse_costs_no_more_than_two(self):
        """三脉冲解总 ΔV 不超过双脉冲解（脉冲集包含双脉冲为特例）"""
        transfer = _make_transfer(TOF_HOHMANN)
        sol2 = transfer.optimize(2)
        sol3 = transfer.optimize(3)
        assert sol3.total_delta_v <= sol2.total_delta_v + 1e-6


class TestValidation:
    """参数校验"""

    def test_missing_dynamics_raises(self):
        with pytest.raises(ValueError, match="mu 或三体 dynamics"):
            MultiImpulseTransfer(
                StateTerminal(_circular(R1), 0.0),
                StateTerminal(_circular(R2, np.pi), TOF_HOHMANN),
            )

    def test_invalid_backend_raises(self):
        transfer = _make_transfer(TOF_HOHMANN)
        with pytest.raises(ValueError, match="backend"):
            transfer.optimize(3, backend="copt")

    def test_too_few_impulses_raises(self):
        transfer = _make_transfer(TOF_HOHMANN)
        with pytest.raises(ValueError, match="n_impulses"):
            transfer.optimize(1)

    def test_bad_x0_shape_raises(self):
        transfer = _make_transfer(TOF_HOHMANN)
        with pytest.raises(ValueError, match="x0"):
            transfer.optimize(3, x0=np.zeros(5))

    def test_zero_arrival_impulse_primer_raises(self):
        """到达脉冲为零时无法确定主矢量端点方向"""
        # 到达终端取霍曼弧的自然到达状态：到达脉冲为零、出发脉冲非零
        from e2m2e.algorithm.transfer import solve_lambert

        r0 = _circular(R1)[:3]
        rf = _circular(R2, np.pi)[:3]
        lam = solve_lambert(r0, rf, TOF_HOHMANN, MU_EARTH)
        transfer = MultiImpulseTransfer(
            StateTerminal(_circular(R1), 0.0),
            StateTerminal(np.concatenate([rf, lam.vf]), TOF_HOHMANN),
            mu=MU_EARTH,
        )
        sol = transfer.optimize(2)
        assert sol.arrival_delta_v < 1e-9
        with pytest.raises(ValueError, match="端点方向"):
            transfer.check_primer_vector(sol)
