"""MGA 链连续精化（refine_mga_chain）的编排测试（#726，multisegment 第二实例）。

oracle 口径（ADR 0055）：

- ① 带定义性公式：从候选历元/TOF 出发独立解 Lambert 重建 flyby 观测量
  （等模残差、总 ΔV、近心点半径），与精化候选逐项对照；
- ② 不变量：精化后总 ΔV 不增（可行流形上的局部最优 ≤ 近可行起点）；
- 接口：不可行以软失败三元组表达（不抛异常），入参非法抛 ``ValueError``。

合成圆共面星历（模式抄 ``test_mga_search.py``），无 SPICE。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from e2m2e.algorithm.transfer import (
    ConvergenceState,
    FailureCause,
    MgaRefinementResult,
    refine_mga_chain,
    search_mga_chains,
)
from e2m2e.algorithm.transfer.lambert import solve_lambert
from e2m2e.algorithm.transfer.mga import flyby_pericenter_radius
from e2m2e.data.constants import SECONDS_PER_DAY

pytestmark = pytest.mark.orchestration

MU_SUN = 1.32712440018e11  # km³/s²

#: 合成行星 GM 与轨道（与 test_mga_search.py 同源）。
_GM = {"SUN": MU_SUN, "EARTH": 3.986004418e5, "VENUS": 3.24859e5}
_AU = 1.495978707e8
_PLANETS: dict[str, tuple[float, float]] = {
    "EARTH": (_AU, 0.0),
    "VENUS": (0.7233 * _AU, 0.3),
}


class _CircularEphemeris:
    """圆共面解析星历（与 test_mga_search.py 相同的两方法协议）。"""

    def get_gm(self, body: str) -> float:
        return _GM[body.upper()]

    def get_body_state(self, target: str, et: float, frame: str, observer: str) -> np.ndarray:
        assert frame == "J2000" and observer == "SUN"
        semi_major, phase = _PLANETS[target.upper()]
        mean_motion = math.sqrt(MU_SUN / semi_major**3)
        theta = phase + mean_motion * float(et)
        return np.array(
            [
                semi_major * math.cos(theta),
                semi_major * math.sin(theta),
                0.0,
                -semi_major * mean_motion * math.sin(theta),
                semi_major * mean_motion * math.cos(theta),
                0.0,
            ]
        )


#: 近可行候选标定网格：launch (10,12,14) 天 × TOF [175,180,185]×[95,100,105] 天，
#: 等模容差 0.05 km/s（(12 天, 185, 100) 格的等模差 ~0.008 km/s），
#: r_p_min = 50 km（该格所需近心点半径 ~89 km，转角约束留有余量）。
_R_P_MIN = 50.0
_MIN_RP = [_R_P_MIN]


@pytest.fixture
def candidate():
    """近可行的网格候选（标定网格下搜索的第一名）。"""
    result = search_mga_chains(
        ["EARTH", "VENUS", "EARTH"],
        [day * SECONDS_PER_DAY for day in (10.0, 12.0, 14.0)],
        [[175.0, 180.0, 185.0], [95.0, 100.0, 105.0]],
        _MIN_RP,
        ephemeris=_CircularEphemeris(),
        mu_sun_km3_s2=MU_SUN,
        v_inf_match_tol_km_s=0.05,
    )
    assert result.status is ConvergenceState.CONVERGED
    assert result.candidates
    return result.candidates[0]


def _refine(cand, *, min_rp=_MIN_RP, **overrides):
    """精化的便捷入口（缺省界围绕候选值 ±3 天 / ±(5~10) 天）。"""
    kwargs = dict(
        min_flyby_pericenter_km=list(min_rp),
        launch_epoch_bounds_s=(
            cand.launch_epoch_et - 3.0 * SECONDS_PER_DAY,
            cand.launch_epoch_et + 3.0 * SECONDS_PER_DAY,
        ),
        tof_bounds_s=[
            [170.0 * SECONDS_PER_DAY, 190.0 * SECONDS_PER_DAY],
            [90.0 * SECONDS_PER_DAY, 110.0 * SECONDS_PER_DAY],
        ],
        ephemeris=_CircularEphemeris(),
        mu_sun_km3_s2=MU_SUN,
    )
    kwargs.update(overrides)
    return refine_mga_chain(["EARTH", "VENUS", "EARTH"], cand, **kwargs)


class TestRefineConvergence:
    def test_refinement_tightens_equality_and_keeps_delta_v(self, candidate):
        """①+② 收敛：等模残差 ≤ 1e-6 km/s、总 ΔV 不增、近心点不低于下限。

        观测量（等模/ΔV/近心点）从精化候选的历元/TOF 出发独立解 Lambert
        重建，不复用被测路径内部的任何中间值。
        """
        result = _refine(candidate)
        assert result.status is ConvergenceState.CONVERGED, result.message
        assert result.cause is FailureCause.NONE
        refined = result.candidate
        assert refined is not None

        ephemeris = _CircularEphemeris()
        tofs = [tof_days * SECONDS_PER_DAY for tof_days in refined.leg_tofs_days]
        t_nodes = [
            refined.launch_epoch_et,
            refined.launch_epoch_et + tofs[0],
            refined.launch_epoch_et + tofs[0] + tofs[1],
        ]
        leg1 = solve_lambert(
            ephemeris.get_body_state("EARTH", t_nodes[0], "J2000", "SUN")[:3],
            ephemeris.get_body_state("VENUS", t_nodes[1], "J2000", "SUN")[:3],
            tofs[0],
            MU_SUN,
        )
        leg2 = solve_lambert(
            ephemeris.get_body_state("VENUS", t_nodes[1], "J2000", "SUN")[:3],
            ephemeris.get_body_state("EARTH", t_nodes[2], "J2000", "SUN")[:3],
            tofs[1],
            MU_SUN,
        )
        v_body = ephemeris.get_body_state("VENUS", t_nodes[1], "J2000", "SUN")[3:]
        v_in = leg1.vf - v_body
        v_out = leg2.v0 - v_body
        n_in = float(np.linalg.norm(v_in))
        n_out = float(np.linalg.norm(v_out))
        assert abs(n_in - n_out) <= 1e-6
        dep = float(
            np.linalg.norm(
                leg1.v0 - ephemeris.get_body_state("EARTH", t_nodes[0], "J2000", "SUN")[3:]
            )
        )
        arr = float(
            np.linalg.norm(
                leg2.vf - ephemeris.get_body_state("EARTH", t_nodes[2], "J2000", "SUN")[3:]
            )
        )
        total = dep + arr
        assert refined.total_delta_v_km_s == pytest.approx(total, rel=1e-12)
        assert total <= result.initial_total_delta_v_km_s + 1e-6
        delta = float(np.arccos(np.clip(np.dot(v_in, v_out) / (n_in * n_out), -1.0, 1.0)))
        r_p = flyby_pericenter_radius(delta, 0.5 * (n_in + n_out), _GM["VENUS"])
        assert r_p >= _R_P_MIN


class TestInfeasibleSemantics:
    def test_unreachable_pericenter_reports_infeasible_triple(self, candidate):
        """r_p_min 抬到不可达 → (INFEASIBLE, CONSTRAINT_VIOLATION) 三元组而非异常。"""
        result = _refine(candidate, min_rp=[1e12])
        assert result.status is ConvergenceState.INFEASIBLE
        assert result.cause is FailureCause.CONSTRAINT_VIOLATION
        assert result.candidate is None
        assert result.initial_total_delta_v_km_s == pytest.approx(candidate.total_delta_v_km_s)


class TestInputValidation:
    """入参校验：bounds 形状 / 下界非正 / 界序颠倒 / 历元区间不含候选值。"""

    @pytest.mark.parametrize(
        "overrides",
        [
            {"tof_bounds_s": [[100.0 * SECONDS_PER_DAY, 190.0 * SECONDS_PER_DAY]]},
            {"tof_bounds_s": [[0.0, 1.0], [1.0, 2.0]]},
            {"tof_bounds_s": [[200.0 * SECONDS_PER_DAY, 100.0 * SECONDS_PER_DAY], [1.0, 2.0]]},
            {
                "launch_epoch_bounds_s": (
                    50.0 * SECONDS_PER_DAY,
                    60.0 * SECONDS_PER_DAY,
                )
            },
            {"min_flyby_pericenter_km": [1.0, 2.0]},
        ],
    )
    def test_invalid_arguments_raise_value_error(self, candidate, overrides):
        with pytest.raises(ValueError):
            _refine(candidate, **overrides)


class TestSoftFailureContract:
    def test_result_status_validates_cause_status_pair(self):
        """软失败三元组过 ResultStatus 构造校验：不一致组合拒绝。"""
        with pytest.raises(ValueError, match="状态与原因不一致"):
            MgaRefinementResult(
                ConvergenceState.CONVERGED,
                FailureCause.CONSTRAINT_VIOLATION,
                "不一致组合",
                None,
                0,
                0.0,
            )
