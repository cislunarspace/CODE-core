"""行星际 MGA 网格搜索（``search_mga_chains``）编排测试。

合成圆共面解析星历注入（不碰 SPICE）：日心 ``(6,)`` 状态由 ``a``、相位与圆速度
闭式给出，Lambert 走真实 Rust 批量内核。验收包含链级 Lambert 一致性（链在任意
借力强度下逐 leg 就是 Lambert 解，ADR 0055 决策 5 ③ 的链级形态）、不可行语义
（无解 vs 约束剔除）与进度/记忆化契约。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from e2m2e.algorithm.transfer.lambert import solve_lambert
from e2m2e.algorithm.transfer.mga import (
    flyby_pericenter_radius,
    heliocentric_tisserand,
    search_mga_chains,
)
from e2m2e.algorithm.transfer.qlaw import rv_to_keplerian
from e2m2e.data.templates import ConvergenceState, FailureCause

pytestmark = pytest.mark.orchestration

MU_SUN = 1.32712440018e11  # km³/s²
AU = 1.495978707e8  # km
SECONDS_PER_DAY = 86400.0

#: 合成行星 GM：太阳与地球/金星量级（飞越天体 GM 只用其量级）。
_GM = {"SUN": MU_SUN, "EARTH": 3.986004418e5, "VENUS": 3.24859e5}

#: 合成行星轨道：圆、共面（日心 x-y 面）→ 名 → (半长轴 km, 初相位 rad)。
_PLANETS: dict[str, tuple[float, float]] = {
    "EARTH": (AU, 0.0),
    "VENUS": (0.7233 * AU, 0.3),
}


class _CircularEphemeris:
    """圆共面解析星历：``get_body_state``/``get_gm`` 两个方法即算法全部依赖。"""

    def __init__(self, planets: dict[str, tuple[float, float]] | None = None) -> None:
        self._planets = dict(_PLANETS if planets is None else planets)
        self.calls: list[tuple[str, float]] = []

    def get_gm(self, body: str) -> float:
        return _GM[body.upper()]

    def get_body_state(self, target: str, et: float, frame: str, observer: str) -> np.ndarray:
        assert frame == "J2000" and observer == "SUN", "MGA 链一律日心 J2000 几何"
        name = target.upper()
        et = float(et)
        self.calls.append((name, et))
        semi_major, phase = self._planets[name]
        mean_motion = math.sqrt(MU_SUN / semi_major**3)
        theta = phase + mean_motion * et
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


class _FrozenEphemeris:
    """位置完全冻结的星历（速度非零）：任意 (dep, arr) 弦长逐位为零。

    用于验证“网格内 Lambert 无可行 leg”分支：Rust 侧对零弦长返回错误，
    批量解全为 NaN。
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def get_gm(self, body: str) -> float:
        return _GM[body.upper()]

    def get_body_state(self, target: str, et: float, frame: str, observer: str) -> np.ndarray:
        assert frame == "J2000" and observer == "SUN"
        self.calls.append((target.upper(), float(et)))
        if target.upper() == "SUN":
            return np.zeros(6)
        return np.array([AU, 0.0, 0.0, 0.0, 29.7847, 0.0])


def _run(body_sequence, launch_epochs, tof_grids, min_rp, *, ephemeris=None, **kwargs):
    return search_mga_chains(
        body_sequence,
        launch_epochs,
        tof_grids,
        min_rp,
        ephemeris=ephemeris if ephemeris is not None else _CircularEphemeris(),
        mu_sun_km3_s2=MU_SUN,
        **kwargs,
    )


class TestSearchConvergence:
    def test_earth_venus_chain_converges_sorted_and_truncated(self):
        """两体链：全格可行 → CONVERGED、按总 ΔV 升序、top_n 截断。"""
        result = _run(
            ["EARTH", "VENUS"],
            [0.0, 5.0 * SECONDS_PER_DAY],
            [[100.0, 110.0, 120.0]],
            [],
            top_n=2,
        )
        assert result.status is ConvergenceState.CONVERGED
        assert result.cause is FailureCause.NONE
        assert len(result.candidates) == 2  # 6 格全部可行，截到 top_n=2
        assert "/ 6 格" in result.message
        totals = [candidate.total_delta_v_km_s for candidate in result.candidates]
        assert totals == sorted(totals)
        for candidate in result.candidates:
            assert candidate.total_delta_v_km_s == pytest.approx(
                candidate.departure_v_inf_km_s + candidate.arrival_v_inf_km_s, rel=1e-15
            )
            assert candidate.flybys == ()  # 两体链无飞越
            assert candidate.leg_tofs_days[0] in (100.0, 110.0, 120.0)

    def test_candidate_legs_reproduce_direct_lambert_solution(self):
        """链级一致性（③）：候选逐 leg 与直接 Lambert 复算逐位一致。"""
        ephemeris = _CircularEphemeris()
        result = _run(
            ["EARTH", "VENUS"],
            [0.0, 3.0 * SECONDS_PER_DAY],
            [[105.0, 115.0]],
            [],
            ephemeris=ephemeris,
        )
        assert result.status is ConvergenceState.CONVERGED
        candidate = result.candidates[0]
        launch_et = candidate.launch_epoch_et
        tof_sec = candidate.leg_tofs_days[0] * SECONDS_PER_DAY
        earth = ephemeris.get_body_state("EARTH", launch_et, "J2000", "SUN")
        venus = ephemeris.get_body_state("VENUS", launch_et + tof_sec, "J2000", "SUN")
        solution = solve_lambert(earth[:3], venus[:3], tof_sec, MU_SUN, direction="short")
        assert candidate.departure_v_inf_km_s == pytest.approx(
            float(np.linalg.norm(solution.v0 - earth[3:])), rel=1e-12
        )
        assert candidate.arrival_v_inf_km_s == pytest.approx(
            float(np.linalg.norm(solution.vf - venus[3:])), rel=1e-12
        )

    def test_three_body_chain_reports_flyby_diagnostics(self):
        """含飞越的链：flyby 诊断字段有限且近心点半径不低于下限。

        网格由合成星历下的经验匹配点给出（等模差 ~0.06 km/s），容差取 3 km/s。
        """
        result = _run(
            ["EARTH", "VENUS", "EARTH"],
            [day * SECONDS_PER_DAY for day in (0.0, 4.0, 16.0, 20.0)],
            [[180.0], [100.0]],
            [1.0],
            v_inf_match_tol_km_s=3.0,
            top_n=5,
        )
        assert result.status is ConvergenceState.CONVERGED
        assert result.candidates
        for candidate in result.candidates:
            assert len(candidate.leg_tofs_days) == 2
            assert len(candidate.flybys) == 1
            flyby = candidate.flybys[0]
            assert flyby.body == "VENUS"
            assert flyby.pericenter_radius_km >= 1.0
            diagnostics = (
                flyby.v_inf_km_s,
                flyby.turn_angle_rad,
                flyby.pericenter_radius_km,
                flyby.tisserand_before,
                flyby.tisserand_after,
            )
            assert all(math.isfinite(value) for value in diagnostics)
            assert 0.0 <= flyby.turn_angle_rad < math.pi


class TestFlybyAssemblyOracle:
    """用候选自身历元独立解相邻两条 leg 的 Lambert，重建 flyby 观测量。

    这是装配层（混合进制 leg 索引映射 + flyby 评估）的端到端 oracle：只有索引
    与两腿速度对应正确时，重建值才能与响应逐项相等。等模容差取极大值，使全部
    有解格都进入候选（剔除面另测）。
    """

    @staticmethod
    def _leg_velocities(ephemeris, dep_body, arr_body, epochs_et, k):
        """第 k 条 leg 的 (v0, vf)：独立 Lambert 复算。"""
        r0 = ephemeris.get_body_state(dep_body, epochs_et[k], "J2000", "SUN")[:3]
        rf = ephemeris.get_body_state(arr_body, epochs_et[k + 1], "J2000", "SUN")[:3]
        solution = solve_lambert(r0, rf, epochs_et[k + 1] - epochs_et[k], MU_SUN)
        return solution.v0, solution.vf

    def _recompute(self, ephemeris, bodies, candidate, index):
        """重建中间天体 ``bodies[index]`` 处 flyby 的 (vin, vout, state)。"""
        epochs_et = [candidate.launch_epoch_et]
        for tof in candidate.leg_tofs_days:
            epochs_et.append(epochs_et[-1] + tof * SECONDS_PER_DAY)
        state = ephemeris.get_body_state(bodies[index], epochs_et[index], "J2000", "SUN")
        _, v_arrival = self._leg_velocities(
            ephemeris, bodies[index - 1], bodies[index], epochs_et, index - 1
        )
        v_departure, _ = self._leg_velocities(
            ephemeris, bodies[index], bodies[index + 1], epochs_et, index
        )
        return v_arrival - state[3:], v_departure - state[3:], state

    @pytest.mark.parametrize(
        "bodies", [["EARTH", "VENUS", "EARTH"], ["EARTH", "VENUS", "EARTH", "VENUS"]]
    )
    def test_reported_flybys_match_independent_reconstruction(self, bodies):
        ephemeris = _CircularEphemeris()
        grids = [[180.0], [100.0], [160.0]][: len(bodies) - 1]
        result = _run(
            bodies,
            [0.0, 4.0 * SECONDS_PER_DAY],
            grids,
            [1.0] * (len(bodies) - 2),
            ephemeris=ephemeris,
            v_inf_match_tol_km_s=1e6,  # 等模不设限：剔除面由专门用例覆盖
            top_n=10,
        )
        assert result.status is ConvergenceState.CONVERGED
        assert len(result.candidates) == 2  # 每个发射历元恰一格
        for candidate in result.candidates:
            assert len(candidate.flybys) == len(bodies) - 2
            for index, flyby in enumerate(candidate.flybys, start=1):
                v_in, v_out, state = self._recompute(ephemeris, bodies, candidate, index)
                norm_in = float(np.linalg.norm(v_in))
                norm_out = float(np.linalg.norm(v_out))
                assert flyby.body == bodies[index]
                assert flyby.v_inf_km_s == pytest.approx(0.5 * (norm_in + norm_out), rel=1e-12)
                expected_angle = math.acos(float(np.dot(v_in, v_out)) / (norm_in * norm_out))
                assert flyby.turn_angle_rad == pytest.approx(expected_angle, rel=1e-12)
                a_ref, *_ = rv_to_keplerian(state[:3], state[3:], MU_SUN)
                # Tisserand 参考面 = 飞越天体轨道面（此处合成天体在 x-y 面内，n̂ = ẑ）
                ref_normal = np.cross(state[:3], state[3:])
                ref_normal = ref_normal / np.linalg.norm(ref_normal)
                expected_radius = flyby_pericenter_radius(
                    expected_angle, flyby.v_inf_km_s, _GM[bodies[index]]
                )
                assert flyby.pericenter_radius_km == pytest.approx(expected_radius, rel=1e-9)
                assert flyby.tisserand_before == pytest.approx(
                    heliocentric_tisserand(
                        state[:3], state[3:] + v_in, MU_SUN, float(a_ref), ref_normal
                    ),
                    rel=1e-12,
                )
                assert flyby.tisserand_after == pytest.approx(
                    heliocentric_tisserand(
                        state[:3], state[3:] + v_out, MU_SUN, float(a_ref), ref_normal
                    ),
                    rel=1e-12,
                )


class TestInfeasibleSemantics:
    def test_all_cells_rejected_by_flyby_constraint(self):
        """有解格全被 flyby 约束剔除 → (INFEASIBLE, CONSTRAINT_VIOLATION)。"""
        result = _run(
            ["EARTH", "VENUS", "EARTH"],
            [day * SECONDS_PER_DAY for day in (0.0, 4.0, 16.0, 20.0)],
            [[180.0], [100.0]],
            [1e12],  # 任何合理近心点半径都不达标
            v_inf_match_tol_km_s=3.0,
        )
        assert result.status is ConvergenceState.INFEASIBLE
        assert result.cause is FailureCause.CONSTRAINT_VIOLATION
        assert result.candidates == ()

    def test_degenerate_geometry_yields_no_intersection(self):
        """位置冻结（弦长逐位为零）→ 每格 Lambert 无解 → NO_INTERSECTION。"""
        result = _run(
            ["EARTH", "VENUS"],
            [0.0, 2.0 * SECONDS_PER_DAY],
            [[100.0, 120.0]],
            [],
            ephemeris=_FrozenEphemeris(),
        )
        assert result.status is ConvergenceState.INFEASIBLE
        assert result.cause is FailureCause.NO_INTERSECTION
        assert result.candidates == ()


class TestInputValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"body_sequence": ["EARTH"]},
            {"body_sequence": ["EARTH", "VENUS"], "leg_tof_grids_days": [[100.0], [120.0]]},
            {"body_sequence": ["EARTH", "VENUS"], "leg_tof_grids_days": [[]]},
            {"body_sequence": ["EARTH", "VENUS"], "leg_tof_grids_days": [[0.0]]},
            {"body_sequence": ["EARTH", "VENUS"], "leg_tof_grids_days": [[100.0, -5.0]]},
            {"body_sequence": ["EARTH", "VENUS"], "min_flyby_pericenter_km": [1.0]},
            {"body_sequence": ["EARTH", "VENUS"], "min_flyby_pericenter_km": [-1.0]},
            {"body_sequence": ["EARTH", "VENUS"], "launch_epochs_et": []},
            {"body_sequence": ["EARTH", "VENUS"], "top_n": 0},
            {"body_sequence": ["EARTH", "VENUS"], "v_inf_match_tol_km_s": -0.5},
        ],
    )
    def test_invalid_arguments_raise_value_error(self, kwargs):
        params = {
            "body_sequence": ["EARTH", "VENUS"],
            "launch_epochs_et": [0.0],
            "leg_tof_grids_days": [[100.0]],
            "min_flyby_pericenter_km": [],
        }
        params.update(kwargs)
        with pytest.raises(ValueError):
            _run(
                params["body_sequence"],
                params["launch_epochs_et"],
                params["leg_tof_grids_days"],
                params["min_flyby_pericenter_km"],
                top_n=kwargs.get("top_n", 5),
                v_inf_match_tol_km_s=kwargs.get("v_inf_match_tol_km_s", 0.1),
            )


class TestSearchContracts:
    def test_progress_callback_counts_every_launch_leg_batch(self):
        seen: list[int] = []
        _run(
            ["EARTH", "VENUS", "EARTH"],
            [0.0, 2.0 * SECONDS_PER_DAY, 4.0 * SECONDS_PER_DAY],
            [[130.0], [150.0]],
            [1.0],
            v_inf_match_tol_km_s=3.0,
            progress_callback=seen.append,
        )
        assert seen == [1] * 6  # 3 发射历元 × 2 leg

    def test_body_states_are_memoized_per_epoch(self):
        ephemeris = _CircularEphemeris()
        _run(
            ["EARTH", "VENUS", "EARTH"],
            [0.0],
            [[120.0, 130.0], [140.0, 150.0]],
            [1.0],
            v_inf_match_tol_km_s=3.0,
            ephemeris=ephemeris,
        )
        assert len(ephemeris.calls) == len(set(ephemeris.calls)), "同一 (天体, 历元) 应只查一次"
