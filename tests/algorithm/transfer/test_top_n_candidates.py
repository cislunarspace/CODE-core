"""top-N 可行解契约（#583，ADR 0040 增补）：候选列表 + 选中解标记。

契约（opt-in ``top_n``，默认关闭时结果不含候选）：

- 收敛运行返回至多 N 个可行候选，按上报 Δv 升序；每候选携带
  Δv / tli_epoch / tof_sec / 轨迹快照（含 ADR 0040 ``state_frame`` 词汇
  的数据系标注）、选中解标记与 Δv 口径（精化后 vs 网格估计）
- 选中解与默认路径会选出的最优解一致（同 Δv、同轨迹）
- LGA/WSB 未精化候选的快照 = 搜索候选的自由飞行弧（网格估计 Δv）；
  HMN/low_thrust 无搜索-精化两级，单候选（权威解数值）
- 快照组装失败降级为该候选无轨迹，不影响其余候选与顶层结果

真值独立构造：合成候选（与 test_trajectory_gcrs.py 同款）+ 手写传播
时长换算（times[-1] = arrival_time_dim × TU）；精化用替身返回同一
候选 + 合成到达弧（与既有编排测试同模式）。
"""

from __future__ import annotations

import math
import warnings
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_System
from e2m2e.algorithm.results import CandidateSearchResult
from e2m2e.algorithm.transfer import (
    STATE_FRAME_SYNODIC_BARYCENTRIC_KM,
    transfer_orbit,
)
from e2m2e.algorithm.transfer.hohmann import MU_EARTH, R_EARTH, TliParams
from e2m2e.algorithm.transfer.lga import LgaCandidate
from e2m2e.algorithm.transfer.wsb import WsbCandidate
from e2m2e.data.templates import ConvergenceState, FailureCause
from e2m2e.exceptions import PropagationFailure

pytestmark = pytest.mark.orchestration

#: 合成候选的已知字段（与 test_trajectory_gcrs.py 同款）
PERILUNE_FRACTION = 0.35
TOF_SEC = 10.0 * 86400.0

#: 合成候选的网格 Δv 分档（升序，dimensionless）
GRID_TOTAL_DVS = (4.06, 4.60, 5.10)


def _canonical_system() -> CR3BP_System:
    """地月 CR3BP 系统（mu 与编排器硬编码值同源，特征尺度 DE421 默认）。"""
    return CR3BP_System(mu=1.21506683e-2, primary="Earth", secondary="Moon")._with_default_scales()


def _make_leo_departure_dim(system):
    """典型 LEO 出发态（无量纲会合系），可被真实传播。"""
    r_park = R_EARTH + 200.0
    v_circ = math.sqrt(MU_EARTH / r_park)
    departure_phys = np.array([r_park, 0.0, 0.0, 0.0, v_circ, 0.0])
    return system.physical_to_dimensionless(departure_phys)


def _fake_arrival_arc(system, n: int = 10):
    """精化到达弧替身：从近月点延续到到达（形状满足拼接契约即可）。"""
    tu = system.characteristic_time
    remaining_dim = (1.0 - PERILUNE_FRACTION) * (TOF_SEC / tu)
    states = np.tile(np.array([0.9, 0.1, 0.0, 0.0, 1.2, 0.0]), (n, 1))
    times = np.linspace(0.0, remaining_dim * tu, n)
    return SimpleNamespace(states=states, times=times)


def _perturbed_departure(base, factor: float):
    """微小速度摄动，让候选出发态彼此可区分（降级测试定点用）。"""
    out = np.array(base, dtype=float)
    out[3:] *= factor
    return out


@contextmanager
def _ignore_warnings():
    """合成候选不保证真实几何，抑制传播/组装路径的 UserWarning。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _synthetic_lga_candidate(system, total_dv: float, departure_state=None) -> LgaCandidate:
    tu = system.characteristic_time
    arrival_time_dim = TOF_SEC / tu
    dv_arr = total_dv - 3.19
    return LgaCandidate(
        departure_phase=1.5,
        out_of_plane_angle=0.0,
        tof_sec=TOF_SEC,
        departure_state=(
            _make_leo_departure_dim(system) if departure_state is None else departure_state
        ),
        perilune_state=np.array([0.9, 0.1, 0.0, 0.0, 1.2, 0.0]),
        perilune_alt_km=1500.0,
        perilune_time_dim=PERILUNE_FRACTION * arrival_time_dim,
        arrival_state=np.array([0.5, 0.5, 0.0, 0.0, 1.0, 0.0]),
        dv_departure=3.19,
        dv_arrival=dv_arr,
        total_dv=total_dv,
        jacobi_departure=4.05,
        jacobi_arrival=4.05,
        arrival_time_dim=arrival_time_dim,
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="synthetic",
    )


class TestLgaTopNCandidates:
    """LGA top-N：候选浮出、排序、选中解与顶层一致。"""

    @pytest.fixture
    def result(self, monkeypatch):
        system = _canonical_system()
        candidates = tuple(_synthetic_lga_candidate(system, dv) for dv in GRID_TOTAL_DVS)
        monkeypatch.setattr(
            "e2m2e.algorithm.transfer.search_lga_trajectories",
            lambda *args, **kwargs: CandidateSearchResult(
                candidates, ConvergenceState.CONVERGED, FailureCause.NONE, "synthetic"
            ),
        )
        monkeypatch.setattr(
            "e2m2e.algorithm.transfer.lga._refine_lga_candidate",
            lambda *args, **kwargs: (candidates[0], _fake_arrival_arc(system)),
        )
        target_phys = system.dimensionless_to_physical(np.array([0.5, 0.5, 0.0, 0.0, 1.0, 0.0]))
        with _ignore_warnings():
            return transfer_orbit(
                "LGA",
                tli_params=TliParams(parking_alt_km=200.0, inclination_deg=0.0),
                target_ephemeris=target_phys.reshape(1, 6),
                top_n=5,
            )

    def test_emits_all_feasible_candidates_below_n(self, result):
        """候选数 = min(N, 可行数)；此处 3 个全数浮出。"""
        assert len(result.candidates) == 3

    def test_exactly_one_selected_and_matches_top_level(self, result):
        """恰一个选中解；Δv 与轨迹与顶层结果一致（默认路径同一解）。"""
        selected = [c for c in result.candidates if c.selected]
        assert len(selected) == 1
        assert selected[0].delta_v_km_s == pytest.approx(result.delta_v)
        assert np.allclose(np.asarray(selected[0].trajectory), np.asarray(result.trajectory))
        assert np.allclose(
            np.asarray(selected[0].trajectory_times), np.asarray(result.trajectory_times)
        )

    def test_candidates_sorted_by_reported_delta_v_ascending(self, result):
        dvs = [c.delta_v_km_s for c in result.candidates]
        assert dvs == sorted(dvs)

    def test_unrefined_candidates_carry_grid_estimate_and_free_flight_snapshot(self, result):
        """未精化候选：Δv = 网格估计（total_dv × 特征速度），快照为自由
        飞行弧（时长换算独立核对：times[-1] = arrival_time_dim × TU）。"""
        system = _canonical_system()
        vu = system.characteristic_velocity
        unrefined = [c for c in result.candidates if not c.selected]
        assert len(unrefined) == 2
        for cand, grid_dv in zip(unrefined, GRID_TOTAL_DVS[1:], strict=True):
            assert cand.refined is False
            assert cand.delta_v_km_s == pytest.approx(grid_dv * vu)
            assert cand.state_frame == STATE_FRAME_SYNODIC_BARYCENTRIC_KM
            states = np.asarray(cand.trajectory)
            times = np.asarray(cand.trajectory_times)
            assert states.shape == (200, 6)
            assert times.shape == (200,)
            assert times[0] == pytest.approx(0.0, abs=1e-6)
            assert times[-1] == pytest.approx(TOF_SEC, rel=1e-9)

    def test_selected_candidate_refined_flag(self, result):
        """精化弧在场时选中解标记为已精化（Δv 与顶层同口径）。"""
        selected = [c for c in result.candidates if c.selected][0]
        assert selected.refined is True
        assert selected.state_frame == STATE_FRAME_SYNODIC_BARYCENTRIC_KM


class TestTopNContractInvariants:
    """契约不变量：截断含选中解、默认关闭零变化、零结果不携带候选。"""

    def _run_lga(
        self,
        monkeypatch,
        candidates: tuple[LgaCandidate, ...],
        status: ConvergenceState = ConvergenceState.CONVERGED,
        **kwargs,
    ):
        system = _canonical_system()
        monkeypatch.setattr(
            "e2m2e.algorithm.transfer.search_lga_trajectories",
            lambda *args, **kwargs: CandidateSearchResult(
                candidates, status, FailureCause.NONE, "synthetic"
            ),
        )
        monkeypatch.setattr(
            "e2m2e.algorithm.transfer.lga._refine_lga_candidate",
            lambda *args, **kwargs: (
                (candidates[0], _fake_arrival_arc(system)) if candidates else None
            ),
        )
        target_phys = system.dimensionless_to_physical(np.array([0.5, 0.5, 0.0, 0.0, 1.0, 0.0]))
        with _ignore_warnings():
            return transfer_orbit(
                "LGA",
                tli_params=TliParams(parking_alt_km=200.0, inclination_deg=0.0),
                target_ephemeris=target_phys.reshape(1, 6),
                **kwargs,
            )

    def test_truncation_keeps_selected_and_fills_by_grid_order(self, monkeypatch):
        """N 小于可行数时截断到 N；选中解（网格最优）必在场。"""
        system = _canonical_system()
        candidates = tuple(_synthetic_lga_candidate(system, dv) for dv in GRID_TOTAL_DVS)
        result = self._run_lga(monkeypatch, candidates, top_n=2)
        assert len(result.candidates) == 2
        assert sum(c.selected for c in result.candidates) == 1
        selected_dv = next(c.delta_v_km_s for c in result.candidates if c.selected)
        assert selected_dv == pytest.approx(result.delta_v)

    def test_top_n_below_one_is_rejected(self, monkeypatch):
        system = _canonical_system()
        candidates = (_synthetic_lga_candidate(system, GRID_TOTAL_DVS[0]),)
        with pytest.raises(ValueError, match="top_n"):
            self._run_lga(monkeypatch, candidates, top_n=0)

    def test_default_top_n_none_carries_no_candidates(self, monkeypatch):
        """默认（不开）与单解契约逐字段一致：结果不含候选。"""
        system = _canonical_system()
        candidates = tuple(_synthetic_lga_candidate(system, dv) for dv in GRID_TOTAL_DVS)
        result = self._run_lga(monkeypatch, candidates)
        assert result.candidates == ()

    def test_zero_result_search_carries_no_candidates(self, monkeypatch):
        """搜索零结果：状态三元组不变，不开候选，即使 top_n 已传。"""
        system = _canonical_system()
        monkeypatch.setattr(
            "e2m2e.algorithm.transfer.search_lga_trajectories",
            lambda *args, **kwargs: CandidateSearchResult(
                (),
                ConvergenceState.INFEASIBLE,
                FailureCause.NO_INTERSECTION,
                "搜索未找到可行候选",
            ),
        )
        target_phys = system.dimensionless_to_physical(np.array([0.5, 0.5, 0.0, 0.0, 1.0, 0.0]))
        with _ignore_warnings():
            result = transfer_orbit(
                "LGA",
                tli_params=TliParams(parking_alt_km=200.0, inclination_deg=0.0),
                target_ephemeris=target_phys.reshape(1, 6),
                top_n=5,
            )
        assert result.status is ConvergenceState.INFEASIBLE
        assert result.candidates == ()


# ---------------------------------------------------------------------------
# WSB：top-N 候选（BCR4BP 自由飞行快照）
# ---------------------------------------------------------------------------


def _synthetic_wsb_candidate(system, total_dv: float) -> WsbCandidate:
    tu = system.characteristic_time
    arrival_time_dim = TOF_SEC / tu
    return WsbCandidate(
        sun_phase0=0.7,
        departure_phase=2.1,
        tof_sec=TOF_SEC,
        departure_state=_make_leo_departure_dim(system),
        perilune_state=np.array([0.9, 0.1, 0.0, 0.0, 1.2, 0.0]),
        perilune_alt_km=2500.0,
        perilune_time_dim=PERILUNE_FRACTION * arrival_time_dim,
        arrival_state=np.array([0.5, 0.5, 0.0, 0.0, 1.0, 0.0]),
        h2_kepler=-0.5,
        dv_departure=3.19,
        dv_arrival=total_dv - 3.19,
        total_dv=total_dv,
        arrival_time_dim=arrival_time_dim,
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="synthetic",
    )


class TestWsbTopNCandidates:
    """WSB top-N：候选浮出与 LGA 同契约，快照走 BCR4BP 自由飞行弧。"""

    @pytest.fixture
    def result(self, monkeypatch):
        from e2m2e.algorithm.dynamics.bcr4bp_system import BCR4BPSystem

        system = BCR4BPSystem.earth_moon()
        candidates = tuple(_synthetic_wsb_candidate(system, dv) for dv in (4.06, 4.60))
        monkeypatch.setattr(
            "e2m2e.algorithm.transfer.search_wsb_trajectories",
            lambda *args, **kwargs: CandidateSearchResult(
                candidates, ConvergenceState.CONVERGED, FailureCause.NONE, "synthetic"
            ),
        )
        monkeypatch.setattr(
            "e2m2e.algorithm.transfer.wsb._refine_wsb_candidate",
            lambda *args, **kwargs: (candidates[0], _fake_arrival_arc(system)),
        )
        target_phys = system.dimensionless_to_physical(np.array([0.5, 0.5, 0.0, 0.0, 1.0, 0.0]))
        with _ignore_warnings():
            return transfer_orbit(
                "WSB",
                tli_params=TliParams(parking_alt_km=200.0, inclination_deg=0.0),
                target_ephemeris=target_phys.reshape(1, 6),
                top_n=5,
            )

    def test_emits_candidates_with_selected_matching_top_level(self, result):
        assert len(result.candidates) == 2
        selected = [c for c in result.candidates if c.selected]
        assert len(selected) == 1
        assert selected[0].delta_v_km_s == pytest.approx(result.delta_v)
        assert np.allclose(np.asarray(selected[0].trajectory), np.asarray(result.trajectory))
        assert selected[0].refined is True

    def test_unrefined_candidate_snapshot_is_bcr4bp_free_flight(self, result):
        """未精化候选：BCR4BP 网格估计 Δv + 自由飞行弧（时长独立核对）。"""
        from e2m2e.algorithm.dynamics.bcr4bp_system import BCR4BPSystem

        vu = BCR4BPSystem.earth_moon().characteristic_velocity
        unrefined = [c for c in result.candidates if not c.selected]
        assert len(unrefined) == 1
        cand = unrefined[0]
        assert cand.refined is False
        assert cand.delta_v_km_s == pytest.approx(4.60 * vu)
        assert cand.state_frame == STATE_FRAME_SYNODIC_BARYCENTRIC_KM
        states = np.asarray(cand.trajectory)
        times = np.asarray(cand.trajectory_times)
        assert states.shape == (200, 6)
        assert times[-1] == pytest.approx(TOF_SEC, rel=1e-6)
        assert cand.trajectory_times is not None


# ---------------------------------------------------------------------------
# 快照组装失败降级：单候选无轨迹，其余与顶层结果不受影响
# ---------------------------------------------------------------------------


class TestSnapshotFailureDegradation:
    def test_failing_candidate_snapshot_degrades_to_none(self, monkeypatch):
        """某个未精化候选传播失败 → 该候选无轨迹，其余候选与顶层不变。"""
        system = _canonical_system()
        base = _make_leo_departure_dim(system)
        candidates = (
            _synthetic_lga_candidate(system, GRID_TOTAL_DVS[0], departure_state=base),
            _synthetic_lga_candidate(
                system, GRID_TOTAL_DVS[1], departure_state=_perturbed_departure(base, 1.001)
            ),
            _synthetic_lga_candidate(
                system, GRID_TOTAL_DVS[2], departure_state=_perturbed_departure(base, 1.002)
            ),
        )
        poisoned = candidates[2].departure_state
        real_propagate = None
        from e2m2e.algorithm.transfer import _propagate_synodic_leg

        real_propagate = _propagate_synodic_leg

        def flaky(dynamics, sys_, x0_dim, t_end_dim, n_samples: int = 200):
            if np.allclose(np.asarray(x0_dim, dtype=float), poisoned):
                raise PropagationFailure("合成的快照传播失败")
            return real_propagate(dynamics, sys_, x0_dim, t_end_dim, n_samples)

        monkeypatch.setattr(
            "e2m2e.algorithm.transfer.search_lga_trajectories",
            lambda *args, **kwargs: CandidateSearchResult(
                candidates, ConvergenceState.CONVERGED, FailureCause.NONE, "synthetic"
            ),
        )
        monkeypatch.setattr(
            "e2m2e.algorithm.transfer.lga._refine_lga_candidate",
            lambda *args, **kwargs: (candidates[0], _fake_arrival_arc(system)),
        )
        monkeypatch.setattr(
            "e2m2e.algorithm.transfer._propagate_synodic_leg",
            flaky,
        )
        target_phys = system.dimensionless_to_physical(np.array([0.5, 0.5, 0.0, 0.0, 1.0, 0.0]))
        with _ignore_warnings():
            result = transfer_orbit(
                "LGA",
                tli_params=TliParams(parking_alt_km=200.0, inclination_deg=0.0),
                target_ephemeris=target_phys.reshape(1, 6),
                top_n=5,
            )

        assert result.status is ConvergenceState.CONVERGED
        assert np.allclose(np.asarray(result.trajectory), result.trajectory)  # 顶层轨迹在场
        by_dv = {round(c.delta_v_km_s, 6): c for c in result.candidates}
        assert len(by_dv) == 3
        poisoned_candidates = [
            c for c in result.candidates if not c.selected and c.trajectory is None
        ]
        assert len(poisoned_candidates) == 1
        assert poisoned_candidates[0].trajectory_times is None
        # 未中招的未精化候选仍带快照
        healthy = [c for c in result.candidates if not c.selected and c.trajectory is not None]
        assert len(healthy) == 1


# ---------------------------------------------------------------------------
# HMN / low_thrust：无搜索-精化两级，单候选（权威解数值）
# ---------------------------------------------------------------------------


class TestSingleCandidateBackends:
    """HMN/low_thrust 开启 top_n 时返回恰一个选中解；默认仍无候选。"""

    def test_hmn_single_candidate_matches_top_level(self):
        result = transfer_orbit(
            "HMN",
            tli_params=TliParams(parking_alt_km=200.0, inclination_deg=0.0),
            target_orbit_radius_km=42164.0,
            top_n=5,
        )
        assert len(result.candidates) == 1
        cand = result.candidates[0]
        assert cand.selected is True
        assert cand.refined is True
        assert cand.delta_v_km_s == pytest.approx(result.delta_v)
        assert np.allclose(np.asarray(cand.trajectory), np.asarray(result.trajectory))
        assert np.allclose(np.asarray(cand.trajectory_times), np.asarray(result.trajectory_times))
        assert cand.state_frame == STATE_FRAME_SYNODIC_BARYCENTRIC_KM

    def test_hmn_default_carries_no_candidates(self):
        result = transfer_orbit(
            "HMN",
            tli_params=TliParams(parking_alt_km=200.0, inclination_deg=0.0),
            target_orbit_radius_km=42164.0,
        )
        assert result.candidates == ()

    def test_low_thrust_single_candidate_force_model_frame(self):
        import e2m2e.algorithm.transfer as transfer_pkg

        fake_sol = SimpleNamespace(
            states=np.zeros((3, 7)),
            final_mass=990.0,
            fuel_consumed=10.0,
            status=ConvergenceState.CONVERGED,
            cause=FailureCause.NONE,
            message="synthetic",
            n_iter=5,
            time=np.array([0.0, 1.0, 2.0]),
            segments=(),
        )
        with patch.object(transfer_pkg.LowThrustShooting, "solve_from_qlaw", return_value=fake_sol):
            result = transfer_orbit(
                "low_thrust",
                tli_params=TliParams(parking_alt_km=200.0, inclination_deg=0.0),
                engine_config=transfer_pkg.EngineConfig(t_max=0.1, isp=3000.0),
                initial_mass=1000.0,
                n_segments=2,
                duration_days=1.0,
                target_state=np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0, 0.0]),
                top_n=5,
            )
        assert len(result.candidates) == 1
        cand = result.candidates[0]
        assert cand.selected is True
        assert cand.refined is True
        assert cand.delta_v_km_s == pytest.approx(result.delta_v)
        assert np.allclose(np.asarray(cand.trajectory), np.asarray(result.trajectory))
        # 力模型状态系（top-level 同标注）；low_thrust 契约无 trajectory_times
        assert cand.state_frame == "force_model_state"
        assert cand.trajectory_times is None
