"""机动事件结构化契约测试（#575）：`maneuver_events` 事件列表。

事件 schema：``kind``（departure/perilune/arrival/…）+ ``t_sec``（TLI 起算
秒，与 trajectory_times 同基准）+ ``dv_km_s``（该次脉冲大小）+ ``note``。

填充规则（#575 契约）：
- HMN：departure + arrival 两条（到达点即近月点，不另发 perilune）
- LGA/WSB：departure + perilune + arrival 三条，perilune 的 dv_km_s = 0
  （飞越段无脉冲；速度折点是精化的隐式修正，ADR 0040 不计入 Δv 收账）
- low_thrust：连续推进无脉冲语义，事件列表为空
- 搜索零结果：事件列表为空

预期值一律来自独立真值：HMN 用 hohmann 解析解；LGA/WSB 用合成候选的
已知字段（无量纲 × 特征量换算是 docstring 契约，非实现复算）。
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
from e2m2e.algorithm.transfer import ManeuverEvent, TransferDesignResult, transfer_orbit
from e2m2e.algorithm.transfer.hohmann import (
    MU_EARTH,
    R_EARTH,
    TliParams,
    hohmann_delta_v,
    hohmann_tof,
)
from e2m2e.algorithm.transfer.lga import LgaCandidate
from e2m2e.algorithm.transfer.wsb import WsbCandidate
from e2m2e.data.constants import Datum
from e2m2e.data.templates import ConvergenceState, FailureCause

pytestmark = pytest.mark.orchestration

MU = Datum.DE421.mu

#: 合成候选的已知字段真值（无量纲）
DV_DEPARTURE_DIM = 3.19
DV_ARRIVAL_DIM = 0.87
PERILUNE_FRACTION = 0.35
TOF_SEC = 10.0 * 86400.0


def _make_leo_departure_dim(system):
    """典型 LEO 出发态（无量纲会合系），可被 CR3BP/BCR4BP 真实传播。"""
    r_park = R_EARTH + 200.0
    v_circ = math.sqrt(MU_EARTH / r_park)
    departure_phys = np.array([r_park, 0.0, 0.0, 0.0, v_circ, 0.0])
    return system.physical_to_dimensionless(departure_phys)


def _fake_arrival_arc(system, n: int = 10) -> SimpleNamespace:
    """精化到达弧替身：从近月点延续到到达（形状满足拼接契约即可）。"""
    tu = system.characteristic_time
    remaining_dim = (1.0 - PERILUNE_FRACTION) * (TOF_SEC / tu)
    states = np.tile(np.array([0.9, 0.1, 0.0, 0.0, 1.2, 0.0]), (n, 1))
    times = np.linspace(0.0, remaining_dim * tu, n)
    return SimpleNamespace(states=states, times=times)


@contextmanager
def _ignore_warnings():
    """合成候选不保证真实几何，抑制传播/组装路径的 UserWarning。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# ---------------------------------------------------------------------------
# HMN：解析真值
# ---------------------------------------------------------------------------


class TestHmnManeuverEvents:
    def test_departure_and_arrival_match_analytic_solution(self):
        """HMN 收敛响应含 departure/arrival 两条，t_sec/dv 与解析解一致。"""
        params = TliParams(parking_alt_km=200.0, inclination_deg=0.0)
        r1 = R_EARTH + 200.0
        r2 = 384405.0
        dv1, dv2 = hohmann_delta_v(r1, r2)
        tof = hohmann_tof(r1, r2)

        result = transfer_orbit("HMN", tli_params=params, target_orbit_radius_km=r2)

        events = result.maneuver_events
        assert all(isinstance(e, ManeuverEvent) for e in events)
        assert [e.kind for e in events] == ["departure", "arrival"]
        assert events[0].t_sec == 0.0
        assert events[0].dv_km_s == pytest.approx(dv1, rel=1e-12)
        assert events[1].t_sec == pytest.approx(tof, rel=1e-12)
        assert events[1].dv_km_s == pytest.approx(dv2, rel=1e-12)
        # 到达点即近月点：HMN 不发独立 perilune 事件（#575 已定）

    def test_events_agree_with_legacy_details_fields(self):
        """新旧字段并行期：dv_km_s/t_sec 与 details 旧字段数值一致。"""
        params = TliParams(parking_alt_km=200.0, inclination_deg=0.0)
        result = transfer_orbit("HMN", tli_params=params, target_orbit_radius_km=384405.0)
        events = result.maneuver_events
        assert events[0].dv_km_s == pytest.approx(result.details.dv1_km_s, rel=1e-12)
        assert events[1].dv_km_s == pytest.approx(result.details.dv2_km_s, rel=1e-12)
        assert events[1].t_sec == pytest.approx(result.details.tof_sec, rel=1e-12)


# ---------------------------------------------------------------------------
# LGA：合成候选（确定性真值）
# ---------------------------------------------------------------------------


def _synthetic_lga_candidate(system) -> LgaCandidate:
    """已知字段的 LGA 候选：perilune 位于 0.35×tof，dv 字段为指定无量纲值。"""
    tu = system.characteristic_time
    arrival_time_dim = TOF_SEC / tu
    return LgaCandidate(
        departure_phase=1.5,
        out_of_plane_angle=0.0,
        tof_sec=TOF_SEC,
        departure_state=_make_leo_departure_dim(system),
        perilune_state=np.array([0.9, 0.1, 0.0, 0.0, 1.2, 0.0]),
        perilune_alt_km=1500.0,
        perilune_time_dim=PERILUNE_FRACTION * arrival_time_dim,
        arrival_state=np.array([0.5, 0.5, 0.0, 0.0, 1.0, 0.0]),
        dv_departure=DV_DEPARTURE_DIM,
        dv_arrival=DV_ARRIVAL_DIM,
        total_dv=DV_DEPARTURE_DIM + DV_ARRIVAL_DIM,
        jacobi_departure=4.05,
        jacobi_arrival=4.05,
        arrival_time_dim=arrival_time_dim,
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="synthetic",
    )


class TestLgaManeuverEvents:
    @pytest.fixture
    def result(self, monkeypatch):
        system = CR3BP_System(mu=MU, primary="Earth", secondary="Moon")._with_default_scales()
        candidate = _synthetic_lga_candidate(system)
        monkeypatch.setattr(
            "e2m2e.algorithm.transfer.search_lga_trajectories",
            lambda *args, **kwargs: CandidateSearchResult(
                (candidate,), ConvergenceState.CONVERGED, FailureCause.NONE, "synthetic"
            ),
        )
        monkeypatch.setattr(
            "e2m2e.algorithm.transfer.lga._refine_lga_candidate",
            lambda *args, **kwargs: (candidate, _fake_arrival_arc(system)),
        )
        target_phys = system.dimensionless_to_physical(np.array([0.5, 0.5, 0.0, 0.0, 1.0, 0.0]))
        with _ignore_warnings():
            return transfer_orbit(
                "LGA",
                tli_params=TliParams(parking_alt_km=200.0, inclination_deg=0.0),
                target_ephemeris=target_phys.reshape(1, 6),
            )

    def test_three_events_with_perilune(self, result):
        """LGA 收敛响应含 departure/perilune/arrival 三条，t_sec 单调。"""
        events = result.maneuver_events
        assert [e.kind for e in events] == ["departure", "perilune", "arrival"]
        assert events[0].t_sec == 0.0
        assert 0.0 < events[1].t_sec < events[2].t_sec

    def test_perilune_time_is_candidate_field_times_unit(self, result, monkeypatch):
        """perilune.t_sec = 候选 perilune_time_dim × 特征时间（独立换算真值）。"""
        system = CR3BP_System(mu=MU, primary="Earth", secondary="Moon")._with_default_scales()
        expected_perilune_sec = (
            PERILUNE_FRACTION * (TOF_SEC / system.characteristic_time)
        ) * system.characteristic_time
        events = result.maneuver_events
        assert events[1].t_sec == pytest.approx(expected_perilune_sec, rel=1e-12)
        assert events[1].t_sec == pytest.approx(TOF_SEC * PERILUNE_FRACTION, rel=1e-9)

    def test_perilune_time_matches_trajectory_join_point(self, result):
        """perilune.t_sec 与轨迹出发段末值（拼接点时刻）一致。"""
        times = np.asarray(result.trajectory_times)
        events = result.maneuver_events
        # 出发段 200 采样，拼接点 = times[199]
        assert events[1].t_sec == pytest.approx(times[199], rel=1e-9)
        assert times[-1] == pytest.approx(TOF_SEC, rel=1e-9)

    def test_dv_matches_details_legacy_fields(self, result, monkeypatch):
        """dv_km_s 与 details 旧字段同值；perilune 不计脉冲（dv=0）。"""
        system = CR3BP_System(mu=MU, primary="Earth", secondary="Moon")._with_default_scales()
        vu = system.characteristic_velocity
        events = result.maneuver_events
        assert events[0].dv_km_s == pytest.approx(DV_DEPARTURE_DIM * vu, rel=1e-12)
        assert events[0].dv_km_s == pytest.approx(result.details.dv_departure_km_s, rel=1e-12)
        assert events[2].dv_km_s == pytest.approx(DV_ARRIVAL_DIM * vu, rel=1e-12)
        assert events[2].dv_km_s == pytest.approx(result.details.dv_arrival_km_s, rel=1e-12)
        assert events[1].dv_km_s == 0.0
        assert events[2].t_sec == pytest.approx(result.details.tof_sec, rel=1e-12)


# ---------------------------------------------------------------------------
# WSB：合成候选（到达 Δv 按 #566 约定 ×CR3BP 特征速度）
# ---------------------------------------------------------------------------


def _synthetic_wsb_candidate(system) -> WsbCandidate:
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
        dv_departure=DV_DEPARTURE_DIM,
        dv_arrival=DV_ARRIVAL_DIM,
        total_dv=DV_DEPARTURE_DIM + DV_ARRIVAL_DIM,
        arrival_time_dim=arrival_time_dim,
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="synthetic",
    )


class TestWsbManeuverEvents:
    @pytest.fixture
    def result(self, monkeypatch):
        from e2m2e.algorithm.dynamics.bcr4bp_system import BCR4BPSystem

        system = BCR4BPSystem.earth_moon()
        candidate = _synthetic_wsb_candidate(system)
        monkeypatch.setattr(
            "e2m2e.algorithm.transfer.search_wsb_trajectories",
            lambda *args, **kwargs: CandidateSearchResult(
                (candidate,), ConvergenceState.CONVERGED, FailureCause.NONE, "synthetic"
            ),
        )
        monkeypatch.setattr(
            "e2m2e.algorithm.transfer.wsb._refine_wsb_candidate",
            lambda *args, **kwargs: (candidate, _fake_arrival_arc(system)),
        )
        target_phys = system.dimensionless_to_physical(np.array([0.5, 0.5, 0.0, 0.0, 1.0, 0.0]))
        with _ignore_warnings():
            return transfer_orbit(
                "WSB",
                tli_params=TliParams(parking_alt_km=200.0, inclination_deg=0.0),
                target_ephemeris=target_phys.reshape(1, 6),
            )

    def test_three_events_with_perilune(self, result):
        """WSB 收敛响应含 departure/perilune/arrival 三条，t_sec 单调。"""
        events = result.maneuver_events
        assert [e.kind for e in events] == ["departure", "perilune", "arrival"]
        assert events[0].t_sec == 0.0
        assert 0.0 < events[1].t_sec < events[2].t_sec
        assert events[1].dv_km_s == 0.0
        assert events[1].t_sec == pytest.approx(TOF_SEC * PERILUNE_FRACTION, rel=1e-9)

    def test_perilune_time_matches_trajectory_join_point(self, result):
        """perilune.t_sec 与轨迹出发段末值（拼接点时刻）一致。"""
        times = np.asarray(result.trajectory_times)
        events = result.maneuver_events
        assert events[1].t_sec == pytest.approx(times[199], rel=1e-9)
        assert times[-1] == pytest.approx(TOF_SEC, rel=1e-9)

    def test_dv_conventions(self, result, monkeypatch):
        """出发 ×BCR4BP VU、到达 ×CR3BP VU（#566 收账约定），与 details 一致。"""
        from e2m2e.algorithm.dynamics.bcr4bp_system import BCR4BPSystem

        vu_bcr4 = BCR4BPSystem.earth_moon().characteristic_velocity
        vu_cr3 = (
            CR3BP_System(mu=MU, primary="Earth", secondary="Moon")
            ._with_default_scales()
            .characteristic_velocity
        )
        events = result.maneuver_events
        assert events[0].dv_km_s == pytest.approx(DV_DEPARTURE_DIM * vu_bcr4, rel=1e-12)
        assert events[2].dv_km_s == pytest.approx(DV_ARRIVAL_DIM * vu_cr3, rel=1e-12)
        assert events[0].dv_km_s == pytest.approx(result.details.dv_departure_km_s, rel=1e-12)
        assert events[2].dv_km_s == pytest.approx(result.details.dv_arrival_km_s, rel=1e-12)


# ---------------------------------------------------------------------------
# 零结果与 low_thrust：空事件列表
# ---------------------------------------------------------------------------


class TestEmptyManeuverEvents:
    def test_lga_zero_result_has_no_events(self, monkeypatch):
        """LGA 搜索零结果：事件列表为空（不伪造出发/到达脉冲）。"""
        monkeypatch.setattr(
            "e2m2e.algorithm.transfer.search_lga_trajectories",
            lambda *args, **kwargs: CandidateSearchResult(
                (), ConvergenceState.INFEASIBLE, FailureCause.NO_INTERSECTION, "无候选"
            ),
        )
        system = CR3BP_System(mu=MU, primary="Earth", secondary="Moon")._with_default_scales()
        target_phys = system.dimensionless_to_physical(np.array([0.5, 0.5, 0.0, 0.0, 1.0, 0.0]))
        result = transfer_orbit(
            "LGA",
            tli_params=TliParams(parking_alt_km=200.0, inclination_deg=0.0),
            target_ephemeris=target_phys.reshape(1, 6),
        )
        assert result.maneuver_events == ()

    def test_low_thrust_has_no_impulse_events(self, monkeypatch):
        """low_thrust 连续推进：事件列表恒为空（无脉冲语义）。"""
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
            )
        assert isinstance(result, TransferDesignResult)
        assert result.maneuver_events == ()
