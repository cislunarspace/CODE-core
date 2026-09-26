"""PCN patched-conic 目标参数化测试（#635，ADR 0013 定义级验证 + ADR 0040 契约）。

覆盖：出发构造能量不变量、出发模式的真实月交会、到达模式往返收敛
（出发→到达目标按构造可行）、ADR 0040 轨迹契约、失败路径状态三元组、
top_n 不支持。用极小网格（远低于用例时间门禁）。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_System
from e2m2e.algorithm.transfer import (
    STATE_FRAME_SYNODIC_BARYCENTRIC_KM,
    AsymptoteParams,
    PcnBplaneTarget,
    PcnSearchParams,
    PcnSolution,
    TliParams,
    _gcrs_to_synodic,
    _moon_state_gcrs,
    _synodic_to_gcrs,
    solve_pcn,
    transfer_orbit,
)
from e2m2e.algorithm.transfer.hohmann import MU_EARTH
from e2m2e.data.templates import ConvergenceState, FailureCause

pytestmark = pytest.mark.orchestration

_TLI = TliParams(parking_alt_km=200.0, inclination_deg=28.5)
_DEP_ASYM = AsymptoteParams(rha_deg=32.0, dha_deg=0.0, c3_km2_s2=1.0)

#: 极小网格：rha 为相对月球方向的偏移（0..20°）；出发渐近线 (32°, 0°, 1.0)
#: 在出发模式选定的 tof 上对应偏移 ≈ 9°，落在网格内，保证往返目标可行。
_SMALL = PcnSearchParams(
    tof_range_days=(1.5, 2.0),
    n_tof=3,
    rha_grid_deg=(0.0, 20.0),
    n_rha=41,
    dha_grid_deg=(0.0, 0.0),
    n_dha=1,
    c3_grid_km2_s2=(1.0, 1.0),
    n_c3=1,
)


def _system() -> CR3BP_System:
    return CR3BP_System(mu=1.21506683e-2, primary="Earth", secondary="Moon")._with_default_scales()


def _moon_fn(system: CR3BP_System):
    return lambda t_sec: _moon_state_gcrs(system, t_sec)


def _solve_departure(params: PcnSearchParams = _SMALL) -> PcnSolution:
    system = _system()
    return solve_pcn(
        _TLI,
        bplane_target=None,
        departure_asymptote=_DEP_ASYM,
        moon_state_fn=_moon_fn(system),
        system=system,
        params=params,
    )


class TestPcnDeparture:
    def test_departure_state_energy_is_c3_over_two(self):
        """出发构造态的地心比能 == C3/2（双曲出发定义级）。"""
        sol = _solve_departure()
        assert sol.status is ConvergenceState.CONVERGED
        assert sol.departure_state_gcrs is not None
        st = sol.departure_state_gcrs
        energy = 0.5 * float(st[3:] @ st[3:]) - MU_EARTH / float(np.linalg.norm(st[:3]))
        assert abs(energy - _DEP_ASYM.c3_km2_s2 / 2.0) < 1e-9
        assert sol.dv_tli_km_s > 0.0

    def test_departure_produces_hyperbolic_lunar_encounter(self):
        """出发模式给出真实月交会（月心 v∞>0、近月点高度有限、月心段为正）。"""
        sol = _solve_departure()
        assert sol.bplane is not None
        assert sol.bplane.v_inf_km_s > 0.0
        assert sol.moon_leg_tof_sec >= 0.0
        assert sol.perilune_state_moon is not None
        assert sol.dv_loi_km_s > 0.0
        # 近月点应在月面附近量级（真实交会而非远距离飞越）
        alt = sol.bplane.perilune_radius_km - 1737.4
        assert -100.0 < alt < 50000.0

    def test_orchestrator_departure_anchors(self):
        """transfer_orbit('PCN', departure) 物理锚点：Δv_TLI 量级与回显。"""
        result = transfer_orbit("PCN", tli_params=_TLI, departure_asymptote=_DEP_ASYM)
        assert result.status is ConvergenceState.CONVERGED
        assert 3.0 < result.details.dv_tli_km_s < 3.6  # 200 km 停泊轨道 TLI 量级
        assert result.details.dv_loi_km_s > 0.0
        assert result.departure_asymptote == _DEP_ASYM
        assert result.bplane is not None and result.bplane.v_inf_km_s > 0.0


class TestPcnArrivalRoundTrip:
    def test_round_trip_converges_to_constructed_target(self):
        """出发模式达成 B-plane → 作为到达目标 → 打靶收敛（残差 ≤ 容差）。"""
        dep = _solve_departure()
        assert dep.bplane is not None
        bp = dep.bplane
        target = PcnBplaneTarget(
            perilune_alt_km=bp.perilune_radius_km - 1737.4,
            bdot_t_km=bp.bdot_t_km,
            bdot_r_km=bp.bdot_r_km,
        )
        system = _system()
        arr = solve_pcn(
            _TLI,
            bplane_target=target,
            departure_asymptote=None,
            moon_state_fn=_moon_fn(system),
            system=system,
            params=_SMALL,
        )
        assert arr.status is ConvergenceState.CONVERGED
        assert arr.bplane is not None
        achieved = np.array(
            [arr.bplane.bdot_r_km, arr.bplane.bdot_t_km, arr.bplane.perilune_radius_km]
        )
        wanted = np.array([target.bdot_r_km, target.bdot_t_km, bp.perilune_radius_km])
        assert float(np.max(np.abs(achieved - wanted))) <= _SMALL.tolerance_km
        assert arr.dv_tli_km_s > 0.0 and arr.dv_loi_km_s > 0.0


class TestPcnTrajectoryContract:
    """ADR 0040 轨迹契约（transfer_orbit 出发模式，默认参数）。"""

    @pytest.fixture(scope="class")
    def result(self):
        return transfer_orbit("PCN", tli_params=_TLI, departure_asymptote=_DEP_ASYM)

    def test_shapes_and_state_frame(self, result):
        n = len(result.trajectory_times)
        assert result.trajectory.shape == (n, 6)
        assert result.trajectory_gcrs_km.shape == (n, 6)
        assert result.state_frame == STATE_FRAME_SYNODIC_BARYCENTRIC_KM

    def test_times_monotonic(self, result):
        assert np.all(np.diff(np.asarray(result.trajectory_times)) > 0.0)

    def test_gcrs_synodic_round_trip(self, result):
        """主段 = _gcrs_to_synodic(trajectory_gcrs_km)（逆约定唯一来源）。"""
        system = _system()
        syn = _gcrs_to_synodic(
            np.asarray(result.trajectory_gcrs_km),
            np.asarray(result.trajectory_times),
            system,
        )
        np.testing.assert_allclose(syn, np.asarray(result.trajectory), rtol=1e-12, atol=1e-8)
        back = _synodic_to_gcrs(syn, np.asarray(result.trajectory_times), system)
        np.testing.assert_allclose(
            back, np.asarray(result.trajectory_gcrs_km), rtol=1e-12, atol=1e-8
        )

    def test_maneuver_events(self, result):
        events = result.maneuver_events
        assert [e.kind for e in events] == ["departure", "arrival"]
        assert events[0].t_sec == 0.0
        assert events[-1].t_sec == pytest.approx(float(result.trajectory_times[-1]))
        assert events[0].dv_km_s > 0.0 and events[-1].dv_km_s > 0.0


class TestPcnFailurePaths:
    def test_xor_violation_raises(self):
        system = _system()
        with pytest.raises(ValueError, match="恰给一个"):
            solve_pcn(
                _TLI,
                bplane_target=None,
                departure_asymptote=None,
                moon_state_fn=_moon_fn(system),
                system=system,
                params=_SMALL,
            )
        with pytest.raises(ValueError, match="恰给一个"):
            solve_pcn(
                _TLI,
                bplane_target=PcnBplaneTarget(200.0, 0.0),
                departure_asymptote=_DEP_ASYM,
                moon_state_fn=_moon_fn(system),
                system=system,
                params=_SMALL,
            )

    def test_all_grid_infeasible_returns_status_triplet(self):
        """网格全不可行 → INFEASIBLE 状态三元组而非异常。"""
        # C3 网格全零 ⇒ 出发构造全部 ValueError ⇒ 无可行网格点
        bad = PcnSearchParams(
            tof_range_days=(1.5, 2.0),
            n_tof=2,
            rha_grid_deg=(30.0, 34.0),
            n_rha=2,
            dha_grid_deg=(0.0, 0.0),
            n_dha=1,
            c3_grid_km2_s2=(0.0, 0.0),
            n_c3=1,
        )
        system = _system()
        sol = solve_pcn(
            _TLI,
            bplane_target=PcnBplaneTarget(perilune_alt_km=200.0, bdot_t_km=0.0),
            departure_asymptote=None,
            moon_state_fn=_moon_fn(system),
            system=system,
            params=bad,
        )
        assert isinstance(sol, PcnSolution)
        assert sol.status is ConvergenceState.INFEASIBLE
        assert sol.cause is FailureCause.NO_INTERSECTION

    def test_top_n_not_supported(self):
        with pytest.raises(NotImplementedError, match="top_n"):
            transfer_orbit("PCN", tli_params=_TLI, departure_asymptote=_DEP_ASYM, top_n=3)
