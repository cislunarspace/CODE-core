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


def _solve_departure(
    asym: AsymptoteParams = _DEP_ASYM, params: PcnSearchParams = _SMALL
) -> PcnSolution:
    system = _system()
    return solve_pcn(
        _TLI,
        bplane_target=None,
        departure_asymptote=asym,
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
        # 近月点应在月面以上且量级合理（CONVERGED 蕴含通过可行域门禁，见 #698）
        alt = sol.bplane.perilune_radius_km - 1737.4
        assert 0.0 < alt < 50000.0

    def test_orchestrator_departure_anchors(self):
        """transfer_orbit('PCN', departure) 物理锚点：Δv_TLI 量级与回显。"""
        result = transfer_orbit("PCN", tli_params=_TLI, departure_asymptote=_DEP_ASYM)
        assert result.status is ConvergenceState.CONVERGED
        assert 3.0 < result.details.dv_tli_km_s < 3.6  # 200 km 停泊轨道 TLI 量级
        assert result.details.dv_loi_km_s > 0.0
        assert result.departure_asymptote == _DEP_ASYM
        assert result.bplane is not None and result.bplane.v_inf_km_s > 0.0


class TestPcnDepartureFeasibility:
    """出发模式月交会可行域门禁（#698）：撞月/远距离飞越不得冒报 CONVERGED。

    门禁口径（Patched-conic）：近月点半径须在月面以上，且网格最近月心距离须落在
    月球影响球（Laplace–Tisserand 代理，spatiography 黄金值 66010 km）以内；
    不可行解仍回显 B-plane / 渐近线，供调用方诊断。
    """

    def test_far_flyby_is_not_a_rendezvous(self):
        """最近月心距离超影响球 → INFEASIBLE/CONSTRAINT_VIOLATION，几何仍回显。"""
        asym = AsymptoteParams(rha_deg=20.0, dha_deg=0.0, c3_km2_s2=1.0)
        sol = _solve_departure(asym)
        assert sol.status is ConvergenceState.INFEASIBLE
        assert sol.cause is FailureCause.CONSTRAINT_VIOLATION
        assert sol.encounter_state_moon is not None
        closest = float(np.linalg.norm(sol.encounter_state_moon[:3]))
        assert closest > 66010.0, "最近月心距离须确认落在影响球外（黄金值 66010 km）"
        assert sol.bplane is not None, "失败解仍须回显达成的 B-plane"
        assert sol.departure_asymptote == asym

    def test_collision_asymptote_reports_body_collision(self):
        """近月点落在月面以下（撞月）→ COLLISION/BODY_COLLISION。"""
        sol = _solve_departure(AsymptoteParams(rha_deg=28.6, dha_deg=0.0, c3_km2_s2=2.0))
        assert sol.status is ConvergenceState.COLLISION
        assert sol.cause is FailureCause.BODY_COLLISION
        assert sol.bplane is not None
        assert sol.bplane.perilune_radius_km <= 1737.4, "该渐近线的近月点须在月面以下"

    def test_feasible_departure_stays_converged(self):
        """可行域内的出发模式仍 CONVERGED，近月点在月面以上。"""
        sol = _solve_departure()
        assert sol.status is ConvergenceState.CONVERGED
        assert sol.bplane is not None
        assert sol.bplane.perilune_radius_km > 1737.4

    def test_orchestrator_echoes_geometry_on_failure(self):
        """编排器失败路径同样回显渐近线与 B-plane（#698）。"""
        asym = AsymptoteParams(rha_deg=20.0, dha_deg=0.0, c3_km2_s2=1.0)
        with pytest.warns(UserWarning, match="未收敛"):
            result = transfer_orbit(
                "PCN", tli_params=_TLI, departure_asymptote=asym, tof_range=(1.5, 2.0)
            )
        assert result.status is ConvergenceState.INFEASIBLE
        assert result.departure_asymptote == asym
        assert result.bplane is not None
        assert result.trajectory is None


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


class TestPcnCanonicalization:
    """到达模式回显角归一（评审修复）：成功解不因 0/360 接缝被误判非法。"""

    def test_canonical_ranges_and_direction(self):
        from e2m2e.algorithm.transfer.bplane import _unit_from_rha_dha
        from e2m2e.algorithm.transfer.pcn import _canonical_rha_dha

        cases = ((370.0, 10.0), (-10.0, 10.0), (40.0, 100.0), (40.0, -100.0), (500.0, 120.0))
        for rha, dha in cases:
            r2, d2 = _canonical_rha_dha(rha, dha)
            assert 0.0 <= r2 < 360.0
            assert -90.0 <= d2 <= 90.0
            np.testing.assert_allclose(
                _unit_from_rha_dha(r2, d2), _unit_from_rha_dha(rha, dha), atol=1e-12
            )

    def test_arrival_echo_within_response_domain(self):
        """到达模式回显的出发渐近线落在响应模型域内（RHA∈[0,360), DHA∈[-90,90]）。"""
        from e2m2e.api.models import DepartureAsymptote

        dep = _solve_departure()
        bp = dep.bplane
        assert bp is not None
        system = _system()
        arr = solve_pcn(
            _TLI,
            bplane_target=PcnBplaneTarget(
                perilune_alt_km=bp.perilune_radius_km - 1737.4,
                bdot_t_km=bp.bdot_t_km,
                bdot_r_km=bp.bdot_r_km,
            ),
            departure_asymptote=None,
            moon_state_fn=_moon_fn(system),
            system=system,
            params=_SMALL,
        )
        assert arr.departure_asymptote is not None
        # 响应模型约束同源校验：越界即 ValidationError
        echo = DepartureAsymptote(
            rha_deg=arr.departure_asymptote.rha_deg,
            dha_deg=arr.departure_asymptote.dha_deg,
            c3_km2_s2=arr.departure_asymptote.c3_km2_s2,
        )
        assert 0.0 <= echo.rha_deg < 360.0


class TestPcnTofRange:
    def test_tof_range_overrides_search_window(self):
        """请求侧 tof_range 覆盖 PCN 搜索窗口（不再静默忽略默认网格）。"""
        result = transfer_orbit(
            "PCN", tli_params=_TLI, departure_asymptote=_DEP_ASYM, tof_range=(1.7, 1.7)
        )
        assert result.status is ConvergenceState.CONVERGED
        assert result.details.earth_leg_tof_sec / 86400.0 == pytest.approx(1.7, abs=1e-6)


class TestPcnStages:
    def test_departure_grid_stage_applicable(self):
        from e2m2e.algorithm.transfer import _pcn_stages

        sol = _solve_departure()
        stages = {s.name: s for s in _pcn_stages(sol, "departure")}
        assert stages["grid_search"].applicable and stages["grid_search"].executed
        assert stages["grid_search"].result_status is ConvergenceState.CONVERGED
        assert not stages["newton"].applicable

    def test_infeasible_arrival_grid_stage_reflects_status(self):
        from e2m2e.algorithm.transfer import _pcn_stages

        bad = PcnSearchParams(
            tof_range_days=(1.5, 2.0),
            n_tof=2,
            rha_grid_deg=(0.0, 20.0),
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
        assert sol.status is ConvergenceState.INFEASIBLE
        stages = {s.name: s for s in _pcn_stages(sol, "arrival")}
        assert stages["grid_search"].result_status is ConvergenceState.INFEASIBLE
        assert not stages["newton"].executed
