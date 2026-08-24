"""LGA 月球引力辅助转移测试（ADR 0013 物理定义验证）。

测试策略：按物理定义验证，不用黄金样本。
CR3BP 纯数值测试不需要 SPICE，用 CR3BP_System(mu=MU)._with_default_scales() 初始化。
"""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.algorithm.results import CandidateSearchResult
from e2m2e.algorithm.transfer import (
    LgaSearchParams,
    LgaTransferDetails,
    TransferDesignResult,
    transfer_orbit,
)
from e2m2e.algorithm.transfer.hohmann import TliParams
from e2m2e.algorithm.transfer.lga import (
    _compute_jacobi,
    _refine_lga_candidate,
    search_lga_trajectories,
)
from e2m2e.data.constants import Datum
from e2m2e.data.templates import ConvergenceState, FailureCause
from e2m2e.exceptions import PropagationFailure

pytestmark = pytest.mark.orchestration


# 地月 CR3BP 参数（DE421 基准）
MU = Datum.DE421.mu
DU = 384405.0  # km


def _make_cr3bp_system():
    """创建地月 CR3BP 系统并初始化特征尺度。"""
    system = CR3BP_System(mu=MU, primary="Earth", secondary="Moon")._with_default_scales()
    dynamics = CR3BP_Dynamics(system)
    return system, dynamics


def _make_departure_state(system):
    """构造典型 LEO 出发态的 CR3BP 无量纲态。"""
    from e2m2e.algorithm.transfer.hohmann import MU_EARTH, R_EARTH

    r_park = R_EARTH + 200.0
    v_circ = math.sqrt(MU_EARTH / r_park)
    r0 = np.array([r_park, 0.0, 0.0])
    v0 = np.array([0.0, v_circ, 0.0])
    departure_phys = np.concatenate([r0, v0])
    return system.physical_to_dimensionless(departure_phys)


def _make_target_state(system):
    """构造近月轨道目标态的 CR3BP 无量纲态。"""
    mu = system.mu
    moon_x = 1.0 - mu
    r_target_km = 2000.0
    r_target_du = r_target_km / DU
    x_target = moon_x + r_target_du
    v_circ_dim = math.sqrt(mu / r_target_du)
    return np.array([x_target, 0.0, 0.0, 0.0, v_circ_dim, 0.0])


# 通用搜索参数：角度网格 90 点（4° 间距），确保覆盖 ~83-85° 区间
# 360 → 90：360 是默认 50 的 7 倍，全文件超过 4 分钟超时被杀（#534 审计
# 发现）。倾角覆盖测试（test_inclination_*）依赖网格密度，90 保留面外
# 采样能力；多个候选排序类断言（len >= 2）在 90 下依然满足。
_SEARCH_PARAMS = LgaSearchParams(
    n_departure_phase=90,
    n_tof=5,
    max_total_dv=25.0,
    perilune_alt_min=100.0,
    perilune_alt_max=10000.0,
)


# ---------------------------------------------------------------------------
# TestLgaSearchParams：搜索参数验证
# ---------------------------------------------------------------------------


class TestLgaSearchParams:
    """搜索参数验证。"""

    def test_default_params_valid(self):
        """默认参数范围合理：tof 5-45天，perilune 100-10000km，max_dv 25km/s。"""
        params = LgaSearchParams()
        assert params.tof_range == (5.0, 45.0)
        assert params.perilune_alt_min == 100.0
        assert params.perilune_alt_max == 10000.0
        assert params.max_total_dv == 25.0
        assert params.n_departure_phase > 0
        assert params.n_tof > 0

    def test_frozen_dataclass(self):
        """LgaSearchParams 是 frozen dataclass。"""
        params = LgaSearchParams()
        with pytest.raises(AttributeError):
            params.n_departure_phase = 100  # type: ignore[misc]

    def test_custom_params(self):
        """自定义参数可正确构造。"""
        params = LgaSearchParams(
            departure_phase_range=(0.0, math.pi),
            n_departure_phase=20,
            tof_range=(10.0, 30.0),
            n_tof=20,
            perilune_alt_min=200.0,
            perilune_alt_max=5000.0,
            max_total_dv=30.0,
        )
        assert params.departure_phase_range == (0.0, math.pi)
        assert params.n_departure_phase == 20
        assert params.max_total_dv == 30.0


# ---------------------------------------------------------------------------
# TestLgaSearch：LGA 弹道搜索单元测试
# ---------------------------------------------------------------------------


class TestLgaSearch:
    """LGA 弹道搜索单元测试。"""

    @pytest.fixture
    def cr3bp_setup(self):
        """CR3BP 系统 + 出发/目标态。"""
        system, dynamics = _make_cr3bp_system()
        dep_state = _make_departure_state(system)
        tgt_state = _make_target_state(system)
        return system, dynamics, dep_state, tgt_state

    def test_search_returns_candidates(self, cr3bp_setup):
        """给定典型 LEO→月球参数，搜索应返回非空候选列表。"""
        system, dynamics, dep_state, tgt_state = cr3bp_setup
        candidates = search_lga_trajectories(dep_state, tgt_state, system, dynamics, _SEARCH_PARAMS)
        assert len(candidates) > 0, "LGA 搜索应返回至少一个候选"

    def test_propagation_failure_skips_the_infeasible_grid_point(self, cr3bp_setup):
        system, _, dep_state, tgt_state = cr3bp_setup

        class FailingDynamics:
            def propagate(self, *args, **kwargs):  # noqa: ARG002
                raise PropagationFailure("step size collapsed")

        params = LgaSearchParams(n_departure_phase=1, n_tof=1, n_propagation_samples=2)
        result = search_lga_trajectories(dep_state, tgt_state, system, FailingDynamics(), params)
        assert not result
        assert result.status is ConvergenceState.DIVERGED
        assert result.cause is FailureCause.DIVERGENCE_DETECTED

    def test_candidates_sorted_by_dv(self, cr3bp_setup):
        """候选按 total_dv 升序排列。"""
        system, dynamics, dep_state, tgt_state = cr3bp_setup
        candidates = search_lga_trajectories(dep_state, tgt_state, system, dynamics, _SEARCH_PARAMS)
        if len(candidates) >= 2:
            dvs = [c.total_dv for c in candidates]
            assert dvs == sorted(dvs), "候选应按 total_dv 升序排列"

    def test_periapsis_detected_within_range(self, cr3bp_setup):
        """可行候选的近月点高度在 perilune_alt_range 内。"""
        system, dynamics, dep_state, tgt_state = cr3bp_setup
        candidates = search_lga_trajectories(dep_state, tgt_state, system, dynamics, _SEARCH_PARAMS)
        for c in candidates:
            assert c.perilune_alt_km >= _SEARCH_PARAMS.perilune_alt_min
            assert c.perilune_alt_km <= _SEARCH_PARAMS.perilune_alt_max

    def test_perilune_time_dim_populated(self, cr3bp_setup):
        """候选的 perilune_time_dim 为正且在到达时刻之前（近月点 → 到达段剩余时间 > 0）。"""
        system, dynamics, dep_state, tgt_state = cr3bp_setup
        candidates = search_lga_trajectories(dep_state, tgt_state, system, dynamics, _SEARCH_PARAMS)
        if not candidates:
            pytest.skip("无可行候选，跳过 perilune_time_dim 验证")
        for c in candidates:
            assert c.perilune_time_dim > 0.0, (
                f"perilune_time_dim 应为正，实际 {c.perilune_time_dim}"
            )
            assert c.arrival_time_dim > c.perilune_time_dim, (
                f"到达时刻 {c.arrival_time_dim} 应晚于近月点时刻 {c.perilune_time_dim}"
            )


# ---------------------------------------------------------------------------
# TestLgaRefine：_refine_lga_candidate 精化测试
# ---------------------------------------------------------------------------


class TestLgaRefine:
    """_refine_lga_candidate ThreeBodyLambert 打靶精化测试。"""

    @pytest.fixture
    def cr3bp_setup(self):
        system, dynamics = _make_cr3bp_system()
        dep_state = _make_departure_state(system)
        tgt_state = _make_target_state(system)
        return system, dynamics, dep_state, tgt_state

    def test_refine_candidate_converges(self, cr3bp_setup):
        """最优候选经 _refine_lga_candidate 精化后状态为收敛。"""
        system, dynamics, dep_state, tgt_state = cr3bp_setup
        candidates = search_lga_trajectories(dep_state, tgt_state, system, dynamics, _SEARCH_PARAMS)
        if not candidates:
            pytest.skip("无可行候选，跳过精化测试")

        best = candidates[0]
        refined = _refine_lga_candidate(best, system, dynamics, tgt_state)
        tof_arr = (best.arrival_time_dim - best.perilune_time_dim) * system.characteristic_time
        assert refined.status is ConvergenceState.CONVERGED, (
            f"精化应收敛：best.total_dv={best.total_dv:.4f}, tof_arrival={tof_arr:.2f}s"
        )

    def test_refine_tof_arrival_correct(self, cr3bp_setup):
        """精化使用的 tof_arrival = (arrival - perilune) * char_time，而非总 TOF。"""
        system, dynamics, dep_state, tgt_state = cr3bp_setup
        candidates = search_lga_trajectories(dep_state, tgt_state, system, dynamics, _SEARCH_PARAMS)
        if not candidates:
            pytest.skip("无可行候选，跳过 tof_arrival 验证")

        best = candidates[0]
        char_time = system.characteristic_time
        tof_arrival_expected = (best.arrival_time_dim - best.perilune_time_dim) * char_time
        tof_total = best.arrival_time_dim * char_time

        # 到达段 TOF 必须严格小于总 TOF（否则说明 perilune 时刻未被减去）
        assert tof_arrival_expected < tof_total, (
            f"到达段 TOF ({tof_arrival_expected:.2f}s) 应小于总 TOF ({tof_total:.2f}s)"
        )
        assert tof_arrival_expected > 0.0


# ---------------------------------------------------------------------------
# TestLgaPhysics：LGA 物理不变量验证
# ---------------------------------------------------------------------------


class TestLgaPhysics:
    """LGA 物理不变量验证。"""

    @pytest.fixture
    def cr3bp_setup(self):
        system, dynamics = _make_cr3bp_system()
        dep_state = _make_departure_state(system)
        tgt_state = _make_target_state(system)
        return system, dynamics, dep_state, tgt_state

    def test_jacobi_constant_conservation(self, cr3bp_setup):
        """Jacobi 常数沿 CR3BP 轨道守恒（传播一段后差 < 1e-8）。"""
        system, dynamics, dep_state, _ = cr3bp_setup
        mu = system.mu

        t_span = (0.0, 1.0)
        result = dynamics.propagate(dep_state, t_span, t_eval=np.linspace(0.0, 1.0, 100))
        states = result["states"]

        c_start = _compute_jacobi(states[0], mu)
        c_end = _compute_jacobi(states[-1], mu)
        assert abs(c_start - c_end) < 1e-8, (
            f"Jacobi 常数不守恒：C_start={c_start:.10e}, C_end={c_end:.10e}, "
            f"差={abs(c_start - c_end):.2e}"
        )

    def test_perilune_speed_exceeds_escape(self, cr3bp_setup):
        """近月点速度 > 月球逃逸速度（确认飞越，非捕获）。"""
        system, dynamics, dep_state, tgt_state = cr3bp_setup
        mu = system.mu
        vu_km_s = system.characteristic_velocity

        candidates = search_lga_trajectories(dep_state, tgt_state, system, dynamics, _SEARCH_PARAMS)
        if not candidates:
            pytest.skip("无可行候选，跳过逃逸速度验证")

        best = candidates[0]
        moon_pos = np.array([1.0 - mu, 0.0, 0.0])
        r_peri = np.linalg.norm(best.perilune_state[:3] - moon_pos)
        v_peri_rel = float(np.linalg.norm(best.perilune_state[3:]))
        v_esc = math.sqrt(2.0 * mu / r_peri)
        assert v_peri_rel > v_esc, (
            f"近月点速度 {v_peri_rel * vu_km_s:.3f} km/s < "
            f"逃逸速度 {v_esc * vu_km_s:.3f} km/s（无量纲：{v_peri_rel:.6f} < {v_esc:.6f}）"
        )

    def test_compute_jacobi_at_l1(self):
        """L1 点处静止状态的 Jacobi 常数验证。"""
        system, _ = _make_cr3bp_system()
        mu = system.mu
        gamma = mu ** (1.0 / 3.0)
        x_l1 = 1.0 - mu - gamma
        state_l1 = np.array([x_l1, 0.0, 0.0, 0.0, 0.0, 0.0])
        c_l1 = _compute_jacobi(state_l1, mu)
        assert c_l1 > 0, "L1 处 Jacobi 常数应为正"
        assert 2.0 < c_l1 < 4.0, f"L1 处 Jacobi 常数异常：{c_l1}"

    def test_jacobi_conservation_across_propagation(self, cr3bp_setup):
        """Jacobi 常数在长时间传播后仍守恒（差 < 1e-6）。"""
        system, dynamics, dep_state, _ = cr3bp_setup
        mu = system.mu

        t_end = 10.0 * 86400.0 / system.characteristic_time
        result = dynamics.propagate(dep_state, (0.0, t_end), t_eval=np.linspace(0.0, t_end, 200))
        states = result["states"]

        c_values = np.array([_compute_jacobi(s, mu) for s in states])
        c_max = np.max(c_values)
        c_min = np.min(c_values)
        assert c_max - c_min < 1e-6, (
            f"Jacobi 常数在长时间传播中变化过大：max-min={c_max - c_min:.2e}"
        )

    def test_search_detects_perilune_within_range(self, cr3bp_setup):
        """search_lga_trajectories 2D 搜索检测到的近月点高度在 perilune_alt_range 内。"""
        system, dynamics, dep_state, tgt_state = cr3bp_setup

        candidates = search_lga_trajectories(dep_state, tgt_state, system, dynamics, _SEARCH_PARAMS)
        if not candidates:
            pytest.skip("无可行候选，跳过近月点检测验证")

        for c in candidates:
            assert c.perilune_state.shape == (6,)
            assert np.all(np.isfinite(c.perilune_state))
            assert c.perilune_alt_km >= _SEARCH_PARAMS.perilune_alt_min
            assert c.perilune_alt_km <= _SEARCH_PARAMS.perilune_alt_max


# ---------------------------------------------------------------------------
# TestLgaTransferOrbit：transfer_orbit("LGA") 端到端测试
# ---------------------------------------------------------------------------


class TestLgaTransferOrbit:
    """transfer_orbit("LGA") 端到端测试。"""

    def _make_target_ephemeris(self):
        system, _ = _make_cr3bp_system()
        target_state = _make_target_state(system)
        du = system.characteristic_length
        vu = system.characteristic_velocity
        target_phys = np.array(
            [
                target_state[0] * du,
                target_state[1] * du,
                target_state[2] * du,
                target_state[3] * vu,
                target_state[4] * vu,
                target_state[5] * vu,
            ]
        )
        return target_phys.reshape(1, 6)

    def test_returns_transfer_design_result(self):
        """transfer_orbit("LGA", ...) 返回 TransferDesignResult，transfer_type == "LGA"。"""
        tli_params = TliParams(parking_alt_km=200.0, inclination_deg=0.0)
        target_ephemeris = self._make_target_ephemeris()

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = transfer_orbit(
                "LGA",
                tli_params=tli_params,
                target_ephemeris=target_ephemeris,
                lga_search_params=_SEARCH_PARAMS,
            )

        assert isinstance(result, TransferDesignResult)
        assert result.transfer_type == "LGA"
        assert [stage.name for stage in result.stages] == ["search", "refinement", "shooting"]
        assert result.stages[0].applicable and result.stages[0].executed
        assert result.stages[0].result_status in (
            ConvergenceState.CONVERGED,
            ConvergenceState.INFEASIBLE,
        )
        if result.stages[0].result_status is ConvergenceState.INFEASIBLE:
            assert result.stages[1].executed is False
            assert result.stages[1].result_status is None
            assert result.stages[2].executed is False
            assert result.stages[2].result_status is None

    def test_search_failure_leaves_followup_stages_unexecuted(self):
        """搜索无候选时，精化和打靶不以失败原因占位。"""
        tli_params = TliParams(parking_alt_km=200.0, inclination_deg=0.0)
        with patch(
            "e2m2e.algorithm.transfer.search_lga_trajectories",
            return_value=CandidateSearchResult(
                (),
                ConvergenceState.INFEASIBLE,
                FailureCause.NO_INTERSECTION,
                "搜索未找到可行候选",
            ),
        ):
            result = transfer_orbit(
                "LGA",
                tli_params=tli_params,
                target_ephemeris=self._make_target_ephemeris(),
            )

        search, refinement, shooting = result.stages
        assert search.result_status is ConvergenceState.INFEASIBLE
        assert refinement.applicable and not refinement.executed
        assert refinement.result_status is None
        assert shooting.applicable and not shooting.executed
        assert shooting.result_status is None

    def test_lga_details_populated(self):
        """details 包含 LgaTransferDetails 全部字段。"""
        tli_params = TliParams(parking_alt_km=200.0, inclination_deg=0.0)
        target_ephemeris = self._make_target_ephemeris()

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = transfer_orbit(
                "LGA",
                tli_params=tli_params,
                target_ephemeris=target_ephemeris,
                lga_search_params=_SEARCH_PARAMS,
            )

        details = result.details
        assert isinstance(details, LgaTransferDetails)
        assert isinstance(details.tof_sec, float)
        assert isinstance(details.perilune_alt_km, float)
        assert isinstance(details.perilune_vel_km_s, float)
        assert isinstance(details.dv_departure_km_s, float)
        assert isinstance(details.dv_arrival_km_s, float)
        assert isinstance(details.n_candidates_searched, int)
        assert isinstance(details.n_candidates_feasible, int)
        assert isinstance(details.status, ConvergenceState)
        assert isinstance(details.search_params, LgaSearchParams)

    def test_unsupported_type_still_raises(self):
        """transfer_orbit("low_thrust") without engine_config 抛出 ValueError。"""
        with pytest.raises(ValueError, match="low_thrust"):
            transfer_orbit("low_thrust")

    def test_lga_missing_params_raises(self):
        """transfer_orbit("LGA") 缺少必要参数时报错。"""
        with pytest.raises(ValueError, match="tli_params"):
            transfer_orbit("LGA", target_ephemeris=np.zeros((1, 6)))

        with pytest.raises(ValueError, match="target_ephemeris"):
            transfer_orbit("LGA", tli_params=TliParams(parking_alt_km=200.0, inclination_deg=0.0))


# ---------------------------------------------------------------------------
# TestLgaOutOfPlane：面外搜索维度（issue #512）
# ---------------------------------------------------------------------------


class TestLgaOutOfPlane:
    """非零出发倾角下 LGA 搜索应通过面外角维度找到可行候选（issue #512）。"""

    @pytest.fixture
    def inclined_setup(self):
        """20°/28.5° 倾角出发态（经 construct_departure_state → 无量纲）。"""
        from e2m2e.algorithm.transfer.hohmann import construct_departure_state

        system, dynamics = _make_cr3bp_system()
        tgt_state = _make_target_state(system)
        dep_states = {}
        for incl in (20.0, 28.5):
            r0, v0 = construct_departure_state(
                TliParams(parking_alt_km=200.0, inclination_deg=incl)
            )
            dep_states[incl] = system.physical_to_dimensionless(np.concatenate([r0, v0]))
        return system, dynamics, dep_states, tgt_state

    def test_search_inclined_departure_finds_candidates(self, inclined_setup):
        """倾角 20° 与 28.5° 的搜索都应返回非空候选。"""
        system, dynamics, dep_states, tgt_state = inclined_setup
        for incl, dep_state in dep_states.items():
            candidates = search_lga_trajectories(
                dep_state, tgt_state, system, dynamics, _SEARCH_PARAMS
            )
            assert len(candidates) > 0, f"倾角 {incl}° 应找到可行候选（issue #512）"

    def test_out_of_plane_grid_centered_keeps_coplanar_candidates(self, inclined_setup):
        """共面出发态（倾角 0）在默认面外网格下仍找到候选，且不劣于纯共面网格。"""
        system, dynamics, _, tgt_state = inclined_setup
        dep_state = _make_departure_state(system)
        candidates = search_lga_trajectories(dep_state, tgt_state, system, dynamics, _SEARCH_PARAMS)
        assert len(candidates) > 0, "共面出发态不应因面外网格退化"


# ---------------------------------------------------------------------------
# TestLgaMaxTotalDvUnits：max_total_dv 的 km/s 语义（issue #512）
# ---------------------------------------------------------------------------


class TestLgaMaxTotalDvUnits:
    """max_total_dv 以 km/s 语义参与筛选。"""

    @pytest.fixture
    def cr3bp_setup(self):
        system, dynamics = _make_cr3bp_system()
        dep_state = _make_departure_state(system)
        tgt_state = _make_target_state(system)
        return system, dynamics, dep_state, tgt_state

    def test_candidates_physical_dv_within_limit(self, cr3bp_setup):
        """所有候选的物理总 Δv（total_dv × 特征速度）≤ max_total_dv (km/s)。"""
        system, dynamics, dep_state, tgt_state = cr3bp_setup
        vu = system.characteristic_velocity
        candidates = search_lga_trajectories(dep_state, tgt_state, system, dynamics, _SEARCH_PARAMS)
        for c in candidates:
            assert c.total_dv * vu <= _SEARCH_PARAMS.max_total_dv + 1e-9, (
                f"物理总 Δv {c.total_dv * vu:.3f} km/s 超过 max_total_dv="
                f"{_SEARCH_PARAMS.max_total_dv} km/s"
            )

    def test_tighter_dv_limit_reduces_candidates(self, cr3bp_setup):
        """把 max_total_dv 压到最优候选物理 Δv 之下，搜索应无候选。"""
        system, dynamics, dep_state, tgt_state = cr3bp_setup
        vu = system.characteristic_velocity
        baseline = search_lga_trajectories(dep_state, tgt_state, system, dynamics, _SEARCH_PARAMS)
        assert len(baseline) > 0
        best_physical_km_s = baseline[0].total_dv * vu
        tight = LgaSearchParams(
            n_departure_phase=_SEARCH_PARAMS.n_departure_phase,
            n_tof=_SEARCH_PARAMS.n_tof,
            max_total_dv=best_physical_km_s - 0.1,
            perilune_alt_min=_SEARCH_PARAMS.perilune_alt_min,
            perilune_alt_max=_SEARCH_PARAMS.perilune_alt_max,
        )
        reduced = search_lga_trajectories(dep_state, tgt_state, system, dynamics, tight)
        assert len(reduced) == 0, (
            f"max_total_dv={best_physical_km_s - 0.1:.3f} km/s 应滤掉全部候选"
            f"（最优物理 Δv={best_physical_km_s:.3f} km/s），实际剩 {len(reduced)} 个"
        )


# ---------------------------------------------------------------------------
# TestLgaInclinedEndToEnd：transfer_orbit("LGA") 非零倾角端到端（issue #512）
# ---------------------------------------------------------------------------


class TestLgaInclinedEndToEnd:
    """issue #512 复现场景：发射倾角 28.5°（文昌纬度）端到端收敛。"""

    def test_transfer_orbit_inclination_28_5_converges(self):
        """incl=28.5° 时 transfer_orbit("LGA") 应 CONVERGED 且候选 > 0。"""
        import warnings

        system, _ = _make_cr3bp_system()
        target_state = _make_target_state(system)
        du, vu = system.characteristic_length, system.characteristic_velocity
        target_ephemeris = np.array(
            [
                target_state[0] * du,
                0.0,
                0.0,
                0.0,
                target_state[4] * vu,
                0.0,
            ]
        ).reshape(1, 6)
        params = LgaSearchParams(
            n_departure_phase=120,
            n_tof=5,
            max_total_dv=25.0,
            perilune_alt_min=100.0,
            perilune_alt_max=10000.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = transfer_orbit(
                "LGA",
                tli_params=TliParams(parking_alt_km=200.0, inclination_deg=28.5),
                target_ephemeris=target_ephemeris,
                lga_search_params=params,
            )
        assert result.status is ConvergenceState.CONVERGED, (
            f"28.5° 倾角应收敛（issue #512），实际 {result.status}: {result.message}"
        )
        assert result.details.n_candidates_feasible > 0
        assert result.delta_v <= params.max_total_dv + 1e-6

    def test_low_inclination_dv_not_degraded(self):
        """倾角 0°/5°/10°/20° 端到端收敛且 Δv ≤ max_total_dv（issue #512 验收）。

        修复前低倾角基线（精化后）总 Δv 为 50~70 km/s；修复后应低于
        max_total_dv=25 km/s，满足不劣于基线 5%（基线×1.05 ≈ 52~74）。
        """
        import warnings

        system, _ = _make_cr3bp_system()
        target_state = _make_target_state(system)
        du, vu = system.characteristic_length, system.characteristic_velocity
        target_ephemeris = np.array(
            [
                target_state[0] * du,
                0.0,
                0.0,
                0.0,
                target_state[4] * vu,
                0.0,
            ]
        ).reshape(1, 6)
        params = LgaSearchParams(
            n_departure_phase=120,
            n_tof=5,
            max_total_dv=25.0,
            perilune_alt_min=100.0,
            perilune_alt_max=10000.0,
        )
        for incl in (0.0, 5.0, 10.0, 20.0):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = transfer_orbit(
                    "LGA",
                    tli_params=TliParams(parking_alt_km=200.0, inclination_deg=incl),
                    target_ephemeris=target_ephemeris,
                    lga_search_params=params,
                )
            assert result.status is ConvergenceState.CONVERGED, (
                f"倾角 {incl}° 应收敛，实际 {result.status}: {result.message}"
            )
            assert result.delta_v <= params.max_total_dv + 1e-6, (
                f"倾角 {incl}° 总 Δv {result.delta_v:.3f} km/s 超过 max_total_dv=25"
            )

    def test_search_high_inclination_iss_finds_candidates(self):
        """大倾角 51.6°（ISS 倾角）搜索层应找到可行候选（经验面外带覆盖 0°~90°）。"""
        from e2m2e.algorithm.transfer.hohmann import construct_departure_state

        system, dynamics = _make_cr3bp_system()
        tgt_state = _make_target_state(system)
        r0, v0 = construct_departure_state(TliParams(parking_alt_km=200.0, inclination_deg=51.6))
        dep_state = system.physical_to_dimensionless(np.concatenate([r0, v0]))
        candidates = search_lga_trajectories(dep_state, tgt_state, system, dynamics, _SEARCH_PARAMS)
        assert len(candidates) > 0, "倾角 51.6° 应找到可行候选（面外带覆盖 0°~90°）"
