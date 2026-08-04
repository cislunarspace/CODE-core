"""LGA 月球引力辅助转移测试（ADR 0013 物理定义验证）。

测试策略：按物理定义验证，不用黄金样本。
CR3BP 纯数值测试不需要 SPICE，用 CR3BP_System(mu=MU)._with_default_scales() 初始化。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.algorithm.transfer import (
    LgaSearchParams,
    LgaTransferDetails,
    TransferDesignResult,
    transfer_orbit,
)
from e2m2e.algorithm.transfer.hohmann import TliParams
from e2m2e.algorithm.transfer.lga import (
    _compute_jacobi,
    search_lga_trajectories,
)

# 地月 CR3BP 参数
MU = 1.21506683e-2
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
_SEARCH_PARAMS = LgaSearchParams(
    n_departure_phase=360,
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
        assert isinstance(details.converged, bool)
        assert isinstance(details.search_params, LgaSearchParams)

    def test_unsupported_type_still_raises(self):
        """transfer_orbit("WSB") 仍抛出 NotImplementedError。"""
        with pytest.raises(NotImplementedError, match="WSB"):
            transfer_orbit("WSB")

    def test_lga_missing_params_raises(self):
        """transfer_orbit("LGA") 缺少必要参数时报错。"""
        with pytest.raises(ValueError, match="tli_params"):
            transfer_orbit("LGA", target_ephemeris=np.zeros((1, 6)))

        with pytest.raises(ValueError, match="target_ephemeris"):
            transfer_orbit("LGA", tli_params=TliParams(parking_alt_km=200.0, inclination_deg=0.0))
