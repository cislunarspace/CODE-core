"""LGA 月球引力辅助转移测试（ADR 0013 物理定义验证）。

测试策略：只保留 API 契约与数学推导类快测试——搜索参数校验、
Jacobi 常数解析点验证与守恒律（单次传播）、搜索失败编排契约
（mock 注入，1×1 网格）、转移弧段拼接契约。驱动真实网格搜索
与端到端收敛的测试已按维护决策移除（全量预算超 ADR 0037 上限）。

CR3BP 纯数值测试不需要 SPICE，用 CR3BP_System(mu=MU)._with_default_scales() 初始化。
"""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.algorithm.results import CandidateSearchResult
from e2m2e.algorithm.transfer import LgaSearchParams, transfer_orbit
from e2m2e.algorithm.transfer.hohmann import TliParams
from e2m2e.algorithm.transfer.lga import _compute_jacobi, search_lga_trajectories
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
# TestLgaSearch：搜索失败契约（mock 注入，不驱动真实网格搜索）
# ---------------------------------------------------------------------------


class TestLgaSearch:
    """search_lga_trajectories 失败编排契约。"""

    @pytest.fixture
    def cr3bp_setup(self):
        """CR3BP 系统 + 出发/目标态。"""
        system, dynamics = _make_cr3bp_system()
        dep_state = _make_departure_state(system)
        tgt_state = _make_target_state(system)
        return system, dynamics, dep_state, tgt_state

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


# ---------------------------------------------------------------------------
# TestLgaPhysics：Jacobi 常数数学性质（解析点 + 守恒律，单次传播）
# ---------------------------------------------------------------------------


class TestLgaPhysics:
    """Jacobi 常数数学性质验证。"""

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


# ---------------------------------------------------------------------------
# TestLgaTransferOrbit：transfer_orbit("LGA") 接口契约（不驱动真实搜索）
# ---------------------------------------------------------------------------


class TestLgaTransferOrbit:
    """transfer_orbit("LGA") 接口契约测试。"""

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
# TestLgaOutOfPlane / TestLgaMaxTotalDvUnits / TestLgaInclinedEndToEnd：
# 驱动真实网格搜索的面外/单位语义/端到端测试已按维护决策移除。
# ---------------------------------------------------------------------------


class TestTransferLegHelpers:
    """_propagate_synodic_leg + _join_transfer_legs（ADR 0040 拼接契约）。"""

    def test_leg_units_and_join_continuity(self):
        """弧段输出为会合系物理 km + 秒；拼接去重衔接点、时刻偏移、位置连续。"""
        from e2m2e.algorithm.transfer import _join_transfer_legs, _propagate_synodic_leg

        system, dynamics = _make_cr3bp_system()
        du = system.characteristic_length
        vu = system.characteristic_velocity
        tu = system.characteristic_time
        x0 = _make_target_state(system)

        t_dep = 0.25
        dep_states, dep_times = _propagate_synodic_leg(dynamics, system, x0, t_dep, n_samples=50)
        # 出发段终点的真值状态（无量纲）作为到达段初值，模拟拼接语义
        full = dynamics.propagate(x0, (0.0, t_dep), t_eval=np.array([t_dep]))
        x_mid = np.asarray(full["states"][-1], dtype=float)
        arr_states, arr_times = _propagate_synodic_leg(dynamics, system, x_mid, 0.5, n_samples=100)

        # 单位换算：位置 ×DU、时刻 ×TU，从 0 起算
        assert dep_states.shape == (50, 6)
        assert np.allclose(dep_states[0, :3], x0[:3] * du, rtol=1e-9)
        assert np.allclose(dep_states[0, 3:], x0[3:] * vu, rtol=1e-9)
        assert dep_times[0] == 0.0
        assert dep_times[-1] == pytest.approx(t_dep * tu, rel=1e-9)

        joined, joined_times = _join_transfer_legs(dep_states, dep_times, arr_states, arr_times)
        assert joined.shape == (50 + 100 - 1, 6)
        assert joined_times.shape == (joined.shape[0],)
        # 到达段整体偏移到出发段时间轴，衔接点无重复时刻
        assert joined_times[49] == pytest.approx(dep_times[-1])
        assert joined_times[50] > joined_times[49]
        assert joined_times[50] == pytest.approx(dep_times[-1] + arr_times[1])
        # 拼接处位置连续：出发段末行与被去重的到达段首行是同一时刻的
        # 同一状态（出发段 t_eval 采样 vs dense 输出重传播），差在容差内
        seam_km = float(np.linalg.norm(dep_states[-1, :3] - arr_states[0, :3]))
        assert seam_km < 10.0, f"拼接位置跳变 {seam_km:.3f} km 超过 10 km"
