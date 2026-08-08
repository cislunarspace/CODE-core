"""DPO（Direct Prograde Orbit）轨道族端到端 + 物理不变量测试。

覆盖 design_dpo 收敛性、Jacobi 守恒、xy 平面约束、周期闭合、注册表，
以及区分 DPO 与 DRO 的特征测试（DPO 顺行、Jacobi 更高、更靠近月球）。
"""

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family import registry
from e2m2e.algorithm.family.cr3bp_orbits import (
    _moon_distance_minmax,
    design_dpo,
    design_dro,
    earth_moon_system,
)
from e2m2e.data.types.orbit import Orbit

pytestmark = [pytest.mark.l4]

CHAR_LENGTH_KM = 384400.0


@pytest.fixture(scope="module")
def orbit():
    """共享一条 design_dpo(20000.0) 轨道。"""
    return design_dpo(20000.0)


@pytest.fixture(scope="module")
def dynamics(orbit):
    return CR3BP_Dynamics(orbit.system)


@pytest.fixture(scope="module")
def dro_ref():
    """共享一条 design_dro(20000.0) 轨道（DPO vs DRO 对比用）。"""
    return design_dro(20000.0)


# =============================================================================
# Registry
# =============================================================================


class TestRegistry:
    """注册表应包含 DPO 条目。"""

    def test_dpo_in_registry(self):
        """registry 应包含 'DPO' 键。"""
        assert "DPO" in registry

    def test_dpo_callable(self):
        """registry['DPO'] 应可调用。"""
        assert callable(registry["DPO"])

    def test_dpo_same_as_design_dpo(self):
        """registry['DPO'] 应指向 design_dpo。"""
        assert registry["DPO"] is design_dpo


# =============================================================================
# End-to-end convergence
# =============================================================================


class TestDesignDpoConvergence:
    """design_dpo 端到端收敛测试。"""

    def test_design_dpo_20000km_converges(self):
        """design_dpo(20000.0) 应返回收敛的 Orbit。"""
        orbit = design_dpo(20000.0)
        assert orbit is not None
        assert orbit.period is not None
        assert orbit.period > 0

    def test_design_dpo_amplitude_matches_target(self):
        """振幅（距月 min/max 均值）应接近 20000 km（容差 25 km，与 design tol 对齐）。"""
        orbit = design_dpo(20000.0)
        dynamics = CR3BP_Dynamics(orbit.system)
        d_min, d_max = _moon_distance_minmax(dynamics, orbit)
        amp_km = 0.5 * (d_min + d_max) * CHAR_LENGTH_KM
        assert abs(amp_km - 20000.0) < 25.0

    def test_design_dpo_state_on_x_axis(self):
        """初始状态应在 xy 平面的 x 轴上：y=0, z=0, vx=0。"""
        orbit = design_dpo(20000.0)
        s0 = orbit.states[0]
        assert s0[1] == pytest.approx(0.0, abs=1e-10)  # y=0
        assert s0[2] == pytest.approx(0.0, abs=1e-10)  # z=0
        assert s0[3] == pytest.approx(0.0, abs=1e-10)  # vx=0

    def test_design_dpo_vy0_negative(self):
        """DPO 初始 vy0 应为负（顺行：旋转坐标系下逆时针）。"""
        orbit = design_dpo(20000.0)
        assert orbit.states[0, 4] < 0.0

    def test_correct_dpo_raises_on_pseudo_solution(self):
        """_correct_dpo 应在周期跳变超过 2× 时抛 Cr3bpOrbitError。"""
        from e2m2e.algorithm.family.cr3bp_orbits import Cr3bpOrbitError, _correct_dpo

        dynamics = CR3BP_Dynamics(earth_moon_system())
        # 构造一个 period 偏小的 orbit 作为 guess，使修正结果超过 2×
        fake_state = np.array([0.90, 0.0, 0.0, 0.0, -0.25, 0.0])
        fake = Orbit(
            states=fake_state.reshape(1, -1),
            times=np.array([0.0]),
            system=dynamics.system,
        )
        fake.period = 1.0  # 种子 period≈2.50，2.50 > 2×1.0 触发伪解拒绝
        with pytest.raises(Cr3bpOrbitError, match="DPO"):
            _correct_dpo(dynamics, 0.90, fake)


# =============================================================================
# Physical invariants
# =============================================================================


class TestPhysicalInvariants:
    """收敛轨道的物理不变量检验。"""

    def test_jacobi_conservation(self, orbit, dynamics):
        """Jacobi 常数在一个周期内漂移 < 1e-10。"""
        T = orbit.period
        result = dynamics.propagate(
            orbit.states[0],
            (0, T),
            t_eval=np.linspace(0, T, 1000),
            with_jacobi=True,
        )
        jacobi = result["jacobi"]
        drift = abs(jacobi[-1] - jacobi[0])
        assert drift < 1e-10

    def test_periodic_closure(self, orbit, dynamics):
        """全周期闭合误差 < 1e-6。"""
        T = orbit.period
        result = dynamics.propagate(orbit.states[0], (0, T), t_eval=np.linspace(0, T, 1000))
        closure = np.linalg.norm(result["states"][-1] - orbit.states[0])
        assert closure < 1e-6

    def test_xy_plane_constraint(self, orbit, dynamics):
        """DPO 为 xy 平面轨道：整个周期内 z≈0。"""
        T = orbit.period
        result = dynamics.propagate(orbit.states[0], (0, T), t_eval=np.linspace(0, T, 500))
        z_max = np.max(np.abs(result["states"][:, 2]))
        assert z_max < 1e-8

    def test_half_period_on_x_axis(self, orbit, dynamics):
        """半周期处状态应在 x 轴上：y=0, z=0, vx=0。"""
        T = orbit.period
        result = dynamics.propagate(orbit.states[0], (0, T / 2), t_eval=[T / 2])
        state_half = result["states"][-1]
        assert state_half[1] == pytest.approx(0.0, abs=1e-8)  # y=0
        assert state_half[2] == pytest.approx(0.0, abs=1e-8)  # z=0
        assert state_half[3] == pytest.approx(0.0, abs=1e-8)  # vx=0


# =============================================================================
# DPO vs DRO distinguishing tests
# =============================================================================


class TestDpoNotDro:
    """区分 DPO 与 DRO 的特征测试。

    DPO 顺行（vy0 < 0）、Jacobi 更高（更靠近月球）、周期更短；
    DRO 逆行（vy0 > 0）、Jacobi 更低。
    """

    def test_dpo_jacobi_higher_than_dro(self, dro_ref, orbit):
        """同振幅下 DPO 的 Jacobi 常数应高于 DRO。"""
        C_dro = float(dro_ref.system.get_jacobi_constant(dro_ref.states[0]))
        C_dpo = float(orbit.system.get_jacobi_constant(orbit.states[0]))
        assert C_dpo > C_dro, f"DPO C={C_dpo:.4f} 应高于 DRO C={C_dro:.4f}"

    def test_dpo_closer_to_moon_than_dro(self, dro_ref, orbit):
        """同振幅下 DPO 的近月距应小于 DRO（DPO 更靠近月球）。"""
        d_dro = CR3BP_Dynamics(dro_ref.system)
        d_dpo = CR3BP_Dynamics(orbit.system)
        min_dro, _ = _moon_distance_minmax(d_dro, dro_ref)
        min_dpo, _ = _moon_distance_minmax(d_dpo, orbit)
        assert min_dpo < min_dro, f"DPO 近月距 {min_dpo:.4f} DU 应小于 DRO {min_dro:.4f} DU"

    def test_dpo_covers_moon_proximity(self, orbit):
        """DPO 轨道的近月距应显著小于远月距（绕月特征）。"""
        dynamics = CR3BP_Dynamics(orbit.system)
        d_min, d_max = _moon_distance_minmax(dynamics, orbit)
        assert d_min < d_max, "DPO 应有明显的近月/远月区分"
        assert d_min < 0.2, f"DPO 近月距 {d_min:.4f} DU 应 < 0.2 DU（绕月轨道）"


# =============================================================================
# design_orbit facade dispatch
# =============================================================================


class TestDpoDesignOrbit:
    """design_orbit 入口分发 DPO。"""

    def test_design_orbit_validates_dpo_params(self):
        """design_orbit("DPO", ...) 参数校验先于实现：duration 校验优先。"""
        from e2m2e.algorithm.design import design_orbit

        with pytest.raises(ValueError, match="duration"):
            design_orbit("DPO", duration=0.0)

    def test_design_orbit_rejects_amplitude_out_of_range(self):
        """amplitude < 1737 km 应抛 ValueError。"""
        from e2m2e.algorithm.design import design_orbit

        with pytest.raises(ValueError, match="amplitude"):
            design_orbit("DPO", amplitude=500.0, duration=0.5)

    def test_design_orbit_rejects_bad_phase(self):
        """phase 超界应抛 ValueError。"""
        from e2m2e.algorithm.design import design_orbit

        with pytest.raises(ValueError, match="phase"):
            design_orbit("DPO", phase=1.5, duration=0.5)

    def test_validate_params_dpo_defaults(self):
        """_validate_params 对 DPO 填默认值（amplitude=20000, phase=0.5001）。"""
        from e2m2e.algorithm.design.design_orbit import _validate_params

        params = _validate_params(
            "DPO",
            amplitude=None,
            phase=None,
            collinear_point=None,
            north_south=None,
            perilune_height=None,
            amplitude_in=None,
            amplitude_out=None,
            phase_in=None,
            phase_out=None,
        )
        assert params == {"amplitude": 20000.0, "phase": 0.5001}

    def test_cr3bp_orbit_for_dpo_end_to_end(self):
        """_cr3bp_orbit_for("DPO", ...) 端到端调用 design_dpo 返回 Orbit。"""
        from e2m2e.algorithm.design.design_orbit import _cr3bp_orbit_for, _validate_params

        params = _validate_params(
            "DPO",
            amplitude=20000.0,
            phase=None,
            collinear_point=None,
            north_south=None,
            perilune_height=None,
            amplitude_in=None,
            amplitude_out=None,
            phase_in=None,
            phase_out=None,
        )
        dynamics = CR3BP_Dynamics(earth_moon_system())
        orbit = _cr3bp_orbit_for("DPO", params, dynamics)
        assert orbit is not None
        assert orbit.period is not None
        assert orbit.period > 0
