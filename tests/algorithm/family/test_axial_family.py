"""Axial 轨道族端到端 + 物理不变量测试。

覆盖 design_axial 收敛性、Jacobi 守恒、x 轴对称、周期闭合、注册表，
以及区分 Axial 与 Vertical Lyapunov 的特征测试。
"""

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family import registry
from e2m2e.algorithm.family.cr3bp_orbits import (
    _z_amplitude_max,
    design_axial,
    earth_moon_system,
)

pytestmark = pytest.mark.orchestration

CHAR_LENGTH_KM = 384400.0


@pytest.fixture(scope="module")
def orbit():
    """共享一条 design_axial(1, 5000.0) 轨道。"""
    return design_axial(1, 5000.0)


@pytest.fixture(scope="module")
def dynamics(orbit):
    return CR3BP_Dynamics(orbit.system)


# =============================================================================
# Registry
# =============================================================================


class TestRegistry:
    """注册表应包含 AXIAL 条目。"""

    def test_axial_in_registry(self):
        """registry 应包含 'AXIAL' 键。"""
        assert "AXIAL" in registry

    def test_axial_callable(self):
        """registry['AXIAL'] 应可调用。"""
        assert callable(registry["AXIAL"])

    def test_axial_same_as_design_axial(self):
        """registry['AXIAL'] 应指向 design_axial。"""
        assert registry["AXIAL"] is design_axial


# =============================================================================
# End-to-end convergence
# =============================================================================


class TestDesignAxialConvergence:
    """design_axial 端到端收敛测试。"""

    def test_design_axial_l1_5000km_converges(self):
        """design_axial(1, 5000.0) 应返回收敛的 Orbit。"""
        orbit = design_axial(1, 5000.0)
        assert orbit is not None
        assert orbit.period is not None
        assert orbit.period > 0

    def test_design_axial_amplitude_matches_target(self):
        """max|z| 应接近 5000 km（容差 500 km）。"""
        orbit = design_axial(1, 5000.0)
        dynamics = CR3BP_Dynamics(orbit.system)
        z_max = _z_amplitude_max(dynamics, orbit)
        z_max_km = z_max * CHAR_LENGTH_KM
        assert abs(z_max_km - 5000.0) < 500.0

    def test_design_axial_state_on_x_axis(self):
        """初始状态应在 x 轴上：z0=0, y0=0, vx0=0。"""
        orbit = design_axial(1, 5000.0)
        s0 = orbit.states[0]
        assert s0[1] == pytest.approx(0.0, abs=1e-10)  # y=0
        assert s0[2] == pytest.approx(0.0, abs=1e-10)  # z=0
        assert s0[3] == pytest.approx(0.0, abs=1e-10)  # vx=0

    def test_design_axial_vz0_nonzero(self):
        """初始 z 方向速度应非零（Type B 特征）。"""
        orbit = design_axial(1, 5000.0)
        s0 = orbit.states[0]
        assert abs(s0[5]) > 1e-6


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

    def test_x_axis_symmetry(self, orbit, dynamics):
        """x 轴对称：state(t) = Sigma(state(T-t))，
        Sigma = (x,-y,-z,-vx,vy,vz)（旋转 π + 时间反演）。"""
        T = orbit.period
        result = dynamics.propagate(orbit.states[0], (0, T), t_eval=np.linspace(0, T, 500))
        states = result["states"]
        n = len(states)
        max_err = 0.0
        for i in range(n):
            j = n - 1 - i
            reflected = np.array(
                [
                    states[j, 0],
                    -states[j, 1],
                    -states[j, 2],
                    -states[j, 3],
                    states[j, 4],
                    states[j, 5],
                ]
            )
            err = np.linalg.norm(states[i] - reflected)
            max_err = max(max_err, err)
        assert max_err < 1e-6

    def test_half_period_on_x_axis(self, orbit, dynamics):
        """半周期处状态应在 x 轴上：y=0, z=0, vx=0。"""
        T = orbit.period
        result = dynamics.propagate(orbit.states[0], (0, T / 2), t_eval=[T / 2])
        state_half = result["states"][-1]
        assert state_half[1] == pytest.approx(0.0, abs=1e-8)  # y=0
        assert state_half[2] == pytest.approx(0.0, abs=1e-8)  # z=0
        assert state_half[3] == pytest.approx(0.0, abs=1e-8)  # vx=0


# =============================================================================
# design_orbit facade dispatch
# =============================================================================


class TestAxialDesignOrbit:
    """DesignOrbitRequest 校验 AXIAL 参数。"""

    def test_design_orbit_validates_axial_params(self):
        """duration=0 应在模型层被拒。"""
        from e2m2e.api.models import DesignOrbitRequest

        with pytest.raises(ValueError):
            DesignOrbitRequest(orbit_type="AXIAL", duration=0.0)

    def test_design_orbit_rejects_bad_collinear_point(self):
        """collinear_point=4 应抛 ValueError。"""
        from e2m2e.api.models import DesignOrbitRequest

        with pytest.raises(ValueError):
            DesignOrbitRequest(orbit_type="AXIAL", collinear_point=4)

    def test_design_orbit_rejects_amplitude_out_of_range(self):
        """|amplitude| > 60000 km 应抛 ValueError。"""
        from e2m2e.api.models import DesignOrbitRequest

        with pytest.raises(ValueError, match="amplitude"):
            DesignOrbitRequest(orbit_type="AXIAL", amplitude=80000.0)

    def test_design_orbit_rejects_bad_phase(self):
        """phase 超界应抛 ValueError。"""
        from e2m2e.api.models import DesignOrbitRequest

        with pytest.raises(ValueError):
            DesignOrbitRequest(orbit_type="AXIAL", phase=1.5)

    def test_validate_params_axial_defaults(self):
        """_validate_params 对 AXIAL 填默认值（collinear_point=2, amplitude=5000, phase=0）。"""
        from e2m2e.algorithm.design.design_orbit import _validate_params

        params = _validate_params(
            "AXIAL",
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
        assert params == {"collinear_point": 2, "amplitude": 5000.0, "phase": 0.0}

    def test_cr3bp_orbit_for_axial_end_to_end(self):
        """_cr3bp_orbit_for("AXIAL", ...) 端到端调用 design_axial 返回 Orbit。"""
        from e2m2e.algorithm.design.design_orbit import _cr3bp_orbit_for, _validate_params

        params = _validate_params(
            "AXIAL",
            amplitude=5000.0,
            phase=None,
            collinear_point=1,
            north_south=None,
            perilune_height=None,
            amplitude_in=None,
            amplitude_out=None,
            phase_in=None,
            phase_out=None,
        )
        dynamics = CR3BP_Dynamics(earth_moon_system())
        orbit = _cr3bp_orbit_for("AXIAL", params, dynamics)
        assert orbit is not None
        assert orbit.period is not None
        assert orbit.period > 0


# =============================================================================
# Axial vs Vertical Lyapunov distinguishing tests
# =============================================================================


class TestAxialNotVerticalLyapunov:
    """区分真 Axial 轨道与 Vertical Lyapunov 的特征测试。

    Vertical Lyapunov 种子 (x_L, 0, 0, 0, 0, vz0) 紧邻平动点，
    Jacobi ≈ C_L ≈ 3.19，面内振幅趋零。真 Axial 轨道从 Lyapunov
    垂直临界轨道分岔，Jacobi ∈ [2.991, 3.021]（Haapala & Howell 2016），
    面内振幅显著（|y|_max > 1000 km）。
    """

    def test_axial_jacobi_in_haapala_range(self):
        """Axial 轨道的 Jacobi 应落在 Haapala 区间 [2.95, 3.05]，

        而非紧邻 C_L1=3.188（Vertical Lyapunov 特征）。"""
        orbit = design_axial(1, 5000.0)
        C = float(orbit.system.get_jacobi_constant(orbit.states[0]))
        assert 2.95 < C < 3.05, f"Axial Jacobi {C:.4f} 不在 Haapala 区间，可能是 Vertical Lyapunov"

    def test_axial_has_nontrivial_inplane_amplitude(self):
        """Axial 继承 planar Lyapunov 父支的面内振幅，|y|_max 不应趋零。"""
        orbit = design_axial(1, 5000.0)
        du = orbit.system.characteristic_length
        dynamics = CR3BP_Dynamics(orbit.system)
        result = dynamics.propagate(
            orbit.states[0],
            (0, orbit.period),
            t_eval=np.linspace(0, orbit.period, 1000),
        )
        y_max = float(np.max(np.abs(result["states"][:, 1]))) * du  # km
        assert y_max > 1000.0, f"|y|_max={y_max:.0f} km 过小，可能不是真 Axial"
