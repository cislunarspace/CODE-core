"""LPO（Long-Period Orbit）轨道族端到端 + 物理不变量测试。

覆盖 design_lpo 收敛性、Jacobi 守恒、xy 平面约束、周期闭合、注册表，
以及 LPO 的特征测试（不稳定、周期约 91 天、围绕 L4/L5）。

References:
    Gómez et al. (2001). Dynamics and mission design near libration
    points, Vol. II. ESA Contract Report.
"""

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family import registry
from e2m2e.algorithm.family.cr3bp_orbits import (
    _l45_distance,
    design_lpo,
    earth_moon_system,
)

pytestmark = pytest.mark.orchestration

CHAR_LENGTH_KM = 384400.0

# design_orbit 需要 Rust 积分器，缺失时跳过相关测试
_DESIGN_ORBIT_AVAILABLE = False
try:
    from e2m2e.algorithm.design import design_orbit  # noqa: F401

    _DESIGN_ORBIT_AVAILABLE = True
except (ImportError, AttributeError):
    pass

_design_orbit_skip = pytest.mark.skipif(
    not _DESIGN_ORBIT_AVAILABLE,
    reason="design_orbit requires Rust integrators (RkMethod not available)",
)


@pytest.fixture(scope="module")
def l4_orbit():
    """共享一条 design_lpo(4, 50000.0) 轨道。"""
    return design_lpo(4, 50000.0)


@pytest.fixture(scope="module")
def l5_orbit():
    """共享一条 design_lpo(5, 50000.0) 轨道。"""
    return design_lpo(5, 50000.0)


@pytest.fixture(scope="module")
def dynamics(l4_orbit):
    return CR3BP_Dynamics(l4_orbit.system)


# =============================================================================
# Registry
# =============================================================================


class TestRegistry:
    """注册表应包含 LPO 条目。"""

    def test_l4_lpo_in_registry(self):
        assert "L4_LPO" in registry

    def test_l5_lpo_in_registry(self):
        assert "L5_LPO" in registry

    def test_l4_lpo_callable(self):
        assert callable(registry["L4_LPO"])

    def test_l5_lpo_callable(self):
        assert callable(registry["L5_LPO"])

    def test_l4_lpo_returns_orbit(self):
        orbit = registry["L4_LPO"](50000.0)
        assert orbit is not None
        assert orbit.period is not None


# =============================================================================
# End-to-end convergence
# =============================================================================


class TestDesignLpoConvergence:
    """design_lpo 端到端收敛测试。"""

    def test_design_lpo_l4_converges(self, l4_orbit):
        orbit = l4_orbit
        assert orbit is not None
        assert orbit.period is not None
        assert orbit.period > 0

    def test_design_lpo_l5_converges(self, l5_orbit):
        orbit = l5_orbit
        assert orbit is not None
        assert orbit.period is not None
        assert orbit.period > 0

    @pytest.mark.parametrize("libration_point", [4, 5])
    def test_declared_upper_amplitude_converges(self, libration_point):
        orbit = design_lpo(libration_point, 110000.0)
        assert orbit is not None
        assert orbit.period is not None
        dynamics = CR3BP_Dynamics(orbit.system)
        d_min, d_max = _l45_distance(dynamics, orbit, libration_point)
        amplitude_km = 0.5 * (d_min + d_max) * CHAR_LENGTH_KM
        assert abs(amplitude_km - 110000.0) < 1500.0

    @pytest.mark.parametrize("amplitude", [999.0, 110001.0])
    def test_rejects_out_of_range_amplitude_before_search(self, amplitude, monkeypatch):
        from e2m2e.algorithm.family import cr3bp_orbits

        monkeypatch.setattr(
            cr3bp_orbits,
            "_correct_lpo",
            lambda *args, **kwargs: pytest.fail("越界振幅不应进入 LPO 搜索"),
        )
        with pytest.raises(ValueError, match="amplitude"):
            design_lpo(4, amplitude)

    def test_amplitude_matches_target(self, l4_orbit):
        """振幅（距 L4 径向距离均值）应接近 50000 km（容差 50 km）。"""
        dynamics = CR3BP_Dynamics(l4_orbit.system)
        d_min, d_max = _l45_distance(dynamics, l4_orbit, 4)
        amp_km = 0.5 * (d_min + d_max) * CHAR_LENGTH_KM
        assert abs(amp_km - 50000.0) < 50.0

    def test_initial_state_plane(self, l4_orbit):
        """初始状态应在 xy 平面：z=0, z-dot=0。"""
        s0 = l4_orbit.states[0]
        assert s0[2] == pytest.approx(0.0, abs=1e-10)
        assert s0[5] == pytest.approx(0.0, abs=1e-10)

    def test_period_in_lpo_range(self, l4_orbit):
        """LPO 周期应在 80-120 天范围内。"""
        T_days = l4_orbit.period / (2 * np.pi) * 27.32
        assert 80.0 < T_days < 120.0, f"周期 {T_days:.2f} 天不在 80-120 天范围内"


# =============================================================================
# Physical invariants
# =============================================================================


class TestPhysicalInvariants:
    """收敛轨道的物理不变量检验。"""

    def test_jacobi_conservation(self, l4_orbit, dynamics):
        """Jacobi 常数在一个周期内漂移 < 1e-10。"""
        T = l4_orbit.period
        result = dynamics.propagate(
            l4_orbit.states[0],
            (0, T),
            t_eval=np.linspace(0, T, 2000),
            with_jacobi=True,
        )
        jacobi = result["jacobi"]
        drift = abs(jacobi[-1] - jacobi[0])
        assert drift < 1e-10

    def test_periodic_closure(self, l4_orbit, dynamics):
        """全周期闭合误差 < 1e-6。"""
        T = l4_orbit.period
        result = dynamics.propagate(l4_orbit.states[0], (0, T), t_eval=np.linspace(0, T, 2000))
        closure = np.linalg.norm(result["states"][-1] - l4_orbit.states[0])
        assert closure < 1e-6

    def test_xy_plane_constraint(self, l4_orbit, dynamics):
        """LPO 为 xy 平面轨道：整个周期内 z≈0。"""
        T = l4_orbit.period
        result = dynamics.propagate(l4_orbit.states[0], (0, T), t_eval=np.linspace(0, T, 1000))
        z_max = np.max(np.abs(result["states"][:, 2]))
        assert z_max < 1e-8

    def test_jacobi_in_expected_range(self, l4_orbit):
        """Jacobi 常数应在 LPO 典型范围。"""
        C = float(l4_orbit.system.get_jacobi_constant(l4_orbit.states[0]))
        assert 2.8 < C < 3.1, f"Jacobi C={C:.4f} 不在预期范围 2.8-3.1"


# =============================================================================
# L4/L5 symmetry
# =============================================================================


class TestL4L5Symmetry:
    """L5 轨道应与 L4 轨道具有相同的标量特征（周期、Jacobi、振幅）。

    注：由于 design_lpo 独立搜索 L4/L5 的 x₀，初始条件不保证精确
    镜像。但标量特征应一致。
    """

    def test_same_period(self, l4_orbit, l5_orbit):
        """L4/L5 周期应一致。"""
        assert abs(l4_orbit.period - l5_orbit.period) < 1e-3

    def test_same_jacobi(self, l4_orbit, l5_orbit):
        """L4/L5 Jacobi 常数应接近（独立搜索有微小差异）。"""
        C4 = float(l4_orbit.system.get_jacobi_constant(l4_orbit.states[0]))
        C5 = float(l5_orbit.system.get_jacobi_constant(l5_orbit.states[0]))
        assert abs(C4 - C5) < 0.01, f"L4 C={C4:.6f}, L5 C={C5:.6f}"

    def test_l4_orbit_near_l4(self, l4_orbit):
        """L4 LPO 的 y₀ 应为正（在 L4 附近）。"""
        assert l4_orbit.states[0, 1] > 0, f"L4 y₀={l4_orbit.states[0, 1]:.6f} 应为正"

    def test_l5_orbit_near_l5(self, l5_orbit):
        """L5 LPO 的 y₀ 应为负（在 L5 附近）。"""
        assert l5_orbit.states[0, 1] < 0, f"L5 y₀={l5_orbit.states[0, 1]:.6f} 应为负"


# =============================================================================
# design_orbit facade dispatch
# =============================================================================


@_design_orbit_skip
class TestLpoDesignOrbit:
    """design_orbit 的 LPO 参数规范化与端到端分派（请求模型校验见 tests/api）。"""

    def test_validate_params_l4_lpo_defaults(self):
        from e2m2e.algorithm.design.design_orbit import _validate_params

        params = _validate_params(
            "L4_LPO",
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
        assert params == {"amplitude": 50000.0, "phase": 0.0}

    @pytest.mark.parametrize("libration_point", [4, 5])
    def test_default_amplitude_dispatch_converges(self, libration_point):
        from e2m2e.algorithm.design.design_orbit import _cr3bp_orbit_for, _validate_params

        sel = f"L{libration_point}_LPO"
        params = _validate_params(
            sel,
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
        dynamics = CR3BP_Dynamics(earth_moon_system())
        orbit = _cr3bp_orbit_for(sel, params, dynamics)
        assert orbit is not None
        assert orbit.period is not None
        assert orbit.period > 0
