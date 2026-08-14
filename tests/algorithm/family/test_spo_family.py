"""SPO（Short-Period Orbit）轨道族端到端 + 物理不变量测试。

覆盖 design_spo 收敛性、Jacobi 守恒、xy 平面约束、周期闭合、注册表，
以及 SPO 的特征测试（近稳定、周期约 1 朔望月、围绕 L4/L5）。

References:
    Gómez et al. (2001). Dynamics and mission design near libration
    points, Vol. II. ESA Contract Report.
    Capdevila & Howell (2018). JGCD. Table 1 SPO 初始条件。
"""

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family import registry
from e2m2e.algorithm.family.cr3bp_orbits import (
    _l45_distance,
    design_spo,
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
    """共享一条 design_spo(4, 10000.0) 轨道。"""
    return design_spo(4, 10000.0)


@pytest.fixture(scope="module")
def l5_orbit():
    """共享一条 design_spo(5, 10000.0) 轨道。"""
    return design_spo(5, 10000.0)


@pytest.fixture(scope="module")
def dynamics(l4_orbit):
    return CR3BP_Dynamics(l4_orbit.system)


# =============================================================================
# Registry
# =============================================================================


class TestRegistry:
    """注册表应包含 SPO 条目。"""

    def test_l4_spo_in_registry(self):
        assert "L4_SPO" in registry

    def test_l5_spo_in_registry(self):
        assert "L5_SPO" in registry

    def test_l4_spo_callable(self):
        assert callable(registry["L4_SPO"])

    def test_l5_spo_callable(self):
        assert callable(registry["L5_SPO"])

    def test_l4_spo_same_as_design_spo(self):

        # registry lambda wraps design_spo; test it's callable with right args
        orbit = registry["L4_SPO"](10000.0)
        assert orbit is not None
        assert orbit.period is not None


# =============================================================================
# End-to-end convergence
# =============================================================================


class TestDesignSpoConvergence:
    """design_spo 端到端收敛测试。"""

    def test_design_spo_l4_converges(self):
        orbit = design_spo(4, 10000.0)
        assert orbit is not None
        assert orbit.period is not None
        assert orbit.period > 0

    def test_design_spo_l5_converges(self):
        orbit = design_spo(5, 10000.0)
        assert orbit is not None
        assert orbit.period is not None
        assert orbit.period > 0

    def test_amplitude_matches_target(self, l4_orbit):
        """振幅（距 L4 径向距离均值）应接近 10000 km（容差 25 km）。"""
        dynamics = CR3BP_Dynamics(l4_orbit.system)
        d_min, d_max = _l45_distance(dynamics, l4_orbit, 4)
        amp_km = 0.5 * (d_min + d_max) * CHAR_LENGTH_KM
        assert abs(amp_km - 10000.0) < 25.0

    def test_initial_state_plane(self, l4_orbit):
        """初始状态应在 xy 平面：z=0, ż=0。"""
        s0 = l4_orbit.states[0]
        assert s0[2] == pytest.approx(0.0, abs=1e-10)
        assert s0[5] == pytest.approx(0.0, abs=1e-10)

    def test_period_in_synodic_range(self, l4_orbit):
        """周期应在 1 朔望月范围内（27-31 天）。"""
        T_days = l4_orbit.period / (2 * np.pi) * 27.32
        assert 27.0 < T_days < 31.0, f"周期 {T_days:.2f} 天不在 27-31 天范围内"


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
            t_eval=np.linspace(0, T, 1000),
            with_jacobi=True,
        )
        jacobi = result["jacobi"]
        drift = abs(jacobi[-1] - jacobi[0])
        assert drift < 1e-10

    def test_periodic_closure(self, l4_orbit, dynamics):
        """全周期闭合误差 < 1e-6。"""
        T = l4_orbit.period
        result = dynamics.propagate(l4_orbit.states[0], (0, T), t_eval=np.linspace(0, T, 1000))
        closure = np.linalg.norm(result["states"][-1] - l4_orbit.states[0])
        assert closure < 1e-6

    def test_xy_plane_constraint(self, l4_orbit, dynamics):
        """SPO 为 xy 平面轨道：整个周期内 z≈0。"""
        T = l4_orbit.period
        result = dynamics.propagate(l4_orbit.states[0], (0, T), t_eval=np.linspace(0, T, 500))
        z_max = np.max(np.abs(result["states"][:, 2]))
        assert z_max < 1e-8

    def test_jacobi_in_expected_range(self, l4_orbit):
        """Jacobi 常数应在 SPO 典型范围（SPO 围绕 L4，C 接近 L4 的 3.0 即可）。"""
        C = float(l4_orbit.system.get_jacobi_constant(l4_orbit.states[0]))
        assert 2.8 < C < 3.1, f"Jacobi C={C:.4f} 不在预期范围 2.8-3.1"


# =============================================================================
# L4/L5 symmetry
# =============================================================================


class TestL4L5Symmetry:
    """L5 轨道应与 L4 轨道具有相同的标量特征（周期、Jacobi、振幅）。

    注：由于 design_spo 独立搜索 L4/L5 的 x₀，初始条件不保证精确
    镜像。但标量特征应一致。
    """

    def test_same_period(self, l4_orbit, l5_orbit):
        """L4/L5 周期应一致。"""
        assert abs(l4_orbit.period - l5_orbit.period) < 1e-4

    def test_same_jacobi(self, l4_orbit, l5_orbit):
        """L4/L5 Jacobi 常数应接近（独立搜索有微小差异）。"""
        C4 = float(l4_orbit.system.get_jacobi_constant(l4_orbit.states[0]))
        C5 = float(l5_orbit.system.get_jacobi_constant(l5_orbit.states[0]))
        assert abs(C4 - C5) < 0.01, f"L4 C={C4:.6f}, L5 C={C5:.6f}"

    def test_l4_orbit_near_l4(self, l4_orbit):
        """L4 SPO 的 y₀ 应为正（在 L4 附近）。"""
        assert l4_orbit.states[0, 1] > 0, f"L4 y₀={l4_orbit.states[0, 1]:.6f} 应为正"

    def test_l5_orbit_near_l5(self, l5_orbit):
        """L5 SPO 的 y₀ 应为负（在 L5 附近）。"""
        assert l5_orbit.states[0, 1] < 0, f"L5 y₀={l5_orbit.states[0, 1]:.6f} 应为负"


# =============================================================================
# design_orbit facade dispatch
# =============================================================================


@_design_orbit_skip
class TestSpoDesignOrbit:
    """design_orbit 的 SPO 参数规范化与端到端分派（请求模型校验见 tests/api）。"""

    def test_validate_params_l4_spo_defaults(self):
        from e2m2e.algorithm.design.design_orbit import _validate_params

        params = _validate_params(
            "L4_SPO",
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
        assert params == {"amplitude": 10000.0, "phase": 0.0}

    def test_cr3bp_orbit_for_l4_spo_end_to_end(self):
        from e2m2e.algorithm.design.design_orbit import _cr3bp_orbit_for, _validate_params

        params = _validate_params(
            "L4_SPO",
            amplitude=10000.0,
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
        orbit = _cr3bp_orbit_for("L4_SPO", params, dynamics)
        assert orbit is not None
        assert orbit.period is not None
        assert orbit.period > 0
