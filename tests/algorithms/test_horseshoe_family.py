"""Horseshoe 马蹄周期轨道测试。

Horseshoe 是 LPO 长周期族的大振幅成员（Marchal 1990, Brown C.2），
本测试验证 design_horseshoe 便捷封装和注册表。

References:
    Taylor (1981). A&A 103, 288. Sun-Jupiter 马蹄周期轨道。
    Marchal (1990). The Three-Body Problem. Brown 猜想 C.2 证实。
"""

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family import registry
from e2m2e.algorithm.family.cr3bp_orbits import (
    _l45_distance,
    design_horseshoe,
)

pytestmark = [pytest.mark.l4]

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
def l4_horseshoe():
    """共享一条 design_horseshoe(4, 100000.0) 轨道。"""
    return design_horseshoe(4, 100000.0)


@pytest.fixture(scope="module")
def dynamics(l4_horseshoe):
    return CR3BP_Dynamics(l4_horseshoe.system)


# =============================================================================
# Registry
# =============================================================================


class TestRegistry:
    """注册表应包含 Horseshoe 条目。"""

    def test_l4_horseshoe_in_registry(self):
        assert "L4_HORSESHOE" in registry

    def test_l5_horseshoe_in_registry(self):
        assert "L5_HORSESHOE" in registry

    def test_l4_horseshoe_callable(self):
        assert callable(registry["L4_HORSESHOE"])

    def test_l5_horseshoe_callable(self):
        assert callable(registry["L5_HORSESHOE"])


# =============================================================================
# End-to-end convergence
# =============================================================================


class TestDesignHorseshoe:
    """design_horseshoe 端到端收敛测试。"""

    def test_converges_l4(self):
        orbit = design_horseshoe(4, 100000.0)
        assert orbit is not None
        assert orbit.period is not None

    def test_converges_l5(self):
        orbit = design_horseshoe(5, 100000.0)
        assert orbit is not None
        assert orbit.period is not None

    def test_amplitude_matches(self, l4_horseshoe):
        """振幅应接近 100000 km（容差 1500 km，LPO 非单调区域精度有限）。"""
        dynamics = CR3BP_Dynamics(l4_horseshoe.system)
        d_min, d_max = _l45_distance(dynamics, l4_horseshoe, 4)
        amp_km = 0.5 * (d_min + d_max) * CHAR_LENGTH_KM
        assert abs(amp_km - 100000.0) < 1500.0

    def test_period_in_lpo_range(self, l4_horseshoe):
        """Horseshoe 周期应在 LPO 范围内（80-150 天）。"""
        T_days = l4_horseshoe.period / (2 * np.pi) * 27.32
        assert 80.0 < T_days < 150.0, f"周期 {T_days:.2f} 天不在 80-150 天范围内"

    def test_xy_plane(self, l4_horseshoe, dynamics):
        """Horseshoe 为 xy 平面轨道：整个周期内 z≈0。"""
        T = l4_horseshoe.period
        result = dynamics.propagate(l4_horseshoe.states[0], (0, T), t_eval=np.linspace(0, T, 1000))
        z_max = np.max(np.abs(result["states"][:, 2]))
        assert z_max < 1e-8

    def test_jacobi_conservation(self, l4_horseshoe, dynamics):
        """Jacobi 常数在一个周期内漂移 < 1e-10。"""
        T = l4_horseshoe.period
        result = dynamics.propagate(
            l4_horseshoe.states[0],
            (0, T),
            t_eval=np.linspace(0, T, 2000),
            with_jacobi=True,
        )
        drift = abs(result["jacobi"][-1] - result["jacobi"][0])
        assert drift < 1e-10

    def test_periodic_closure(self, l4_horseshoe, dynamics):
        """全周期闭合误差 < 1e-6。"""
        T = l4_horseshoe.period
        result = dynamics.propagate(l4_horseshoe.states[0], (0, T), t_eval=np.linspace(0, T, 2000))
        closure = np.linalg.norm(result["states"][-1] - l4_horseshoe.states[0])
        assert closure < 1e-6


# =============================================================================
# design_orbit facade dispatch
# =============================================================================


@_design_orbit_skip
class TestHorseshoeDesignOrbit:
    """design_orbit 入口分发 Horseshoe。"""

    def test_validate_params_l4_horseshoe_defaults(self):
        from e2m2e.algorithm.design.design_orbit import _validate_params

        params = _validate_params(
            "L4_HORSESHOE",
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
        assert params == {"amplitude": 150000.0, "phase": 0.0}

    def test_validate_params_rejects_small_amplitude(self):
        from e2m2e.algorithm.design.design_orbit import _validate_params

        with pytest.raises(ValueError, match="amplitude"):
            _validate_params(
                "L4_HORSESHOE",
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
