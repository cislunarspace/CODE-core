"""NRHO 轨道族端到端 + 物理不变量测试。

覆盖 design_nrho 的两条实现路径：L1 经伪弧长延拓（PAL）行走到目标
近月距，L2 经固定 x0 族行走。检验收敛性、近月距命中、周期闭合、
Jacobi 守恒、近直线特征与注册表。

L1 用例是 #434 的回归测试：#351 结果契约迁移漏改 PAL 编排层的结果
读取，L1 NRHO 必然 AttributeError 崩溃，而套件曾对此零覆盖。

References:
    Lee (2019). Lunar Destination Analysis for Human Exploration Missions.
    NRHO 近直线特征：近月距远小于远月距。
"""

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family import registry
from e2m2e.algorithm.family.cr3bp_orbits import (
    _moon_distance_minmax,
    design_nrho,
)

pytestmark = pytest.mark.orchestration

MOON_RADIUS_KM = 1737.4
CHAR_LENGTH_KM = 384400.0

# design_nrho 默认 tol_km=10：近月距命中容差。断言留少许裕度。
PERILUNE_TOL_KM = 15.0


@pytest.fixture(scope="module")
def l1_north_orbit():
    """共享一条 L1 北族 NRHO（近月高 6000 km，PAL 路径）。"""
    return design_nrho(1, 1, 6000.0)


@pytest.fixture(scope="module")
def l2_south_orbit():
    """共享一条 L2 南族 NRHO（近月高 2500 km，固定 x0 行走路径）。"""
    return design_nrho(2, 2, 2500.0)


@pytest.fixture(scope="module")
def dynamics(l1_north_orbit):
    return CR3BP_Dynamics(l1_north_orbit.system)


def _perilune_height_km(dynamics: CR3BP_Dynamics, orbit) -> float:
    """一个周期内距月心最小距离换算的近月点高度（km）。"""
    d_min, _ = _moon_distance_minmax(dynamics, orbit)
    return d_min * CHAR_LENGTH_KM - MOON_RADIUS_KM


# =============================================================================
# Registry
# =============================================================================


class TestRegistry:
    """注册表应包含 NRHO 条目。"""

    def test_nrho_in_registry(self):
        assert "NRHO" in registry

    def test_nrho_callable(self):
        assert callable(registry["NRHO"])

    def test_nrho_same_as_design_nrho(self):
        assert registry["NRHO"] is design_nrho


# =============================================================================
# End-to-end convergence
# =============================================================================


class TestDesignNrhoConvergence:
    """design_nrho 端到端收敛测试（L1 PAL 路径 + L2 固定 x0 路径）。"""

    def test_design_nrho_l1_north_converges(self, l1_north_orbit):
        assert l1_north_orbit is not None
        assert l1_north_orbit.period is not None
        assert l1_north_orbit.period > 0

    def test_design_nrho_l2_south_converges(self, l2_south_orbit):
        assert l2_south_orbit is not None
        assert l2_south_orbit.period is not None
        assert l2_south_orbit.period > 0

    def test_perilune_matches_target_l1(self, l1_north_orbit, dynamics):
        """L1 北族近月高应命中 6000 km（设计容差 10 km）。"""
        height = _perilune_height_km(dynamics, l1_north_orbit)
        assert abs(height - 6000.0) < PERILUNE_TOL_KM, f"近月高 {height:.1f} km 未命中 6000 km"

    def test_perilune_matches_target_l2(self, l2_south_orbit, dynamics):
        """L2 南族近月高应命中 2500 km（设计容差 10 km）。"""
        height = _perilune_height_km(dynamics, l2_south_orbit)
        assert abs(height - 2500.0) < PERILUNE_TOL_KM, f"近月高 {height:.1f} km 未命中 2500 km"

    def test_north_south_orientation(self, l1_north_orbit, l2_south_orbit):
        """北族初始状态 z>0，南族 z<0（北/南约定与 design_halo 一致）。"""
        assert l1_north_orbit.states[0, 2] > 0
        assert l2_south_orbit.states[0, 2] < 0

    def test_period_in_nrho_range(self, l1_north_orbit, l2_south_orbit):
        """NRHO 周期约 1-3 TU（约 4-13 天，覆盖 9:2 共振邻域）。"""
        for orbit in (l1_north_orbit, l2_south_orbit):
            assert 1.0 < orbit.period < 3.0, f"周期 {orbit.period:.3f} TU 不在 NRHO 典型范围"


# =============================================================================
# Physical invariants
# =============================================================================


class TestPhysicalInvariants:
    """收敛轨道的物理不变量检验。"""

    def test_jacobi_conservation(self, l1_north_orbit, dynamics):
        """Jacobi 常数在一个周期内漂移 < 1e-10。"""
        T = l1_north_orbit.period
        result = dynamics.propagate(
            l1_north_orbit.states[0],
            (0, T),
            t_eval=np.linspace(0, T, 2000),
            with_jacobi=True,
        )
        jacobi = result["jacobi"]
        drift = abs(jacobi[-1] - jacobi[0])
        assert drift < 1e-10

    def test_periodic_closure_l1(self, l1_north_orbit, dynamics):
        """全周期闭合误差 < 1e-6（L1 PAL 路径）。"""
        T = l1_north_orbit.period
        result = dynamics.propagate(
            l1_north_orbit.states[0], (0, T), t_eval=np.linspace(0, T, 2000)
        )
        closure = np.linalg.norm(result["states"][-1] - l1_north_orbit.states[0])
        assert closure < 1e-6

    def test_periodic_closure_l2(self, l2_south_orbit, dynamics):
        """全周期闭合误差 < 1e-6（L2 固定 x0 路径）。"""
        T = l2_south_orbit.period
        result = dynamics.propagate(
            l2_south_orbit.states[0], (0, T), t_eval=np.linspace(0, T, 2000)
        )
        closure = np.linalg.norm(result["states"][-1] - l2_south_orbit.states[0])
        assert closure < 1e-6

    def test_near_rectilinear(self, l1_north_orbit, l2_south_orbit, dynamics):
        """近直线特征：远月距应数倍于近月距（NRHO 定义性几何）。"""
        for orbit in (l1_north_orbit, l2_south_orbit):
            d_min, d_max = _moon_distance_minmax(dynamics, orbit)
            assert d_max > 3.0 * d_min, f"远月距/近月距 = {d_max / d_min:.2f}，不具备近直线特征"
