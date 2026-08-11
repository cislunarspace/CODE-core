"""ForceModel 二体传播测试。

验证点质量圆轨道一个周期闭合与能量守恒。
"""

import numpy as np
import pytest

from e2m2e.algorithm.forces import ForceModel

pytestmark = pytest.mark.force


class _FakeSystem:
    """仅用于传播测试的最小 System 桩。"""

    def __init__(self):
        self.coordinate_system = object()
        self.origin = "EARTH"

    @property
    def frame(self):
        from e2m2e.mbse.data.enums import ReferenceFrame

        return ReferenceFrame.J2000

    @property
    def unit_system(self):
        from e2m2e.mbse.data.enums import UnitSystem

        return UnitSystem.SI

    def gravitational_parameter(self, body):
        return 398600.4415


def test_propagate_circular_orbit_one_period(point_mass_force):
    """点质量圆轨道传播一个周期应回到近似初始状态。"""
    system = _FakeSystem()
    fm = ForceModel(system, forces=[point_mass_force])

    mu = 398600.4415
    r = 6778.0  # km
    v = np.sqrt(mu / r)
    period = 2.0 * np.pi * np.sqrt(r**3 / mu)

    y0 = np.array([r, 0.0, 0.0, 0.0, v, 0.0])
    result = fm.propagate(y0, (0.0, period), t_eval=np.linspace(0.0, period, 5))

    final = result["states"][-1]
    np.testing.assert_allclose(final[:3], y0[:3], rtol=1e-4, atol=1e-8)
    np.testing.assert_allclose(final[3:], y0[3:], rtol=1e-4, atol=1e-8)


def test_propagate_energy_conservation(point_mass_force):
    """点质量传播中比机械能应近似守恒。"""
    system = _FakeSystem()
    fm = ForceModel(system, forces=[point_mass_force])

    mu = 398600.4415
    r = 6778.0
    v = np.sqrt(mu / r) * 1.1  # 稍快，椭圆轨道
    y0 = np.array([r, 0.0, 0.0, 0.0, v, 0.0])

    period = 2.0 * np.pi * np.sqrt(r**3 / mu)
    result = fm.propagate(y0, (0.0, period), t_eval=np.linspace(0.0, period, 11))

    def specific_energy(state):
        r_vec = state[:3]
        v_vec = state[3:]
        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)
        return 0.5 * v * v - mu / r

    energies = np.array([specific_energy(s) for s in result["states"]])
    np.testing.assert_allclose(energies, energies[0], rtol=1e-6)
