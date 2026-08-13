"""PointMassGravity 物理规律验证。

点质量圆轨道一个周期闭合（二体闭式解对照）与比机械能守恒。契约测试见
``contract/test_point_mass_gravity.py``。
"""

import numpy as np
import pytest

from e2m2e.algorithm.forces import ForceModel
from tests.numerical.forces.conftest import FakeSystem

pytestmark = pytest.mark.force


def test_point_mass_gravity_rust_propagation_matches_two_body_solution(point_mass_force):
    """Rust compiled 传播点质量圆轨道，一个周期后回到初值（二体闭式解）。"""
    system = FakeSystem()
    fm = ForceModel(system, forces=[point_mass_force])

    mu = point_mass_force.mu
    r = 6778.0  # km
    v = np.sqrt(mu / r)
    period = 2.0 * np.pi * np.sqrt(r**3 / mu)
    y0 = np.array([r, 0.0, 0.0, 0.0, v, 0.0])

    result = fm.propagate(y0, (0.0, period), t_eval=np.array([0.0, period]))

    final = result["states"][-1]
    np.testing.assert_allclose(final[:3], y0[:3], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(final[3:], y0[3:], rtol=1e-6, atol=1e-8)


def test_propagate_energy_conservation(point_mass_force):
    """点质量传播中比机械能应近似守恒。"""
    system = FakeSystem()
    fm = ForceModel(system, forces=[point_mass_force])

    mu = point_mass_force.mu
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
