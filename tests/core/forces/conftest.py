"""测试用点质量力模型 fixture。"""

import numpy as np

import pytest

from e2m2e.core.forces import PhysicalModel


class PointMassTestForce(PhysicalModel):
    """测试用中心引力模型，返回 -mu/r^3 * r。"""

    def __init__(self, mu: float):
        self.mu = float(mu)

    def compute_acceleration(self, t, state, system):
        r = np.asarray(state[:3], dtype=float)
        rr = np.dot(r, r)
        return -self.mu / (rr * np.sqrt(rr)) * r


@pytest.fixture
def point_mass_force():
    """地球引力参数下的点质量力 fixture。"""
    return PointMassTestForce(398600.4415)
