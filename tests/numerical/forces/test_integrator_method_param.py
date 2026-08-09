"""ForceModel 积分器选择参数测试。

验证 propagate / propagate_maneuvers 接受 method 参数，
PD45 与 PD78 传播同一轨道结果量级一致。
"""

import numpy as np
import pytest

from e2m2e.algorithm.forces import ForceModel, PhysicalModel
from e2m2e.algorithm.forces.thrust import ImpulsiveBurn
from e2m2e.integrators import RkMethod

pytestmark = pytest.mark.force


class PointMassTestForce(PhysicalModel):
    """测试用中心引力模型，返回 -mu/r^3 * r。"""

    def __init__(self, mu: float):
        self.mu = float(mu)

    def compute_acceleration(self, t, state, system):
        r = np.asarray(state[:3], dtype=float)
        rr = np.dot(r, r)
        return -self.mu / (rr * np.sqrt(rr)) * r


class _FakeSystem:
    """仅用于传播测试的最小 System 桩。"""

    def __init__(self):
        self.coordinate_system = object()

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


def test_propagate_pd78_works():
    """PD78 积分器可正常传播。"""
    system = _FakeSystem()
    mu = 398600.4415
    fm = ForceModel(system, forces=[PointMassTestForce(mu)])

    r = 6778.0
    v = np.sqrt(mu / r)
    period = 2.0 * np.pi * np.sqrt(r**3 / mu)
    y0 = np.array([r, 0.0, 0.0, 0.0, v, 0.0])

    result = fm.propagate(y0, (0.0, period), method=RkMethod.PD78)

    final = result["states"][-1]
    # 一个周期应回到近似初始位置
    np.testing.assert_allclose(final[:3], y0[:3], rtol=1e-4, atol=1e-8)
    np.testing.assert_allclose(final[3:], y0[3:], rtol=1e-4, atol=1e-8)


def test_propagate_pd45_and_pd78_similar_results():
    """同一初值分别用 PD45 和 PD78 传播，末状态应相近。"""
    system = _FakeSystem()
    mu = 398600.4415
    fm = ForceModel(system, forces=[PointMassTestForce(mu)])

    r = 6778.0
    v = np.sqrt(mu / r)
    period = 2.0 * np.pi * np.sqrt(r**3 / mu)
    y0 = np.array([r, 0.0, 0.0, 0.0, v, 0.0])

    result_45 = fm.propagate(y0, (0.0, period), method=RkMethod.PD45)
    result_78 = fm.propagate(y0, (0.0, period), method=RkMethod.PD78)

    final_45 = result_45["states"][-1]
    final_78 = result_78["states"][-1]

    # 不同积分器有数值差异，但量级应一致
    np.testing.assert_allclose(final_45, final_78, rtol=1e-3, atol=1e-4)


def test_propagate_default_method_is_pd45():
    """默认参数不指定 method 时，行为与显式传 PD45 一致。"""
    system = _FakeSystem()
    mu = 398600.4415
    fm = ForceModel(system, forces=[PointMassTestForce(mu)])

    r = 6778.0
    v = np.sqrt(mu / r)
    period = 2.0 * np.pi * np.sqrt(r**3 / mu)
    y0 = np.array([r, 0.0, 0.0, 0.0, v, 0.0])

    result_default = fm.propagate(y0, (0.0, period))
    result_explicit = fm.propagate(y0, (0.0, period), method=RkMethod.PD45)

    np.testing.assert_array_equal(result_default["states"], result_explicit["states"])


def test_propagate_maneuvers_method_param():
    """propagate_maneuvers 支持 method 参数并透传。"""
    system = _FakeSystem()
    fm = ForceModel(system)  # 零外力

    v0 = np.array([1.0, 0.0, 0.0])
    dv = np.array([0.0, 0.5, 0.0])
    y0 = np.array([0.0, 0.0, 0.0, v0[0], v0[1], v0[2]])

    burn = ImpulsiveBurn(epoch=0.5, delta_v=dv)
    result = fm.propagate_maneuvers(y0, (0.0, 1.0), burns=[burn], method=RkMethod.PD78)

    # 零外力下末速度 = 初速度 + Δv
    final = result["states"][-1]
    np.testing.assert_allclose(final[3:6], v0 + dv, rtol=1e-12)
