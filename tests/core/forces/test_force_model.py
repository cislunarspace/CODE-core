"""ForceModel 容器测试。"""

import numpy as np
import pytest

from e2m2e.core.forces import ForceModel, PhysicalModel


class _FakeSystem:
    """仅用于聚合测试的最小 System 桩。"""

    def __init__(self, has_coordinate_system=True):
        self.coordinate_system = object() if has_coordinate_system else None

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


class ConstantForce(PhysicalModel):
    """测试用恒力模型。"""

    def __init__(self, acceleration):
        self._acceleration = np.asarray(acceleration, dtype=float)

    def compute_acceleration(self, t, state, system):
        return self._acceleration.copy()


def test_force_model_requires_coordinate_system():
    """ForceModel 构造需要 system.coordinate_system 非 None。"""
    system = _FakeSystem(has_coordinate_system=False)
    with pytest.raises(ValueError, match="coordinate_system"):
        ForceModel(system)


def test_force_model_adds_forces():
    """ForceModel 可以聚合多个力模型并叠加加速度。"""
    system = _FakeSystem()

    fm = ForceModel(system)
    fm.add_force(ConstantForce([1.0, 0.0, 0.0]))
    fm.add_force(ConstantForce([0.0, 2.0, 0.0]))

    assert len(fm.forces) == 2

    acc = fm._compute_total_acceleration(0.0, np.zeros(6))
    np.testing.assert_array_equal(acc, np.array([1.0, 2.0, 0.0]))


def test_force_model_remove_force_by_index():
    """可以通过索引移除力模型。"""
    system = _FakeSystem()

    f1 = ConstantForce([1.0, 0.0, 0.0])
    f2 = ConstantForce([0.0, 2.0, 0.0])

    fm = ForceModel(system, forces=[f1, f2])
    fm.remove_force(0)

    assert len(fm.forces) == 1
    assert fm.forces[0] is f2
