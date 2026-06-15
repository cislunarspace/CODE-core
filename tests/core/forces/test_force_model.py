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


def test_add_force_with_name_and_get_by_name():
    """add_force 带 name 后可用 get_force 按名取回同一实例。"""
    system = _FakeSystem()
    fm = ForceModel(system)
    force = ConstantForce([1.0, 0.0, 0.0])

    fm.add_force(force, name="primary")

    assert fm.get_force("primary") is force


def test_list_forces_exposes_name_and_enabled():
    """list_forces 返回 entry，暴露 name / force / enabled 三个维度。"""
    system = _FakeSystem()
    fm = ForceModel(system)
    f1 = ConstantForce([1.0, 0.0, 0.0])
    fm.add_force(f1, name="primary")

    entries = fm.list_forces()

    assert len(entries) == 1
    assert entries[0].name == "primary"
    assert entries[0].force is f1
    assert entries[0].enabled is True


def test_enable_disable_toggles_entry():
    """disable(name) 关闭力，enable(name) 重新打开。"""
    system = _FakeSystem()
    fm = ForceModel(system)
    fm.add_force(ConstantForce([1.0, 0.0, 0.0]), name="primary")

    fm.disable("primary")
    assert fm.list_forces()[0].enabled is False

    fm.enable("primary")
    assert fm.list_forces()[0].enabled is True


def test_disabled_force_skipped_but_kept_in_forces():
    """disable 后加速度计算跳过该力，但 forces/list_forces 仍含它。"""
    system = _FakeSystem()
    fm = ForceModel(system)
    fm.add_force(ConstantForce([1.0, 0.0, 0.0]), name="a")
    fm.add_force(ConstantForce([0.0, 2.0, 0.0]), name="b")

    fm.disable("a")

    # forces 与 list_forces 仍含两个
    assert len(fm.forces) == 2
    assert len(fm.list_forces()) == 2
    # 只有 b 贡献加速度（经公开 equations_of_motion 验证）
    deriv = fm.equations_of_motion(0.0, np.zeros(6))
    np.testing.assert_array_equal(deriv[3:6], np.array([0.0, 2.0, 0.0]))


def test_auto_name_disambiguates_same_type():
    """不传 name 时，同类力自动消歧（Foo、Foo_2）。"""
    system = _FakeSystem()
    fm = ForceModel(system)
    fm.add_force(ConstantForce([1.0, 0.0, 0.0]))
    fm.add_force(ConstantForce([0.0, 2.0, 0.0]))

    names = [e.name for e in fm.list_forces()]
    assert names == ["ConstantForce", "ConstantForce_2"]


def test_auto_name_skips_occupied_suffix():
    """显式占用 Foo_2 后，自动命名不再回退到 Foo，而是取 Foo_3。"""
    system = _FakeSystem()
    fm = ForceModel(system)
    fm.add_force(ConstantForce([1.0, 0.0, 0.0]), name="ConstantForce_2")
    fm.add_force(ConstantForce([0.0, 2.0, 0.0]))

    names = [e.name for e in fm.list_forces()]
    assert names == ["ConstantForce_2", "ConstantForce_3"]


def test_explicit_duplicate_name_raises():
    """显式给出重名时抛 ValueError。"""
    system = _FakeSystem()
    fm = ForceModel(system)
    fm.add_force(ConstantForce([1.0, 0.0, 0.0]), name="primary")
    with pytest.raises(ValueError, match="already exists"):
        fm.add_force(ConstantForce([0.0, 2.0, 0.0]), name="primary")


def test_remove_force_by_name():
    """可按名字移除力模型。"""
    system = _FakeSystem()
    f1 = ConstantForce([1.0, 0.0, 0.0])
    fm = ForceModel(system)
    fm.add_force(f1, name="primary")
    fm.add_force(ConstantForce([0.0, 2.0, 0.0]), name="secondary")

    fm.remove_force("primary")

    assert len(fm.forces) == 1
    assert fm.forces[0] is fm.get_force("secondary")


def test_remove_force_by_index():
    """可按索引移除力模型。"""
    system = _FakeSystem()
    fm = ForceModel(system)
    fm.add_force(ConstantForce([1.0, 0.0, 0.0]), name="primary")
    fm.add_force(ConstantForce([0.0, 2.0, 0.0]), name="secondary")

    fm.remove_force(0)

    assert len(fm.forces) == 1
    assert fm.forces[0] is fm.get_force("secondary")


def test_get_force_missing_raises_keyerror():
    """get_force 不存在的名字抛 KeyError。"""
    system = _FakeSystem()
    fm = ForceModel(system)
    with pytest.raises(KeyError):
        fm.get_force("nope")


def test_disable_missing_name_raises_keyerror():
    """disable/enable 不存在的名字抛 KeyError。"""
    system = _FakeSystem()
    fm = ForceModel(system)
    with pytest.raises(KeyError):
        fm.disable("nope")
    with pytest.raises(KeyError):
        fm.enable("nope")


def test_multi_force_acceleration_equals_vector_sum():
    """多力组合的总加速度 = 各力单独加速度的矢量和（< 1e-14）。"""
    system = _FakeSystem()
    f1 = ConstantForce([1.5, -2.0, 0.5])
    f2 = ConstantForce([0.3, 4.0, -1.0])
    f3 = ConstantForce([-0.8, 1.0, 2.5])
    fm = ForceModel(system, forces=[f1, f2, f3])
    state = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])

    total = fm.equations_of_motion(0.0, state)[3:6]
    manual_sum = (
        f1.compute_acceleration(0.0, state, system)
        + f2.compute_acceleration(0.0, state, system)
        + f3.compute_acceleration(0.0, state, system)
    )
    np.testing.assert_allclose(total, manual_sum, atol=1e-14)

