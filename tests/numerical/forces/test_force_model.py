"""ForceModel 容器测试。

覆盖坐标系要求、力聚合、增删、启用禁用、自动命名与重复名检测。
Python 侧 ``_compute_total_acceleration`` / ``equations_of_motion`` 已按
issue #378 删除；聚合对传播的影响通过 Rust 端到端传播验证。
"""

import numpy as np
import pytest

from e2m2e.algorithm.forces import ForceModel, PhysicalModel, PointMassGravity

pytestmark = pytest.mark.force


class _FakeSystem:
    """仅用于聚合测试的最小 System 桩。"""

    def __init__(self, has_coordinate_system=True):
        self.coordinate_system = object() if has_coordinate_system else None
        self.origin = "EARTH"
        self.spice = object()  # 模拟有 SPICE：力无 Rust spec 属能力缺失

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
    """测试用恒力模型（无 Rust spec）。"""

    def __init__(self, acceleration):
        self._acceleration = np.asarray(acceleration, dtype=float)


def test_force_model_requires_coordinate_system():
    """ForceModel 构造需要 system.coordinate_system 非 None。"""
    system = _FakeSystem(has_coordinate_system=False)
    with pytest.raises(ValueError, match="coordinate_system"):
        ForceModel(system)


def test_force_model_adds_forces():
    """ForceModel 可以聚合多个力模型。"""
    system = _FakeSystem()

    fm = ForceModel(system)
    fm.add_force(ConstantForce([1.0, 0.0, 0.0]))
    fm.add_force(ConstantForce([0.0, 2.0, 0.0]))

    assert len(fm.forces) == 2


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
    """disable 后 forces/list_forces 仍含它，传播时该力被跳过。"""
    system = _FakeSystem()
    fm = ForceModel(system)
    fm.add_force(ConstantForce([1.0, 0.0, 0.0]), name="a")
    fm.add_force(ConstantForce([0.0, 2.0, 0.0]), name="b")

    fm.disable("a")

    # forces 与 list_forces 仍含两个
    assert len(fm.forces) == 2
    assert len(fm.list_forces()) == 2
    # 禁用无 Rust spec 的力后，传播不再因它报能力错误；剩下的 b 仍报错
    with pytest.raises(NotImplementedError, match="无 Rust 实现"):
        fm.propagate(np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0]), (0.0, 1.0))


def test_disabled_force_with_rust_spec_is_skipped_in_propagation():
    """禁用有 Rust spec 的力后，传播结果与空力模型一致。"""
    system = _FakeSystem()
    fm = ForceModel(system, forces=[PointMassGravity("EARTH", mu=398600.4415)])
    fm.disable("PointMassGravity")

    y0 = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    result = fm.propagate(y0, (0.0, 10.0), t_eval=np.array([0.0, 10.0]))

    # 无启用力 → 匀速直线
    expected = y0.copy()
    expected[:3] += y0[3:] * 10.0
    np.testing.assert_allclose(result["states"][-1], expected, atol=1e-12)


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


def test_multi_force_rust_propagation_superposes():
    """多力组合的 Rust 传播与单力传播差等价于各力贡献叠加（定性验证）。"""
    system = _FakeSystem()
    fm_single = ForceModel(system, forces=[PointMassGravity("EARTH", mu=398600.4415)])
    fm_double = ForceModel(
        system,
        forces=[
            PointMassGravity("EARTH", mu=398600.4415),
            PointMassGravity("EARTH", mu=398600.4415),
        ],
    )

    y0 = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    t_eval = np.array([0.0, 10.0])
    r1 = fm_single.propagate(y0, (0.0, 10.0), t_eval=t_eval)["states"][-1]
    r2 = fm_double.propagate(y0, (0.0, 10.0), t_eval=t_eval)["states"][-1]

    # 双点质量等效 mu 加倍，末态应不同且可重复
    assert not np.allclose(r1, r2)
    r2_again = fm_double.propagate(y0, (0.0, 10.0), t_eval=t_eval)["states"][-1]
    np.testing.assert_allclose(r2, r2_again, atol=1e-12)
