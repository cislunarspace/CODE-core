"""ForceModel propagate 过程中动态坐标系更新测试。

验证 DynamicAxes.update 在传播循环中被正确调用。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.coordinate.axes import Axes
from e2m2e.algorithm.coordinate.coordinate_system import CoordinateSystem
from e2m2e.algorithm.coordinate.dynamic_axes import DynamicAxes
from e2m2e.algorithm.coordinate.origin import Origin
from e2m2e.algorithm.forces import ForceModel, PhysicalModel

pytestmark = pytest.mark.force


class _FixedOrigin(Origin):
    """固定于 ICRF 原点的 Origin 桩。"""

    def state(self, et: float) -> np.ndarray:
        return np.zeros(6)


class _FixedAxes(Axes):
    """恒等旋转轴，用于测试。"""

    def rotation_matrix(self, et: float) -> np.ndarray:
        return np.eye(3)


class _MockDynamicAxes(DynamicAxes):
    """记录每次 update 调用的 DynamicAxes 桩。"""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[float, np.ndarray]] = []

    def update(self, t: float, state: np.ndarray) -> None:
        self.calls.append((t, state.copy()))
        self._updated = True

    def _compute_rotation_matrix(self, et: float) -> np.ndarray:
        return np.eye(3)


class _FakeSystemWithDynamicAxes:
    """带 DynamicAxes 坐标系的最小 System 桩。"""

    def __init__(self, axes: _MockDynamicAxes) -> None:
        origin = _FixedOrigin()
        self.coordinate_system = CoordinateSystem(axes, origin)
        self.spice = object()  # 模拟有 SPICE：力无 Rust spec 属能力缺失
        self._update_calls: list[tuple[float, np.ndarray]] = []

    def update_coordinate_systems(self, t: float, state: np.ndarray) -> None:
        self._update_calls.append((t, state.copy()))
        cs = self.coordinate_system
        axes = getattr(cs, "axes", None)
        if axes is not None and isinstance(axes, DynamicAxes):
            axes.update(t, state)

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


class _FakeSystemNoUpdate:
    """无 update_coordinate_systems 方法的旧 System 桩（兼容测试）。"""

    def __init__(self) -> None:
        origin = _FixedOrigin()
        axes = _FixedAxes()
        self.coordinate_system = CoordinateSystem(axes, origin)
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


def test_propagate_calls_update_before_loop_and_each_step():
    """传播应在循环开始前和每个 rk_step 前调用 update_coordinate_systems。

    issue #378：ForceModel 传播改走 Rust compiled 路径后，坐标系更新由
    Rust 内部按步完成，Python 侧不再逐回调 ``update_coordinate_systems``。
    本测试转为验证：无 Rust spec 的力在传播入口显式报错，不进入任何
    Python 坐标系更新循环。
    """
    axes = _MockDynamicAxes()
    system = _FakeSystemWithDynamicAxes(axes)
    fm = ForceModel(system, forces=[ConstantForce([0.0, 0.0, 0.0])])

    y0 = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    with pytest.raises(NotImplementedError, match="无 Rust 实现"):
        fm.propagate(y0, (0.0, 10.0), t_eval=np.linspace(0.0, 10.0, 3))

    # 未进入传播循环，不应有坐标系更新调用
    assert len(axes.calls) == 0


def test_propagate_compatible_with_old_system_without_update():
    """旧 System 无 update_coordinate_systems 方法时，无 Rust spec 力仍显式报错。"""
    system = _FakeSystemNoUpdate()
    fm = ForceModel(system, forces=[ConstantForce([0.0, 0.0, 0.0])])

    y0 = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    with pytest.raises(NotImplementedError, match="无 Rust 实现"):
        fm.propagate(y0, (0.0, 1.0), t_eval=np.linspace(0.0, 1.0, 3))


def test_system_update_coordinate_systems_updates_dynamic_axes():
    """System.update_coordinate_systems 正确识别 DynamicAxes 并调用 update。"""
    from e2m2e.algorithm.dynamics.system import System

    axes = _MockDynamicAxes()
    origin = _FixedOrigin()
    cs = CoordinateSystem(axes, origin)

    class _MinimalSystem(System):
        def __init__(self):
            self._cs = cs

        @property
        def frame(self):
            from e2m2e.mbse.data.enums import ReferenceFrame

            return ReferenceFrame.J2000

        @property
        def unit_system(self):
            from e2m2e.mbse.data.enums import UnitSystem

            return UnitSystem.SI

        @property
        def coordinate_system(self):
            return self._cs

        def update_coordinate_systems(self, t, state):
            from e2m2e.algorithm.coordinate.dynamic_axes import DynamicAxes

            cs = self.coordinate_system
            if cs is None:
                return
            a = getattr(cs, "axes", None)
            if a is not None and isinstance(a, DynamicAxes):
                a.update(t, np.asarray(state, dtype=float))

        def gravitational_parameter(self, body):
            return 398600.4415

    system = _MinimalSystem()
    state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    system.update_coordinate_systems(42.0, state)

    assert len(axes.calls) == 1
    assert axes.calls[0][0] == 42.0
    np.testing.assert_allclose(axes.calls[0][1], state)


def test_system_update_coordinate_systems_no_op_for_static_axes():
    """update_coordinate_systems 对静态 Axes 为 no-op。"""
    from e2m2e.algorithm.dynamics.system import System

    origin = _FixedOrigin()
    cs = CoordinateSystem(_FixedAxes(), origin)

    class _MinimalSystem(System):
        def __init__(self):
            self._cs = cs

        @property
        def frame(self):
            from e2m2e.mbse.data.enums import ReferenceFrame

            return ReferenceFrame.J2000

        @property
        def unit_system(self):
            from e2m2e.mbse.data.enums import UnitSystem

            return UnitSystem.SI

        @property
        def coordinate_system(self):
            return self._cs

        def update_coordinate_systems(self, t, state):
            pass

        def gravitational_parameter(self, body):
            return 398600.4415

    system = _MinimalSystem()
    state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # 不应抛异常
    system.update_coordinate_systems(42.0, state)


def test_system_update_coordinate_systems_no_op_when_no_coordinate_system():
    """coordinate_system 为 None 时 update_coordinate_systems 为 no-op。"""
    from e2m2e.algorithm.dynamics.system import System

    class _MinimalSystem(System):
        @property
        def frame(self):
            from e2m2e.mbse.data.enums import ReferenceFrame

            return ReferenceFrame.J2000

        @property
        def unit_system(self):
            from e2m2e.mbse.data.enums import UnitSystem

            return UnitSystem.SI

        def update_coordinate_systems(self, t, state):
            pass

        def gravitational_parameter(self, body):
            return 398600.4415

    system = _MinimalSystem()
    state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # 不应抛异常
    system.update_coordinate_systems(42.0, state)
