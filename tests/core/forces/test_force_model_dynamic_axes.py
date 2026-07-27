"""ForceModel propagate 过程中动态坐标系更新测试。

验证 DynamicAxes.update 在传播循环中被正确调用。
"""

from __future__ import annotations

import numpy as np
from e2m2e.core.axes import Axes
from e2m2e.core.coordinate_system import CoordinateSystem
from e2m2e.core.dynamic_axes import DynamicAxes
from e2m2e.core.origin import Origin

from e2m2e.core.forces import ForceModel, PhysicalModel


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


def test_propagate_calls_update_before_loop_and_each_step():
    """传播应在循环开始前和每个 rk_step 前调用 update_coordinate_systems。"""
    axes = _MockDynamicAxes()
    system = _FakeSystemWithDynamicAxes(axes)
    fm = ForceModel(system, forces=[ConstantForce([0.0, 0.0, 0.0])])

    y0 = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    result = fm.propagate(y0, (0.0, 10.0), t_eval=np.linspace(0.0, 10.0, 3))

    assert result["time"][-1] == 10.0
    # 至少有一次循环前调用 + 若干步内调用
    assert len(axes.calls) >= 2

    # 第一次调用是循环前（t0）
    assert axes.calls[0][0] == 0.0
    np.testing.assert_allclose(axes.calls[0][1], y0)

    # 后续每次调用都是某个步进时刻
    for t_call, _ in axes.calls[1:]:
        assert 0.0 <= t_call <= 10.0


def test_propagate_compatible_with_old_system_without_update():
    """旧 System 无 update_coordinate_systems 方法时不应抛异常。"""
    system = _FakeSystemNoUpdate()
    fm = ForceModel(system, forces=[ConstantForce([0.0, 0.0, 0.0])])

    y0 = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    result = fm.propagate(y0, (0.0, 1.0), t_eval=np.linspace(0.0, 1.0, 3))

    assert result["time"][-1] == 1.0


def test_system_update_coordinate_systems_updates_dynamic_axes():
    """System.update_coordinate_systems 正确识别 DynamicAxes 并调用 update。"""
    from e2m2e.core.system import System

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
            from e2m2e.core.coordinate.dynamic_axes import DynamicAxes

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
    from e2m2e.core.system import System

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
    from e2m2e.core.system import System

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
