"""ForceModel 传播边界行为测试。

覆盖零时间跨度、终止事件与 t_eval 精确输出。
"""

import numpy as np
import pytest

from e2m2e.algorithm.forces import ForceModel, PhysicalModel, PointMassGravity

pytestmark = pytest.mark.force


class ConstantForce(PhysicalModel):
    """测试用恒力模型（无 Rust spec）。"""

    def __init__(self, acceleration):
        self._acceleration = np.asarray(acceleration, dtype=float)


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


def test_propagate_zero_span_returns_initial_state():
    """t_span 两端相等时返回初始状态（无启用力时直接返回）。"""
    system = _FakeSystem()
    fm = ForceModel(system)
    y0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    result = fm.propagate(y0, (2.0, 2.0))

    np.testing.assert_array_equal(result["time"], np.array([2.0]))
    np.testing.assert_array_equal(result["states"][0], y0)


def test_propagate_zero_span_with_rust_force_returns_initial_state():
    """t_span 两端相等且含 Rust 力时仍返回初始状态。"""
    system = _FakeSystem()
    fm = ForceModel(system, forces=[PointMassGravity("EARTH", mu=398600.4415)])
    y0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    result = fm.propagate(y0, (2.0, 2.0))

    np.testing.assert_array_equal(result["time"], np.array([2.0]))
    np.testing.assert_array_equal(result["states"][0], y0)


def test_propagate_rejects_backward_integration():
    """反向传播抛 NotImplementedError。"""
    system = _FakeSystem()
    fm = ForceModel(system)
    with pytest.raises(NotImplementedError, match="forward integration"):
        fm.propagate(np.zeros(6), (1.0, 0.0))


def test_propagate_with_stm_returns_kinematic_stm():
    """with_stm=True 返回 STM；恒力下 STM 为纯运动学 [[I, tI],[0, I]]。

    无外力（自由质点）时 ∂a/∂r = 0、∂a/∂v = 0，STM 退化为
    [[I, t·I], [0, I]]。
    """
    system = _FakeSystem()
    fm = ForceModel(system)
    y0 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    dt = 2.0

    result = fm.propagate(y0, (0.0, dt), with_stm=True)

    assert "stm" in result
    stm = result["stm"][-1]
    expected = np.eye(6)
    expected[:3, 3:] = dt * np.eye(3)  # ∂r_f/∂v_0 = t·I
    np.testing.assert_allclose(stm, expected, atol=1e-6)


def test_propagate_rejects_with_jacobi():
    """with_jacobi=True 抛 NotImplementedError。"""
    system = _FakeSystem()
    fm = ForceModel(system)
    with pytest.raises(NotImplementedError, match="Jacobi"):
        fm.propagate(np.zeros(6), (0.0, 1.0), with_jacobi=True)


def test_propagate_rejects_t_eval_out_of_bounds():
    """t_eval 超出 t_span 时抛 ValueError。"""
    system = _FakeSystem()
    fm = ForceModel(system)
    with pytest.raises(ValueError, match="within t_span"):
        fm.propagate(np.zeros(6), (0.0, 1.0), t_eval=np.array([-0.1, 0.5]))


def test_propagate_rejects_non_monotonic_t_eval():
    """t_eval 非单调时抛 ValueError。"""
    system = _FakeSystem()
    fm = ForceModel(system)
    with pytest.raises(ValueError, match="monotonically increasing"):
        fm.propagate(np.zeros(6), (0.0, 1.0), t_eval=np.array([0.0, 0.5, 0.3, 1.0]))


def test_propagate_rejects_invalid_initial_state_shape():
    """初始状态形状错误时抛 ValueError。"""
    system = _FakeSystem()
    fm = ForceModel(system)
    with pytest.raises(ValueError, match="initial_state"):
        fm.propagate(np.zeros(5), (0.0, 1.0))


def test_propagate_rejects_invalid_initial_step():
    """initial_step <= 0 时抛 ValueError。"""
    system = _FakeSystem()
    fm = ForceModel(system)
    with pytest.raises(ValueError, match="initial_step"):
        fm.propagate(np.zeros(6), (0.0, 1.0), initial_step=0.0)


def test_add_force_rejects_non_physical_model():
    """add_force 只能接受 PhysicalModel 实例。"""
    system = _FakeSystem()
    fm = ForceModel(system)
    with pytest.raises(TypeError, match="PhysicalModel"):
        fm.add_force("not a force")
