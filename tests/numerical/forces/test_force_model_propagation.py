"""ForceModel 传播循环测试。

覆盖恒力抛物线轨迹、t_eval 精确输出与终止事件。
"""

import numpy as np
import pytest

from e2m2e.algorithm.forces import ForceModel, PhysicalModel

pytestmark = pytest.mark.force


class ConstantForce(PhysicalModel):
    """测试用恒力模型。"""

    def __init__(self, acceleration):
        self._acceleration = np.asarray(acceleration, dtype=float)

    def compute_acceleration(self, t, state, system):
        return self._acceleration.copy()


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


def test_propagate_constant_force_matches_parabola():
    """恒力下传播轨迹应为抛物线。"""
    system = _FakeSystem()
    fm = ForceModel(system, forces=[ConstantForce([0.0, 0.0, 1.0])])

    y0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    result = fm.propagate(y0, (0.0, 1.0), t_eval=np.linspace(0.0, 1.0, 11))

    time = result["time"]
    states = result["states"]

    # 解析解：x = t, z = 0.5 * t^2
    expected_x = time
    expected_z = 0.5 * time**2

    np.testing.assert_allclose(states[:, 0], expected_x, rtol=1e-6)
    np.testing.assert_allclose(states[:, 2], expected_z, rtol=1e-6)
    assert result["terminal_event_index"] is None


def test_propagate_records_t_eval_points():
    """输出 time 应精确等于 t_eval。"""
    system = _FakeSystem()
    fm = ForceModel(system, forces=[ConstantForce([0.0, 0.0, 0.0])])

    y0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    t_eval = np.array([0.0, 0.1, 0.5, 1.0])
    result = fm.propagate(y0, (0.0, 1.0), t_eval=t_eval)

    np.testing.assert_array_almost_equal(result["time"], t_eval)


def test_propagate_termination_event():
    """终止事件在高度穿过零时停止。"""
    system = _FakeSystem()
    fm = ForceModel(system, forces=[ConstantForce([0.0, 0.0, -1.0])])

    y0 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    def hit_ground(t, y):
        return y[2]

    result = fm.propagate(
        y0,
        (0.0, 5.0),
        t_eval=np.linspace(0.0, 5.0, 6),
        events=[hit_ground],
    )

    assert result["terminal_event_index"] == 0
    # 事件在 t=sqrt(2) 附近触发；由于 t_eval 钳制，最后输出点可能到 2.0
    assert result["time"][-1] <= 2.0
    assert result["time"][-1] > 1.3


def test_propagate_termination_event_refined_by_rust():
    """Rust solve_ivp_events 路径：事件时刻步内求精，末点落在事件面上。"""
    integrators = pytest.importorskip("e2m2e._integrators")
    if not hasattr(integrators, "solve_ivp_events_py"):
        pytest.skip("需要带 solve_ivp_events_py 的 Rust 扩展")

    system = _FakeSystem()
    fm = ForceModel(system, forces=[ConstantForce([0.0, 0.0, -1.0])])
    # 步内求精基于线性插值，误差 ~h²/8；max_step=0.01 时约 1e-5
    fm.max_step = 0.01

    y0 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    def hit_ground(t, y):
        return y[2]

    result = fm.propagate(
        y0,
        (0.0, 5.0),
        t_eval=np.linspace(0.0, 5.0, 6),
        events=[hit_ground],
    )

    assert result["terminal_event_index"] == 0
    # z(t) = 1 - t²/2，零点在 t = √2
    assert result["time"][-1] == pytest.approx(np.sqrt(2.0), abs=1e-3)
    assert abs(result["states"][-1][2]) < 1e-3
    assert len(result["t_events"][0]) == 1
    assert result["t_events"][0][0] == pytest.approx(np.sqrt(2.0), abs=1e-3)
