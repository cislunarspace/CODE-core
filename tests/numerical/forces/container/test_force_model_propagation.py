"""ForceModel 编译传播的能力边界测试。

自定义 Python 力和事件都没有对应的 compiled-forces Rust API，必须显式报错，
不能退回 Python 力循环或 scipy 事件积分。
"""

import numpy as np
import pytest

from e2m2e.algorithm.forces import ForceModel, PhysicalModel
from e2m2e.algorithm.forces.point_mass_gravity import PointMassGravity
from e2m2e.algorithm.forces.thrust import FiniteBurn, VariableMassFiniteBurn
from tests.numerical.forces.conftest import FakeSystem

pytestmark = pytest.mark.force


class ConstantForce(PhysicalModel):
    """无 Rust spec 的测试力。"""

    def __init__(self, acceleration):
        self._acceleration = np.asarray(acceleration, dtype=float)


def test_propagate_rejects_force_without_rust_spec():
    """有 SPICE 但力无 Rust spec（能力缺失）→ NotImplementedError。"""
    force_model = ForceModel(FakeSystem(), forces=[ConstantForce([0.0, 0.0, 1.0])])

    with pytest.raises(NotImplementedError, match="无 Rust 实现"):
        force_model.propagate(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), (0.0, 1.0))


def test_propagate_reports_spice_missing_as_resource_error():
    """system 无 spice（资源缺失）→ RustExtensionUnavailableError（ADR 0020 决策 4 分流）。"""
    from e2m2e.exceptions import RustExtensionUnavailableError

    force_model = ForceModel(FakeSystem(spice=None), forces=[ConstantForce([0.0, 0.0, 1.0])])

    with pytest.raises(RustExtensionUnavailableError, match="需要 SPICE"):
        force_model.propagate(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), (0.0, 1.0))


def test_propagate_point_mass_uses_rust_and_honors_t_eval():
    """PointMass 的 compiled propagation 返回真实轨迹与规范化输出时间。"""
    force_model = ForceModel(FakeSystem(), forces=[PointMassGravity("EARTH", mu=398600.4418)])
    y0 = np.array([7000.0, 0.0, 0.0, 0.0, 7.546049108166282, 0.0])
    t_eval = np.array([0.0, 10.0, 30.0])

    result = force_model.propagate(y0, (0.0, 30.0), t_eval=t_eval)

    np.testing.assert_array_equal(result["time"], t_eval)
    assert result["states"].shape == (3, 6)
    np.testing.assert_allclose(result["states"][0], y0, atol=0.0)
    assert not np.allclose(result["states"][-1], y0)
    assert result["terminal_event_index"] is None


def test_finite_burn_propagation_reports_unsupported_rust_capability():
    """FiniteBurn 没有 Rust spec 时，传播必须显式报告能力边界。"""
    burn = FiniteBurn(
        thrust_profile=lambda _t: 10.0,
        direction=[1.0, 0.0, 0.0],
        mass=1000.0,
    )
    force_model = ForceModel(FakeSystem(), forces=[burn])

    with pytest.raises(NotImplementedError, match="FiniteBurn.*Rust"):
        force_model.propagate(
            np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0]),
            (0.0, 1.0),
        )


def test_variable_mass_propagate_rejects_events_before_rust_path():
    """低推力路径不能忽略 events 或退回 Python 力传播。"""
    burn = VariableMassFiniteBurn(
        thrust=0.1,
        isp=3000.0,
        initial_mass=1000.0,
        direction=np.array([1.0, 0.0, 0.0]),
    )
    force_model = ForceModel(FakeSystem(), forces=[burn])

    with pytest.raises(NotImplementedError, match="事件传播"):
        force_model.propagate(
            np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0, 1000.0]),
            (0.0, 1.0),
            events=[lambda _t, state: float(state[0])],
        )


def test_propagate_rejects_events_without_compiled_forces_api():
    """事件传播不进入 Python 力循环。"""
    force_model = ForceModel(FakeSystem())

    with pytest.raises(NotImplementedError, match="事件传播"):
        force_model.propagate(
            np.zeros(6),
            (0.0, 1.0),
            events=[lambda _t, state: float(state[2])],
        )
