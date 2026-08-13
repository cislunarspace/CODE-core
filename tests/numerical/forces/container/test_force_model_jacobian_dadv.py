"""ForceModel 速度依赖力（阻尼型）的 Rust STM 能力边界测试。

issue #378：Python 侧自定义力（无 Rust spec）不再支持传播；``with_stm=True``
对无 spec 的力显式报能力错误。阻尼力的 ∂a/∂v 行为由 Rust drag 力的
``drag_accel_and_jacobian`` 在编译路径内处理，Python 不再保留 FD 兜底路径。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.forces import ForceModel
from e2m2e.algorithm.forces.physical_model import PhysicalModel
from tests.numerical.forces.conftest import FakeSystem

pytestmark = pytest.mark.force


class _DampingForce(PhysicalModel):
    """速度依赖力 ``a = -k·v``（无 Rust spec）。"""

    def __init__(self, k: float = 0.5) -> None:
        self._k = float(k)


class _PositionOnlyForce(PhysicalModel):
    """位置型力 ``a = -k·r``（无 Rust spec）。"""

    def __init__(self, k: float = 1.0) -> None:
        self._k = float(k)


def test_damping_force_propagation_rejected_without_rust_spec():
    """速度依赖的自定义阻尼力无 Rust spec，传播必须显式报错。"""
    fm = ForceModel(FakeSystem(), forces=[_DampingForce(k=0.5)])

    with pytest.raises(NotImplementedError, match="无 Rust 实现"):
        fm.propagate(np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]), (0.0, 1.0))


def test_damping_force_stm_propagation_rejected_without_rust_spec():
    """速度依赖的自定义阻尼力无 Rust spec，STM 传播必须显式报错。"""
    fm = ForceModel(FakeSystem(), forces=[_DampingForce(k=0.5)])

    with pytest.raises(NotImplementedError, match="无 Rust 实现"):
        fm.propagate(np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]), (0.0, 1.0), with_stm=True)


def test_position_only_force_propagation_rejected_without_rust_spec():
    """位置型自定义力无 Rust spec，传播同样显式报错。"""
    fm = ForceModel(FakeSystem(), forces=[_PositionOnlyForce(k=1.0)])

    with pytest.raises(NotImplementedError, match="无 Rust 实现"):
        fm.propagate(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), (0.0, 1.0))
