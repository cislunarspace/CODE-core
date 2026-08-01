"""PhysicalModel ABC 子类化契约测试。

验证抽象基类不能直接实例化、子类可提供恒定力。
"""

import numpy as np
import pytest

from e2m2e.algorithm.forces import PhysicalModel


class ConstantForce(PhysicalModel):
    """测试用恒力模型。"""

    def __init__(self, acceleration: np.ndarray) -> None:
        self._acceleration = np.asarray(acceleration, dtype=float)

    def compute_acceleration(self, t, state, system):
        return self._acceleration.copy()


def test_physical_model_is_abstract():
    """PhysicalModel 不能直接实例化。"""
    with pytest.raises(TypeError):
        PhysicalModel()


def test_physical_model_can_be_subclassed():
    """子类化后可以提供恒定力。"""
    force = ConstantForce(np.array([1.0, 2.0, 3.0]))
    acc = force.compute_acceleration(0.0, np.zeros(6), None)

    np.testing.assert_array_equal(acc, np.array([1.0, 2.0, 3.0]))
    # 应返回新数组，不修改内部状态
    acc[0] = 99.0
    np.testing.assert_array_equal(force._acceleration, np.array([1.0, 2.0, 3.0]))
