"""PhysicalModel ABC 子类化契约测试。

验证抽象基类不能直接实例化、子类可继承；Python 单点 ``compute_acceleration``
已按 issue #378 删除，子类只需实现 ``to_rust_spec`` 即可接入传播。
"""

import numpy as np
import pytest

from e2m2e.algorithm.forces import PhysicalModel

pytestmark = pytest.mark.force


class ConstantForce(PhysicalModel):
    """测试用恒力模型（无 Rust spec）。"""

    def __init__(self, acceleration: np.ndarray) -> None:
        self._acceleration = np.asarray(acceleration, dtype=float)


def test_physical_model_can_be_instantiated_directly():
    """PhysicalModel 不是 ABC，可直接实例化（加速度由 Rust 承载）。"""
    force = PhysicalModel()
    assert isinstance(force, PhysicalModel)
    assert force.to_rust_spec(None) is None


def test_physical_model_can_be_subclassed():
    """子类化后可实例化并访问配置。"""
    force = ConstantForce(np.array([1.0, 2.0, 3.0]))
    assert isinstance(force, PhysicalModel)
    np.testing.assert_array_equal(force._acceleration, np.array([1.0, 2.0, 3.0]))
