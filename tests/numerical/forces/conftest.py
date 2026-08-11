"""力模型测试共享 fixture。

提供点质量力 fixture（基于 Rust 支持的 ``PointMassGravity``）。
"""

import pytest

from e2m2e.algorithm.forces import PointMassGravity


@pytest.fixture
def point_mass_force():
    """地球引力参数下的点质量力 fixture。"""
    return PointMassGravity("EARTH", mu=398600.4415)
