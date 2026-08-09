"""SynodicAxes 缓存容量注入 + 淘汰回归测试。

验证：cache_capacity 参数注入后，LRU 缓存在容量满时正确淘汰最老条目，
不会因两个缓存 key 集合不同步（rotation vs rate）而抛 StopIteration。
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from e2m2e.algorithm.coordinate.synodic_axes import SynodicAxes

pytestmark = pytest.mark.data


def _make_mock_spice(moon_pos=None, moon_vel=None):
    """构造返回固定月球状态的 mock SPICE。"""
    spice = MagicMock()
    if moon_pos is None:
        moon_pos = np.array([384400.0, 0.0, 0.0])
    if moon_vel is None:
        moon_vel = np.array([0.0, 1.022, 0.0])
    state = np.concatenate([moon_pos, moon_vel])
    spice.get_body_state.return_value = state
    return spice


class TestSynodicAxesCacheEviction:
    """cache_capacity 注入 + LRU 淘汰正确性。"""

    def test_default_capacity(self):
        """默认 capacity=256 不改变既有行为。"""
        spice = _make_mock_spice()
        axes = SynodicAxes(spice)
        assert axes._CACHE_CAPACITY == 256

    def test_custom_capacity(self):
        """构造时注入小容量 → _CACHE_CAPACITY 为注入值。"""
        spice = _make_mock_spice()
        axes = SynodicAxes(spice, cache_capacity=4)
        assert axes._CACHE_CAPACITY == 4

    def test_rotation_cache_eviction(self):
        """rotation_cache 超容量时淘汰最老条目，不抛异常。"""
        spice = _make_mock_spice()
        capacity = 3
        axes = SynodicAxes(spice, cache_capacity=capacity)

        # 写入超过容量的条目
        for i in range(capacity + 2):
            axes.rotation_matrix(float(i * 100.0))

        assert len(axes._rotation_cache) <= capacity

    def test_rate_cache_eviction(self):
        """rotation_and_rate 同时填充 rotation_cache + rate_cache，
        两者各自独立淘汰，不抛 StopIteration。"""
        spice = _make_mock_spice()
        capacity = 4
        axes = SynodicAxes(spice, cache_capacity=capacity)

        # 写入超过容量的条目（rotation_and_rate 内部调 rotation_matrix 3 次）
        for i in range(capacity + 3):
            axes.rotation_and_rate(float(i * 100.0))

        # 两个缓存各自不超容量
        assert len(axes._rotation_cache) <= capacity
        assert len(axes._rate_cache) <= capacity

    def test_cache_hit_after_eviction(self):
        """淘汰旧条目后，新条目仍可命中缓存（返回相同对象）。"""
        spice = _make_mock_spice()
        axes = SynodicAxes(spice, cache_capacity=2)

        r1 = axes.rotation_matrix(100.0)
        r1_again = axes.rotation_matrix(100.0)
        assert r1 is r1_again  # 同一对象（缓存命中）

        # 写入新条目触发淘汰
        axes.rotation_matrix(200.0)
        axes.rotation_matrix(300.0)

        # 100.0 可能已被淘汰，但 300.0 一定命中
        r3 = axes.rotation_matrix(300.0)
        assert r3 is axes._rotation_cache[300.0]

    def test_capacity_floor_at_one(self):
        """cache_capacity < 1 时自动钳位到 1（避免除零/空缓存问题）。"""
        spice = _make_mock_spice()
        axes = SynodicAxes(spice, cache_capacity=0)
        assert axes._CACHE_CAPACITY == 1
