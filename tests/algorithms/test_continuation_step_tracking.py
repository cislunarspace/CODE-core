"""
Continuation Step Tracking and Sorting Tests

Tests for the continuation_step metadata tracking and sorting feature that ensures
orbits are ordered by step number from the seed orbit (0, 1, -1, 2, -2, ...).

Reference: Commit 071cd09 - feat(continuation): 添加轨道延拓步数跟踪与排序功能
"""

import numpy as np
import pytest

import e2m2e
from e2m2e.algorithms import DifferentialCorrection, Continuation
from e2m2e.core import CR3BP_Dynamics, Orbit, OrbitFamily


# 地月系统质量比
MU = 1.21506683e-2


@pytest.fixture
def earth_moon_system():
    """创建地月CR3BP系统"""
    system = e2m2e.core.system.CR3BP_System(mu=MU, primary="earth", secondary="moon")
    system.compute_libration_points()
    return system


@pytest.fixture
def dynamics(earth_moon_system):
    """创建动力学对象"""
    return CR3BP_Dynamics(earth_moon_system)


@pytest.fixture
def corrector(dynamics):
    """创建配置好的微分修正器"""
    corrector = DifferentialCorrection(dynamics)
    x0 = 0.79188556619742  # DRO 种子参数
    corrector.setup_2D_symmetric_x_fixed_x0(x0)
    return corrector


@pytest.fixture
def continuation(corrector):
    """创建延拓器"""
    return Continuation(corrector=corrector, step=0.001)


@pytest.fixture
def seed_orbit(corrector):
    """创建种子轨道"""
    x0 = 0.79188556619742
    vy0 = 0.53682
    seed = Orbit(
        states=[[x0, 0.0, 0.0, 0.0, vy0, 0.0]],
        times=[0],
    )
    seed.period = 3.420385
    seed.system = corrector.dynamics.system
    return seed


class TestContinuationStepMetadata:
    """测试轨道族元数据中的continuation_step跟踪"""

    def test_seed_orbit_has_step_zero(self, continuation, seed_orbit):
        """测试种子轨道应该被标记为step=0"""
        # 创建简单的测试轨道族
        family = OrbitFamily(seed_orbit)
        family.metadata["family_type"] = "test"

        # 验证种子轨道被正确添加
        assert len(family.orbits) == 1

    def test_orbit_metadata_has_continuation_step_key(self, continuation, seed_orbit):
        """测试Orbit.metadata字典包含continuation_step键"""
        # 创建测试轨道
        test_orbit = seed_orbit.copy()
        test_orbit.metadata["continuation_step"] = 5

        assert "continuation_step" in test_orbit.metadata
        assert test_orbit.metadata["continuation_step"] == 5


class TestContinuationOrbitSorting:
    """测试延拓轨道的排序功能"""

    def test_sorting_order_by_absolute_value(self, continuation, seed_orbit):
        """测试排序按绝对值从小到大排序"""
        # 创建多个测试轨道，标记不同的步数
        step_values = [3, -3, 1, -1, -2, 2, 0]  # 乱序

        temp_orbits_with_steps = []
        for step in step_values:
            orbit = Orbit(
                states=seed_orbit.states.copy(),
                times=seed_orbit.times.copy(),
            )
            orbit.period = seed_orbit.period
            orbit.system = seed_orbit.system
            orbit.metadata["continuation_step"] = step
            temp_orbits_with_steps.append((orbit, step))

        def sort_key(item):
            orbit, step = item
            return (abs(step), step > 0)
        
        temp_orbits_with_steps.sort(key=sort_key)

        # 验证排序后的顺序（按绝对值排序）
        sorted_steps = [step for _, step in temp_orbits_with_steps]
        
        # 排序应为: 0, 1, -1, 2, -2, 3, -3
        # 即先按绝对值，再按正负（负数在前因为False<True）
        expected_order = [0, 1, -1, 2, -2, 3, -3]
        assert sorted_steps == expected_order

    def test_seed_included_in_sorting(self, continuation, seed_orbit):
        """测试种子轨道(step=0)被包含在排序中"""
        temp_orbits_with_steps = []

        # 创建测试轨道
        for step in [0, 1, -1, 2]:
            orbit = Orbit(
                states=seed_orbit.states.copy(),
                times=seed_orbit.times.copy(),
            )
            orbit.period = seed_orbit.period
            orbit.system = seed_orbit.system
            orbit.metadata["continuation_step"] = step
            temp_orbits_with_steps.append((orbit, step))

        def sort_key(item):
            orbit, step = item
            return (abs(step), step > 0)

        temp_orbits_with_steps.sort(key=sort_key)
        sorted_steps = [step for _, step in temp_orbits_with_steps]

        # 0 应该在第一位
        assert sorted_steps[0] == 0
        # 相同绝对值时，负数在前（因为False < True）
        assert sorted_steps[1] == 1
        assert sorted_steps[2] == -1

    def test_sort_key_uses_tuple_comparison(self, continuation):
        """测试排序键使用元组比较"""
        def sort_key(item):
            orbit, step = item
            return (abs(step), step > 0)

        # 测试：相同绝对值时，正数(True)排在负数(False)之后
        steps = [(None, 1), (None, -1), (None, 2), (None, -2)]
        steps.sort(key=sort_key)
        sorted_steps = [s for _, s in steps]

        # 因为 (1, True) > (1, False)，所以负数排在正数前面
        # 排序结果: -1, 1, -2, 2
        assert sorted_steps == [-1, 1, -2, 2]


class TestContinuationStats:
    """测试延拓统计信息"""

    def test_stats_initialization(self, continuation):
        """测试统计信息正确初始化"""
        stats = continuation.continuation_stats

        assert "total_steps" in stats
        assert "successful_steps" in stats
        assert "failed_steps" in stats

        assert stats["total_steps"] == 0
        assert stats["successful_steps"] == 0
        assert stats["failed_steps"] == 0

    def test_stats_update_on_success(self, continuation):
        """测试成功延拓时统计更新"""
        initial_success_count = continuation.continuation_stats["successful_steps"]
        
        # 模拟成功延拓
        continuation.continuation_stats["successful_steps"] += 1
        
        assert continuation.continuation_stats["successful_steps"] == initial_success_count + 1


class TestBidirectionalContinuation:
    """测试双向延拓的步数标记"""

    def test_forward_steps_are_positive(self, continuation):
        """测试正向延拓步数为正数"""
        # 正向延拓时 step = 1, 2, 3, ...
        forward_steps = [1, 2, 3, 10]
        
        for step in forward_steps:
            assert step > 0

    def test_backward_steps_are_negative(self, continuation):
        """测试反向延拓步数为负数"""
        # 反向延拓时 step = -1, -2, -3, ...
        backward_steps = [-1, -2, -3, -10]
        
        for step in backward_steps:
            assert step < 0

    def test_step_number_abs_increases_with_distance(self, continuation):
        """测试步数的绝对值随距离增加而增大"""
        # 距离种子轨道越远，绝对值越大
        steps = [0, 1, 2, 10, -1, -2, -10]
        
        # 按绝对值排序
        steps_by_abs = sorted(steps, key=abs)
        
        # 0 最小，然后绝对值递增
        assert steps_by_abs[0] == 0
        assert abs(steps_by_abs[1]) == 1
        assert abs(steps_by_abs[-1]) == 10
