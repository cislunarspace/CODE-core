"""保持点/安全分析测试（主题 3）。"""

import numpy as np
import pytest

from e2m2e.proximity.safety import (
    SafetyRegion,
    check_passive_safety,
    max_collision_probability,
)


class TestSafetyRegion:
    """安全域几何。"""

    def test_sphere_contains(self):
        """球安全域包含判断。"""
        region = SafetyRegion(
            kind="keep_out",
            shape="sphere",
            center=np.array([0.0, 0.0, 0.0]),
            radius=1.0,
        )
        assert region.contains(np.array([0.5, 0.0, 0.0]))
        assert not region.contains(np.array([2.0, 0.0, 0.0]))

    def test_sphere_distance(self):
        """球安全域距离。"""
        region = SafetyRegion(
            kind="keep_out",
            shape="sphere",
            center=np.array([0.0, 0.0, 0.0]),
            radius=1.0,
        )
        assert region.distance_to(np.array([2.0, 0.0, 0.0])) == pytest.approx(1.0)
        assert region.distance_to(np.array([0.5, 0.0, 0.0])) == pytest.approx(-0.5)

    def test_cone_contains(self):
        """锥安全域包含判断。"""
        region = SafetyRegion(
            kind="approach",
            shape="cone",
            center=np.array([0.0, 0.0, 0.0]),
            radius=1.0,
            cone_axis=np.array([1.0, 0.0, 0.0]),
            cone_half_angle=np.pi / 6,  # 30°
        )
        # 锥内：轴向 1.0，径向 0.3 < 1.0 * tan(30°) ≈ 0.577
        assert region.contains(np.array([1.0, 0.3, 0.0]))
        # 锥外：径向 0.8 > 0.577
        assert not region.contains(np.array([1.0, 0.8, 0.0]))
        # 负轴向：锥外
        assert not region.contains(np.array([-1.0, 0.0, 0.0]))


class TestPassiveSafety:
    """被动安全校验。"""

    def test_safe_trajectory(self):
        """全程安全轨迹。"""
        region = SafetyRegion(
            kind="keep_out",
            shape="sphere",
            center=np.array([0.0, 0.0, 0.0]),
            radius=1.0,
        )
        times = np.linspace(0, 10, 100)
        # 轨迹始终在球外（半径 2.0）
        positions = np.column_stack([2.0 * np.ones(100), np.zeros(100), np.zeros(100)])
        report = check_passive_safety(times, positions, region)
        assert report.safe
        assert len(report.violation_intervals) == 0
        assert report.min_distance > 0

    def test_violation_detected(self):
        """违背检测：轨迹穿过禁区。"""
        region = SafetyRegion(
            kind="keep_out",
            shape="sphere",
            center=np.array([0.0, 0.0, 0.0]),
            radius=1.0,
        )
        times = np.linspace(0, 10, 100)
        # 轨迹从球外进入球内再出来
        x = np.linspace(2.0, -2.0, 100)
        positions = np.column_stack([x, np.zeros(100), np.zeros(100)])
        report = check_passive_safety(times, positions, region)
        assert not report.safe
        assert len(report.violation_intervals) == 1
        assert report.min_distance < 0  # 进入球内

    def test_collision_probability(self):
        """碰撞概率计算。"""
        region = SafetyRegion(
            kind="keep_out",
            shape="sphere",
            center=np.array([0.0, 0.0, 0.0]),
            radius=1.0,
        )
        times = np.linspace(0, 10, 100)
        x = np.linspace(2.0, -2.0, 100)
        positions = np.column_stack([x, np.zeros(100), np.zeros(100)])
        cov = np.eye(3) * 0.01  # 0.01 km² 协方差
        report = check_passive_safety(times, positions, region, cov=cov)
        assert not report.safe
        assert report.collision_probability > 0.0


class TestChanFormula:
    """Chan 最大碰撞概率公式。"""

    def test_zero_distance(self):
        """零距离：最大碰撞概率。"""
        cov = np.eye(3) * 0.01
        pc = max_collision_probability(0.0, cov, 5.0, 5.0)
        assert 0.0 < pc <= 1.0

    def test_far_distance(self):
        """远距离：碰撞概率趋近零。"""
        cov = np.eye(3) * 0.01
        pc = max_collision_probability(100.0, cov, 5.0, 5.0)
        assert pc < 1e-10

    def test_larger_cov_higher_pc(self):
        """协方差越大，碰撞概率越高。"""
        pc_small = max_collision_probability(1.0, np.eye(3) * 0.01, 5.0, 5.0)
        pc_large = max_collision_probability(1.0, np.eye(3) * 0.1, 5.0, 5.0)
        assert pc_large > pc_small
