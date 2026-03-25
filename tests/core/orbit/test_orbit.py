"""
Orbit 类测试

测试轨道类的核心功能，包括轨道创建、属性计算、插值、保存/加载等。
"""

import numpy as np
import pytest
import os
import tempfile

from e2m2e.core import Orbit, CR3BP_System


class TestOrbitCreation:
    """测试轨道创建"""

    def test_orbit_creation(self):
        """测试轨道基本创建"""
        states = np.random.randn(10, 6)
        times = np.linspace(0, 1, 10)

        orbit = Orbit(states, times)

        assert orbit.states.shape == (10, 6)
        assert orbit.times.shape == (10,)

    def test_orbit_with_system(self):
        """测试带系统的轨道创建"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")

        states = np.random.randn(10, 6)
        times = np.linspace(0, 1, 10)

        orbit = Orbit(states, times, system=system)

        assert orbit.system is system

    def test_orbit_invalid_states(self):
        """测试无效状态维度"""
        with pytest.raises(ValueError):
            # 状态维度不正确
            states = np.random.randn(10, 5)  # 应该是6维
            times = np.linspace(0, 1, 10)
            Orbit(states, times)

    def test_orbit_times_mismatch(self):
        """测试时间与状态长度不匹配"""
        with pytest.raises(ValueError):
            states = np.random.randn(10, 6)
            times = np.linspace(0, 1, 5)  # 长度不匹配
            Orbit(states, times)


class TestComputeBasicProperties:
    """测试基本属性计算"""

    def test_jacobi_constant_computed(self):
        """测试Jacobi常数计算"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")

        states = np.random.randn(10, 6)
        times = np.linspace(0, 1, 10)

        orbit = Orbit(states, times, system=system)

        assert orbit.jacobi_constants is not None
        assert len(orbit.jacobi_constants) == 10

    def test_amplitudes_computed(self):
        """测试振幅计算"""
        states = np.random.randn(10, 6)
        times = np.linspace(0, 1, 10)

        orbit = Orbit(states, times)

        assert "x" in orbit.amplitudes
        assert "y" in orbit.amplitudes
        assert "z" in orbit.amplitudes

    def test_extrema_computed(self):
        """测试极值计算"""
        states = np.random.randn(10, 6)
        times = np.linspace(0, 1, 10)

        orbit = Orbit(states, times)

        assert "x_max" in orbit.extrema
        assert "x_min" in orbit.extrema


class TestSaveAndLoad:
    """测试保存和加载"""

    def test_save_to_file(self):
        """测试保存轨道数据"""
        states = np.random.randn(10, 6)
        times = np.linspace(0, 1, 10)

        orbit = Orbit(states, times)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_file = f.name

        try:
            orbit.save_to_file(temp_file)

            # 检查文件是否创建
            assert os.path.exists(temp_file)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_load_from_file(self):
        """测试加载轨道数据"""
        # 创建并保存轨道
        states = np.random.randn(10, 6)
        times = np.linspace(0, 1, 10)

        orbit = Orbit(states, times)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_file = f.name

        try:
            orbit.save_to_file(temp_file)

            # 加载轨道
            loaded_orbit = Orbit.load_from_file(temp_file)

            assert loaded_orbit.states.shape == orbit.states.shape
            assert np.allclose(loaded_orbit.times, orbit.times)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


class TestOrbitGeometricFeatures:
    """测试轨道几何特征"""

    def test_periodicity_check(self):
        """测试周期性检测"""
        # 创建近似周期轨道
        states = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        orbit = Orbit(states, times)

        # 检查轨道属性
        assert orbit.period is not None


class TestOrbitFamilyType:
    """测试轨道族类型"""

    def test_set_family_type(self):
        """测试设置轨道族类型"""
        states = np.random.randn(10, 6)
        times = np.linspace(0, 1, 10)

        orbit = Orbit(states, times)

        # 设置轨道族类型
        orbit.family_type = "halo"

        assert orbit.family_type == "halo"

    def test_invalid_family_type(self):
        """测试无效轨道族类型"""
        states = np.random.randn(10, 6)
        times = np.linspace(0, 1, 10)

        orbit = Orbit(states, times)

        # 应该不设置无效类型（或者可以被设置为任何字符串）
        orbit.family_type = "invalid_type"

        assert orbit.family_type == "invalid_type"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
