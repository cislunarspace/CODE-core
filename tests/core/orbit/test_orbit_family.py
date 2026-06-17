"""OrbitFamily 类测试。

覆盖轨道族创建、属性计算、添加轨道、保存/加载与系统关联。
"""

import os
import tempfile

import numpy as np
import pytest

from e2m2e.core import CR3BP_System, Orbit, OrbitFamily


class TestOrbitFamilyCreation:
    """测试轨道族创建"""

    def test_empty_family_creation(self):
        """测试创建空轨道族"""
        family = OrbitFamily()

        assert len(family) == 0
        assert family.family_type is None

    def test_family_creation_with_type(self):
        """测试带类型的轨道族创建"""
        family = OrbitFamily(family_type="halo")

        assert family.family_type == "halo"
        assert len(family) == 0

    def test_family_creation_with_orbits(self):
        """测试带轨道列表的轨道族创建"""
        # 创建测试轨道
        states = np.random.randn(10, 6)
        times = np.linspace(0, 1, 10)
        orbit1 = Orbit(states, times)
        orbit1.period = 1.0

        states2 = np.random.randn(10, 6)
        times2 = np.linspace(0, 2, 10)
        orbit2 = Orbit(states2, times2)
        orbit2.period = 2.0

        family = OrbitFamily(orbits=[orbit1, orbit2], family_type="lyapunov")

        assert len(family.orbits) == 2
        assert family.family_type == "lyapunov"


class TestOrbitFamilyProperties:
    """测试轨道族属性"""

    @pytest.fixture
    def sample_family(self):
        """创建测试用轨道族"""
        family = OrbitFamily(family_type="halo")

        # 添加周期轨道
        for i in range(5):
            states = np.random.randn(20, 6)
            times = np.linspace(0, 1 + i * 0.1, 20)
            orbit = Orbit(states, times)
            orbit.period = 1.0 + i * 0.1
            orbit.family_type = "halo"
            orbit.amplitudes = {"x": 0.1 + i * 0.02, "z": 0.05 + i * 0.01}
            family.add_orbit(orbit)

        return family

    def test_periods_property(self, sample_family):
        """测试 periods 属性"""
        periods = sample_family.get_periods()

        assert len(periods) == 5
        assert np.allclose(periods, [1.0, 1.1, 1.2, 1.3, 1.4])

    def test_states_property(self, sample_family):
        """测试 states 属性"""
        states = sample_family.get_states()

        assert states.shape == (5, 6)

    def test_get_periods_method(self, sample_family):
        """测试 get_periods 方法"""
        periods = sample_family.get_periods()

        assert len(periods) == 5

    def test_get_amplitudes(self, sample_family):
        """测试获取各轨道振幅"""
        amplitudes_x = [orbit.amplitudes.get("x", 0) for orbit in sample_family]

        assert len(amplitudes_x) == 5
        assert np.allclose(amplitudes_x, [0.1, 0.12, 0.14, 0.16, 0.18])


class TestOrbitFamilyAccess:
    """测试轨道族访问方法"""

    @pytest.fixture
    def sample_family(self):
        """创建测试用轨道族"""
        family = OrbitFamily(family_type="lyapunov")

        for i in range(3):
            states = np.random.randn(20, 6)
            times = np.linspace(0, 1, 20)
            orbit = Orbit(states, times)
            orbit.period = float(i + 1)
            family.add_orbit(orbit)

        return family

    def test_add_orbit(self, sample_family):
        """测试添加轨道"""
        assert len(sample_family) == 3

        new_states = np.random.randn(20, 6)
        new_times = np.linspace(0, 1, 20)
        new_orbit = Orbit(new_states, new_times)
        new_orbit.period = 4.0

        sample_family.add_orbit(new_orbit)

        assert len(sample_family) == 4

    def test_get_orbit(self, sample_family):
        """测试获取单个轨道"""
        orbit = sample_family[0]
        assert orbit is not None
        assert orbit.period == 1.0

    def test_indexing(self, sample_family):
        """测试索引访问"""
        assert sample_family[0].period == 1.0
        assert sample_family[1].period == 2.0
        assert sample_family[2].period == 3.0

    def test_iteration(self, sample_family):
        """测试迭代"""
        periods = [orbit.period for orbit in sample_family]
        assert periods == [1.0, 2.0, 3.0]


class TestOrbitFamilyPersistence:
    """测试轨道族持久化"""

    @pytest.fixture
    def sample_family(self):
        """创建测试用轨道族"""
        family = OrbitFamily(family_type="halo")

        for i in range(3):
            states = np.random.randn(20, 6)
            times = np.linspace(0, 1 + i * 0.5, 20)
            orbit = Orbit(states, times)
            orbit.period = 1.0 + i * 0.5
            orbit.family_type = "halo"
            orbit.is_periodic = True
            orbit.amplitudes = {"x": 0.1 + i * 0.05, "z": 0.05}
            family.add_orbit(orbit)

        return family

    def test_save_and_load(self, sample_family):
        """测试保存和加载"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            # 保存
            sample_family.save_to_file(temp_file)

            # 加载
            loaded_family = OrbitFamily.load_from_file(temp_file)

            # 验证
            assert loaded_family.family_type == sample_family.family_type
            assert len(loaded_family.orbits) == len(sample_family.orbits)
            assert len(loaded_family.get_periods()) == len(sample_family.get_periods())
            assert np.allclose(loaded_family.get_periods(), sample_family.get_periods())

        finally:
            # 清理
            if os.path.exists(temp_file):
                os.remove(temp_file)


class TestOrbitFamilyWithSystem:
    """测试带系统的轨道族"""

    def test_jacobi_constants_with_system(self):
        """测试Jacobi常数计算"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")

        # 创建轨道时关联系统
        for i in range(3):
            # 创建在L1附近的测试状态
            state = np.array([0.8 - i * 0.01, 0, 0, 0, 0.1 + i * 0.01, 0])
            states = np.tile(state, (20, 1))
            times = np.linspace(0, 1, 20)

            orbit = Orbit(states, times, system=system)
            orbit.period = 1.0 + i * 0.1

        # 创建带系统的轨道族
        family = OrbitFamily(family_type="halo", system=system)

        # 重新创建轨道并添加到族中
        for i in range(3):
            state = np.array([0.8 - i * 0.01, 0, 0, 0, 0.1 + i * 0.01, 0])
            states = np.tile(state, (20, 1))
            times = np.linspace(0, 1, 20)
            orbit = Orbit(states, times, system=system)
            orbit.period = 1.0 + i * 0.1
            family.add_orbit(orbit)

        # 获取Jacobi常数
        jacobi = family.get_jacobi_constants()

        assert len(jacobi) == 3
        # 所有值应该是有效的数值
        assert not np.any(np.isnan(jacobi))
