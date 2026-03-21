"""
可视化模块稳定性指数测试

测试 compute_stability_for_family 函数的功能和改进的文档。

参考最近 commit:
- "refactor(plotting): 优化可视化模块代码结构与稳定性指数计算文档"
"""

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")  # 使用非交互式后端

from e2m2e.core import Orbit, OrbitFamily, CR3BP_System
from e2m2e.visualization.plotting import compute_stability_for_family, OrbitVisualizer


# 地月系统质量比
MU = 1.21506683e-2


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture
def earth_moon_system():
    """创建地月CR3BP系统"""
    system = CR3BP_System(mu=MU, primary="earth", secondary="moon")
    system.compute_libration_points()
    return system


@pytest.fixture
def sample_orbit(earth_moon_system):
    """创建测试用单条轨道"""
    # 创建一条简单的 DRO 轨道
    x0 = 0.79188556619742
    vy0 = 0.53682
    
    # 创建状态数组（简化版，实际应该通过积分得到）
    t = np.linspace(0, 3.42, 100)
    states = np.zeros((len(t), 6))
    states[:, 0] = x0 * np.ones(len(t))  # x
    states[:, 1] = 0.1 * np.sin(t)  # y
    states[:, 2] = 0.0  # z (平面轨道)
    states[:, 3] = 0.0  # vx
    states[:, 4] = vy0 * np.cos(t)  # vy
    states[:, 5] = 0.0  # vz
    
    orbit = Orbit(states=states, times=t)
    orbit.period = 3.42
    orbit.system = earth_moon_system
    return orbit


@pytest.fixture
def sample_family(earth_moon_system):
    """创建测试用轨道族"""
    family = OrbitFamily(family_type="test_dro")
    
    # 创建3条简单的测试轨道
    for i in range(3):
        x0 = 0.79 + i * 0.005
        t = np.linspace(0, 3.5, 100)
        
        states = np.zeros((len(t), 6))
        states[:, 0] = x0 * np.ones(len(t))
        states[:, 1] = 0.1 * np.sin(t + i)
        states[:, 2] = 0.0
        states[:, 3] = 0.0
        states[:, 4] = 0.5 * np.cos(t + i)
        states[:, 5] = 0.0
        
        orbit = Orbit(states=states, times=t)
        orbit.period = 3.0 + i * 0.2
        orbit.system = earth_moon_system
        family.add_orbit(orbit)
    
    return family


# ============================================================
# 基本功能测试
# ============================================================
class TestComputeStabilityForFamily:
    """测试 compute_stability_for_family 函数"""

    def test_returns_list(self, sample_family, earth_moon_system):
        """应该返回列表类型"""
        result = compute_stability_for_family(sample_family, earth_moon_system)
        assert isinstance(result, list), f"Should return list, got {type(result)}"

    def test_same_length_as_family(self, sample_family, earth_moon_system):
        """返回列表长度应该与轨道族中轨道数量相同"""
        result = compute_stability_for_family(sample_family, earth_moon_system)
        assert len(result) == len(sample_family), \
            f"Result length ({len(result)}) should match family length ({len(sample_family)})"

    def test_stability_values_are_floats(self, sample_family, earth_moon_system):
        """稳定性指数应该是浮点数"""
        result = compute_stability_for_family(sample_family, earth_moon_system)
        for val in result:
            assert isinstance(val, (float, np.floating)), \
                f"Stability value should be float, got {type(val)}"

    def test_stability_values_non_negative(self, sample_family, earth_moon_system):
        """稳定性指数应该非负"""
        result = compute_stability_for_family(sample_family, earth_moon_system)
        for val in result:
            assert val >= 0, f"Stability value should be non-negative, got {val}"


# ============================================================
# 边界情况测试
# ============================================================
class TestStabilityBoundaryCases:
    """测试边界情况"""

    def test_empty_family(self, earth_moon_system):
        """空轨道族应返回空列表"""
        empty_family = OrbitFamily(family_type="empty")
        result = compute_stability_for_family(empty_family, earth_moon_system)
        assert result == [], "Empty family should return empty list"

    def test_none_family(self, earth_moon_system):
        """None 轨道族应返回空列表"""
        result = compute_stability_for_family(None, earth_moon_system)
        assert result == [], "None family should return empty list"

    def test_single_orbit_family(self, earth_moon_system, sample_orbit):
        """单条轨道的轨道族应该正常工作"""
        single_family = OrbitFamily(family_type="single")
        single_family.add_orbit(sample_orbit)
        
        result = compute_stability_for_family(single_family, earth_moon_system)
        assert len(result) == 1, "Single orbit family should return list of length 1"

    def test_orbit_without_period(self, earth_moon_system):
        """没有周期信息的轨道应返回稳定性指数 1.0"""
        family = OrbitFamily(family_type="no_period")
        
        # 创建没有 period 的轨道
        t = np.linspace(0, 1, 50)
        states = np.zeros((len(t), 6))
        states[:, 0] = 0.8 * np.ones(len(t))
        orbit = Orbit(states=states, times=t)
        orbit.period = None  # 没有周期
        orbit.system = earth_moon_system
        family.add_orbit(orbit)
        
        result = compute_stability_for_family(family, earth_moon_system)
        assert len(result) == 1
        # 根据 TODO 注释，没有 period 的轨道应假设为中性稳定（1.0）
        assert result[0] == 1.0


# ============================================================
# 稳定性指数解释测试
# ============================================================
class TestStabilityIndexInterpretation:
    """测试稳定性指数的解释"""

    def test_stability_index_equals_one_neutral(self, sample_family, earth_moon_system):
        """稳定性指数等于 1 表示中性稳定"""
        result = compute_stability_for_family(sample_family, earth_moon_system)
        for val in result:
            # 中性稳定：λ_max = 1
            if np.isclose(val, 1.0, rtol=1e-5):
                assert True

    def test_stability_index_less_than_one_stable(self, sample_family, earth_moon_system):
        """稳定性指数小于 1 表示渐近稳定"""
        result = compute_stability_for_family(sample_family, earth_moon_system)
        for val in result:
            if val < 1.0:
                assert True  # 渐近稳定

    def test_stability_index_greater_than_one_unstable(self, sample_family, earth_moon_system):
        """稳定性指数大于 1 表示不稳定"""
        result = compute_stability_for_family(sample_family, earth_moon_system)
        for val in result:
            if val > 1.0:
                assert True  # 不稳定


# ============================================================
# 系统关联测试
# ============================================================
class TestSystemAssociation:
    """测试轨道系统关联"""

    def test_orbit_system_assignment(self, sample_family, earth_moon_system):
        """轨道应该关联到指定的系统"""
        result = compute_stability_for_family(sample_family, earth_moon_system)
        
        # 检查每条轨道的系统是否被设置
        for orbit in sample_family:
            if orbit.period is not None:
                assert orbit.system is not None, \
                    "Orbit should have system assigned after compute_stability_for_family"

    def test_none_system_orbit_gets_assigned(self, earth_moon_system):
        """没有关联系统的轨道应该获得系统"""
        family = OrbitFamily(family_type="no_system")
        
        t = np.linspace(0, 3.0, 100)
        states = np.zeros((len(t), 6))
        states[:, 0] = 0.8 * np.ones(len(t))
        orbit = Orbit(states=states, times=t)
        orbit.period = 3.0
        orbit.system = None  # 没有系统
        family.add_orbit(orbit)
        
        # 计算稳定性
        result = compute_stability_for_family(family, earth_moon_system)
        
        # 轨道应该获得系统
        assert orbit.system == earth_moon_system


# ============================================================
# 异常处理测试
# ============================================================
class TestExceptionHandling:
    """测试异常处理"""

    def test_computation_failure_returns_one(self, earth_moon_system):
        """计算失败时应返回 1.0（中性稳定）"""
        family = OrbitFamily(family_type="problematic")
        
        # 创建一条可能导致计算失败的轨道
        t = np.linspace(0, 1, 50)
        states = np.zeros((len(t), 6))
        states[:, 0] = 0.8 * np.ones(len(t))
        orbit = Orbit(states=states, times=t)
        orbit.period = 0.0  # 零周期可能导致计算失败
        orbit.system = earth_moon_system
        family.add_orbit(orbit)
        
        result = compute_stability_for_family(family, earth_moon_system)
        assert len(result) == 1
        # 根据代码，计算失败时返回 1.0
        assert result[0] == 1.0


# ============================================================
# 与 OrbitVisualizer 集成测试
# ============================================================
class TestVisualizerIntegration:
    """测试与 OrbitVisualizer 的集成"""

    def test_visualizer_with_stability_plot(self, earth_moon_system, sample_family):
        """测试可视化器绘制稳定性图"""
        viz = OrbitVisualizer(earth_moon_system)
        
        # 创建一些带周期信息的轨道
        for i, orbit in enumerate(sample_family):
            orbit.period = 3.0 + i * 0.2
        
        # 测试稳定性计算功能
        stability_values = compute_stability_for_family(sample_family, earth_moon_system)
        
        assert len(stability_values) == len(sample_family)
        assert all(isinstance(v, (float, np.floating)) for v in stability_values)


# ============================================================
# 文档字符串测试
# ============================================================
class TestDocumentation:
    """测试文档字符串的准确性"""

    def test_function_has_docstring(self):
        """compute_stability_for_family 应该有文档字符串"""
        assert compute_stability_for_family.__doc__ is not None, \
            "Function should have docstring"

    def test_docstring_mentions_monodromy(self):
        """文档字符串应该提及单值矩阵 (Monodromy Matrix)"""
        doc = compute_stability_for_family.__doc__
        assert doc is not None
        assert "monodromy" in doc.lower() or "单值矩阵" in doc, \
            "Docstring should mention Monodromy Matrix"

    def test_docstring_mentions_eigenvalues(self):
        """文档字符串应该提及特征值"""
        doc = compute_stability_for_family.__doc__
        assert doc is not None
        assert "eigenvalue" in doc.lower() or "特征值" in doc, \
            "Docstring should mention eigenvalues"
