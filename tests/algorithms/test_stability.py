"""StabilityAnalysis 稳定性分析测试。

覆盖单值矩阵、Floquet 乘子、稳定性分类、分岔分析与枚举值。
"""

import numpy as np
import pytest

from e2m2e.algorithms import StabilityAnalysis
from e2m2e.algorithms.stability import BifurcationType, StabilityType
from e2m2e.core import CR3BP_Dynamics, CR3BP_System, Orbit


class TestStabilityAnalysisCreation:
    """测试稳定性分析创建"""

    def test_stability_analysis_creation(self):
        """测试稳定性分析创建"""
        states = np.random.randn(10, 6)
        times = np.linspace(0, 1, 10)

        orbit = Orbit(states, times)

        stability = StabilityAnalysis(orbit)

        assert stability.orbit is orbit
        assert stability.dynamics is None

    def test_stability_analysis_with_dynamics(self):
        """测试带动力学模型的稳定性分析"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        dynamics = CR3BP_Dynamics(system)

        states = np.random.randn(10, 6)
        times = np.linspace(0, 1, 10)

        orbit = Orbit(states, times)

        stability = StabilityAnalysis(orbit, dynamics)

        assert stability.dynamics is dynamics


class TestComputeMonodromy:
    """测试单值矩阵计算"""

    def test_compute_monodromy_requires_dynamics(self):
        """测试计算单值矩阵需要动力学模型"""
        # 创建一个有周期的轨道
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
        orbit.period = 1.0  # 设置周期

        stability = StabilityAnalysis(orbit)

        # 尝试计算单值矩阵应该报错
        with pytest.raises(ValueError):
            stability.compute_monodromy()


class TestComputeFloquetMultipliers:
    """测试Floquet乘子计算"""

    def test_floquet_multipliers_requires_monodromy(self):
        """测试Floquet乘子计算需要先有单值矩阵"""
        # 创建一个有周期的轨道
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

        stability = StabilityAnalysis(orbit)

        # 尝试计算Floquet乘子应该触发自动计算单值矩阵
        # 由于没有dynamics，应该会报错
        with pytest.raises(ValueError):
            stability.compute_floquet_multipliers()


class TestStabilityClassification:
    """测试稳定性分类"""

    def test_classify_orbit_requires_eigenvalues(self):
        """测试轨道分类需要先有特征值"""
        states = np.random.randn(10, 6)
        times = np.linspace(0, 1, 10)

        orbit = Orbit(states, times)

        stability = StabilityAnalysis(orbit)

        # 尝试分类应该会触发特征值计算
        with pytest.raises(ValueError):
            stability.classify_orbit()


class TestAnalyzeBifurcation:
    """测试分岔分析"""

    def test_analyze_bifurcation_initialization(self):
        """测试分岔分析初始化"""
        states = np.random.randn(10, 6)
        times = np.linspace(0, 1, 10)

        orbit = Orbit(states, times)

        stability = StabilityAnalysis(orbit)

        assert stability.bifurcation_type == BifurcationType.NONE
        assert not stability.bifurcation_detected


class TestStabilityIndices:
    """测试稳定性指数"""

    def test_stability_indices_initialization(self):
        """测试稳定性指数初始化"""
        states = np.random.randn(10, 6)
        times = np.linspace(0, 1, 10)

        orbit = Orbit(states, times)

        stability = StabilityAnalysis(orbit)

        assert stability.stability_indices["nu1"] is None
        assert stability.stability_indices["nu2"] is None
        assert stability.stability_indices["broucke"] is None


class TestNumericalErrors:
    """测试数值误差跟踪"""

    def test_numerical_errors_initialization(self):
        """测试数值误差初始化"""
        states = np.random.randn(10, 6)
        times = np.linspace(0, 1, 10)

        orbit = Orbit(states, times)

        stability = StabilityAnalysis(orbit)

        assert stability.numerical_errors["determinant_error"] is None
        assert stability.numerical_errors["symplectic_error"] is None


class TestStringRepresentations:
    """测试字符串表示"""

    def test_str_representation(self):
        """测试字符串表示"""
        states = np.random.randn(10, 6)
        times = np.linspace(0, 1, 10)

        orbit = Orbit(states, times)

        stability = StabilityAnalysis(orbit)

        str_repr = str(stability)

        assert "StabilityAnalysis" in str_repr

    def test_repr_representation(self):
        """测试详细字符串表示"""
        states = np.random.randn(10, 6)
        times = np.linspace(0, 1, 10)

        orbit = Orbit(states, times)

        stability = StabilityAnalysis(orbit)

        repr_str = repr(stability)

        assert "StabilityAnalysis" in repr_str


class TestStabilityType:
    """测试稳定性类型枚举"""

    def test_stability_type_values(self):
        """测试稳定性类型枚举值"""
        assert StabilityType.STABLE.value == "stable"
        assert StabilityType.UNSTABLE.value == "unstable"
        assert StabilityType.MARGINALLY_STABLE.value == "marginally_stable"
        assert StabilityType.HYPERBOLIC.value == "hyperbolic"
        assert StabilityType.ELLIPTIC.value == "elliptic"
        assert StabilityType.PARABOLIC.value == "parabolic"


class TestBifurcationType:
    """测试分岔类型枚举"""

    def test_bifurcation_type_values(self):
        """测试分岔类型枚举值"""
        assert BifurcationType.NONE.value == "none"
        assert BifurcationType.PERIOD_DOUBLING.value == "period_doubling"
        assert BifurcationType.SADDLE_NODE.value == "saddle_node"
        assert BifurcationType.TORUS.value == "torus"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
