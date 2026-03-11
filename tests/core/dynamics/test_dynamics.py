"""
CR3BP_Dynamics 类测试

测试动力学模型的核心功能，包括运动方程、轨迹传播、状态转移矩阵等。
"""

import numpy as np
import pytest

from e2m2e.core import CR3BP_System, CR3BP_Dynamics


class TestEquationsOfMotion:
    """测试运动方程"""

    def test_equations_of_motion_shape(self):
        """测试运动方程返回维度正确"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        dynamics = CR3BP_Dynamics(system)

        state = [0.5, 0, 0, 0, 0.5, 0]
        deriv = dynamics.equations_of_motion(0, state)

        assert len(deriv) == 6

    def test_equations_of_motion_values(self):
        """测试运动方程计算值正确"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        dynamics = CR3BP_Dynamics(system)

        # 在原点处的加速度（科里奥利项起作用）
        state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        deriv = dynamics.equations_of_motion(0, state)

        # 位置导数应为速度
        assert deriv[0] == 0.0
        assert deriv[1] == 0.0
        assert deriv[2] == 0.0


class TestPropagate:
    """测试轨迹传播"""

    def test_propagate_basic(self):
        """测试基本轨迹传播"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        dynamics = CR3BP_Dynamics(system)

        state = [0.5, 0, 0, 0, 0.5, 0]
        result = dynamics.propagate(state, [0, 1.0])

        assert "time" in result
        assert "states" in result
        assert "jacobi" in result
        assert "jacobi_error" in result

    def test_propagate_with_stm(self):
        """测试带状态转移矩阵的轨迹传播"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        dynamics = CR3BP_Dynamics(system)

        state = [0.5, 0, 0, 0, 0.5, 0]
        result = dynamics.propagate(state, [0, 1.0], with_stm=True)

        assert "stm" in result
        assert result["stm"].shape[-2:] == (6, 6)


class TestComputeJacobiConstant:
    """测试Jacobi常数计算"""

    def test_jacobi_constant(self):
        """测试Jacobi常数计算"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        dynamics = CR3BP_Dynamics(system)

        state = [0.5, 0, 0, 0, 0.5, 0]
        C = dynamics.compute_jacobi_constant(state)

        assert isinstance(C, float)
        assert C > 0

    def test_jacobi_constant_at_lagrange_point(self):
        """测试平动点处的Jacobi常数"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        system.compute_libration_points()
        dynamics = CR3BP_Dynamics(system)

        # L1点处速度为零
        L1_state = list(system.L1) + [0, 0, 0]
        C = dynamics.compute_jacobi_constant(L1_state)

        assert isinstance(C, float)
        assert C > 2.5  # 平动点处Jacobi常数通常较大


class TestCheckCrossSection:
    """测试截面检测"""

    def test_check_cross_section(self):
        """测试截面穿越检测"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        dynamics = CR3BP_Dynamics(system)

        state = [0.5, 0, 0, 0, 0.5, 0]

        # 在x=0.5处
        is_crossing = dynamics.check_cross_section(state, "x", 0.5)
        assert is_crossing == True

        # 不在x=0处
        is_crossing = dynamics.check_cross_section(state, "x", 0.0)
        assert is_crossing == False


class TestStateTransitionMatrix:
    """测试状态转移矩阵"""

    def test_compute_stm(self):
        """测试状态转移矩阵计算"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        dynamics = CR3BP_Dynamics(system)

        initial_state = [0.5, 0, 0, 0, 0.5, 0]
        stm = dynamics.compute_state_transition_matrix(initial_state, 1.0)

        assert stm.shape == (6, 6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
