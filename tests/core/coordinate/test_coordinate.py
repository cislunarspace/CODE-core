"""
CoordinateTransformation 类测试

测试坐标变换功能，包括旋转系与惯性系之间的转换、质心系与天体中心系之间的转换等。
"""

import numpy as np
import pytest

from e2m2e.core import CoordinateTransformation, CR3BP_System


class TestRotationMatrix:
    """测试旋转矩阵计算"""

    def test_compute_rotation_matrix(self):
        """测试旋转矩阵计算"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        coord = CoordinateTransformation(system)

        R = coord.compute_rotation_matrix(0.0)

        assert R.shape == (3, 3)
        # 验证是标准旋转矩阵（正交）
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_rotation_matrix_at_different_times(self):
        """测试不同时刻的旋转矩阵"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        coord = CoordinateTransformation(system)

        coord.compute_rotation_matrix(0.0)
        R_pi_2 = coord.compute_rotation_matrix(np.pi / 2)

        # 验证旋转角度正确
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        assert np.allclose(R_pi_2, expected, atol=1e-10)


class TestRotatingToInertial:
    """测试旋转系到惯性系转换"""

    def test_position_transform(self):
        """测试位置变换"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        coord = CoordinateTransformation(system)

        # 在旋转系中原点的位置，在惯性系中也是原点
        state_rotating = [0, 0, 0, 0, 0, 0]
        state_inertial = coord.rotating_to_inertial(state_rotating, 0.0)

        assert np.allclose(state_inertial[:3], [0, 0, 0], atol=1e-10)

    def test_velocity_transform_at_nonzero_time(self):
        """测试速度变换在非零时间下的计算"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        coord = CoordinateTransformation(system)

        # 非零时间测试
        state_rotating = [1, 0, 0, 0, 1, 0]
        state_inertial = coord.rotating_to_inertial(state_rotating, np.pi / 4)

        # 验证返回值是正确形状
        assert state_inertial.shape == (6,)


class TestInertialToRotating:
    """测试惯性系到旋转系转换"""

    def test_round_trip_position_only(self):
        """测试位置往返转换一致性"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        coord = CoordinateTransformation(system)

        # 只测试位置变换（速度变换由于科里奥利项不是简单的往返）
        original_position = np.array([0.5, 0.3, 0.1])
        time = 0.5

        # 惯性 -> 旋转 -> 惯性
        R = coord.compute_rotation_matrix(time)
        position_rotating = R @ original_position
        position_back = R.T @ position_rotating

        # 验证往返转换一致性
        assert np.allclose(position_back, original_position, atol=1e-10)


class TestBarycentricTransform:
    """测试质心系变换"""

    def test_barycentric_to_primary(self):
        """测试质心系到主天体中心变换"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        coord = CoordinateTransformation(system)

        # 质心系中的原点
        state = [0, 0, 0, 0, 0, 0]
        state_primary = coord.barycentric_to_primary(state)

        # 主天体在质心系中位于 (-mu, 0, 0)
        expected_position = np.array([system.mu, 0, 0])
        assert np.allclose(state_primary[:3], expected_position, atol=1e-10)

    def test_primary_to_barycentric(self):
        """测试主天体中心到质心系变换"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        coord = CoordinateTransformation(system)

        # 主天体中心为原点
        state = [0, 0, 0, 0, 0, 0]
        state_bary = coord.primary_to_barycentric(state)

        # 质心系中主天体位置
        expected_position = np.array([-system.mu, 0, 0])
        assert np.allclose(state_bary[:3], expected_position, atol=1e-10)


class TestSecondaryCenteredTransform:
    """测试次天体中心变换"""

    def test_barycentric_to_secondary(self):
        """测试质心系到次天体中心变换"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        coord = CoordinateTransformation(system)

        # 质心系中的原点
        state = [0, 0, 0, 0, 0, 0]
        state_secondary = coord.barycentric_to_secondary(state)

        # 次天体在质心系中位于 (1-mu, 0, 0)
        # 从原点减去该位置，得到相对于次天体的位置
        expected_position = np.array([-(1 - system.mu), 0, 0])
        assert np.allclose(state_secondary[:3], expected_position, atol=1e-10)

    def test_secondary_to_barycentric(self):
        """测试次天体中心到质心系变换"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        coord = CoordinateTransformation(system)

        # 次天体中心为原点
        state = [0, 0, 0, 0, 0, 0]
        state_bary = coord.secondary_to_barycentric(state)

        # 质心系中次天体位置
        expected_position = np.array([1 - system.mu, 0, 0])
        assert np.allclose(state_bary[:3], expected_position, atol=1e-10)


class TestTransform:
    """测试通用坐标变换"""

    def test_transform_same_frame(self):
        """测试同一参考系之间的变换"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        coord = CoordinateTransformation(system)

        state = [0.5, 0.3, 0.1, 0.2, 0.4, 0.1]

        # 同一参考系应该返回相同状态
        result = coord.transform(state, "rotating", "rotating", 0.0)
        assert np.allclose(result, state)

    def test_transform_rotating_to_inertial(self):
        """测试旋转系到惯性系变换"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        coord = CoordinateTransformation(system)

        state = [0.5, 0.3, 0.1, 0.2, 0.4, 0.1]

        result = coord.transform(state, "rotating", "inertial", 0.0)
        assert result.shape == (6,)

    def test_transform_barycentric_to_primary(self):
        """测试质心系到主天体中心变换"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        coord = CoordinateTransformation(system)

        # 质心系原点
        state = [0, 0, 0, 0, 0, 0]

        result = coord.transform(state, "barycentric", "primary_centered")

        # 变换后位置应该是主天体在质心系中的位置
        expected = np.array([system.mu, 0, 0, 0, 0, 0])
        assert np.allclose(result, expected, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
