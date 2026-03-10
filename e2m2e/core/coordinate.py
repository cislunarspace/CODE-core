"""
坐标变换模块

包含CoordinateTransformation类，用于在不同参考系之间转换轨道状态。
"""

from __future__ import annotations

import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy.typing as npt

from .system import CR3BP_System


class ReferenceFrame(Enum):
    """参考系枚举"""

    ROTATING = "rotating"  # 旋转系
    INERTIAL = "inertial"  # 惯性系
    BARYCENTRIC = "barycentric"  # 质心系
    PRIMARY_CENTERED = "primary_centered"  # 主天体中心系
    SECONDARY_CENTERED = "secondary_centered"  # 次天体中心系
    SYNODIC = "synodic"  # 会合系（同旋转系）


class CoordinateTransformation:
    """坐标系变换

    属性：
    - system: CR3BP_System对象
    - rotation_matrices: 旋转矩阵缓存

    方法：
    - __init__(system): 初始化变换器
    - rotating_to_inertial(state, time): 旋转系到惯性系
    - inertial_to_rotating(state, time): 惯性系到旋转系
    - barycentric_to_primary(state): 质心系到主天体中心
    - primary_to_barycentric(state): 主天体中心到质心系
    - compute_rotation_matrix(time): 计算旋转矩阵
    - transform_velocity(state, from_frame, to_frame): 速度变换
    """

    # 类属性
    VELOCITY_TRANSFORM_INCLUDE_CORIOLIS = True  # 速度变换是否包含科里奥利项
    CACHE_ROTATION_MATRICES = True  # 是否缓存旋转矩阵
    MAX_CACHE_SIZE = 1000  # 最大缓存大小

    def __init__(self, system: CR3BP_System) -> None:
        """初始化变换器

        参数：
        - system: CR3BP_System对象
        """
        # 关联系统
        self.system = system
        self.mu = system.mu if hasattr(system, "mu") else None

        # 旋转矩阵缓存
        self.rotation_matrices = {}  # 旋转矩阵缓存 {time: matrix}
        self.rotation_matrix_derivatives = {}  # 旋转矩阵导数缓存

        # 变换状态
        self.initialized = True  # 初始化完成标志

    def compute_rotation_matrix(self, time: float) -> npt.NDArray[np.floating]:
        """计算旋转矩阵

        参数：
        - time: 时间（无量纲）

        返回：
        - 旋转矩阵 (3x3)
        """
        # 检查缓存
        if self.CACHE_ROTATION_MATRICES and time in self.rotation_matrices:
            return self.rotation_matrices[time]

        # 计算旋转角度（假设平均角速度为1）
        angle = time  # 无量纲时间对应无量纲角度

        # 构建旋转矩阵（绕z轴旋转）
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        rotation_matrix = np.array(
            [[cos_angle, -sin_angle, 0], [sin_angle, cos_angle, 0], [0, 0, 1]]
        )

        # 计算旋转矩阵导数
        rotation_matrix_derivative = np.array(
            [[-sin_angle, -cos_angle, 0], [cos_angle, -sin_angle, 0], [0, 0, 0]]
        )

        # 缓存结果
        if self.CACHE_ROTATION_MATRICES:
            if len(self.rotation_matrices) >= self.MAX_CACHE_SIZE:
                # 移除最旧的缓存项
                oldest_key = next(iter(self.rotation_matrices))
                del self.rotation_matrices[oldest_key]
                del self.rotation_matrix_derivatives[oldest_key]

            self.rotation_matrices[time] = rotation_matrix
            self.rotation_matrix_derivatives[time] = rotation_matrix_derivative

        return rotation_matrix

    def rotating_to_inertial(
        self, state: npt.ArrayLike, time: float
    ) -> npt.NDArray[np.floating]:
        """旋转系到惯性系

        参数：
        - state: 状态向量 [x, y, z, vx, vy, vz]（旋转系）
        - time: 时间（无量纲）

        返回：
        - 惯性系状态向量
        """
        # 解包状态
        position = state[:3]
        velocity = state[3:]

        # 获取旋转矩阵及其导数
        R = self.compute_rotation_matrix(time)
        R_dot = self.rotation_matrix_derivatives[time]

        # 位置变换
        position_inertial = R.T @ position

        # 速度变换（包含科里奥利项）
        if self.VELOCITY_TRANSFORM_INCLUDE_CORIOLIS:
            # 旋转系速度 + 旋转效应
            velocity_inertial = R.T @ velocity + R_dot.T @ position
        else:
            velocity_inertial = R.T @ velocity

        return np.concatenate([position_inertial, velocity_inertial])

    def inertial_to_rotating(
        self, state: npt.ArrayLike, time: float
    ) -> npt.NDArray[np.floating]:
        """惯性系到旋转系

        参数：
        - state: 状态向量 [x, y, z, vx, vy, vz]（惯性系）
        - time: 时间（无量纲）

        返回：
        - 旋转系状态向量
        """
        # 解包状态
        position = state[:3]
        velocity = state[3:]

        # 获取旋转矩阵及其导数
        R = self.compute_rotation_matrix(time)
        R_dot = self.rotation_matrix_derivatives[time]

        # 位置变换
        position_rotating = R @ position

        # 速度变换（包含科里奥利项）
        if self.VELOCITY_TRANSFORM_INCLUDE_CORIOLIS:
            # 惯性系速度 - 旋转效应
            velocity_rotating = R @ velocity - R_dot @ position_rotating
        else:
            velocity_rotating = R @ velocity

        return np.concatenate([position_rotating, velocity_rotating])

    def barycentric_to_primary(self, state: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """质心系到主天体中心

        参数：
        - state: 状态向量 [x, y, z, vx, vy, vz]（质心系）

        返回：
        - 主天体中心系状态向量
        """
        if self.mu is None:
            raise ValueError("系统未初始化，无法进行坐标变换")

        # 解包状态
        position = state[:3]
        velocity = state[3:]

        # 主天体在质心系中的位置（在旋转系中位于(-mu, 0, 0)）
        primary_position = np.array([-self.mu, 0, 0])

        # 位置变换
        position_primary = position - primary_position

        # 速度变换（主天体在质心系中静止）
        velocity_primary = velocity

        return np.concatenate([position_primary, velocity_primary])

    def primary_to_barycentric(self, state: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """主天体中心到质心系

        参数：
        - state: 状态向量 [x, y, z, vx, vy, vz]（主天体中心系）

        返回：
        - 质心系状态向量
        """
        if self.mu is None:
            raise ValueError("系统未初始化，无法进行坐标变换")

        # 解包状态
        position = state[:3]
        velocity = state[3:]

        # 主天体在质心系中的位置
        primary_position = np.array([-self.mu, 0, 0])

        # 位置变换
        position_barycentric = position + primary_position

        # 速度变换
        velocity_barycentric = velocity

        return np.concatenate([position_barycentric, velocity_barycentric])

    def barycentric_to_secondary(self, state: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """质心系到次天体中心

        参数：
        - state: 状态向量 [x, y, z, vx, vy, vz]（质心系）

        返回：
        - 次天体中心系状态向量
        """
        if self.mu is None:
            raise ValueError("系统未初始化，无法进行坐标变换")

        # 解包状态
        position = state[:3]
        velocity = state[3:]

        # 次天体在质心系中的位置（在旋转系中位于(1-mu, 0, 0)）
        secondary_position = np.array([1 - self.mu, 0, 0])

        # 位置变换
        position_secondary = position - secondary_position

        # 速度变换
        velocity_secondary = velocity

        return np.concatenate([position_secondary, velocity_secondary])

    def secondary_to_barycentric(self, state: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """次天体中心到质心系

        参数：
        - state: 状态向量 [x, y, z, vx, vy, vz]（次天体中心系）

        返回：
        - 质心系状态向量
        """
        if self.mu is None:
            raise ValueError("系统未初始化，无法进行坐标变换")

        # 解包状态
        position = state[:3]
        velocity = state[3:]

        # 次天体在质心系中的位置
        secondary_position = np.array([1 - self.mu, 0, 0])

        # 位置变换
        position_barycentric = position + secondary_position

        # 速度变换
        velocity_barycentric = velocity

        return np.concatenate([position_barycentric, velocity_barycentric])

    def transform(
        self,
        state: npt.ArrayLike,
        from_frame: Union[ReferenceFrame, str],
        to_frame: Union[ReferenceFrame, str],
        time: float = 0.0,
    ) -> npt.NDArray[np.floating]:
        """通用坐标变换

        参数：
        - state: 状态向量
        - from_frame: 源参考系（ReferenceFrame枚举或字符串）
        - to_frame: 目标参考系（ReferenceFrame枚举或字符串）
        - time: 时间（仅对涉及旋转的变换需要）

        返回：
        - 变换后的状态向量
        """
        # 转换字符串为枚举
        if isinstance(from_frame, str):
            from_frame = ReferenceFrame(from_frame)
        if isinstance(to_frame, str):
            to_frame = ReferenceFrame(to_frame)

        # 如果源和目标相同，直接返回
        if from_frame == to_frame:
            return state

        # 处理变换链
        if from_frame == ReferenceFrame.ROTATING and to_frame == ReferenceFrame.INERTIAL:
            return self.rotating_to_inertial(state, time)
        elif from_frame == ReferenceFrame.INERTIAL and to_frame == ReferenceFrame.ROTATING:
            return self.inertial_to_rotating(state, time)
        elif (
            from_frame == ReferenceFrame.BARYCENTRIC and to_frame == ReferenceFrame.PRIMARY_CENTERED
        ):
            return self.barycentric_to_primary(state)
        elif (
            from_frame == ReferenceFrame.PRIMARY_CENTERED and to_frame == ReferenceFrame.BARYCENTRIC
        ):
            return self.primary_to_barycentric(state)
        elif (
            from_frame == ReferenceFrame.BARYCENTRIC
            and to_frame == ReferenceFrame.SECONDARY_CENTERED
        ):
            return self.barycentric_to_secondary(state)
        elif (
            from_frame == ReferenceFrame.SECONDARY_CENTERED
            and to_frame == ReferenceFrame.BARYCENTRIC
        ):
            return self.secondary_to_barycentric(state)
        else:
            # 对于更复杂的变换，可能需要中间步骤
            # 这里可以扩展支持更多变换组合
            raise NotImplementedError(f"不支持从 {from_frame} 到 {to_frame} 的变换")

    def __str__(self):
        """字符串表示"""
        return f"CoordinateTransformation(system={self.system})"

    def __repr__(self):
        """详细表示"""
        return (
            f"CoordinateTransformation(system={self.system}, "
            f"cache_size={len(self.rotation_matrices)})"
        )
