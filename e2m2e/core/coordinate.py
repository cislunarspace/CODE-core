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
    """参考系枚举

    Attributes:
        ROTATING: 旋转系
        INERTIAL: 惯性系
        BARYCENTRIC: 质心系
        PRIMARY_CENTERED: 主天体中心系
        SECONDARY_CENTERED: 次天体中心系
        SYNODIC: 会合系（同旋转系）
    """

    ROTATING = "rotating"
    INERTIAL = "inertial"
    BARYCENTRIC = "barycentric"
    PRIMARY_CENTERED = "primary_centered"
    SECONDARY_CENTERED = "secondary_centered"
    SYNODIC = "synodic"


class CoordinateTransformation:
    """坐标系变换器

    在CR3BP系统的不同参考系之间进行状态向量变换，支持旋转系/惯性系、
    质心系/天体中心系之间的转换，并内置旋转矩阵缓存机制。

    Attributes:
        system: 关联的CR3BP_System对象
        mu: 系统的质量参数
        rotation_matrices: 旋转矩阵缓存，键为时间，值为3x3旋转矩阵
        rotation_matrix_derivatives: 旋转矩阵导数缓存
        initialized: 初始化完成标志
    """

    VELOCITY_TRANSFORM_INCLUDE_CORIOLIS = True
    CACHE_ROTATION_MATRICES = True
    MAX_CACHE_SIZE = 1000

    def __init__(self, system: CR3BP_System) -> None:
        """初始化变换器

        Args:
            system: CR3BP_System对象，提供质量参数等信息
        """
        self.system = system
        self.mu = system.mu if hasattr(system, "mu") else None
        self.rotation_matrices = {}
        self.rotation_matrix_derivatives = {}
        self.initialized = True

    def compute_rotation_matrix(self, time: float) -> npt.NDArray[np.floating]:
        """计算给定时刻的旋转矩阵及其导数

        Args:
            time: 时间（无量纲）

        Returns:
            3x3旋转矩阵
        """
        if self.CACHE_ROTATION_MATRICES and time in self.rotation_matrices:
            return self.rotation_matrices[time]

        # 计算旋转角度（假设平均角速度为1）
        angle = time

        # 构建旋转矩阵（绕z轴旋转）
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        rotation_matrix = np.array(
            [[cos_angle, -sin_angle, 0], [sin_angle, cos_angle, 0], [0, 0, 1]]
        )

        rotation_matrix_derivative = np.array(
            [[-sin_angle, -cos_angle, 0], [cos_angle, -sin_angle, 0], [0, 0, 0]]
        )

        if self.CACHE_ROTATION_MATRICES:
            if len(self.rotation_matrices) >= self.MAX_CACHE_SIZE:
                oldest_key = next(iter(self.rotation_matrices))
                del self.rotation_matrices[oldest_key]
                del self.rotation_matrix_derivatives[oldest_key]

            self.rotation_matrices[time] = rotation_matrix
            self.rotation_matrix_derivatives[time] = rotation_matrix_derivative

        return rotation_matrix

    def rotating_to_inertial(self, state: npt.ArrayLike, time: float) -> npt.NDArray[np.floating]:
        """将状态向量从旋转系转换到惯性系

        Args:
            state: 旋转系状态向量 [x, y, z, vx, vy, vz]
            time: 时间（无量纲）

        Returns:
            惯性系状态向量 [x, y, z, vx, vy, vz]
        """
        position = state[:3]
        velocity = state[3:]

        R = self.compute_rotation_matrix(time)
        R_dot = self.rotation_matrix_derivatives[time]

        position_inertial = R.T @ position

        if self.VELOCITY_TRANSFORM_INCLUDE_CORIOLIS:
            velocity_inertial = R.T @ velocity + R_dot.T @ position
        else:
            velocity_inertial = R.T @ velocity

        return np.concatenate([position_inertial, velocity_inertial])

    def inertial_to_rotating(self, state: npt.ArrayLike, time: float) -> npt.NDArray[np.floating]:
        """将状态向量从惯性系转换到旋转系

        Args:
            state: 惯性系状态向量 [x, y, z, vx, vy, vz]
            time: 时间（无量纲）

        Returns:
            旋转系状态向量 [x, y, z, vx, vy, vz]
        """
        position = state[:3]
        velocity = state[3:]

        R = self.compute_rotation_matrix(time)
        R_dot = self.rotation_matrix_derivatives[time]

        position_rotating = R @ position

        if self.VELOCITY_TRANSFORM_INCLUDE_CORIOLIS:
            velocity_rotating = R @ velocity - R_dot @ position_rotating
        else:
            velocity_rotating = R @ velocity

        return np.concatenate([position_rotating, velocity_rotating])

    def barycentric_to_primary(self, state: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """将状态向量从质心系转换到主天体中心系

        Args:
            state: 质心系状态向量 [x, y, z, vx, vy, vz]

        Returns:
            主天体中心系状态向量 [x, y, z, vx, vy, vz]

        Raises:
            ValueError: 系统未初始化（mu为None）时抛出
        """
        if self.mu is None:
            raise ValueError("系统未初始化，无法进行坐标变换")

        position = state[:3]
        velocity = state[3:]

        # 主天体在质心系中的位置（在旋转系中位于(-mu, 0, 0)）
        primary_position = np.array([-self.mu, 0, 0])

        position_primary = position - primary_position
        velocity_primary = velocity

        return np.concatenate([position_primary, velocity_primary])

    def primary_to_barycentric(self, state: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """将状态向量从主天体中心系转换到质心系

        Args:
            state: 主天体中心系状态向量 [x, y, z, vx, vy, vz]

        Returns:
            质心系状态向量 [x, y, z, vx, vy, vz]

        Raises:
            ValueError: 系统未初始化（mu为None）时抛出
        """
        if self.mu is None:
            raise ValueError("系统未初始化，无法进行坐标变换")

        position = state[:3]
        velocity = state[3:]

        # 主天体在质心系中的位置（在旋转系中位于(-mu, 0, 0)）
        primary_position = np.array([-self.mu, 0, 0])

        position_barycentric = position + primary_position
        velocity_barycentric = velocity

        return np.concatenate([position_barycentric, velocity_barycentric])

    def barycentric_to_secondary(self, state: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """将状态向量从质心系转换到次天体中心系

        Args:
            state: 质心系状态向量 [x, y, z, vx, vy, vz]

        Returns:
            次天体中心系状态向量 [x, y, z, vx, vy, vz]

        Raises:
            ValueError: 系统未初始化（mu为None）时抛出
        """
        if self.mu is None:
            raise ValueError("系统未初始化，无法进行坐标变换")

        position = state[:3]
        velocity = state[3:]

        # 次天体在质心系中的位置（在旋转系中位于(1-mu, 0, 0)）
        secondary_position = np.array([1 - self.mu, 0, 0])

        position_secondary = position - secondary_position
        velocity_secondary = velocity

        return np.concatenate([position_secondary, velocity_secondary])

    def secondary_to_barycentric(self, state: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """将状态向量从次天体中心系转换到质心系

        Args:
            state: 次天体中心系状态向量 [x, y, z, vx, vy, vz]

        Returns:
            质心系状态向量 [x, y, z, vx, vy, vz]

        Raises:
            ValueError: 系统未初始化（mu为None）时抛出
        """
        if self.mu is None:
            raise ValueError("系统未初始化，无法进行坐标变换")

        position = state[:3]
        velocity = state[3:]

        # 次天体在质心系中的位置（在旋转系中位于(1-mu, 0, 0)）
        secondary_position = np.array([1 - self.mu, 0, 0])

        position_barycentric = position + secondary_position
        velocity_barycentric = velocity

        return np.concatenate([position_barycentric, velocity_barycentric])

    def transform(
        self,
        state: npt.ArrayLike,
        from_frame: Union[ReferenceFrame, str],
        to_frame: Union[ReferenceFrame, str],
        time: float = 0.0,
    ) -> npt.NDArray[np.floating]:
        """通用坐标变换接口

        Args:
            state: 状态向量 [x, y, z, vx, vy, vz]
            from_frame: 源参考系（ReferenceFrame枚举或字符串）
            to_frame: 目标参考系（ReferenceFrame枚举或字符串）
            time: 时间（仅对涉及旋转系/惯性系的变换需要）

        Returns:
            变换后的状态向量 [x, y, z, vx, vy, vz]

        Raises:
            NotImplementedError: 不支持指定的坐标系变换组合时抛出
        """
        if isinstance(from_frame, str):
            from_frame = ReferenceFrame(from_frame)
        if isinstance(to_frame, str):
            to_frame = ReferenceFrame(to_frame)

        if from_frame == to_frame:
            return state

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
            raise NotImplementedError(f"不支持从 {from_frame} 到 {to_frame} 的变换")

    def __str__(self):
        """字符串表示

        Returns:
            变换器的简短字符串描述
        """
        return f"CoordinateTransformation(system={self.system})"

    def __repr__(self):
        """详细表示

        Returns:
            包含缓存大小等详细信息的字符串
        """
        return (
            f"CoordinateTransformation(system={self.system}, "
            f"cache_size={len(self.rotation_matrices)})"
        )
