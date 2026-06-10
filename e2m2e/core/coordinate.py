"""
坐标变换模块

包含CoordinateTransformation类，用于在不同参考系之间转换轨道状态。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..mbse.data.enums import ReferenceFrame
from .cr3bp_system import CR3BP_System

_TU_SECONDS_DEFAULT = 4.34811305 * 86400


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
        self.rotation_matrices: dict[float, npt.NDArray[np.floating]] = {}
        self.rotation_matrix_derivatives: dict[float, npt.NDArray[np.floating]] = {}
        self.initialized = True

    def compute_rotation_matrix(self, time: float) -> npt.NDArray[np.floating]:
        """计算给定时刻的旋转矩阵及其导数

        Args:
            time: 时间（无量纲）

        Returns:
            3x3旋转矩阵
        """
        # 优先从缓存中获取旋转矩阵，避免重复计算
        if self.CACHE_ROTATION_MATRICES and time in self.rotation_matrices:
            return self.rotation_matrices[time]

        # 在 CR3BP 旋转系中，角速度归一化为 1，故旋转角度等于无量纲时间
        angle = time

        # 绕 z 轴的旋转矩阵 R(t) 及其导数 dR/dt
        # R(t) = [[cos(t), -sin(t), 0], [sin(t), cos(t), 0], [0, 0, 1]]
        # dR/dt = [[-sin(t), -cos(t), 0], [cos(t), -sin(t), 0], [0, 0, 0]]
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        rotation_matrix = np.array(
            [[cos_angle, -sin_angle, 0], [sin_angle, cos_angle, 0], [0, 0, 1]]
        )

        rotation_matrix_derivative = np.array(
            [[-sin_angle, -cos_angle, 0], [cos_angle, -sin_angle, 0], [0, 0, 0]]
        )

        # LRU 缓存：达到上限时删除最早缓存的项
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
        state = np.asarray(state, dtype=float)
        position = state[:3]
        velocity = state[3:]

        R = self.compute_rotation_matrix(time)
        R_dot = self.rotation_matrix_derivatives[time]

        # 位置变换：r_inertial = R^T * r_rotating
        position_inertial = R.T @ position

        # 速度变换：v_inertial = R^T * v + dR^T/dt * r（含科里奥利项）
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
        state = np.asarray(state, dtype=float)
        position = state[:3]
        velocity = state[3:]

        R = self.compute_rotation_matrix(time)
        R_dot = self.rotation_matrix_derivatives[time]

        # 位置变换：r_rotating = R * r_inertial
        position_rotating = R @ position

        # 速度变换：v_rotating = R * v - dR/dt * R * r（逆变换的科里奥利项）
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

        state = np.asarray(state, dtype=float)
        position = state[:3]
        velocity = state[3:]

        # 主天体在质心系中的位置：(-μ, 0, 0)
        primary_position = np.array([-self.mu, 0, 0])

        # 平移变换（无速度分量，因两参考系间无相对旋转）
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

        state = np.asarray(state, dtype=float)
        position = state[:3]
        velocity = state[3:]

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

        state = np.asarray(state, dtype=float)
        position = state[:3]
        velocity = state[3:]

        # 次天体在质心系中的位置：(1-μ, 0, 0)
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

        state = np.asarray(state, dtype=float)
        position = state[:3]
        velocity = state[3:]

        secondary_position = np.array([1 - self.mu, 0, 0])

        position_barycentric = position + secondary_position
        velocity_barycentric = velocity

        return np.concatenate([position_barycentric, velocity_barycentric])

    def transform(
        self,
        state: npt.ArrayLike,
        from_frame: ReferenceFrame | str,
        to_frame: ReferenceFrame | str,
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
            return np.asarray(state, dtype=float)

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


class SynodicJ2000Transformation:
    """CR3BP synodic坐标系与J2000惯性坐标系之间的转换器。

    该类提供了CR3BP归一化synodic坐标系与J2000惯性坐标系之间的双向转换，
    使用SPICE获取月球瞬时状态来构建旋转矩阵，支持单个状态和批量转换。

    Attributes:
        cr3bp_system: CR3BP系统对象，提供质量参数等信息
        spice: SPICEManager对象，用于获取天体状态
    """

    def __init__(self, cr3bp_system: CR3BP_System, spice) -> None:
        """初始化转换器

        Args:
            cr3bp_system: CR3BP系统对象
            spice: SPICEManager对象
        """
        self.cr3bp_system = cr3bp_system
        self.spice = spice

    def _get_time_unit(self) -> float:
        """获取时间单位（秒）

        优先使用CR3BP系统的特征时间，否则使用默认值。

        Returns:
            时间单位（秒）
        """
        if (
            hasattr(self.cr3bp_system, "characteristic_time")
            and self.cr3bp_system.characteristic_time is not None
        ):
            return self.cr3bp_system.characteristic_time
        return _TU_SECONDS_DEFAULT

    def _build_rotation_matrix(self, r_m: npt.NDArray, v_m: npt.NDArray) -> npt.NDArray:
        """构建旋转矩阵

        基于月球位置和速度构建瞬时旋转矩阵：
        - e1: 指向月球的方向（x轴）
        - e3: 轨道法线方向（z轴）
        - e2: 完成右手坐标系（y轴）

        Args:
            r_m: 月球位置向量（km）
            v_m: 月球速度向量（km/s）

        Returns:
            3x3旋转矩阵
        """
        e1 = r_m / np.linalg.norm(r_m)  # e1 指向月球方向
        h = np.cross(r_m, v_m)  # 轨道角动量方向
        e3 = h / np.linalg.norm(h)  # e3 为轨道法线（角动量单位向量）
        e2 = np.cross(e3, e1)  # e2 = e3 × e1，完成右手坐标系
        return np.column_stack([e1, e2, e3])

    def _get_moon_frame(self, et: float):
        """获取月球参考系参数

        在给定ET时刻获取月球状态并计算参考系参数。

        Args:
            et: SPICE ephemeris time（秒）

        Returns:
            Tuple: (r_m, v_m, l_c, R, omega)
                - r_m: 月球位置（km）
                - v_m: 月球速度（km/s）
                - l_c: 特征长度（km）
                - R: 旋转矩阵
                - omega: 角速度向量（rad/s）
        """
        moon_state = self.spice.get_body_state("MOON", et, "J2000", "EARTH")
        r_m = moon_state[:3]  # 月球相对地球位置 (km)
        v_m = moon_state[3:]  # 月球相对地球速度 (km/s)
        l_c = np.linalg.norm(r_m)  # 特征长度 = 月地距离
        R = self._build_rotation_matrix(r_m, v_m)  # 瞬时旋转矩阵
        # 角速度 ω = r × v / |r|²（瞬时轨道角速度向量）
        omega = np.cross(r_m, v_m) / np.dot(r_m, r_m)
        return r_m, v_m, l_c, R, omega

    def synodic_to_j2000(self, state_syn: npt.ArrayLike, t_syn: float, et0: float) -> npt.NDArray:
        """将状态从synodic坐标系转换到J2000惯性坐标系

        Args:
            state_syn: synodic坐标系状态向量 [x, y, z, vx, vy, vz]（无量纲）
            t_syn: synodic时间（TU，无量纲）
            et0: 参考历元的SPICE ephemeris time（秒）

        Returns:
            J2000坐标系状态向量 [x, y, z, vx, vy, vz]（km, km/s）
        """
        state_syn = np.asarray(state_syn, dtype=float)
        mu = self.cr3bp_system.mu
        t_c = self._get_time_unit()

        et = et0 + t_syn * t_c  # 计算绝对 SPICE 时间
        r_m, v_m, l_c, R, omega = self._get_moon_frame(et)

        # synodic → 有量纲：r_dim = r_syn * l_c
        r_dim = state_syn[:3] * l_c
        # 质心系 → 地心系：减去地球在质心系的位置偏移
        r_from_earth = r_dim - np.array([-mu, 0.0, 0.0]) * l_c
        # 旋转系 → J2000：应用旋转矩阵 R
        r_j2000 = R @ r_from_earth

        # 速度量纲化：v_dim = v_syn * l_c / t_c
        v_dim = state_syn[3:] * l_c / t_c
        # 速度变换：含科里奥利项 ω × r
        v_j2000 = R @ v_dim + np.cross(omega, r_j2000)

        return np.concatenate([r_j2000, v_j2000])

    def j2000_to_synodic(self, state_j2000: npt.ArrayLike, t_syn: float, et0: float) -> npt.NDArray:
        """将状态从J2000惯性坐标系转换到synodic坐标系

        Args:
            state_j2000: J2000坐标系状态向量 [x, y, z, vx, vy, vz]（km, km/s）
            t_syn: synodic时间（TU，无量纲）
            et0: 参考历元的SPICE ephemeris time（秒）

        Returns:
            synodic坐标系状态向量 [x, y, z, vx, vy, vz]（无量纲）
        """
        state_j2000 = np.asarray(state_j2000, dtype=float)
        mu = self.cr3bp_system.mu
        t_c = self._get_time_unit()

        et = et0 + t_syn * t_c
        r_m, v_m, l_c, R, omega = self._get_moon_frame(et)

        # J2000 → 地心系：逆旋转
        r_from_earth = R.T @ state_j2000[:3]
        # 地心系 → 质心系：加上地球偏移
        r_dim = r_from_earth + np.array([-mu, 0.0, 0.0]) * l_c
        # 质心系 → synodic 无量纲：除以特征长度
        r_syn = r_dim / l_c

        # 速度逆变换：先消除科里奥利项，再旋转回 synodic 系
        v_from_earth = R.T @ (state_j2000[3:] - np.cross(omega, state_j2000[:3]))
        v_syn = v_from_earth * t_c / l_c

        return np.concatenate([r_syn, v_syn])

    def batch_synodic_to_j2000(
        self,
        states_syn: npt.ArrayLike,
        t_syn_arr: npt.ArrayLike,
        et0: float,
    ) -> npt.NDArray:
        """批量将状态从synodic坐标系转换到J2000惯性坐标系

        Args:
            states_syn: synodic坐标系状态数组，形状 (N, 6)（无量纲）
            t_syn_arr: synodic时间数组，形状 (N,)（TU，无量纲）
            et0: 参考历元的SPICE ephemeris time（秒）

        Returns:
            J2000坐标系状态数组，形状 (N, 6)（km, km/s）
        """
        states_syn = np.asarray(states_syn, dtype=float)
        t_syn_arr = np.asarray(t_syn_arr, dtype=float)
        n = len(t_syn_arr)
        results = np.empty((n, 6))
        for i in range(n):
            results[i] = self.synodic_to_j2000(states_syn[i], t_syn_arr[i], et0)
        return results

    def batch_j2000_to_synodic(
        self,
        states_j2000: npt.ArrayLike,
        t_syn_arr: npt.ArrayLike,
        et0: float,
    ) -> npt.NDArray:
        """批量将状态从J2000惯性坐标系转换到synodic坐标系

        Args:
            states_j2000: J2000坐标系状态数组，形状 (N, 6)（km, km/s）
            t_syn_arr: synodic时间数组，形状 (N,)（TU，无量纲）
            et0: 参考历元的SPICE ephemeris time（秒）

        Returns:
            synodic坐标系状态数组，形状 (N, 6)（无量纲）
        """
        states_j2000 = np.asarray(states_j2000, dtype=float)
        t_syn_arr = np.asarray(t_syn_arr, dtype=float)
        n = len(t_syn_arr)
        results = np.empty((n, 6))
        for i in range(n):
            results[i] = self.j2000_to_synodic(states_j2000[i], t_syn_arr[i], et0)
        return results
