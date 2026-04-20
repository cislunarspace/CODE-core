"""
三体问题系统模块

包含 CR3BP_System 类，用于定义和操作圆型限制性三体问题系统。
"""

from __future__ import annotations

import warnings
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import fsolve


class LibrationPoint(Enum):
    """平动点枚举"""

    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4
    L5 = 5


class CR3BP_System:
    """圆型限制性三体问题系统

    Attributes:
        mu: 质量参数 μ = m2/(m1+m2)
        primary_body: 主天体名称
        secondary_body: 次天体名称
        L_points: 平动点位置字典
        L1: L1平动点坐标
        L2: L2平动点坐标
        L3: L3平动点坐标
        L4: L4平动点坐标
        L5: L5平动点坐标
        characteristic_length: 特征长度 (km)
        characteristic_time: 特征时间 (s)
        characteristic_velocity: 特征速度 (km/s)
        mass_primary: 主天体质量 (kg)
        mass_secondary: 次天体质量 (kg)
        total_mass: 总质量 (kg)
        semi_major_axis: 半长轴 (km)
        orbital_period: 轨道周期 (s)
        mean_motion: 平均角速度 (rad/s)
        has_L_points: 是否已计算平动点
    """

    # 天文常量
    EARTH_MOON_DISTANCE_KM = 384400.0  # 地月平均距离 (km)
    AU = 149597870.7  # 天文单位 (km)
    G = 6.67430e-20  # 万有引力常数 (km^3 / (kg * s^2))
    DAY = 86400  # 一天的秒数
    YEAR = 365.25 * 86400  # 一年的秒数（儒略年）

    # 预定义的常见三体系统参数
    KNOWN_SYSTEMS: dict[str, dict[str, str | float]] = {
        "earth_moon": {
            "primary": "Earth",
            "secondary": "Moon",
            "mu": 0.01215,
            "distance": EARTH_MOON_DISTANCE_KM,
            "period": 27.32 * 86400,
        },
        "sun_earth": {
            "primary": "Sun",
            "secondary": "Earth",
            "mu": 3.0039e-6,
            "distance": AU,
            "period": 365.25 * 86400,
        },
        "sun_jupiter": {
            "primary": "Sun",
            "secondary": "Jupiter",
            "mu": 0.0009535,
            "distance": 5.2 * AU,
            "period": 11.86 * 365.25 * 86400,
        },
    }

    @classmethod
    def from_known_system(cls, system_name: str) -> CR3BP_System:
        """从已知系统创建CR3BP系统

        Args:
            system_name: 系统名称，如 "earth_moon", "sun_earth", "sun_jupiter"

        Returns:
            CR3BP_System实例

        Raises:
            ValueError: 系统名称不在已知系统中
        """
        if system_name not in cls.KNOWN_SYSTEMS:
            raise ValueError(f"未知系统: {system_name}。可用系统: {list(cls.KNOWN_SYSTEMS.keys())}")

        system_params = cls.KNOWN_SYSTEMS[system_name]
        return cls(
            mu=float(system_params["mu"]),
            primary=str(system_params["primary"]),
            secondary=str(system_params["secondary"]),
        )

    def __init__(self, mu: float, primary: str, secondary: str) -> None:
        """初始化系统参数

        Args:
            mu: 质量参数 μ = m2/(m1+m2)
            primary: 主天体名称
            secondary: 次天体名称
        """
        self.primary_body: str = primary
        self.secondary_body: str = secondary
        self.mu: float = mu

        self.characteristic_length: float | None = None
        self.characteristic_time: float | None = None
        self.characteristic_velocity: float | None = None

        self.L_points: dict[LibrationPoint, npt.NDArray[np.floating]] | None = None
        self.L1: npt.NDArray[np.floating] | None = None
        self.L2: npt.NDArray[np.floating] | None = None
        self.L3: npt.NDArray[np.floating] | None = None
        self.L4: npt.NDArray[np.floating] | None = None
        self.L5: npt.NDArray[np.floating] | None = None

        self.mass_primary: float | None = None
        self.mass_secondary: float | None = None
        self.total_mass: float | None = None

        self.semi_major_axis: float | None = None
        self.orbital_period: float | None = None
        self.mean_motion: float | None = None

        self.has_L_points: bool = False
        self.is_initialized: bool = False

    def set_characteristic_scales(self, distance: float, period: float) -> None:
        """设置特征尺度

        Args:
            distance: 两天体之间的距离 (km)
            period: 轨道周期 (s)
        """
        self.characteristic_length = distance
        self.characteristic_time = period / (2 * np.pi)
        self.characteristic_velocity = distance / self.characteristic_time

        self.mean_motion = 2 * np.pi / period

        self.semi_major_axis = distance
        self.orbital_period = period

        self.is_initialized = True

    def compute_libration_points(self) -> dict[LibrationPoint, npt.NDArray[np.floating]]:
        """计算五个平动点

        Returns:
            平动点位置字典，键为 LibrationPoint 枚举，值为坐标数组
        """
        mu = self.mu

        # L1点：位于两天体之间，引力与离心力平衡
        # 平衡方程: x - (1-μ)/(x+μ)² + μ/(x-1+μ)² = 0
        def f1(x):
            return x - (1 - mu) / (x + mu) ** 2 + mu / (x - 1 + mu) ** 2

        # L2点：位于次天体外侧，引力与离心力平衡（两体同侧）
        def f2(x):
            return x - (1 - mu) / (x + mu) ** 2 - mu / (x - 1 + mu) ** 2

        # L3点：位于主天体外侧（远离次天体一侧）
        def f3(x):
            return x + (1 - mu) / (x + mu) ** 2 + mu / (x - 1 + mu) ** 2

        # 初始猜测值：基于级数展开的近似解
        L1_guess = 1 - mu ** (1 / 3)  # L1 约在 1 - μ^(1/3) 处
        L2_guess = 1 + mu ** (1 / 3)  # L2 约在 1 + μ^(1/3) 处
        L3_guess = -1 - (5 / 12) * mu  # L3 约在 -1 - 5μ/12 处

        # 使用 scipy.fsolve 迭代求解非线性方程
        L1_x = fsolve(f1, L1_guess)[0]
        L2_x = fsolve(f2, L2_guess)[0]
        L3_x = fsolve(f3, L3_guess)[0]

        # L4和L5点：等边三角形点，解析解精确
        # 位于与两天体构成等边三角形的顶点
        L4_x = 0.5 - mu
        L4_y = np.sqrt(3) / 2  # 上三角（北）

        L5_x = 0.5 - mu
        L5_y = -np.sqrt(3) / 2  # 下三角（南）

        self.L1 = np.array([L1_x, 0.0, 0.0])
        self.L2 = np.array([L2_x, 0.0, 0.0])
        self.L3 = np.array([L3_x, 0.0, 0.0])
        self.L4 = np.array([L4_x, L4_y, 0.0])
        self.L5 = np.array([L5_x, L5_y, 0.0])

        self.L_points = {
            LibrationPoint.L1: self.L1,
            LibrationPoint.L2: self.L2,
            LibrationPoint.L3: self.L3,
            LibrationPoint.L4: self.L4,
            LibrationPoint.L5: self.L5,
        }

        self.has_L_points = True
        return self.L_points

    def get_libration_point(self, point: LibrationPoint) -> npt.NDArray[np.floating]:
        """获取指定平动点

        若尚未计算平动点，会自动调用 compute_libration_points()。

        Args:
            point: LibrationPoint枚举值

        Returns:
            平动点坐标数组

        Raises:
            ValueError: 平动点无效
        """
        if not self.has_L_points:
            self.compute_libration_points()

        assert self.L_points is not None
        if point not in self.L_points:
            raise ValueError(f"无效的平动点: {point}")

        return self.L_points[point]

    def get_jacobi_constant(self, state: npt.ArrayLike) -> float:
        """计算Jacobi常数

        Args:
            state: 状态向量 [x, y, z, vx, vy, vz]

        Returns:
            Jacobi常数
        """
        x, y, z, vx, vy, vz = np.asarray(state, dtype=float)

        r1 = np.sqrt((x + self.mu) ** 2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + self.mu) ** 2 + y**2 + z**2)

        if r1 < 1e-12 or r2 < 1e-12:
            warnings.warn(
                "State at singularity in Jacobi constant calculation (r1={:.2e}, r2={:.2e})".format(r1, r2),
                RuntimeWarning,
                stacklevel=2,
            )
            return float("nan")

        U = (x**2 + y**2) / 2 + (1 - self.mu) / r1 + self.mu / r2

        v2 = vx**2 + vy**2 + vz**2

        C = 2 * U - v2

        return C

    def dimensionless_to_physical(self, state: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """无量纲化转物理单位

        Args:
            state: 无量纲状态向量 [x, y, z, vx, vy, vz]

        Returns:
            物理状态向量 [km, km/s]

        Raises:
            ValueError: 系统未初始化特征尺度
        """
        if not self.is_initialized:
            raise ValueError("系统未初始化，请先设置特征尺度")

        state = np.asarray(state, dtype=float)
        position = state[:3] * self.characteristic_length
        velocity = state[3:] * self.characteristic_velocity

        return np.concatenate([position, velocity])

    def physical_to_dimensionless(self, state: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """物理单位转无量纲化

        Args:
            state: 物理状态向量 [x, y, z, vx, vy, vz] (km, km/s)

        Returns:
            无量纲状态向量

        Raises:
            ValueError: 系统未初始化特征尺度
        """
        if not self.is_initialized:
            raise ValueError("系统未初始化，请先设置特征尺度")

        state = np.asarray(state, dtype=float)
        position = state[:3] / self.characteristic_length
        velocity = state[3:] / self.characteristic_velocity

        return np.concatenate([position, velocity])

    def compute_stability_index(self, L_point: LibrationPoint) -> dict[str, Any]:
        """计算平动点稳定性指标

        通过线性化运动方程的特征值分析平动点稳定性。

        Args:
            L_point: LibrationPoint枚举值

        Returns:
            稳定性指标字典，包含 is_stable、max_real_part、max_imag_part、
            eigenvalues 和 linear_matrix
        """
        if not self.has_L_points:
            self.compute_libration_points()

        point = self.get_libration_point(L_point)
        x, y, z = point

        r1 = np.sqrt((x + self.mu) ** 2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + self.mu) ** 2 + y**2 + z**2)

        U_xx = (
            1
            - (1 - self.mu) * (1 / r1**3 - 3 * (x + self.mu) ** 2 / r1**5)
            - self.mu * (1 / r2**3 - 3 * (x - 1 + self.mu) ** 2 / r2**5)
        )

        U_yy = (
            1
            - (1 - self.mu) * (1 / r1**3 - 3 * y**2 / r1**5)
            - self.mu * (1 / r2**3 - 3 * y**2 / r2**5)
        )

        U_zz = -(1 - self.mu) / r1**3 - self.mu / r2**3

        U_xy = (
            3 * (1 - self.mu) * (x + self.mu) * y / r1**5
            + 3 * self.mu * (x - 1 + self.mu) * y / r2**5
        )

        A = np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [U_xx, U_xy, 0, 0, 2, 0],
                [U_xy, U_yy, 0, -2, 0, 0],
                [0, 0, U_zz, 0, 0, 0],
            ]
        )

        eigenvalues = np.linalg.eigvals(A)

        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)

        # 特征值实部大于零表示不稳定
        is_stable = np.all(real_parts <= 0)

        max_real = np.max(real_parts)
        max_imag = np.max(np.abs(imag_parts))

        return {
            "is_stable": is_stable,
            "max_real_part": max_real,
            "max_imag_part": max_imag,
            "eigenvalues": eigenvalues,
            "linear_matrix": A,
        }

    def __str__(self):
        """字符串表示"""
        return (
            f"CR3BP_System(mu={self.mu}, "
            f"primary='{self.primary_body}', secondary='{self.secondary_body}')"
        )

    def __repr__(self):
        """详细表示"""
        return (
            f"CR3BP_System(mu={self.mu}, "
            f"primary='{self.primary_body}', secondary='{self.secondary_body}', "
            f"initialized={self.is_initialized}, has_L_points={self.has_L_points})"
        )

    def info(self, mode: str = "default") -> None:
        """输出系统信息

        Args:
            mode: 信息模式，"default" 为基础信息，"all" 为详细信息
        """
        # 1. 公共部分：头部与基础参数
        print("=" * 60)
        print("CR3BP 系统信息")
        print("=" * 60)
        print(f"系统名称：{self.primary_body}-{self.secondary_body}")
        print(f"质量参数 μ: {self.mu:.6e}")
        print(f"主天体：{self.primary_body}")
        print(f"次天体：{self.secondary_body}")

        # 2. 扩展信息（仅 mode="all" 时输出）
        if mode == "all":
            print("系统状态:")
            print(f"  是否初始化：{self.is_initialized}")
            print(f"  是否已计算平动点：{self.has_L_points}")
            print()

            if self.is_initialized and self.characteristic_length is not None:
                print("特征尺度:")
                print(f"  特征长度：{self.characteristic_length:.2f} km")
                print(f"  特征时间：{self.characteristic_time:.2f} s")
                print(f"  特征速度：{self.characteristic_velocity:.2f} km/s")
                print(f"  平均角速度：{self.mean_motion:.6e} rad/s")
                assert self.orbital_period is not None
                print(
                    f"  轨道周期：{self.orbital_period:.2f} s "
                    f"({self.orbital_period / 86400:.2f} 天)"
                )
                print(f"  半长轴：{self.semi_major_axis:.2f} km")
            else:
                print("特征尺度：未设置 (请使用 set_characteristic_scales() 方法设置)")
            print()

            if self.has_L_points and self.L_points is not None:
                print("平动点位置 (无量纲坐标):")
                for point_name, point_coords in self.L_points.items():
                    name = point_name.name
                    x, y, z = point_coords
                    print(f"  {name}: ({x:.6f}, {y:.6f}, {z:.6f})")
            else:
                print("平动点：未计算 (请使用 compute_libration_points() 方法计算)")
            print()

            if self.mass_primary is not None and self.mass_secondary is not None:
                print("质量信息:")
                print(f"  主天体质量：{self.mass_primary:.3e} kg")
                print(f"  次天体质量：{self.mass_secondary:.3e} kg")
                print(f"  总质量：{self.total_mass:.3e} kg")
            else:
                print("质量信息：未设置")
            print()

        # 3. 尾部分隔线
        print("=" * 60)
