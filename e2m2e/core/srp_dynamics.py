"""
太阳辐射压动力学模块

实现 CR3BP 框架下的太阳辐射压 (SRP) 扰动。
基于 EXOSIMS 的 equationsOfMotion_CRTBP 实现，采用光学系数模型。

物理背景
--------
太阳辐射压是光子撞击航天器表面产生的力。对于非完美反射表面，
SRP 力可分解为径向和切向分量，由光学系数 b1, b2, b3 决定。

光学系数模型（来自 EXOSIMS）：
  - b1 = 0.5 * (1 - s * p)：漫反射分量
  - b2 = s * p：镜面反射分量
  - b3 = 0.5 * (Bf * (1-s) * p + (1-p) * (ef*Bf - eb*Bb) / (ef + eb))：非朗伯分量

其中：
  - s: 镜面反射因子
  - p: 反射系数
  - Bf, Bb: 前/后表面非朗伯系数
  - ef, eb: 前/后表面发射系数

References:
    - EXOSIMS ObservatoryL2Halo.equationsOfMotion_CRTBP
    - Vallado, D. A. (2013). Fundamentals of Astrodynamics and Applications.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from .dynamics import CR3BP_Dynamics
from .cr3bp_system import CR3BP_System

NUMERICAL_JACOBIAN_EPS = 1e-8


class CR3BP_SRP_Dynamics(CR3BP_Dynamics):
    """带太阳辐射压的 CR3BP 动力学

    在标准 CR3BP 运动方程基础上添加太阳辐射压扰动力。
    SRP 力模型基于 EXOSIMS 实现，支持非完美反射表面的光学系数。

    Attributes:
        area: 航天器截面积 (m²)
        mass: 航天器质量 (kg)
        Cr: 反射系数 (1=完全吸收, 2=完全反射)
        b1: 漫反射光学系数
        b2: 镜面反射光学系数
        b3: 非朗伯光学系数
        beta: SRP 加速度系数 (solar sail parameter)
        P_srp: 太阳辐射压常数 (N/m² at 1 AU)

    Example:
        >>> from e2m2e.core.cr3bp_system import CR3BP_System
        >>> from e2m2e.core.srp_dynamics import CR3BP_SRP_Dynamics
        >>> system = CR3BP_System(mu=0.0121506683, primary="Earth", secondary="Moon")._with_default_scales()
        >>> dynamics = CR3BP_SRP_Dynamics(system, area=100.0, mass=1000.0)
        >>> result = dynamics.propagate([0.8, 0, 0, 0, 0.6, 0], (0, 1))
    """

    # 默认光学系数（来自 EXOSIMS）
    DEFAULT_NON_LAMBERTIAN_FRONT = 0.038
    DEFAULT_NON_LAMBERTIAN_BACK = 0.004
    DEFAULT_SPECULAR_REFLECTION = 0.975
    DEFAULT_NREFLECTION_COEFF = 0.999
    DEFAULT_EMISSION_FRONT = 0.8
    DEFAULT_EMISSION_BACK = 0.2
    DEFAULT_P_SRP = 4.56e-6  # N/m² at 1 AU

    def __init__(
        self,
        system: CR3BP_System,
        area: float = 1.0,
        mass: float = 1000.0,
        Cr: float = 1.5,
        non_lambertian_front: float = DEFAULT_NON_LAMBERTIAN_FRONT,
        non_lambertian_back: float = DEFAULT_NON_LAMBERTIAN_BACK,
        specular_reflection: float = DEFAULT_SPECULAR_REFLECTION,
        nreflection_coeff: float = DEFAULT_NREFLECTION_COEFF,
        emission_front: float = DEFAULT_EMISSION_FRONT,
        emission_back: float = DEFAULT_EMISSION_BACK,
        P_srp: float = DEFAULT_P_SRP,
    ) -> None:
        """初始化 SRP 动力学

        Args:
            system: CR3BP_System 对象
            area: 航天器截面积 (m²)
            mass: 航天器质量 (kg)
            Cr: 反射系数 (1=完全吸收, 2=完全反射)
            non_lambertian_front: 前表面非朗伯系数
            non_lambertian_back: 后表面非朗伯系数
            specular_reflection: 镜面反射因子
            nreflection_coeff: 反射系数
            emission_front: 前表面发射系数
            emission_back: 后表面发射系数
            P_srp: 太阳辐射压常数 (N/m² at 1 AU)
        """
        super().__init__(system)

        self.area = area
        self.mass = mass
        self.Cr = Cr
        self.P_srp = P_srp

        # 计算光学系数 (来自 EXOSIMS)
        s = specular_reflection
        p = nreflection_coeff
        Bf = non_lambertian_front
        Bb = non_lambertian_back
        ef = emission_front
        eb = emission_back

        self.b1 = 0.5 * (1.0 - s * p)
        self.b2 = s * p
        self.b3 = 0.5 * (Bf * (1.0 - s) * p + (1.0 - p) * (ef * Bf - eb * Bb) / (ef + eb))

        # SRP 加速度系数
        self.beta = self._compute_beta()

    def _compute_beta(self) -> float:
        """计算 SRP 加速度系数 (solar sail parameter)

        beta = P_srp * A * Cr / (2 * m)

        Returns:
            SRP 加速度系数
        """
        if self.mass <= 0:
            raise ValueError("航天器质量必须为正数")
        return self.P_srp * self.area * self.Cr / (2.0 * self.mass)

    def _get_eom_func(self, with_stm: bool) -> Callable:
        """返回带 SRP 扰动的运动方程函数

        Args:
            with_stm: 是否需要 STM 版本

        Returns:
            ODE 右端函数
        """
        if with_stm:
            return self._equations_with_stm_srp
        return self._equations_of_motion_srp

    def _equations_of_motion_srp(
        self, t: float, state: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """带 SRP 的 6 维运动方程

        Args:
            t: 时间
            state: 状态向量 [x, y, z, vx, vy, vz]

        Returns:
            状态导数 [vx, vy, vz, ax, ay, az]
        """
        # 基础 CR3BP 方程
        base = super().equations_of_motion(t, state)

        # 如果 beta 为 0，跳过 SRP 计算
        if self.beta == 0.0:
            return base

        x, y, z = state[:3]
        mu = self.system.mu

        # 航天器到主天体（较大天体，位于 x=-mu）的距离
        r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)

        # 径向单位矢量（从主天体指向航天器）
        ur = np.array([x + mu, y, z]) / r1

        # 切向单位矢量（在旋转平面内垂直于径向）
        r_xy = np.sqrt((x + mu) ** 2 + y**2)
        ut = np.array([-y, x + mu, 0.0]) / r_xy if r_xy > 1e-15 else np.array([0.0, 0.0, 0.0])

        # 径向和切向 SRP 力分量（来自 EXOSIMS 光学系数模型）
        # F_radial = b1 + b2·cos²(α_avg) + b3·cos(α_avg)
        #   其中 α_avg = 60°（假设航天器面法线与太阳方向的平均入射角），
        #   cos²(60°) = 0.25，cos(60°) = 0.5
        # F_tangential = b2·cos(α_avg)·sin(α_avg) + b3·sin(α_avg)
        #   sin(60°) = √3/2，故 sin(60°)·cos(60°) = √3/4
        F_radial = self.b1 + 0.25 * self.b2 + 0.5 * self.b3
        F_tangential = (np.sqrt(3) * 0.25) * (self.b2 + 2.0 * self.b3)

        # SRP 扰动加速度
        a_srp = self.beta * (F_radial * ur + F_tangential * ut)

        # 构造新数组，避免修改基类返回值
        return np.array(
            [base[0], base[1], base[2], base[3] + a_srp[0], base[4] + a_srp[1], base[5] + a_srp[2]]
        )

    def _equations_with_stm_srp(
        self, t: float, augmented_state: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """带 SRP 的 42 维增广运动方程

        Args:
            t: 时间
            augmented_state: 增广状态 [6状态 + 36个STM元素]

        Returns:
            增广状态导数
        """
        state = augmented_state[:6]
        stm = augmented_state[6:].reshape((6, 6))

        # 带 SRP 的状态导数
        state_derivative = self._equations_of_motion_srp(t, state)

        # 使用数值差分计算雅可比矩阵（因为 SRP 力依赖于位置）
        A = self._compute_jacobian_numerical(state)

        stm_dot = A @ stm

        return np.concatenate([state_derivative, stm_dot.flatten()])

    def _compute_jacobian_numerical(self, state: npt.NDArray[np.floating]) -> np.ndarray:
        """数值计算雅可比矩阵

        使用中心差分计算包含 SRP 的雅可比矩阵。

        Args:
            state: 状态向量

        Returns:
            6x6 雅可比矩阵
        """
        n = 6
        A = np.zeros((n, n))

        for j in range(n):
            state_plus = state.copy()
            state_plus[j] += NUMERICAL_JACOBIAN_EPS
            f_plus = self._equations_of_motion_srp(0.0, state_plus)

            state_minus = state.copy()
            state_minus[j] -= NUMERICAL_JACOBIAN_EPS
            f_minus = self._equations_of_motion_srp(0.0, state_minus)

            A[:, j] = (f_plus - f_minus) / (2.0 * NUMERICAL_JACOBIAN_EPS)

        return A

    def __str__(self) -> str:
        return (
            f"CR3BP_SRP_Dynamics(system={self.system}, "
            f"area={self.area}, mass={self.mass}, Cr={self.Cr})"
        )

    def __repr__(self) -> str:
        return (
            f"CR3BP_SRP_Dynamics("
            f"system={self.system}, "
            f"area={self.area}, "
            f"mass={self.mass}, "
            f"Cr={self.Cr}, "
            f"beta={self.beta:.6e})"
        )
