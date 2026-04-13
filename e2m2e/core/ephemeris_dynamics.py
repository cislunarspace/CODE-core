"""星历动力学模型 —— 在多体 N 体引力场中传播航天器轨道。

本模块实现 :class:`EphemerisDynamics`，它继承自 :class:`Dynamics`，利用
SPICE 星历数据提供的高精度天体位置与引力参数，计算多体引力加速度、
状态转移矩阵 (STM) 并完成数值积分传播。

核心物理模型
-----------
采用 **受限 N 体问题** (Restricted N-Body Problem) 建模：

1. 以 ``system.origin`` 为坐标原点（通常是主天体，如地球）。
2. 原点天体对航天器施加中心引力加速度。
3. 其余天体（如月球、太阳等）对航天器施加第三体摄动加速度，
   同时扣除其对原点天体的引力加速度（即间接项），以保持坐标原点
   位于原点天体而非质心。

加速度公式（以原点天体 P₀ 为中心）：

    a = - μ₀ r / |r|³
        - Σᵢ μᵢ [ (r - rᵢ) / |r - rᵢ|³ + rᵢ / |rᵢ|³ ]

重构说明 (v4.0 MBSE)
--------------------
- 调用 ``super().__init__()`` （REQ-005）
- 覆写 ``_get_eom_func()`` 和 ``_get_max_step()`` 钩子方法
- ``propagate()`` 继承自基类，states 形状统一为 ``(n_points, 6)`` （REQ-002）
- ``stm`` 形状统一为 ``(n_points, 6, 6)``

References
----------
- Battin, R. H. *An Introduction to the Methods of Astrodynamics*.
- Gurfil, P., & Seidelmann, P. K. *Celestial Mechanics and Astrodynamics*.
"""

from __future__ import annotations

import warnings
from typing import Callable, Tuple

import numpy as np
import numpy.typing as npt

from .dynamics import Dynamics
from .ephemeris_system import EphemerisSystem


class EphemerisDynamics(Dynamics):
    """星历 N 体动力学类，提供运动方程、STM 变分方程与轨道传播功能。

    Args:
        system: 星历系统配置，包含天体列表、SPICE 内核接口、参考系与原点天体等信息。

    Attributes:
        integrator: SciPy 积分器名称，默认 ``"DOP853"``（8 阶 Runge-Kutta）。
        rtol: 相对积分容差。
        atol: 绝对积分容差。
        max_step: 积分器最大步长（秒）。
    """

    MIN_DISTANCE = 1e-6  # km (1 米)，防止除零的最小距离钳位

    def __init__(self, system: EphemerisSystem) -> None:
        # 调用基类 __init__（REQ-005），基类设置通用属性
        super().__init__(system)
        # 覆写星历动力学特有的配置
        self.integrator = "DOP853"
        self.max_step = 60.0  # 秒（物理单位），非无量纲

    def _get_eom_func(self, with_stm: bool) -> Callable:
        """返回星历 N 体运动方程函数"""
        if with_stm:
            return self.equations_with_stm
        return self.equations_of_motion

    def _get_max_step(self, t_span: Tuple[float, float]) -> float:
        """自适应步长：根据传播时长调整最大步长，防止短弧段步长过大"""
        span_duration = abs(t_span[1] - t_span[0])
        if span_duration > 0:
            return min(self.max_step, span_duration / 10.0)
        return self.max_step

    def _compute_acc_and_jacobian(
        self,
        t: float,
        r_sc: npt.NDArray[np.floating],
        need_jacobian: bool = False,
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating] | None]:
        """单次遍历所有天体，同时计算加速度和（可选）雅可比矩阵。

        将加速度计算和雅可比计算合并到一次循环中，避免对 SPICE 的重复查询。

        Args:
            t: 历元时刻（ephemeris seconds past J2000）。
            r_sc: 航天器位置向量，形状 ``(3,)``。
            need_jacobian: 是否同时计算 ∂a/∂r。

        Returns:
            (acc, dacc_dr) 元组。acc 形状 (3,)，dacc_dr 形状 (3,3) 或 None。
        """
        acc = np.zeros(3)
        dacc_dr = np.zeros((3, 3)) if need_jacobian else None

        for body in self.system.bodies:
            gm = self.system.get_gm(body)
            if body == self.system.origin:
                r_norm = np.linalg.norm(r_sc)
                if r_norm < self.MIN_DISTANCE:
                    warnings.warn(
                        f"Spacecraft at origin body center (|r|={r_norm:.2e} km), "
                        f"clamping to MIN_DISTANCE={self.MIN_DISTANCE} km",
                        stacklevel=3,
                    )
                    r_norm = self.MIN_DISTANCE
                acc -= gm * r_sc / r_norm**3
                if need_jacobian:
                    dacc_dr -= gm * (
                        np.eye(3) / r_norm**3
                        - 3.0 * np.outer(r_sc, r_sc) / r_norm**5
                    )
            else:
                r_ob = self.system.get_body_position(body, t)
                r_bsc = r_sc - r_ob
                r_bsc_norm = np.linalg.norm(r_bsc)
                r_ob_norm = np.linalg.norm(r_ob)
                if r_bsc_norm < self.MIN_DISTANCE:
                    warnings.warn(
                        f"Spacecraft at perturbing body {body} center "
                        f"(|r_bsc|={r_bsc_norm:.2e} km), "
                        f"clamping to MIN_DISTANCE={self.MIN_DISTANCE} km",
                        stacklevel=3,
                    )
                    r_bsc_norm = self.MIN_DISTANCE
                if r_ob_norm < self.MIN_DISTANCE:
                    r_ob_norm = self.MIN_DISTANCE
                acc -= gm * (r_bsc / r_bsc_norm**3 + r_ob / r_ob_norm**3)
                # 间接项 r_ob/r_ob_norm³ 不依赖于航天器位置，故 ∂/∂r_sc = 0，
                # 仅对主项 r_bsc/r_bsc_norm³ 求偏导
                if need_jacobian:
                    dacc_dr -= gm * (
                        np.eye(3) / r_bsc_norm**3
                        - 3.0 * np.outer(r_bsc, r_bsc) / r_bsc_norm**5
                    )

        return acc, dacc_dr

    def equations_of_motion(
        self, t: float, state: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """计算受限 N 体问题的运动方程右端项（加速度）。

        对每个天体分别处理：
        - **原点天体**：中心引力 ``-μ₀ r/|r|³``
        - **摄动天体**：第三体摄动 ``-μᵢ [(r-rᵢ)/|r-rᵢ|³ + rᵢ/|rᵢ|³]``，
          第二项为间接项（扣除摄动天体对原点的引力）。

        Args:
            t: 历元时刻（ephemeris seconds past J2000），用于查询天体星历位置。
            state: 航天器状态向量，形状 ``(6,)``，前 3 个元素为位置 [km]，
                后 3 个元素为速度 [km/s]。

        Returns:
            状态导数向量，形状 ``(6,)``，即 ``[v, a]``。
        """
        r_sc = state[:3]
        v_sc = state[3:]

        acc, _ = self._compute_acc_and_jacobian(t, r_sc, need_jacobian=False)
        return np.concatenate([v_sc, acc])

    def compute_jacobian_A(
        self, t: float, state: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """计算星历 N 体状态方程的雅可比矩阵 A(t)。

        A = | 0₃ₓ₃  I₃ₓ₃ |
            | U₃ₓ₃  0₃ₓ₃ |

        其中 U = ∂a/∂r 是加速度对位置的偏导数 (3x3)。
        N 体问题中无速度相关力，因此 ∂a/∂v = 0。

        Args:
            t: 历元时刻（ephemeris seconds past J2000）。
            state: 航天器状态向量，形状 ``(6,)``。

        Returns:
            雅可比矩阵，形状 ``(6, 6)``。
        """
        r_sc = state[:3]

        _, dacc_dr = self._compute_acc_and_jacobian(t, r_sc, need_jacobian=True)

        A = np.zeros((6, 6))
        A[:3, 3:] = np.eye(3)
        A[3:, :3] = dacc_dr
        return A

    def equations_with_stm(
        self, t: float, augmented_state: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """计算含状态转移矩阵 (STM) 的增广运动方程右端项。

        增广状态向量布局：``[r(3), v(3), Φ(36)]``，共 42 维。
        其中 Φ 为 6×6 状态转移矩阵按行展平。

        Args:
            t: 历元时刻（ephemeris seconds past J2000）。
            augmented_state: 增广状态向量，形状 ``(42,)``。

        Returns:
            增广状态导数向量，形状 ``(42,)``，即 ``[v, a, dΦ/dt_flat]``。
        """
        state = augmented_state[:6]
        r_sc = state[:3]
        stm = augmented_state[6:].reshape((6, 6))

        # 单次遍历同时获取加速度和雅可比，避免重复 SPICE 查询
        acc, dacc_dr = self._compute_acc_and_jacobian(t, r_sc, need_jacobian=True)

        A = np.zeros((6, 6))
        A[:3, 3:] = np.eye(3)
        A[3:, :3] = dacc_dr
        stm_dot = A @ stm

        state_deriv = np.concatenate([state[3:], acc])
        return np.concatenate([state_deriv, stm_dot.flatten()])

    def __str__(self):
        return f"EphemerisDynamics(system={self.system}, integrator='{self.integrator}')"

    def __repr__(self):
        return (
            f"EphemerisDynamics(system={self.system}, integrator='{self.integrator}', "
            f"rtol={self.rtol}, atol={self.atol}, max_step={self.max_step})"
        )
