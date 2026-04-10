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

其中：
- r   : 航天器相对原点天体的位置向量
- rᵢ  : 第 i 个摄动天体相对原点天体的位置向量
- μ₀  : 原点天体的引力常数 (GM)
- μᵢ  : 第 i 个摄动天体的引力常数 (GM)

状态转移矩阵 (STM, Φ) 满足变分方程：

    dΦ/dt = A(t) Φ

其中 A 为线性化的系统矩阵：

    A = | 0₃ₓ₃  I₃ₓ₃ |
        | ∂a/∂r  0₃ₓ₃ |

References
----------
- Battin, R. H. *An Introduction to the Methods of Astrodynamics*.
- Gurfil, P., & Seidelmann, P. K. *Celestial Mechanics and Astrodynamics*.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional, Any

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp

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
        last_trajectory: 最近一次传播的时间与状态轨迹 ``(t, states)``。
        last_stm: 最近一次含 STM 传播的状态转移矩阵序列，形状 ``(6, 6, n_times)``。
        cross_section_tolerance: Poincaré 截面穿越检测容差。
        last_crossing: 最近一次检测到的截面穿越信息。
        jacobi_history: Jacobi 常数历史记录（用于能量守恒检验）。
        jacobi_error: Jacobi 常数最大偏差。
        initialized: 实例是否已完成初始化的标志。
    """

    STM_DIMENSION = 42

    def __init__(self, system: EphemerisSystem) -> None:
        self.system = system
        self.integrator = "DOP853"
        self.rtol = 1e-12
        self.atol = 1e-12
        self.max_step = 60.0
        self.last_trajectory = None
        self.last_stm = None
        self.cross_section_tolerance = 1e-8
        self.last_crossing = None
        self.jacobi_history = []
        self.jacobi_error = 0.0
        self.initialized = True

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

        acc = np.zeros(3)
        for body in self.system.bodies:
            gm = self.system.spice.get_gm(body)
            if body == self.system.origin:
                # 原点天体：中心引力加速度
                r_norm = np.linalg.norm(r_sc)
                acc -= gm * r_sc / r_norm**3
            else:
                # 摄动天体：第三体摄动 + 间接项
                r_ob = self.system.spice.get_body_position(
                    body, t, self.system.frame, self.system.origin
                )
                r_bsc = r_sc - r_ob  # 航天器到摄动天体的位置向量
                r_bsc_norm = np.linalg.norm(r_bsc)
                r_ob_norm = np.linalg.norm(r_ob)
                # 直接项 + 间接项
                acc -= gm * (r_bsc / r_bsc_norm**3 + r_ob / r_ob_norm**3)

        return np.concatenate([v_sc, acc])

    def equations_with_stm(
        self, t: float, augmented_state: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """计算含状态转移矩阵 (STM) 的增广运动方程右端项。

        增广状态向量布局：``[r(3), v(3), Φ(36)]``，共 42 维。
        其中 Φ 为 6×6 状态转移矩阵按行展平。

        STM 满足变分方程 ``dΦ/dt = A Φ``，其中系统矩阵 A 为：

            A = | 0₃ₓ₃  I₃ₓₓ₃ |
                | ∂a/∂r  0₃ₓ₃  |

        加速度对位置的偏导数（重力梯度张量）为：

            ∂a/∂r = -μ (I₃ / |r|³ - 3 r⊗r / |r|⁵)

        Args:
            t: 历元时刻（ephemeris seconds past J2000）。
            augmented_state: 增广状态向量，形状 ``(42,)``。

        Returns:
            增广状态导数向量，形状 ``(42,)``，即 ``[v, a, dΦ/dt_flat]``。
        """
        state = augmented_state[:6]
        r_sc = state[:3]

        # 从增广状态中恢复 6×6 STM 矩阵
        stm = augmented_state[6:].reshape((6, 6))

        acc = np.zeros(3)
        dacc_dr = np.zeros((3, 3))  # 加速度对位置的偏导数（重力梯度张量）

        for body in self.system.bodies:
            gm = self.system.spice.get_gm(body)
            if body == self.system.origin:
                # 原点天体：中心引力及其 Jacobian
                r_norm = np.linalg.norm(r_sc)
                acc -= gm * r_sc / r_norm**3
                # ∂a/∂r = -μ (I/r³ - 3 r⊗r / r⁵)
                dacc_dr -= gm * (np.eye(3) / r_norm**3 - 3.0 * np.outer(r_sc, r_sc) / r_norm**5)
            else:
                # 摄动天体：第三体摄动及其 Jacobian（仅直接项贡献）
                r_ob = self.system.spice.get_body_position(
                    body, t, self.system.frame, self.system.origin
                )
                r_bsc = r_sc - r_ob
                r_bsc_norm = np.linalg.norm(r_bsc)
                r_ob_norm = np.linalg.norm(r_ob)
                acc -= gm * (r_bsc / r_bsc_norm**3 + r_ob / r_ob_norm**3)
                # 间接项 r_ob/|r_ob|³ 与 r_sc 无关，Jacobian 为零
                dacc_dr -= gm * (
                    np.eye(3) / r_bsc_norm**3 - 3.0 * np.outer(r_bsc, r_bsc) / r_bsc_norm**5
                )

        # 状态方程导数：[v, a]
        state_deriv = np.concatenate([state[3:], acc])

        # 构造 6×6 系统矩阵 A
        A = np.zeros((6, 6))
        A[:3, 3:] = np.eye(3)  # dr/dt = v
        A[3:, :3] = dacc_dr  # dv/dt 依赖于位置

        # STM 变分方程：dΦ/dt = A Φ
        stm_dot = A @ stm

        return np.concatenate([state_deriv, stm_dot.flatten()])

    def propagate(
        self,
        initial_state: npt.ArrayLike,
        t_span: Tuple[float, float],
        t_eval: Optional[npt.ArrayLike] = None,
        with_stm: bool = False,
        with_jacobi: bool = False,
    ) -> Dict[str, Any]:
        """数值积分传播航天器轨道。

        使用 SciPy 的 :func:`solve_ivp` 进行数值积分，支持两种模式：

        1. **纯状态传播** (``with_stm=False``)：仅积分 6 维状态向量 ``[r, v]``。
        2. **含 STM 传播** (``with_stm=True``)：积分 42 维增广状态，
           同时求解状态转移矩阵，STM 初始值设为 6×6 单位矩阵。

        Args:
            initial_state: 初始状态向量，形状 ``(6,)``，``[x, y, z, vx, vy, vz]``。
            t_span: 积分时间区间 ``(t0, tf)``，单位为 ephemeris seconds。
            t_eval: 指定输出时间点序列。若为 ``None``，由积分器自动选择。
            with_stm: 是否同时求解状态转移矩阵。
            with_jacobi: 是否沿轨迹逐点计算 Jacobi 常数。

        Returns:
            包含传播结果的字典：
            - ``"time"``: 时间序列，形状 ``(n_times,)``。
            - ``"states"``: 状态序列，形状 ``(6, n_times)``。
            - ``"stm"``: 状态转移矩阵序列，形状 ``(6, 6, n_times)``（仅 ``with_stm=True``）。
        """
        # 根据传播时长自适应调整最大步长，防止短弧段步长过大
        span_duration = abs(t_span[1] - t_span[0])
        if span_duration > 0:
            max_step = min(self.max_step, span_duration / 10.0)
        else:
            max_step = self.max_step

        if with_stm:
            # 初始化 STM 为 6×6 单位矩阵并展平
            initial_stm = np.eye(6).flatten()
            augmented_state = np.concatenate([np.asarray(initial_state, dtype=float), initial_stm])

            result = solve_ivp(
                self.equations_with_stm,
                t_span,
                augmented_state,
                method=self.integrator,
                t_eval=t_eval,
                rtol=self.rtol,
                atol=self.atol,
                max_step=max_step,
            )

            # 提取状态和 STM
            states = result.y[:6, :]
            n_times = states.shape[1]
            stm_flat = result.y[6:, :]
            # 将展平的 STM 重构为 (6, 6, n_times) 张量
            stm_matrices = np.zeros((6, 6, n_times))
            for k in range(n_times):
                stm_matrices[:, :, k] = stm_flat[:, k].reshape(6, 6)
            self.last_trajectory = (result.t, states.T)
            self.last_stm = stm_matrices

            return {
                "time": result.t,
                "states": states,
                "stm": stm_matrices,
            }
        else:
            # 纯状态传播（不含 STM）
            result = solve_ivp(
                self.equations_of_motion,
                t_span,
                np.asarray(initial_state, dtype=float),
                method=self.integrator,
                t_eval=t_eval,
                rtol=self.rtol,
                atol=self.atol,
                max_step=max_step,
            )

            states = result.y

            self.last_trajectory = (result.t, states.T)

            return {
                "time": result.t,
                "states": states,
            }
