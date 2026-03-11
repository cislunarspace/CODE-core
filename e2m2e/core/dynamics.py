"""
三体问题动力学模块

包含CR3BP_Dynamics类，用于计算和积分圆型限制性三体问题的动力学方程。
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, List, Tuple, Optional, Any, Callable

import numpy.typing as npt

from .system import CR3BP_System


class CR3BP_Dynamics:
    """CR3BP动力学方程

    本类封装了CR3BP的动力学模型，提供状态传播、状态转移矩阵计算、Jacobi常数计算等核心功能。
    支持6维状态向量（位置+速度）和42维增广状态向量（状态+状态转移矩阵）的数值积分。

    主要功能：
        - 运动方程定义与数值积分
        - 状态转移矩阵(STM)的计算与传播
        - Jacobi常数的实时监控与守恒性检验
        - 截面穿越检测（用于Poincaré映射和周期轨道搜索）

    属性：
        system (CR3BP_System): CR3BP系统对象，包含质量参数μ等系统常数
        integrator (str): 数值积分器类型（默认'RK45'）
        rtol (float): 相对积分容差
        atol (float): 绝对积分容差
        max_step (float): 最大积分步长
        last_trajectory (array): 最近一次积分的轨迹 [t, y]
        last_stm (array): 最近一次积分的状态转移矩阵
        cross_section_tolerance (float): 截面检测容差
        last_crossing (tuple): 上次穿过截面的点和时间
        jacobi_history (list): Jacobi常数历史记录
        jacobi_error (float): Jacobi常数误差（用于精度检验）
        initialized (bool): 初始化完成标志

    方法：
        __init__(system): 初始化动力学对象
        equations_of_motion(t, state): 6维状态向量的运动方程
        equations_with_stm(t, augmented_state): 42维增广状态向量的运动方程
        propagate(initial_state, t_span, t_eval=None, with_stm=False): 传播轨迹
        compute_state_transition_matrix(initial_state, t): 计算状态转移矩阵
        compute_jacobi_constant(state): 实时计算Jacobi常数
        check_cross_section(state, plane, value): 检查是否穿过指定截面
    """

    # 类属性
    DEFAULT_TOLERANCE = 1e-12  # 默认积分容差
    DEFAULT_MAX_STEP = 0.01  # 默认最大步长
    STM_DIMENSION = 42  # 状态转移矩阵维度 (6x6 + 6状态)

    def __init__(self, system: CR3BP_System) -> None:
        """初始化动力学

        参数：
        - system: CR3BP_System对象
        """
        # 关联的CR3BP系统
        self.system = system

        # 积分器设置
        self.integrator = "RK45"  # 默认积分器
        self.rtol = self.DEFAULT_TOLERANCE  # 相对容差
        self.atol = self.DEFAULT_TOLERANCE  # 绝对容差
        self.max_step = self.DEFAULT_MAX_STEP  # 最大步长

        # 缓存最近的积分结果
        self.last_trajectory = None  # 最近的轨迹 [t, y]
        self.last_stm = None  # 最近的状态转移矩阵

        # 截面检测设置
        self.cross_section_tolerance = 1e-8  # 截面检测容差
        self.last_crossing = None  # 上次穿过截面的点和时间

        # 能量监控
        self.jacobi_history = []  # Jacobi常数历史
        self.jacobi_error = 0.0  # Jacobi常数误差

        # 计算标志
        self.initialized = True  # 初始化完成

    def equations_of_motion(
        self, t: float, state: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """6维状态向量的运动方程

        参数：
        - t: 时间
        - state: 状态向量 [x, y, z, vx, vy, vz]

        返回：
        - 状态导数 [vx, vy, vz, ax, ay, az]
        """
        mu = self.system.mu

        # 解包状态
        x, y, z, vx, vy, vz = state

        # 计算到两个天体的距离
        r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2)

        # 计算加速度
        ax = 2 * vy + x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
        ay = -2 * vx + y - (1 - mu) * y / r1**3 - mu * y / r2**3
        az = -(1 - mu) * z / r1**3 - mu * z / r2**3

        return np.array([vx, vy, vz, ax, ay, az])

    def equations_with_stm(
        self, t: float, augmented_state: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """42维增广状态向量的运动方程（包含状态转移矩阵）

        参数：
        - t: 时间
        - augmented_state: 增广状态向量 [6状态 + 36个STM元素]

        返回：
        - 增广状态导数
        """
        mu = self.system.mu

        # 解包状态
        x, y, z, vx, vy, vz = augmented_state[:6]

        # 计算到两个天体的距离
        r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2)

        # 计算加速度
        ax = 2 * vy + x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
        ay = -2 * vx + y - (1 - mu) * y / r1**3 - mu * y / r2**3
        az = -(1 - mu) * z / r1**3 - mu * z / r2**3

        # 状态导数
        state_derivative = np.array([vx, vy, vz, ax, ay, az])

        # 提取状态转移矩阵
        stm = augmented_state[6:].reshape((6, 6))

        # 计算雅可比矩阵 A(t)
        # 计算二阶导数
        U_xx = (
            1
            - (1 - mu) * (1 / r1**3 - 3 * (x + mu) ** 2 / r1**5)
            - mu * (1 / r2**3 - 3 * (x - 1 + mu) ** 2 / r2**5)
        )
        U_yy = 1 - (1 - mu) * (1 / r1**3 - 3 * y**2 / r1**5) - mu * (1 / r2**3 - 3 * y**2 / r2**5)
        U_zz = -(1 - mu) / r1**3 - mu / r2**3
        U_xy = 3 * (1 - mu) * (x + mu) * y / r1**5 + 3 * mu * (x - 1 + mu) * y / r2**5
        U_xz = 3 * (1 - mu) * (x + mu) * z / r1**5 + 3 * mu * (x - 1 + mu) * z / r2**5
        U_yz = 3 * (1 - mu) * y * z / r1**5 + 3 * mu * y * z / r2**5

        # 构建雅可比矩阵
        A = np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [U_xx, U_xy, U_xz, 0, 2, 0],
                [U_xy, U_yy, U_yz, -2, 0, 0],
                [U_xz, U_yz, U_zz, 0, 0, 0],
            ]
        )

        # 计算STM导数
        stm_dot = A @ stm

        # 组合导数
        derivative = np.concatenate([state_derivative, stm_dot.flatten()])

        return derivative

    def propagate(
        self,
        initial_state: npt.ArrayLike,
        t_span: Tuple[float, float],
        t_eval: Optional[npt.ArrayLike] = None,
        with_stm: bool = False,
    ) -> Dict[str, Any]:
        """传播轨迹

        参数：
        - initial_state: 初始状态向量
        - t_span: 时间区间 [t0, tf]
        - t_eval: 评估时间点数组（可选）
        - with_stm: 是否计算状态转移矩阵

        返回：
        - 轨迹结果对象
        """
        if with_stm:
            # 创建增广状态（初始STM为单位矩阵）
            initial_stm = np.eye(6).flatten()
            augmented_state = np.concatenate([initial_state, initial_stm])

            # 积分增广状态方程
            result = solve_ivp(
                self.equations_with_stm,
                t_span,
                augmented_state,
                method=self.integrator,
                t_eval=t_eval,
                rtol=self.rtol,
                atol=self.atol,
                max_step=self.max_step,
            )

            # 分离状态和STM
            states = result.y[:6, :].T
            stm_matrices = result.y[6:, :].T.reshape(-1, 6, 6)

            # 存储结果
            self.last_trajectory = (result.t, states)
            self.last_stm = stm_matrices

            # 计算Jacobi常数历史
            self.jacobi_history = [self.compute_jacobi_constant(state) for state in states]
            if len(self.jacobi_history) > 1:
                self.jacobi_error = np.max(np.abs(np.diff(self.jacobi_history)))

            return {
                "time": result.t,
                "states": states,
                "stm": stm_matrices,
                "jacobi": self.jacobi_history,
                "jacobi_error": self.jacobi_error,
            }
        else:
            # 积分普通状态方程
            result = solve_ivp(
                self.equations_of_motion,
                t_span,
                initial_state,
                method=self.integrator,
                t_eval=t_eval,
                rtol=self.rtol,
                atol=self.atol,
                max_step=self.max_step,
            )

            states = result.y.T

            # 存储结果
            self.last_trajectory = (result.t, states)

            # 计算Jacobi常数历史
            self.jacobi_history = [self.compute_jacobi_constant(state) for state in states]
            if len(self.jacobi_history) > 1:
                self.jacobi_error = np.max(np.abs(np.diff(self.jacobi_history)))

            return {
                "time": result.t,
                "states": states,
                "jacobi": self.jacobi_history,
                "jacobi_error": self.jacobi_error,
            }

    def compute_state_transition_matrix(
        self, initial_state: npt.ArrayLike, t: float
    ) -> npt.NDArray[np.floating]:
        """计算状态转移矩阵

        参数：
        - initial_state: 初始状态向量
        - t: 时间

        返回：
        - 状态转移矩阵 (6x6)
        """
        # 传播轨迹并计算STM
        result = self.propagate(initial_state, [0, t], with_stm=True)

        # 返回最终时刻的STM
        return result["stm"][-1]

    def compute_jacobi_constant(self, state: npt.ArrayLike) -> float:
        """实时计算Jacobi常数

        参数：
        - state: 状态向量 [x, y, z, vx, vy, vz]

        返回：
        - Jacobi常数
        """
        return self.system.get_jacobi_constant(state)

    def check_cross_section(
        self, state: npt.ArrayLike, plane: str, value: float
    ) -> bool:
        """检查是否穿过指定截面

        参数：
        - state: 状态向量
        - plane: 截面平面 ('x', 'y', 'z')
        - value: 平面值

        返回：
        - 布尔值，表示是否穿过截面
        """
        if plane == "x":
            return abs(state[0] - value) < self.cross_section_tolerance
        elif plane == "y":
            return abs(state[1] - value) < self.cross_section_tolerance
        elif plane == "z":
            return abs(state[2] - value) < self.cross_section_tolerance
        else:
            raise ValueError(f"无效的平面: {plane}。可用平面: 'x', 'y', 'z'")

    def __str__(self):
        """字符串表示"""
        return f"CR3BP_Dynamics(system={self.system}, integrator='{self.integrator}')"

    def __repr__(self):
        """详细表示"""
        return (
            f"CR3BP_Dynamics(system={self.system}, integrator='{self.integrator}', "
            f"rtol={self.rtol}, atol={self.atol}, max_step={self.max_step})"
        )
