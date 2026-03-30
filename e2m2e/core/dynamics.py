"""
三体问题动力学模块

包含通用 Dynamics 基类和 CR3BP_Dynamics 类，用于计算和积分圆型限制性三体问题的动力学方程。
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, List, Tuple, Optional, Any, Callable, TYPE_CHECKING

import numpy.typing as npt

from .system import CR3BP_System

if TYPE_CHECKING:
    from .orbit import Orbit
    from .system import CR3BP_System as SystemType


class Dynamics:
    """通用天体系统动力学基类

    Attributes:
        system: 关联的系统对象
        integrator: 数值积分器类型
        rtol: 相对积分容差
        atol: 绝对积分容差
        max_step: 最大积分步长
        last_trajectory: 最近一次积分的轨迹 [t, y]
        last_stm: 最近一次积分的状态转移矩阵
        cross_section_tolerance: 截面检测容差
        last_crossing: 上次穿过截面的点和时间
        jacobi_history: Jacobi常数历史记录
        jacobi_error: Jacobi常数误差
        initialized: 初始化完成标志
    """

    DEFAULT_TOLERANCE = 1e-12
    DEFAULT_MAX_STEP = 0.01

    def __init__(self, system: CR3BP_System) -> None:
        """初始化动力学

        Args:
            system: CR3BP_System对象
        """
        self.system = system

        self.integrator = "RK45"
        self.rtol = self.DEFAULT_TOLERANCE
        self.atol = self.DEFAULT_TOLERANCE
        self.max_step = self.DEFAULT_MAX_STEP

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
        """运动方程（子类需实现）

        Args:
            t: 时间
            state: 状态向量

        Returns:
            状态导数

        Raises:
            NotImplementedError: 子类未实现此方法
        """
        raise NotImplementedError("子类必须实现此方法")

    def propagate(
        self,
        initial_state: npt.ArrayLike,
        t_span: Tuple[float, float],
        t_eval: Optional[npt.ArrayLike] = None,
        with_stm: bool = False,
        with_jacobi: bool = False,
    ) -> Dict[str, Any]:
        """传播轨迹

        Args:
            initial_state: 初始状态向量
            t_span: 时间区间 [t0, tf]
            t_eval: 评估时间点数组（可选）
            with_stm: 是否计算状态转移矩阵
            with_jacobi: 是否沿轨迹逐点计算 Jacobi 常数并写入
                ``jacobi`` / ``jacobi_error``（默认关闭以减轻粗积分负担）

        Returns:
            轨迹结果字典，包含 ``time`` 和 ``states`` 键；
            仅当 ``with_jacobi=True`` 时额外包含 ``jacobi`` 与 ``jacobi_error`` 键
        """
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

        self.last_trajectory = (result.t, states)

        out: Dict[str, Any] = {
            "time": result.t,
            "states": states,
        }

        if with_jacobi and hasattr(self, "compute_jacobi_constant"):
            self.jacobi_history = [self.compute_jacobi_constant(state) for state in states]
            if len(self.jacobi_history) > 1:
                self.jacobi_error = float(np.max(np.abs(np.diff(self.jacobi_history))))
            else:
                self.jacobi_error = 0.0
            out["jacobi"] = self.jacobi_history
            out["jacobi_error"] = self.jacobi_error
        else:
            self.jacobi_history = []
            self.jacobi_error = 0.0

        return out

    def compute_jacobi_constant(self, state: npt.ArrayLike) -> float:
        """计算能量常数（子类需实现）

        Args:
            state: 状态向量

        Returns:
            能量常数

        Raises:
            NotImplementedError: 子类未实现此方法
        """
        raise NotImplementedError("子类必须实现此方法")

    def check_cross_section(self, state: npt.ArrayLike, plane: str, value: float) -> bool:
        """检查是否穿过指定截面

        Args:
            state: 状态向量
            plane: 截面平面 ('x', 'y', 'z')
            value: 平面值

        Returns:
            是否穿过截面

        Raises:
            ValueError: 无效的平面参数
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
        return f"{self.__class__.__name__}(system={self.system})"

    def __repr__(self):
        return f"{self.__class__.__name__}(system={self.system}, integrator='{self.integrator}', rtol={self.rtol})"


class CR3BP_Dynamics(Dynamics):
    """CR3BP动力学方程

    封装了CR3BP的动力学模型，提供状态传播、状态转移矩阵计算、
    Jacobi常数计算等核心功能。支持6维状态向量（位置+速度）和
    42维增广状态向量（状态+状态转移矩阵）的数值积分。

    Attributes:
        STM_DIMENSION: 增广状态向量维度（6状态 + 36个STM元素 = 42）
    """

    STM_DIMENSION = 42

    def __init__(self, system: CR3BP_System) -> None:
        """初始化CR3BP动力学

        Args:
            system: CR3BP_System对象，包含质量参数μ等系统常数
        """
        super().__init__(system)

    def equations_of_motion(
        self, t: float, state: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """6维状态向量的运动方程

        Args:
            t: 时间
            state: 状态向量 [x, y, z, vx, vy, vz]

        Returns:
            状态导数 [vx, vy, vz, ax, ay, az]
        """
        mu = self.system.mu

        x, y, z, vx, vy, vz = state

        r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2)

        ax = 2 * vy + x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
        ay = -2 * vx + y - (1 - mu) * y / r1**3 - mu * y / r2**3
        az = -(1 - mu) * z / r1**3 - mu * z / r2**3

        return np.array([vx, vy, vz, ax, ay, az])

    def equations_with_stm(
        self, t: float, augmented_state: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """42维增广状态向量的运动方程（包含状态转移矩阵）

        同时积分状态向量和状态转移矩阵(STM)，满足 dΦ/dt = A(t)·Φ。

        Args:
            t: 时间
            augmented_state: 增广状态向量 [6状态 + 36个STM元素]

        Returns:
            增广状态导数
        """
        mu = self.system.mu

        x, y, z, vx, vy, vz = augmented_state[:6]

        r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2)

        ax = 2 * vy + x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
        ay = -2 * vx + y - (1 - mu) * y / r1**3 - mu * y / r2**3
        az = -(1 - mu) * z / r1**3 - mu * z / r2**3

        state_derivative = np.array([vx, vy, vz, ax, ay, az])

        stm = augmented_state[6:].reshape((6, 6))

        # 伪势能二阶偏导数
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

        # 状态方程雅可比矩阵 A(t)
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

        stm_dot = A @ stm

        derivative = np.concatenate([state_derivative, stm_dot.flatten()])

        return derivative

    def propagate(
        self,
        initial_state: npt.ArrayLike,
        t_span: Tuple[float, float],
        t_eval: Optional[npt.ArrayLike] = None,
        with_stm: bool = False,
        with_jacobi: bool = False,
    ) -> Dict[str, Any]:
        """传播轨迹

        Args:
            initial_state: 初始状态向量
            t_span: 时间区间 [t0, tf]
            t_eval: 评估时间点数组（可选）
            with_stm: 是否计算状态转移矩阵
            with_jacobi: 是否逐点计算 Jacobi 常数（默认 ``False``，
                搜索/大量积分时可显著减少开销）

        Returns:
            轨迹结果字典。始终包含 ``time`` 和 ``states`` 键；
            当 ``with_stm=True`` 时额外包含 ``stm`` 键；
            当 ``with_jacobi=True`` 时额外包含 ``jacobi`` 与 ``jacobi_error`` 键
        """
        if with_stm:
            initial_stm = np.eye(6).flatten()
            augmented_state = np.concatenate([initial_state, initial_stm])

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

            states = result.y[:6, :].T
            stm_matrices = result.y[6:, :].T.reshape(-1, 6, 6)

            self.last_trajectory = (result.t, states)
            self.last_stm = stm_matrices

            out: Dict[str, Any] = {
                "time": result.t,
                "states": states,
                "stm": stm_matrices,
            }

            if with_jacobi:
                self.jacobi_history = [self.compute_jacobi_constant(state) for state in states]
                if len(self.jacobi_history) > 1:
                    self.jacobi_error = float(np.max(np.abs(np.diff(self.jacobi_history))))
                else:
                    self.jacobi_error = 0.0
                out["jacobi"] = self.jacobi_history
                out["jacobi_error"] = self.jacobi_error
            else:
                self.jacobi_history = []
                self.jacobi_error = 0.0

            return out
        else:
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

            self.last_trajectory = (result.t, states)

            out = {
                "time": result.t,
                "states": states,
            }

            if with_jacobi:
                self.jacobi_history = [self.compute_jacobi_constant(state) for state in states]
                if len(self.jacobi_history) > 1:
                    self.jacobi_error = float(np.max(np.abs(np.diff(self.jacobi_history))))
                else:
                    self.jacobi_error = 0.0
                out["jacobi"] = self.jacobi_history
                out["jacobi_error"] = self.jacobi_error
            else:
                self.jacobi_history = []
                self.jacobi_error = 0.0

            return out

    def propagate_orbit_state_at_time(
        self,
        orbit: Orbit,
        t: float,
        integration_dt: float = 0.01,
    ) -> npt.NDArray[np.floating]:
        """从轨道首点状态积分到给定时刻对应的相位（周期轨道上对周期取模）

        Args:
            orbit: 周期轨道数据（须含 ``states``、``times``、有效 ``period``）
            t: 与轨道 ``times`` 一致的时间坐标（绝对时间）
            integration_dt: 构造 ``t_eval`` 的步长

        Returns:
            积分末端状态 ``[x, y, z, vx, vy, vz]``

        Raises:
            ValueError: 轨道无状态或周期无效
        """
        if orbit.states.shape[0] < 1:
            raise ValueError("轨道无状态")
        if orbit.period is None or orbit.period <= 0:
            raise ValueError("轨道周期无效，无法沿周期外推")

        t0 = float(orbit.times[0])
        period = float(orbit.period)
        t_rel = float(np.mod(t - t0, period))
        if t_rel < 1e-14:
            return np.asarray(orbit.states[0], dtype=float).copy()

        n_steps = max(int(np.ceil(t_rel / integration_dt)) + 1, 2)
        t_eval = np.linspace(t0, t0 + t_rel, n_steps)
        result = self.propagate(
            initial_state=orbit.states[0],
            t_span=(t0, t0 + t_rel),
            t_eval=t_eval,
            with_stm=False,
            with_jacobi=False,
        )
        return np.asarray(result["states"][-1], dtype=float)

    def compute_state_transition_matrix(
        self, initial_state: npt.ArrayLike, t: float
    ) -> npt.NDArray[np.floating]:
        """计算状态转移矩阵

        Args:
            initial_state: 初始状态向量
            t: 积分终止时间

        Returns:
            状态转移矩阵 (6x6)
        """
        result = self.propagate(initial_state, (0.0, float(t)), with_stm=True, with_jacobi=False)

        return result["stm"][-1]

    def compute_jacobi_constant(self, state: npt.ArrayLike) -> float:
        """计算Jacobi常数

        Args:
            state: 状态向量 [x, y, z, vx, vy, vz]

        Returns:
            Jacobi常数
        """
        return self.system.get_jacobi_constant(state)

    def check_cross_section(self, state: npt.ArrayLike, plane: str, value: float) -> bool:
        """检查是否穿过指定截面

        Args:
            state: 状态向量
            plane: 截面平面 ('x', 'y', 'z')
            value: 平面值

        Returns:
            是否穿过截面

        Raises:
            ValueError: 无效的平面参数
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
        return f"CR3BP_Dynamics(system={self.system}, integrator='{self.integrator}')"

    def __repr__(self):
        return (
            f"CR3BP_Dynamics(system={self.system}, integrator='{self.integrator}', "
            f"rtol={self.rtol}, atol={self.atol}, max_step={self.max_step})"
        )


def propagate_state_at_orbit_time(
    orbit: Any,
    t: float,
    dynamics: CR3BP_Dynamics,
    integration_dt: float = 0.01,
) -> npt.NDArray[np.floating]:
    """委托 :meth:`CR3BP_Dynamics.propagate_orbit_state_at_time`，便于顶层导入兼容

    Args:
        orbit: 周期轨道数据
        t: 目标时间（绝对时间）
        dynamics: CR3BP动力学对象
        integration_dt: 积分步长

    Returns:
        积分末端状态
    """
    return dynamics.propagate_orbit_state_at_time(orbit, t, integration_dt)
