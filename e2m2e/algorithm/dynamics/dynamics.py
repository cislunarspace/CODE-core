"""
三体问题动力学模块

包含通用 Dynamics 基类和 CR3BP_Dynamics 类，用于计算和积分圆型限制性三体问题的动力学方程。

物理背景
--------
在圆型限制性三体问题 (CR3BP) 中，两个主天体（如地球和月球）绕其公共质心做圆周运动，
第三体（航天器）质量小到不影响两个主天体的运动。采用以质心为原点的旋转坐标系，
使得两个主天体固定在 x 轴上。

坐标系约定：
  - 原点：系统质心
  - x 轴：从质心指向较大天体（质量 1-μ）的方向
  - 较大天体位于 x = -μ，较小天体（质量 μ）位于 x = 1-μ
  - y 轴在轨道平面内垂直于 x 轴
  - z 轴与 x-y 平面正交

所有量均采用无量纲化单位（距离单位 DU = 主天体间距，时间单位 TU 使主天体角速度为 1）。
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp

from e2m2e.integrators import require_rust_extension

from .cr3bp_system import CR3BP_System
from .potential import pseudo_potential_hessian
from .system import System

if TYPE_CHECKING:
    from ...data.types.orbit import Orbit

from e2m2e.integrators import propagate_cr3bp_py, propagate_cr3bp_stm_py

# 跨语言契约（issue #317 第 3.1 项）：Rust ``propagate_cr3bp``（cr3bp.rs）步长
# 塌缩错误形如 "step size collapsed below minimum ..."。``EphemerisDynamics
# ._propagate_state_only`` 据此前缀识别该失败并转成空 states（对齐 scipy 失败
# 语义，让上层 NLP 走 dv=1e10 惩罚）。改 Rust 该错误消息须同步本标记——Rust
# 侧对应注释见 cr3bp.rs ``MIN_STEP`` 处。本字符串是 Python↔Rust 的稳定契约，
# 勿当作普通文案随意改写。
_RUST_STEP_COLLAPSED_MARKER = "step size collapsed"


class Dynamics:
    """通用天体系统动力学基类

    采用 Template Method 模式：基类定义 ``propagate()`` 的算法骨架，
    子类通过钩子方法提供具体的 ODE 函数和步长配置。

    契约（对应 MBSE REQ-002）：
    - ``propagate()`` 返回的 ``states`` 形状始终为 ``(n_points, 6)``
    - ``stm``（如果存在）形状为 ``(n_points, 6, 6)``

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
    """

    DEFAULT_TOLERANCE = 1e-12  # 默认积分容差，双精度机器精度量级，确保数值解精度
    DEFAULT_MAX_STEP = 0.01  # 无量纲（CR3BP），EphemerisDynamics 覆写为秒

    STATE_DIM = 6  # 状态向量维度 [x, y, z, vx, vy, vz]
    STM_DIMENSION = STATE_DIM + STATE_DIM * STATE_DIM  # 42 = 6 + 36
    MIN_DISTANCE = 1e-10  # km (dimensionless), prevents division by zero at singularities

    def __init__(self, system: System) -> None:
        """初始化动力学

        Args:
            system: 系统对象（CR3BP_System 或 EphemerisSystem）
        """
        self.system = system

        # --- 积分器配置 ---
        self.integrator: str = "RK45"
        self.rtol: float = self.DEFAULT_TOLERANCE
        self.atol: float = self.DEFAULT_TOLERANCE
        self.max_step: float = self.DEFAULT_MAX_STEP

        # 缓存最近一次积分结果
        self.last_trajectory: tuple[np.ndarray, np.ndarray] | None = None
        self.last_stm: np.ndarray | None = None  # STM 矩阵数组

        # 截面检测参数
        self.cross_section_tolerance = 1e-8
        self.last_crossing = None

    def _get_eom_func(self, with_stm: bool) -> Callable:
        """获取运动方程函数（钩子方法，子类可覆写）

        Args:
            with_stm: 是否需要 STM 版本的运动方程

        Returns:
            ODE 右端函数
        """
        if with_stm:
            raise NotImplementedError("子类须实现 equations_with_stm 或覆写 _get_eom_func")
        return self.equations_of_motion

    def _get_max_step(self, t_span: tuple[float, float]) -> float:
        """获取当前传播的最大步长（钩子方法，子类可覆写）

        默认实现直接返回 self.max_step。子类（如 EphemerisDynamics）
        可覆写此方法以实现自适应步长。

        Args:
            t_span: 积分时间区间

        Returns:
            最大步长
        """
        return self.max_step

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
        t_span: tuple[float, float],
        t_eval: npt.ArrayLike | None = None,
        with_stm: bool = False,
        with_jacobi: bool = False,
        events: Callable[[float, np.ndarray], float]
        | list[Callable[[float, np.ndarray], float]]
        | None = None,
    ) -> dict[str, Any]:
        """传播轨迹（Template Method）

        统一的传播入口，保证：
        - states 形状为 (n_points, 6)（REQ-002）
        - stm 形状为 (n_points, 6, 6)（当 with_stm=True 时）
        - time 数组单调递增

        Args:
            initial_state: 初始状态向量
            t_span: 时间区间 [t0, tf]
            t_eval: 评估时间点数组（可选）
            with_stm: 是否计算状态转移矩阵
            with_jacobi: 是否沿轨迹逐点计算 Jacobi 常数
            events: 事件函数（单个 callable 或列表），scipy ``solve_ivp``
                语义：``g(t, state) -> float``，零点即事件面；可给函数对象
                设 ``terminal = True``（触发即停）与 ``direction``（> 0 只记
                上行穿越，< 0 只记下行，0 双向）属性。
                ``with_stm=True`` 时事件函数接收 42 维增广状态。

        Returns:
            轨迹结果字典，包含 ``time`` 和 ``states`` 键；
            当 ``with_stm=True`` 时额外包含 ``stm`` 键；
            当 ``with_jacobi=True`` 时额外包含 ``jacobi`` 与 ``jacobi_error`` 键；
            当传入 ``events`` 时额外包含 ``t_events`` 与 ``y_events`` 键
            （逐事件的触发时刻与状态数组，scipy 语义）
        """
        initial_state = np.asarray(initial_state, dtype=float)
        max_step = self._get_max_step(t_span)

        if events is not None and callable(events):
            events = [events]

        if with_stm:
            return self._propagate_with_stm(
                initial_state, t_span, t_eval, max_step, with_jacobi, events
            )
        else:
            return self._propagate_state_only(
                initial_state, t_span, t_eval, max_step, with_jacobi, events
            )

    def _propagate_with_stm(
        self,
        initial_state: np.ndarray,
        t_span: tuple[float, float],
        t_eval: npt.ArrayLike | None,
        max_step: float,
        with_jacobi: bool,
        events: list[Callable[[float, np.ndarray], float]] | None = None,
    ) -> dict[str, Any]:
        """增广状态积分（含 STM）

        初始 STM 设为单位矩阵，拼接为 STATE_DIM + STATE_DIM² 维增广状态后积分。
        """
        initial_stm = np.eye(self.STATE_DIM).flatten()
        augmented_state = np.concatenate([initial_state, initial_stm])

        eom_func = self._get_eom_func(with_stm=True)
        result = solve_ivp(
            eom_func,
            t_span,
            augmented_state,
            method=self.integrator,
            t_eval=t_eval,
            rtol=self.rtol,
            atol=self.atol,
            max_step=max_step,
            events=events,
        )

        # 从增广结果中分离状态和 STM
        n = self.STATE_DIM
        states = result.y[:n, :].T
        stm_matrices = result.y[n:, :].T.reshape(-1, n, n)

        self.last_trajectory = (result.t, states)
        self.last_stm = stm_matrices

        out: dict[str, Any] = {
            "time": result.t,
            "states": states,
            "stm": stm_matrices,
        }

        if events is not None:
            out["t_events"] = result.t_events
            out["y_events"] = result.y_events

        if with_jacobi:
            out = self._handle_jacobi(states, out)

        return out

    def _propagate_state_only(
        self,
        initial_state: np.ndarray,
        t_span: tuple[float, float],
        t_eval: npt.ArrayLike | None,
        max_step: float,
        with_jacobi: bool,
        events: list[Callable[[float, np.ndarray], float]] | None = None,
    ) -> dict[str, Any]:
        """纯状态积分（不含 STM）"""
        eom_func = self._get_eom_func(with_stm=False)
        result = solve_ivp(
            eom_func,
            t_span,
            initial_state,
            method=self.integrator,
            t_eval=t_eval,
            rtol=self.rtol,
            atol=self.atol,
            max_step=max_step,
            events=events,
        )

        # result.y 形状为 (6, n_points)，转置为 (n_points, 6) — REQ-002
        states = result.y.T

        self.last_trajectory = (result.t, states)

        out: dict[str, Any] = {
            "time": result.t,
            "states": states,
        }

        if events is not None:
            out["t_events"] = result.t_events
            out["y_events"] = result.y_events

        if with_jacobi:
            out = self._handle_jacobi(states, out)

        return out

    def _handle_jacobi(self, states: np.ndarray, out: dict[str, Any]) -> dict[str, Any]:
        """沿轨迹计算 Jacobi 常数的钩子方法。

        基类默认为 no-op。CR3BP_Dynamics 覆写此方法以计算 Jacobi 常数。
        EphemerisDynamics 继承 no-op（N 体问题无 Jacobi 积分）。

        Args:
            states: 状态序列，形状 (n, 6)
            out: 输出字典

        Returns:
            更新后的输出字典（基类直接返回不修改）
        """
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
        state = np.asarray(state, dtype=float)
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
        return (
            f"{self.__class__.__name__}("
            f"system={self.system}, integrator='{self.integrator}', rtol={self.rtol})"
        )


class CR3BP_Dynamics(Dynamics):
    """CR3BP动力学方程

    封装了CR3BP的动力学模型，提供状态传播、状态转移矩阵计算、
    Jacobi常数计算等核心功能。支持6维状态向量（位置+速度）和
    42维增广状态向量（状态+状态转移矩阵）的数值积分。

    CR3BP 运动方程（旋转坐标系中）：
        ẍ - 2ẏ = ∂Ω/∂x
        ÿ + 2ẋ = ∂Ω/∂y
        z̈       = ∂Ω/∂z

    其中 Ω 为伪势能（见 equations_of_motion 方法的详细注释），
    等号左侧的 2ẏ、-2ẋ 项为科里奥利力（Coriolis），伪势能中
    已包含离心力项 x²/2 + y²/2。

    Attributes:
        system: CR3BP 系统对象，提供 mu 等系统常数与 Jacobi 常数计算。
    """

    system: CR3BP_System

    def __init__(self, system: CR3BP_System) -> None:
        """初始化CR3BP动力学

        Args:
            system: CR3BP_System对象，包含质量参数μ等系统常数
        """
        super().__init__(system)
        # Jacobi 常数监测（仅 CR3BP 有定义，不在基类中）
        self.jacobi_history: list[float] = []
        self.jacobi_error: float = 0.0

    def _get_eom_func(self, with_stm: bool) -> Callable:
        """返回 CR3BP 运动方程函数"""
        if with_stm:
            return self.equations_with_stm
        return self.equations_of_motion

    def equations_of_motion(
        self, t: float, state: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """6维状态向量的运动方程

        实现 CR3BP 在旋转坐标系中的运动方程。旋转坐标系以两个主天体的
        公共质心为原点，与主天体同步旋转（角速度 ω = 1），因此两个主天体
        在坐标系中固定不动。

        在旋转坐标系中，运动方程为：
            ẍ - 2ẏ = ∂Ω/∂x    (x 方向：离心力 + 引力 + 科里奥利力)
            ÿ + 2ẋ = ∂Ω/∂y    (y 方向：离心力 + 引力 + 科里奥利力)
            z̈       = ∂Ω/∂z    (z 方向：仅引力，无科里奥利力)

        伪势能 Ω = (x² + y²)/2 + (1-μ)/r₁ + μ/r₂，其偏导数为：
            ∂Ω/∂x = x - (1-μ)(x+μ)/r₁³ - μ(x-1+μ)/r₂³
            ∂Ω/∂y = y - (1-μ)y/r₁³     - μy/r₂³
            ∂Ω/∂z =   - (1-μ)z/r₁³     - μz/r₂³

        因此加速度各项的物理含义：
          - "x" / "y" 项：离心力（伪势能中的二次项贡献）
          - "(1-μ)(x+μ)/r₁³" 等：较大天体（如地球）的引力加速度
          - "μ(x-1+μ)/r₂³" 等：较小天体（如月球）的引力加速度
          - "2vy" / "-2vx"：科里奥利力（旋转坐标系中的虚拟力）

        Args:
            t: 时间（旋转坐标系中，CR3BP方程不显含时间，即自治系统）
            state: 状态向量 [x, y, z, vx, vy, vz]

        Returns:
            状态导数 [vx, vy, vz, ax, ay, az]
        """
        mu = self.system.mu  # 质量参数 μ = m₂/(m₁+m₂)，m₂ 为较小天体质量

        x, y, z, vx, vy, vz = state

        # r₁：航天器到较大天体（质量 1-μ，位于 x=-μ）的距离
        r1 = max(np.sqrt((x + mu) ** 2 + y**2 + z**2), self.MIN_DISTANCE)
        # r₂：航天器到较小天体（质量 μ，位于 x=1-μ）的距离
        r2 = max(np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2), self.MIN_DISTANCE)

        # --- x 方向加速度 ---
        ax = 2 * vy + x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
        # --- y 方向加速度 ---
        ay = -2 * vx + y - (1 - mu) * y / r1**3 - mu * y / r2**3
        # --- z 方向加速度 ---
        az = -(1 - mu) * z / r1**3 - mu * z / r2**3

        return np.array([vx, vy, vz, ax, ay, az])

    def compute_jacobian_A(self, state: npt.NDArray[np.floating]) -> np.ndarray:
        """计算 CR3BP 状态方程的雅可比矩阵 A(t)

        A(t) 是 6x6 矩阵，满足 dΦ/dt = A(t)·Φ。
        结构如下::

            | 0₃ₓ₃  I₃ₓ₃ |    位置方程的雅可比：∂(v)/∂(r,v) = [0, I]
            | U_ij   Ω   |    速度方程的雅可比：∂(a)/∂(r,v) = [U, Ω]

        此方法提取自 equations_with_stm，供 Continuation 等模块复用（REQ-103）。

        Args:
            state: 状态向量 [x, y, z, vx, vy, vz]

        Returns:
            6x6 雅可比矩阵 A
        """
        mu = self.system.mu
        x, y, z = state[0], state[1], state[2]

        H = pseudo_potential_hessian(mu, x, y, z)

        A = np.zeros((6, 6))
        A[:3, 3:] = np.eye(3)
        A[3:, :3] = H
        A[3, 4] = 2.0
        A[4, 3] = -2.0
        return A

    def equations_with_stm(
        self, t: float, augmented_state: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """42维增广状态向量的运动方程（包含状态转移矩阵）

        同时积分状态向量和状态转移矩阵(STM)，满足 dΦ/dt = A(t)·Φ。

        状态转移矩阵 Φ(t, t₀) 将初始状态的微小扰动映射到当前时刻：
            δx(t) = Φ(t, t₀) · δx(t₀)

        通过将 Φ 拉伸为 36 维向量并与 6 维状态拼接为 42 维增广状态，
        可以用标准的 ODE 积分器同时求解轨道和 STM。

        Args:
            t: 时间
            augmented_state: 增广状态向量 [6状态 + 36个STM元素]

        Returns:
            增广状态导数
        """
        state = augmented_state[:6]
        stm = augmented_state[6:].reshape((6, 6))

        state_derivative = self.equations_of_motion(t, state)
        A = self.compute_jacobian_A(state)
        stm_dot = A @ stm

        return np.concatenate([state_derivative, stm_dot.flatten()])

    def _propagate_with_stm(
        self,
        initial_state: np.ndarray,
        t_span: tuple[float, float],
        t_eval: npt.ArrayLike | None,
        max_step: float,
        with_jacobi: bool,
        events: list[Callable[[float, np.ndarray], float]] | None = None,
    ) -> dict[str, Any]:
        """增广状态积分（含 STM），优先走 Rust 快速路径。

        Rust 路径不支持事件检测；仅当调用者显式传入 ``events`` 时改走
        scipy。这是 ADR 0002 规定的事件语义例外，不取决于扩展是否可用，
        并非 Rust 缺失时的 fallback。无 events 时要求 Rust 扩展可用（issue
        #378：缺失即抛 RustExtensionUnavailableError，不静默降级 scipy）。
        """
        if events is not None:
            warnings.warn(
                "CR3BP 显式传入 events 时使用 scipy 事件积分；这由 events 输入触发，"
                "与 Rust 扩展可用性无关。",
                stacklevel=2,
            )
            return super()._propagate_with_stm(
                initial_state, t_span, t_eval, max_step, with_jacobi, events
            )
        require_rust_extension("propagate_cr3bp_stm_py")
        return self._propagate_with_stm_rust(initial_state, t_span, t_eval, max_step, with_jacobi)

    def _propagate_with_stm_rust(
        self,
        initial_state: np.ndarray,
        t_span: tuple[float, float],
        t_eval: npt.ArrayLike | None,
        max_step: float,
        with_jacobi: bool,
    ) -> dict[str, Any]:
        """Rust 快速路径：调用 propagate_cr3bp_stm_py 完成 STM 传播。

        初始 STM 由 Rust 侧设为单位矩阵，返回的 stm 形状为 (n, 6, 6)，
        ``stm[k][i][j] = ∂state(t_k)[i]/∂state(t0)[j]``。
        """
        mu = float(self.system.mu)
        if t_eval is not None:
            t_eval_list = [float(t) for t in np.asarray(t_eval, dtype=float).ravel()]
        elif t_span[0] == t_span[1]:
            # 零跨度（如 compute_state_transition_matrix(t=0)）：合成 [t0, t0] 会让
            # Rust 核心输出点去重只产 1 点、触发长度校验；退化为单点 [t0]，与 scipy
            # t_eval=None 的零跨度行为一致（STM 即单位阵、状态即初值，无需积分）。
            t_eval_list = [float(t_span[0])]
        else:
            t_eval_list = [float(t_span[0]), float(t_span[1])]

        result = propagate_cr3bp_stm_py(
            mu=mu,
            t_span=(float(t_span[0]), float(t_span[1])),
            t_eval=t_eval_list,
            initial_state=[float(x) for x in initial_state[:6]],
            rtol=self.rtol,
            atol=self.atol,
            max_step=float(max_step),
        )

        states = np.array(result["states"])
        stm = np.array(result["stm"]).reshape(-1, 6, 6)
        time = np.array(result["time"])

        # 防御性校验：Rust 侧任何提前退出都必须在这里暴露，不允许把截断
        # 结果当完整轨迹返回（issue #246，照抄 ephemeris_dynamics.py）。
        if len(time) != len(t_eval_list):
            raise RuntimeError(
                f"Rust STM propagation returned {len(time)} of {len(t_eval_list)} "
                f"requested time points; the trajectory is truncated"
            )

        self.last_trajectory = (time, states)
        self.last_stm = stm

        out: dict[str, Any] = {"time": time, "states": states, "stm": stm}
        if with_jacobi:
            out = self._handle_jacobi(states, out)
        return out

    def _propagate_state_only(
        self,
        initial_state: np.ndarray,
        t_span: tuple[float, float],
        t_eval: npt.ArrayLike | None,
        max_step: float,
        with_jacobi: bool,
        events: list[Callable[[float, np.ndarray], float]] | None = None,
    ) -> dict[str, Any]:
        """纯状态积分（不含 STM），优先走 Rust 快速路径。

        Rust 路径不支持事件检测；仅当调用者显式传入 ``events`` 时改走
        scipy。这是 ADR 0002 规定的事件语义例外，不取决于扩展是否可用，
        并非 Rust 缺失时的 fallback。无 events 时要求 Rust 扩展可用（issue
        #378：缺失即抛 RustExtensionUnavailableError，不静默降级 scipy）。
        """
        if events is not None:
            warnings.warn(
                "CR3BP 显式传入 events 时使用 scipy 事件积分；这由 events 输入触发，"
                "与 Rust 扩展可用性无关。",
                stacklevel=2,
            )
            return super()._propagate_state_only(
                initial_state, t_span, t_eval, max_step, with_jacobi, events
            )
        require_rust_extension("propagate_cr3bp_py")
        mu = float(self.system.mu)
        if t_eval is not None:
            t_eval_list = [float(t) for t in np.asarray(t_eval, dtype=float).ravel()]
        elif t_span[0] == t_span[1]:
            # 零跨度：同 _propagate_with_stm_rust，退化为单点避免重复点触发长度校验。
            t_eval_list = [float(t_span[0])]
        else:
            t_eval_list = [float(t_span[0]), float(t_span[1])]

        # 步长塌缩时 Rust 抛 RuntimeError；这里 catch 后返回空 states，
        # 对齐 scipy 路径失败语义（失败返回空、不 raise），让上层
        # （TransferOptimization._evaluate_all）走 dv=1e10 惩罚，使 NLP
        # 优化器绕开发散点。仅 catch 步长塌缩；其他 RuntimeError（如截断）
        # 仍向上抛——那是真实 bug，该暴露。
        try:
            result = propagate_cr3bp_py(
                mu=mu,
                t_span=(float(t_span[0]), float(t_span[1])),
                t_eval=t_eval_list,
                initial_state=[float(x) for x in initial_state[:6]],
                rtol=self.rtol,
                atol=self.atol,
                max_step=float(max_step),
            )

            states = np.array(result["states"])
            time = np.array(result["time"])

            if len(time) != len(t_eval_list):
                raise RuntimeError(
                    f"Rust propagation returned {len(time)} of {len(t_eval_list)} "
                    f"requested time points; the trajectory is truncated"
                )
        except RuntimeError as e:
            if _RUST_STEP_COLLAPSED_MARKER in str(e):
                return {"time": np.array([]), "states": np.empty((0, 6))}
            raise

        self.last_trajectory = (time, states)

        out: dict[str, Any] = {"time": time, "states": states}
        if with_jacobi:
            out = self._handle_jacobi(states, out)
        return out

    def propagate_orbit_state_at_time(
        self,
        orbit: Orbit,
        t: float,
        integration_dt: float = 0.01,
    ) -> npt.NDArray[np.floating]:
        """从轨道首点状态积分到给定时刻对应的相位（周期轨道上对周期取模）

        利用周期轨道的周期性，将目标时间对周期取模后从轨道起始状态
        重新积分，得到该相位处的精确状态。

        Args:
            orbit: 周期轨道数据（须含 ``states``、``times``、有效 ``period``）
            t: 与轨道 ``times`` 一致的时间坐标（绝对时间）
            integration_dt: 构造 ``t_eval`` 的步长

        Returns:
            积分末端状态 ``[x, y, z, vx, vy, vz]``

        Raises:
            ValueError: 轨道无状态或周期无效
            RuntimeError: 重传播未返回状态
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
        states = result["states"]
        if len(states) > 0:
            return np.asarray(states[-1], dtype=float)
        raise RuntimeError("CR3BP 轨道状态传播失败：未返回状态；Rust 积分器可能发生步长塌缩")

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

    def _handle_jacobi(self, states: np.ndarray, out: dict[str, Any]) -> dict[str, Any]:
        """沿轨迹逐点计算 Jacobi 常数"""
        self.jacobi_history = [self.compute_jacobi_constant(state) for state in states]
        if len(self.jacobi_history) > 1:
            self.jacobi_error = float(np.max(np.abs(np.diff(self.jacobi_history))))
        else:
            self.jacobi_error = 0.0
        out["jacobi"] = self.jacobi_history
        out["jacobi_error"] = self.jacobi_error
        return out

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
