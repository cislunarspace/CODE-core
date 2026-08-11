"""双圆限制性四体问题（BCR4BP）动力学模块。

包含 ``BCR4BP_Dynamics`` 类：在 CR3BP 运动方程上叠加太阳直接项与间接项
摄动。太阳位置由 ``BCR4BPSystem.sun_position(t)`` 解析给出，系统显式含
时间 t（时间周期系统），``propagate`` 接口语义与 ``CR3BP_Dynamics``
一致（无量纲时间、无量纲状态）。

运动方程（地月会合旋转系，无量纲）：

    ẍ - 2ẏ = ∂Ω/∂x + a_sun,x
    ÿ + 2ẋ = ∂Ω/∂y + a_sun,y
    z̈       = ∂Ω/∂z + a_sun,z

其中太阳摄动加速度（与 ``ThirdBodyGravity`` 同一公式）：

    a_sun = ``-m_s · [ (r - r_s)/|r - r_s|³ + r_s/|r_s|³ ]``

第一项为直接项（太阳对航天器的引力），第二项为间接项（扣除太阳对
系统质心的引力）。
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from e2m2e.integrators import propagate_bcr4bp_py, propagate_bcr4bp_stm_py, require_rust_extension

from .bcr4bp_system import BCR4BPSystem
from .dynamics import Dynamics
from .potential import pseudo_potential_hessian


class BCR4BP_Dynamics(Dynamics):
    """BCR4BP 动力学方程（CR3BP + 太阳质点摄动）

    支持 6 维状态向量与 42 维增广状态向量（状态 + 状态转移矩阵）的
    数值积分。STM 变分方程 dΦ/dt = A(t)·Φ 中的雅可比 A(t) 显式依赖
    时间：太阳项对位置的偏导随 r_s(t) 变化。

    BCR4BP 无 Jacobi 积分（太阳项显式含时），``with_jacobi=True`` 会
    抛出 ``NotImplementedError``。

    Attributes:
        system: BCR4BPSystem 对象，提供 mu 与太阳参数。
    """

    system: BCR4BPSystem

    def __init__(self, system: BCR4BPSystem) -> None:
        """初始化 BCR4BP 动力学

        Args:
            system: BCR4BPSystem 对象
        """
        super().__init__(system)

    def _get_eom_func(self, with_stm: bool) -> Callable:
        """返回 BCR4BP 运动方程函数"""
        if with_stm:
            return self.equations_with_stm
        return self.equations_of_motion

    def sun_acceleration(self, t: float, position: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """太阳摄动加速度（直接项 + 间接项，无量纲）

            a_sun = ``-m_s · [ (r - r_s)/|r - r_s|³ + r_s/|r_s|³ ]``

        与 ``ThirdBodyGravity`` 的公式一致（间接项以地月质心为参考点）。

        Args:
            t: 无量纲时间
            position: 航天器位置（无量纲），形状 (3,)

        Returns:
            加速度向量，形状 (3,)
        """
        r = np.asarray(position, dtype=float)
        r_s = self.system.sun_position(t)
        d = r - r_s
        d_norm = max(float(np.linalg.norm(d)), self.MIN_DISTANCE)
        r_s_norm = float(np.linalg.norm(r_s))
        return -self.system.sun_mass * (d / d_norm**3 + r_s / r_s_norm**3)

    def equations_of_motion(
        self, t: float, state: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """6 维状态向量的运动方程（显式含时间 t）

        CR3BP 方程 + 太阳摄动项。太阳位置 r_s(t) 使方程显式依赖时间，
        系统为非自治（时间周期）系统。

        Args:
            t: 无量纲时间（会合系；与 CR3BP 不同，此处不能忽略）
            state: 状态向量 [x, y, z, vx, vy, vz]

        Returns:
            状态导数 [vx, vy, vz, ax, ay, az]
        """
        mu = self.system.mu

        x, y, z, vx, vy, vz = state

        r1 = max(np.sqrt((x + mu) ** 2 + y**2 + z**2), self.MIN_DISTANCE)
        r2 = max(np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2), self.MIN_DISTANCE)

        ax = 2 * vy + x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
        ay = -2 * vx + y - (1 - mu) * y / r1**3 - mu * y / r2**3
        az = -(1 - mu) * z / r1**3 - mu * z / r2**3

        a_sun = self.sun_acceleration(t, state[:3])

        return np.array([vx, vy, vz, ax + a_sun[0], ay + a_sun[1], az + a_sun[2]])

    def compute_jacobian_A(self, t: float, state: npt.NDArray[np.floating]) -> np.ndarray:
        """计算 BCR4BP 状态方程的雅可比矩阵 A(t)

        结构与 CR3BP 相同::

            | 0₃ₓ₃  I₃ₓ₃ |
            | U_ij   Ω   |

        但左下块在 CR3BP 伪势能 Hessian 上叠加太阳项对位置的偏导
        （第三体雅可比，标准公式）：

            J_sun = ``-m_s · ( I/|d|³ - 3·d·dᵀ/|d|⁵ )``,   d = r - r_s(t)

        间接项 ``-m_s·r_s/|r_s|³`` 不依赖航天器位置，偏导为零。

        Args:
            t: 无量纲时间（太阳位置随时间变化，A 显式含时）
            state: 状态向量 [x, y, z, vx, vy, vz]

        Returns:
            6x6 雅可比矩阵 A
        """
        mu = self.system.mu
        x, y, z = state[0], state[1], state[2]

        H = pseudo_potential_hessian(mu, x, y, z)

        r_s = self.system.sun_position(t)
        d = np.array([x, y, z], dtype=float) - r_s
        d_norm = max(float(np.linalg.norm(d)), self.MIN_DISTANCE)
        J_sun = -self.system.sun_mass / d_norm**3 * (np.eye(3) - 3.0 * np.outer(d, d) / d_norm**2)

        A = np.zeros((6, 6))
        A[:3, 3:] = np.eye(3)
        A[3:, :3] = H + J_sun
        A[3, 4] = 2.0
        A[4, 3] = -2.0
        return A

    def equations_with_stm(
        self, t: float, augmented_state: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """42 维增广状态向量的运动方程（包含状态转移矩阵）

        与 CR3BP 的增广方程相同结构，但 A(t) 显式含时（太阳项）。

        Args:
            t: 无量纲时间
            augmented_state: 增广状态向量 [6 状态 + 36 个 STM 元素]

        Returns:
            增广状态导数
        """
        state = augmented_state[:6]
        stm = augmented_state[6:].reshape((6, 6))

        state_derivative = self.equations_of_motion(t, state)
        A = self.compute_jacobian_A(t, state)
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

        BCR4BP 的 Rust 路径不支持事件检测；仅当调用者显式传入 ``events``
        时走 scipy 路径并发出警告。这是 ADR 0002 规定的事件语义例外，
        不取决于 Rust 扩展是否可用，非扩展缺失 fallback。无 events 时要求
        Rust 扩展可用（issue #378：缺失即抛 RustExtensionUnavailableError，
        不静默降级 scipy）。
        """
        if events is not None:
            warnings.warn(
                "BCR4BP 显式传入 events 时回退到 scipy 事件积分；这是由 events 输入"
                "触发的设计例外，与 Rust 扩展可用性无关。",
                stacklevel=2,
            )
            return super()._propagate_with_stm(
                initial_state, t_span, t_eval, max_step, with_jacobi, events
            )
        require_rust_extension("propagate_bcr4bp_stm_py")
        return self._propagate_with_stm_rust(initial_state, t_span, t_eval, max_step, with_jacobi)

    def _propagate_with_stm_rust(
        self,
        initial_state: np.ndarray,
        t_span: tuple[float, float],
        t_eval: npt.ArrayLike | None,
        max_step: float,
        with_jacobi: bool,
    ) -> dict[str, Any]:
        """Rust 快速路径：调用 propagate_bcr4bp_stm_py 完成 STM 传播。

        初始 STM 由 Rust 侧设为单位矩阵，返回的 stm 形状为 (n, 6, 6)，
        ``stm[k][i][j] = ∂state(t_k)[i]/∂state(t0)[j]``。
        """
        mu = float(self.system.mu)
        if t_eval is not None:
            t_eval_list = [float(t) for t in np.asarray(t_eval, dtype=float).ravel()]
        else:
            t_eval_list = [float(t_span[0]), float(t_span[1])]

        result = propagate_bcr4bp_stm_py(
            mu=mu,
            mu_sun=float(self.system.sun_mass),
            sun_distance=float(self.system.sun_distance),
            sun_angular_rate=float(self.system.sun_angular_rate),
            sun_phase0=float(self.system.sun_phase0),
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
        # 结果当完整轨迹返回（issue #246，照抄 cr3bp 的 _propagate_with_stm_rust）。
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

        BCR4BP 的 Rust 路径不支持事件检测；仅当调用者显式传入 ``events``
        时走 scipy 路径并发出警告。这是 ADR 0002 规定的事件语义例外，
        不取决于 Rust 扩展是否可用，非扩展缺失 fallback。无 events 时要求
        Rust 扩展可用（issue #378：缺失即抛 RustExtensionUnavailableError，
        不静默降级 scipy）。
        """
        if events is not None:
            warnings.warn(
                "BCR4BP 显式传入 events 时回退到 scipy 事件积分；这是由 events 输入"
                "触发的设计例外，与 Rust 扩展可用性无关。",
                stacklevel=2,
            )
            return super()._propagate_state_only(
                initial_state, t_span, t_eval, max_step, with_jacobi, events
            )
        require_rust_extension("propagate_bcr4bp_py")
        mu = float(self.system.mu)
        if t_eval is not None:
            t_eval_list = [float(t) for t in np.asarray(t_eval, dtype=float).ravel()]
        else:
            t_eval_list = [float(t_span[0]), float(t_span[1])]

        result = propagate_bcr4bp_py(
            mu=mu,
            mu_sun=float(self.system.sun_mass),
            sun_distance=float(self.system.sun_distance),
            sun_angular_rate=float(self.system.sun_angular_rate),
            sun_phase0=float(self.system.sun_phase0),
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

        self.last_trajectory = (time, states)

        out: dict[str, Any] = {"time": time, "states": states}
        if with_jacobi:
            out = self._handle_jacobi(states, out)
        return out

    def compute_state_transition_matrix(
        self, initial_state: npt.ArrayLike, t: float, t0: float = 0.0
    ) -> npt.NDArray[np.floating]:
        """计算状态转移矩阵 Φ(t, t0)

        BCR4BP 显式含时，STM 依赖起止时刻（而非仅时间跨度），
        故比 CR3BP 版本多一个 ``t0`` 参数。

        Args:
            initial_state: 初始状态向量
            t: 积分终止时间
            t0: 积分起始时间（决定太阳初始相位），默认 0

        Returns:
            状态转移矩阵 (6x6)
        """
        result = self.propagate(initial_state, (float(t0), float(t)), with_stm=True)
        return result["stm"][-1]

    def compute_jacobi_constant(self, state: npt.ArrayLike) -> float:
        """BCR4BP 无 Jacobi 积分，调用抛出 NotImplementedError"""
        raise NotImplementedError("BCR4BP 是时间周期系统，无 Jacobi 积分")

    def _handle_jacobi(self, states: np.ndarray, out: dict[str, Any]) -> dict[str, Any]:
        """BCR4BP 无 Jacobi 积分，with_jacobi=True 时报错"""
        raise NotImplementedError("BCR4BP 是时间周期系统，无 Jacobi 积分")

    def __str__(self):
        return f"BCR4BP_Dynamics(system={self.system}, integrator='{self.integrator}')"

    def __repr__(self):
        return (
            f"BCR4BP_Dynamics(system={self.system}, integrator='{self.integrator}', "
            f"rtol={self.rtol}, atol={self.atol}, max_step={self.max_step})"
        )
