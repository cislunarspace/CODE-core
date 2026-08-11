"""块三对角多重打靶法（Block-Tridiagonal Multiple Shooting）。

对应 qiao ``Code05_DynSubs_Gfunc.py`` 的多重打靶 Newton 迭代部分：

- 在 ``t_Q`` 节点上每段独立积分并组装状态转移矩阵 ``Φ_i``；
- 用块三对角消元（``L``/``D``/``X_d`` 前代 + 回代）解连续性残差
  ``Xf_i - X_Q_{i+1}``；
- 在多次迭代中收敛，得到只含受迫频率的动力学替代轨道 ``b(t)``。

与既有 :class:`e2m2e.algorithms.multiple_shooting.MultipleShooting` 不同，
本模块：

- 不引入可变时间节点；
- 不使用工作池并行（qiao 流水线单进程即可，下游 ``fft``/``w_func``
  也是单线程串行）；
- 暴露 :class:`SubstituteSolver` 协议，把"如何积分一段弧并给出
  ``(Xf, Φ)``"留给调用方注入，便于：

  1. 单元测试注入假动力学；
  2. 未来把现有 :class:`MultipleShooting` 适配到本接口
     （多进程并行路径）。

Public API：

- :class:`SubstituteSolver` —— 协议；
- :class:`ODESubstituteSolver` —— 协议的内置实现（默认走 scipy ODE 积分）；
- :func:`solve_block_tridiagonal` —— 单轮解的纯函数；
- :func:`multiple_shooting_newton` —— 多轮 Newton 迭代封装。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ...data.templates import ConvergenceState, FailureCause
from ..results import ResultStatus

#: ``(N, 6)`` 节点状态数组；每行 ``[q1, q2, q3, p1, p2, p3]`` （rho 坐标）。
PatchStates = npt.NDArray[np.floating]
#: ``(N-1, 6, 6)`` 状态转移矩阵 ``Φ_i = Φ(t_{i+1}; t_i, X_i)``。
STMStack = npt.NDArray[np.floating]


@runtime_checkable
class SubstituteSolver(Protocol):
    """动力学替代打靶求解器协议。

    给定 ``(t0, tf, X0)``，返回 ``(Xf, Phi)``：

    - ``Xf``: ``(6,)`` 段终端状态；
    - ``Phi``: ``(6, 6)`` 段状态转移矩阵 ``∂Xf/∂X0``。
    """

    def propagate_segment(
        self, t0: float, tf: float, x0: npt.ArrayLike
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]: ...


@dataclass(frozen=True)
class ShootingPatch:
    """打靶节点与时间。

    Attributes:
        t_Q: 节点时间序列，形状 ``(N,)``，归一化 TU。
        X_Q: 节点状态序列，形状 ``(N, 6)``，归一化坐标。
    """

    t_Q: npt.NDArray[np.floating]
    X_Q: PatchStates

    def __post_init__(self) -> None:
        if self.t_Q.ndim != 1:
            raise ValueError(f"t_Q 必须是一维数组，得到形状 {self.t_Q.shape}")
        if self.X_Q.ndim != 2 or self.X_Q.shape[1] != 6:
            raise ValueError(f"X_Q 必须是 (N, 6) 形状，得到 {self.X_Q.shape}")
        if self.t_Q.shape[0] != self.X_Q.shape[0]:
            raise ValueError(f"t_Q 与 X_Q 长度不一致：{self.t_Q.shape[0]} vs {self.X_Q.shape[0]}")


@dataclass(frozen=True)
class MultipleShootingResult:
    """多重打靶迭代结果。

    Attributes:
        t_Q: 收敛后节点时间数组。
        X_Q: 收敛后节点状态数组，形状 ``(N, 6)``。
        max_residual: 最终迭代最大连续性残差 ``max_i ‖Xf_i − X_Q_{i+1}‖``。
        mean_residual: 平均连续性残差。
        iterations: 实际迭代轮数。
        status: 算法最终状态。
        cause: 算法终止原因。
        message: 人类可读的终止说明。
        residual_history: 每轮最大残差的历史。
    """

    t_Q: npt.NDArray[np.floating]
    X_Q: PatchStates
    max_residual: float
    mean_residual: float
    iterations: int
    status: ConvergenceState
    cause: FailureCause
    message: str
    residual_history: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)


# ---------------------------------------------------------------------------
# 块三对角消元
# ---------------------------------------------------------------------------


def solve_block_tridiagonal(
    phi_stack: STMStack,
    xf_stack: npt.NDArray[np.floating],
    X_Q: PatchStates,
) -> tuple[npt.NDArray[np.floating], list[float]]:
    """对给定段结果做单轮块三对角 Newton 修正。

    对应 qiao ``Code05_DynSubs_Gfunc.py`` 第 100–145 行。

    Args:
        phi_stack: ``(N-1, 6, 6)`` 状态转移矩阵栈。
        xf_stack: ``(N-1, 6)`` 段终端状态。
        X_Q: ``(N, 6)`` 当前节点状态（原地更新）。

    Returns:
        ``(delta_Q, errors)``：
        ``delta_Q`` 是 ``(N, 6)`` 的修正量；``errors`` 是每段连续性残差范数。
    """
    n_seg = phi_stack.shape[0]
    if xf_stack.shape != (n_seg, 6):
        raise ValueError(f"xf_stack 形状必须为 ({n_seg}, 6)，得到 {xf_stack.shape}")
    if X_Q.shape[0] != n_seg + 1:
        raise ValueError(f"X_Q 行数必须为 phi_stack 行数 + 1，得到 {X_Q.shape[0]} vs {n_seg + 1}")

    I6 = np.eye(6)

    # ---- 前代：构建 L / D / X_d ----
    A: list[npt.NDArray[np.floating]] = []
    L: list[npt.NDArray[np.floating] | None] = []
    D: list[npt.NDArray[np.floating]] = []
    X_d: list[npt.NDArray[np.floating]] = []
    errs: list[float] = []

    A0 = phi_stack[0]
    A.append(A0)
    L.append(None)
    D.append(I6 + A0 @ A0.T)
    X_d.append(xf_stack[0] - X_Q[1])
    errs.append(float(np.linalg.norm(X_d[0])))

    for i in range(1, n_seg):
        Ai = phi_stack[i]
        # L_i = -A_i @ D_{i-1}^{-1}
        Dinv = np.linalg.inv(D[i - 1])
        Li = -Ai @ Dinv
        # D_i = I + A_i A_i^T - L_i D_{i-1} L_i^T
        Di = I6 + Ai @ Ai.T - Li @ D[i - 1] @ Li.T
        # X_d[i] = Xf_i - X_Q_{i+1} - L_i X_d_{i-1}
        Xdi = xf_stack[i] - X_Q[i + 1] - Li @ X_d[i - 1]

        A.append(Ai)
        L.append(Li)
        D.append(Di)
        X_d.append(Xdi)
        errs.append(float(np.linalg.norm(xf_stack[i] - X_Q[i + 1])))

    # ---- 回代：解 Y ----
    # Y 长度 = n_seg = N-1（节点数 - 1）；Y[i] 对应第 i 个约束的乘子。
    # 对应 qiao Code05 第 127–134 行。
    Y: list[npt.NDArray[np.floating]] = [None] * n_seg  # type: ignore[list-item]
    # 末段：Y_{n_seg-1} = D_{n_seg-1}^{-1} X_d_{n_seg-1}
    Y[n_seg - 1] = np.linalg.solve(D[n_seg - 1], X_d[n_seg - 1])
    for i in range(n_seg - 2, -1, -1):
        # L[i+1] 对应 i+1 ∈ [1, n_seg-1]，前代循环已全部赋为非 None 矩阵；
        # 仅 L[0]（前代首段）为 None，本循环不触及。
        l_next: npt.NDArray[np.floating] = L[i + 1]  # type: ignore[assignment]
        Y[i] = np.linalg.solve(D[i], X_d[i]) - l_next.T @ Y[i + 1]

    # ---- 拼装 delta_Q ----
    delta_Q = np.zeros_like(X_Q)
    delta_Q[0] = -A[0].T @ Y[0]
    for i in range(1, n_seg):
        delta_Q[i] = Y[i - 1] - A[i].T @ Y[i]
    delta_Q[n_seg] = Y[n_seg - 1]

    return delta_Q, errs


# ---------------------------------------------------------------------------
# 多轮 Newton 迭代
# ---------------------------------------------------------------------------


def multiple_shooting_newton(
    initial: ShootingPatch,
    solver: SubstituteSolver,
    *,
    max_iter: int = 19,
    tolerance: float = 1e-11,
    min_iter: int = 5,
    early_stop: bool = True,
    phi_stack: STMStack | None = None,
    xf_stack: npt.NDArray[np.floating] | None = None,
) -> MultipleShootingResult:
    """多轮 Newton 迭代收敛动力学替代轨道。

    与 qiao 主循环等价的策略：

    1. 对每段调用 ``solver.propagate_segment`` 得到 ``(Xf, Φ)``；
    2. 用 :func:`solve_block_tridiagonal` 算修正量 ``delta_Q``；
    3. 若 ``max_residual < tolerance`` 或超过 ``min_iter`` 且残差平稳则退出；
    4. 否则 ``X_Q += delta_Q`` 继续迭代。

    Args:
        initial: 初始 :class:`ShootingPatch`（包含 ``t_Q`` 与 ``X_Q``）。
        solver: :class:`SubstituteSolver` 实现，给出每段 ``(Xf, Φ)``。
        max_iter: 最大迭代轮数。
        tolerance: 收敛容差（最大连续性残差）。
        min_iter: 最少迭代轮数；早于此值不接受早停。
        early_stop: 是否在残差平稳时早停。
        phi_stack: 预计算的 ``(N-1, 6, 6)`` STM 栈；为 ``None`` 时内部重算。
            主要用于调用方在重构/FFT 分析之间复用 STM。
        xf_stack: 预计算的 ``(N-1, 6)`` 终端状态栈；同上。

    Returns:
        :class:`MultipleShootingResult`，含收敛后的 ``t_Q`` / ``X_Q`` 与
        残差历史。
    """
    if max_iter < 1:
        raise ValueError(f"max_iter 必须 ≥ 1，得到 {max_iter}")
    if tolerance <= 0:
        raise ValueError(f"tolerance 必须 > 0，得到 {tolerance}")

    t_Q = np.array(initial.t_Q, dtype=float, copy=True)
    X_Q = np.array(initial.X_Q, dtype=float, copy=True)
    n_seg = t_Q.shape[0] - 1
    if n_seg < 1:
        raise ValueError("t_Q 至少需要 2 个节点")

    history: list[float] = []
    status = ConvergenceState.MAX_ITERATIONS
    cause = FailureCause.MAX_ITERATIONS_REACHED
    message = f"达到最大迭代次数 {max_iter}"
    iterations = 0
    max_residual = float("inf")

    for it in range(1, max_iter + 1):
        iterations = it

        if phi_stack is None or xf_stack is None or it == 1:
            # 重算段结果
            phi_stack_curr = np.zeros((n_seg, 6, 6), dtype=float)
            xf_stack_curr = np.zeros((n_seg, 6), dtype=float)
            for i in range(n_seg):
                xf_i, phi_i = solver.propagate_segment(float(t_Q[i]), float(t_Q[i + 1]), X_Q[i])
                phi_stack_curr[i] = phi_i
                xf_stack_curr[i] = xf_i
        else:
            phi_stack_curr = phi_stack
            xf_stack_curr = xf_stack

        delta_Q, errs = solve_block_tridiagonal(phi_stack_curr, xf_stack_curr, X_Q)
        max_residual = float(np.max(errs))
        mean_residual = float(np.mean(errs))
        history.append(max_residual)

        # 收敛判定
        if max_residual < tolerance and it >= min_iter:
            status = ConvergenceState.CONVERGED
            cause = FailureCause.NONE
            message = f"连续性残差 {max_residual:.3e} 已满足容差 {tolerance:.3e}"
            break

        # 平整整轮：连续 3 轮残差下降 < 1% 且 it ≥ min_iter
        if (
            early_stop
            and it >= min_iter
            and len(history) >= 3
            and history[-1] >= history[-2] * 0.99
            and history[-2] >= history[-3] * 0.99
        ):
            status = ConvergenceState.STAGNATED
            cause = FailureCause.STAGNATION_DETECTED
            message = f"连续 3 轮残差未显著下降，最终残差 {max_residual:.3e}"
            break

        X_Q = X_Q + delta_Q

    return MultipleShootingResult(
        t_Q=t_Q,
        X_Q=X_Q,
        max_residual=max_residual,
        mean_residual=mean_residual,
        iterations=iterations,
        status=status,
        cause=cause,
        message=message,
        residual_history=tuple(history),
    )


# ---------------------------------------------------------------------------
# 内置 solver：从右端函数 + 中心差分近似构造 STM
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ODESubstituteSolver:
    """通用 ODE 替代打靶求解器。

    通过注入 ``rhs(t, X) -> (6,)`` 即可工作；STM 用中心差分近似。

    Attributes:
        rhs: 状态右端项 ``f(t, X) -> (6,)``。
        rtol: 相对容差（默认 ``1e-10``）。
        atol: 绝对容差（默认 ``1e-12``）。
        max_step: 最大步长（默认 ``None``，由 solver 自适应）。
        stm_eps: 中心差分步长（默认 ``1e-7``）。
    """

    rhs: Callable[[float, npt.ArrayLike], npt.ArrayLike]
    rtol: float = 1e-10
    atol: float = 1e-12
    max_step: float | None = None
    stm_eps: float = 1e-7

    def propagate_segment(
        self, t0: float, tf: float, x0: npt.ArrayLike
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """积分单段并用中心差分近似 STM。"""
        from ._solve_ivp_rust import solve_ivp_rust

        x0_arr = np.asarray(x0, dtype=float).ravel()
        if x0_arr.shape != (6,):
            raise ValueError(f"x0 必须是形状 (6,)，得到 {x0_arr.shape}")

        opts: dict[str, object] = {
            "rtol": float(self.rtol),
            "atol": float(self.atol),
        }
        if self.max_step is not None:
            opts["max_step"] = float(self.max_step)

        sol = solve_ivp_rust(
            fun=lambda t, X: np.asarray(self.rhs(t, X), dtype=float).ravel(),
            t_span=(float(t0), float(tf)),
            y0=x0_arr,
            rtol=float(self.rtol),
            atol=float(self.atol),
            max_step=float(self.max_step) if self.max_step is not None else None,
        )
        if not sol.success:
            raise RuntimeError(f"ODE 积分失败：{sol.message}")
        xf = sol.y[:, -1]

        # STM：用 6 组扰动分别积分；与 qiao ``Calc_Phi_rho`` 一致
        phi = np.zeros((6, 6), dtype=float)
        eps = float(self.stm_eps)
        _rtol_val = float(self.rtol)
        _atol_val = float(self.atol)
        _max_step_val = float(self.max_step) if self.max_step is not None else None
        for k in range(6):
            xp = x0_arr.copy()
            xp[k] += eps
            xm = x0_arr.copy()
            xm[k] -= eps
            sol_p = solve_ivp_rust(
                fun=lambda t, X: np.asarray(self.rhs(t, X), dtype=float).ravel(),
                t_span=(float(t0), float(tf)),
                y0=xp,
                rtol=_rtol_val,
                atol=_atol_val,
                max_step=_max_step_val,
            )
            sol_m = solve_ivp_rust(
                fun=lambda t, X: np.asarray(self.rhs(t, X), dtype=float).ravel(),
                t_span=(float(t0), float(tf)),
                y0=xm,
                rtol=_rtol_val,
                atol=_atol_val,
                max_step=_max_step_val,
            )
            phi[:, k] = (sol_p.y[:, -1] - sol_m.y[:, -1]) / (2.0 * eps)

        return xf, phi


__all__ = [
    "PatchStates",
    "STMStack",
    "SubstituteSolver",
    "ShootingPatch",
    "MultipleShootingResult",
    "solve_block_tridiagonal",
    "multiple_shooting_newton",
    "ODESubstituteSolver",
]
