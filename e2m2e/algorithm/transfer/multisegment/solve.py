"""多段问题的两档非 NLP 解法 + minimax 组装（#726；无求解器分发）。

- :func:`solve_square`：p = q（方程数 = 变量数）的 Newton-Raphson +
  Broyden 秩一更新档（matchpoint 定向问题的方系统求解器）；
- :func:`solve_minimax`：p ≠ q 的松弛 minimax 档（``min s  s.t. −s ≤ F ≤ s``，
  经 :func:`..nlp_scipy.solve_slsqp` 求解，无显式成本档）。

状态三元组遵循 :class:`e2m2e.status.ResultStatus` 契约（构造时校验
cause↔status 一致）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds

from e2m2e.status import ConvergenceState, FailureCause, ResultStatus

from ...results import scipy_slsqp_status
from ..nlp_core import NLPSpec
from ..nlp_scipy import solve_slsqp

__all__ = [
    "MinimaxSolveResult",
    "SquareSolveResult",
    "solve_minimax",
    "solve_square",
]

#: 方系统求解器的可调用类型：决策向量 → 缺陷向量。
DefectFn = Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]
JacobianFn = Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]

#: Broyden 更新的步长平方下限：低于该值视为零步，跳过秩一更新（除零守卫）。
_DX2_FLOOR = 1e-300


@dataclass(frozen=True)
class SquareSolveResult:
    """方系统求解结果（软失败三元组随解携带，不抛异常）。

    Attributes:
        x: 末次迭代点（不一定可行）。
        defect_norm: ``x`` 处的缺陷 ``∞`` 范数。
        n_iter: Newton 迭代次数（初值即根时为 0）。
        n_jac_evals: 全量雅可比评估次数（Broyden 秩一更新不计入）。
        status: 算法最终状态。
        cause: 算法最终原因码。
        message: 求解器状态消息。
    """

    x: npt.NDArray[np.floating]
    defect_norm: float
    n_iter: int
    n_jac_evals: int
    status: ConvergenceState
    cause: FailureCause
    message: str

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)


@dataclass(frozen=True)
class MinimaxSolveResult:
    """minimax 档求解结果（软失败三元组随解携带，不抛异常）。

    Attributes:
        x: 末次迭代点（不含松弛量 s）。
        max_defect: ``x`` 处缺陷的 ``∞`` 范数（Chebyshev 目标值）。
        n_iter: SLSQP 迭代次数。
        status: 算法最终状态。
        cause: 算法最终原因码。
        message: 求解器状态消息。
    """

    x: npt.NDArray[np.floating]
    max_defect: float
    n_iter: int
    status: ConvergenceState
    cause: FailureCause
    message: str

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)


def solve_square(
    defect: DefectFn,
    jac: JacobianFn,
    x0: npt.ArrayLike,
    *,
    tol: float = 1e-10,
    maxiter: int = 100,
    full_jac_every: int = 8,
    max_backtrack: int = 4,
) -> SquareSolveResult:
    """方系统 ``F(x) = 0`` 的 Newton-Raphson + Broyden 秩一更新求解器。

    迭代：``‖F‖∞ ≤ tol`` 判收敛；解 ``J·δ = −F``（``np.linalg.solve``，
    ``LinAlgError`` 退 ``np.linalg.lstsq``，最小二乘解仍不满足线性方程
    （残差不低于 ``‖F‖∞`` 的 ``1e-8`` 倍以上）判 ``FAILED/SINGULAR_JACOBIAN``）；
    步长回溯 ``α ∈ {1, ½, ¼, …}``（共 ``max_backtrack + 1`` 个试探）要求
    ``‖F(x + αδ)‖∞`` 严格下降，回溯耗尽接受试探中最好的步、重算全量 J，
    连续两次耗尽判 ``STAGNATED/STAGNATION_DETECTED``；接受步后 Broyden 更新
    ``J ← J + (ΔF − JΔx)·Δxᵀ/(ΔxᵀΔx)``；每 ``full_jac_every`` 次迭代重算
    全量 J；``maxiter`` 耗尽判 ``MAX_ITERATIONS/MAX_ITERATIONS_REACHED``。

    Args:
        defect: 缺陷函数 ``x → F(x)``。
        jac: 全量雅可比函数 ``x → ∂F/∂x``。
        x0: 初猜 ``(q,)``（p = q 的方系统）。
        tol: ``‖F‖∞`` 收敛容差。
        maxiter: 最大 Newton 迭代次数。
        full_jac_every: 全量雅可比重算周期（迭代数）。
        max_backtrack: 回溯减半次数上限（α 最小到 ``2^−max_backtrack``）。

    Returns:
        :class:`SquareSolveResult`；初值即根时零迭代收敛。
    """
    x = np.asarray(x0, dtype=float).reshape(-1).copy()
    fx = np.asarray(defect(x), dtype=float).reshape(-1)
    norm = float(np.linalg.norm(fx, ord=np.inf))
    jac_matrix: npt.NDArray[np.floating] | None = None
    n_jac_evals = 0
    n_iter = 0
    stagnation = 0
    while norm > tol:
        if n_iter >= maxiter:
            status, cause = ConvergenceState.MAX_ITERATIONS, FailureCause.MAX_ITERATIONS_REACHED
            message = f"Newton-Broyden 迭代耗尽：maxiter={maxiter}，‖F‖∞ = {norm:.3e}"
            return SquareSolveResult(x, norm, n_iter, n_jac_evals, status, cause, message)
        if jac_matrix is None or n_iter % full_jac_every == 0:
            jac_matrix = np.asarray(jac(x), dtype=float)
            n_jac_evals += 1
        try:
            delta = np.linalg.solve(jac_matrix, -fx)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(jac_matrix, -fx, rcond=None)[0]
            linear_residual = float(np.linalg.norm(jac_matrix @ delta + fx, ord=np.inf))
            if not np.all(np.isfinite(delta)) or linear_residual > 1e-8 * max(1.0, norm):
                status, cause = ConvergenceState.FAILED, FailureCause.SINGULAR_JACOBIAN
                message = (
                    f"雅可比奇异：线性最小二乘解残差 {linear_residual:.3e}（‖F‖∞ = {norm:.3e}）"
                )
                return SquareSolveResult(x, norm, n_iter, n_jac_evals, status, cause, message)

        # 步长回溯：α = 1, 1/2, …, 2^-max_backtrack，要求 ‖F‖∞ 严格下降。
        x_prev, fx_prev = x, fx
        accepted = False
        best_x = x.copy()
        best_fx = fx
        best_norm = norm
        for halving in range(max_backtrack + 1):
            alpha = 0.5**halving
            x_try = x + alpha * delta
            fx_try = np.asarray(defect(x_try), dtype=float).reshape(-1)
            norm_try = float(np.linalg.norm(fx_try, ord=np.inf))
            if norm_try < norm:
                x, fx, norm = x_try, fx_try, norm_try
                accepted = True
                break
            if norm_try < best_norm:
                best_x, best_fx, best_norm = x_try, fx_try, norm_try
        if not accepted:
            # 回溯耗尽：接受当前最好步并重算全量 J（不做 Broyden，避免在
            # 停滞步上叠加秩一噪声）；连续两次耗尽判停滞。
            x, fx, norm = best_x, best_fx, best_norm
            jac_matrix = np.asarray(jac(x), dtype=float)
            n_jac_evals += 1
            stagnation += 1
            if stagnation >= 2:
                status = ConvergenceState.STAGNATED
                cause = FailureCause.STAGNATION_DETECTED
                message = f"回溯连续 {stagnation} 次耗尽：‖F‖∞ = {norm:.3e}"
                return SquareSolveResult(x, norm, n_iter, n_jac_evals, status, cause, message)
        else:
            stagnation = 0
            dx_vec = x - x_prev
            df_vec = fx - fx_prev
            dx2 = float(dx_vec @ dx_vec)
            if dx2 > _DX2_FLOOR:
                # Broyden 秩一更新：J ← J + (ΔF − JΔx)Δxᵀ/(ΔxᵀΔx)。
                assert jac_matrix is not None
                jac_matrix = jac_matrix + np.outer(df_vec - jac_matrix @ dx_vec, dx_vec) / dx2
        n_iter += 1
    status, cause = ConvergenceState.CONVERGED, FailureCause.NONE
    message = (
        f"Newton-Broyden 收敛：{n_iter} 次迭代，‖F‖∞ = {norm:.3e}，全量雅可比 {n_jac_evals} 次"
    )
    return SquareSolveResult(x, norm, n_iter, n_jac_evals, status, cause, message)


def solve_minimax(
    defect: DefectFn,
    x0: npt.ArrayLike,
    *,
    bounds: Bounds | None = None,
    ftol: float = 1e-9,
    maxiter: int = 200,
) -> MinimaxSolveResult:
    """minimax 档：松弛形式 ``min s  s.t. s − Fᵢ(x) ≥ 0, s + Fᵢ(x) ≥ 0``。

    决策变量扩为 ``z = [x, s]``（``s`` 初值取 ``max|F(x0)|``，可行边界），
    经 :func:`..nlp_scipy.solve_slsqp` 求解（SLSQP 数值差分，无解析雅可比；
    p ≠ q 无显式成本档）。``bounds`` 只约束 ``x`` 部分，``s`` 无界。

    Args:
        defect: 缺陷函数 ``x → F(x)``（p 维，p 可 ≠ 变量数 q）。
        x0: 初猜 ``(q,)``。
        bounds: ``x`` 的变量盒；``None`` 无界。
        ftol: SLSQP 目标容差。
        maxiter: SLSQP 最大迭代次数。

    Returns:
        :class:`MinimaxSolveResult`；解处 ``max|F|`` 为 Chebyshev 目标值
        （线性问题的最优具等幅振荡特征）。
    """
    x0_arr = np.asarray(x0, dtype=float).reshape(-1)
    f0 = np.asarray(defect(x0_arr), dtype=float).reshape(-1)
    z0 = np.concatenate([x0_arr, [float(np.max(np.abs(f0)))]])

    def ineq_fn(z: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        f = np.asarray(defect(z[:-1]), dtype=float).reshape(-1)
        s = float(z[-1])
        return np.concatenate([s - f, s + f])

    spec_bounds: Bounds | None = None
    if bounds is not None:
        spec_bounds = Bounds(
            np.concatenate([np.asarray(bounds.lb, dtype=float), [-np.inf]]),
            np.concatenate([np.asarray(bounds.ub, dtype=float), [np.inf]]),
        )
    spec = NLPSpec(objective=lambda z: float(z[-1]), ineq=ineq_fn, bounds=spec_bounds)
    result = solve_slsqp(spec, z0, ftol=ftol, maxiter=maxiter)
    status, cause = scipy_slsqp_status(bool(result.success), int(result.status))
    x = np.asarray(result.x[:-1], dtype=float)
    max_defect = float(np.max(np.abs(np.asarray(defect(x), dtype=float))))
    return MinimaxSolveResult(
        x=x,
        max_defect=max_defect,
        n_iter=int(result.nit),
        status=status,
        cause=cause,
        message=str(result.message),
    )
