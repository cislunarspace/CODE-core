"""normal_form 的 Rust solve_ivp 适配器。

将 ``scipy.integrate.solve_ivp`` 替换为 Rust 的 ``solve_ivp_py`` （DOP853），
消除 normal_form 模块中的 scipy 积分路径依赖。适配器提供与 scipy 兼容的结果对象，
让调用方改动最小。

不支持：
- ``dense_output=True`` （Rust 侧无稠密输出插值器）→ 用密集 ``t_eval`` 替代

复值 Lie 流（QF↔CM）不经本适配器，整链下沉到
``qf_to_cm_py`` / ``cm_to_qf_py``（12 实维分裂）。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from e2m2e.exceptions import RustExtensionUnavailableError


class _RustOdeResult:
    """与 ``scipy.optimize.OdeResult`` 兼容的 Rust 积分结果。

    仅实现 normal_form 调用方实际访问的属性。
    """

    def __init__(
        self,
        success: bool,
        message: str,
        t: npt.NDArray[np.floating],
        y: npt.NDArray[np.floating],
        n_steps: int,
    ):
        self.success = success
        self.message = message
        self.t = t
        self.y = y
        self._n_steps = n_steps

    @property
    def nfev(self) -> int:
        return self._n_steps  # 每步一次 RHS 评估


def solve_ivp_rust(
    fun: Callable[[float, npt.ArrayLike], npt.NDArray[np.floating]],
    t_span: tuple[float, float],
    y0: npt.ArrayLike,
    *,
    method: str | None = None,  # 忽略，Rust 侧固定 DOP853
    t_eval: npt.ArrayLike | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_step: float | None = None,
    dense_output: bool = False,
    **kwargs: object,
) -> _RustOdeResult:
    """用 Rust solve_ivp_py 替代 ``scipy.integrate.solve_ivp``。

    ``method``、``dense_output``、``**kwargs`` 接受但不生效——仅用于兼容 scipy
    调用签名。Rust 侧固定使用 DOP853。

    Args:
        fun: 右端函数 ``f(t, y) -> ndarray``。由 Rust 积分循环回调。
        t_span: ``(t0, tf)`` 积分区间。
        y0: 初始状态 ``(n,)``。
        method: 忽略（Rust 侧固定 DOP853）。
        t_eval: 输出时间点。若为 None 则仅返回首末点。
        rtol: 相对容差。
        atol: 绝对容差。
        max_step: 最大步长。
        dense_output: 若 True，Rust 侧无稠密输出插值器，
          本函数在 t_eval 上直接输出（调用方需自行传入密集网格）。

    Returns:
        ``_RustOdeResult``，含 ``success``、``message``、``t``、``y`` 属性。
    """
    try:
        from e2m2e._integrators import solve_ivp_py
    except ImportError as exc:
        raise RustExtensionUnavailableError(
            "e2m2e._integrators 不可用（Rust 扩展未构建），无法使用 solve_ivp_rust。"
            "请先构建：make dev"
        ) from exc

    y0_arr = np.asarray(y0, dtype=float).ravel()
    t0, tf = float(t_span[0]), float(t_span[1])

    if t_eval is None or dense_output:
        # 无 t_eval 或 dense_output 模式：至少输出首末点
        if t_eval is not None:
            t_arr = np.asarray(t_eval, dtype=float).ravel()
        else:
            t_arr = np.array([t0, tf], dtype=float)
    else:
        t_arr = np.asarray(t_eval, dtype=float).ravel()
        if t_arr.size == 0:
            t_arr = np.array([t0, tf], dtype=float)

    # Rust solve_ivp_py 在 t_eval 稀疏时，自适应步长可能被输出网格绑架
    # （只在 t_eval 点评估，内部步长过大导致误差失控）。当 t_eval 点间距
    # 大于 span 的 1/10 时，显式设 max_step 为 span/20 兜底。
    span = abs(tf - t0)
    effective_max_step = max_step
    if effective_max_step is None and t_arr.size <= 10 and span > 0:
        min_gap = float(np.min(np.diff(t_arr))) if t_arr.size > 1 else span
        if min_gap > 0.1 * span:
            effective_max_step = span / 20.0

    try:
        result = solve_ivp_py(
            (t0, tf),
            y0_arr.tolist(),
            t_arr.tolist(),
            float(rtol),
            float(atol),
            fun,
            float(effective_max_step) if effective_max_step is not None else None,
            None,  # max_steps
            None,  # state_error_dim
        )
    except Exception as exc:
        return _RustOdeResult(
            success=False,
            message=str(exc),
            t=np.array([t0]),
            y=y0_arr.reshape(-1, 1),
            n_steps=0,
        )

    if result is None:
        return _RustOdeResult(
            success=False,
            message="Rust solve_ivp_py 返回 None",
            t=np.array([t0]),
            y=y0_arr.reshape(-1, 1),
            n_steps=0,
        )

    states = np.array(result["states"], dtype=float)
    times = np.array(result["time"], dtype=float)
    n_steps = int(result.get("n_steps", 0))

    return _RustOdeResult(
        success=True,
        message="Rust solve_ivp_py 完成",
        t=times,
        y=states.T,  # scipy 兼容：(dim, N) 形状
        n_steps=n_steps,
    )
