"""MEGNO 混沌指标的 Python 参照实现与调度入口（Primer §7.1 式 142）。

Rust 内核（``integrators.propagate_cr3bp_megno_py`` /
``propagate_bcr4bp_megno_py``）是生产路径；本模块提供：

- :func:`megno_reference`：scipy 参照实现（14 维增广：状态 + 切变分 +
  两时间积分累加器），仅供等价性对照（backend 语义对齐既有 rust/python
  双后端惯例），不在生产路径调用；
- :func:`propagate_cr3bp_megno` / :func:`propagate_bcr4bp_megno`：显式
  backend 调度（``"rust"`` 生产 / ``"python"`` 参照），与
  ``Dynamics.propagate`` 的事件 backend 约定同款：不允许隐式回退。

数学口径（与 Rust 内核逐项一致，测试锁定）：
``Y(t) = (2/t)·I₁``、``Ȳ(t) = I₂/t``，其中
``dI₁/dt = t·(δ·δ̇)/|δ|²``、``dI₂/dt = 2·I₁/t``（t > 0）；
``δ·δ̇`` 是**全相空间 6 维**内积（δr·δv + δv·δv̇）——正则轨迹的
Ȳ → 2 正是来自切向量沿流方向的线性增长项。
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.integrate import solve_ivp

from ...integrators import (
    propagate_bcr4bp_megno_py as _rust_bcr4bp_megno,
)
from ...integrators import (
    propagate_cr3bp_megno_py as _rust_cr3bp_megno,
)
from ...integrators import (
    require_rust_extension,
)

__all__ = [
    "megno_reference",
    "propagate_bcr4bp_megno",
    "propagate_cr3bp_megno",
]


def _pack_result(sol: solve_ivp, times: np.ndarray) -> dict[str, Any]:
    """scipy 解 → MEGNO 结果字典（键与 Rust FFI 返回一致）。"""
    y14 = sol.y
    t_safe = np.maximum(times, 1e-300)
    return {
        "time": times,
        "states": y14[:6].T,
        "deltas": y14[6:12].T,
        "y": np.where(times > 0.0, 2.0 * y14[12] / t_safe, 0.0),
        "ybar": np.where(times > 0.0, y14[13] / t_safe, 0.0),
        # 参照实现不统计积分步：n_steps 取输出点数占位（仅键形一致）。
        "n_steps": sol.t.size,
        "n_rejected": 0,
    }


def megno_reference(
    eom: Any,
    t_span: tuple[float, float],
    t_eval: np.ndarray,
    initial_state: np.ndarray,
    initial_delta: np.ndarray | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """通用 14 维 MEGNO 参照积分（scipy DOP853）。

    Args:
        eom: 右端函数 ``f(t, y14) -> y14'``（由调用方按力模型构造，
            含累加器方程）。
        t_span / t_eval / rtol / atol: scipy 语义。
        initial_state: 6 维初态。
        initial_delta: 6 维切向量初值；None = 单位 x 向量。

    Returns:
        字典（键与 Rust FFI 一致）：time/states/y/ybar/n_steps/n_rejected。
    """
    delta0 = np.zeros(6) if initial_delta is None else np.asarray(initial_delta, dtype=float)
    if initial_delta is None:
        delta0[0] = 1.0
    y0 = np.concatenate([np.asarray(initial_state, dtype=float), delta0, [0.0, 0.0]])
    sol = solve_ivp(
        eom,
        t_span,
        y0,
        method="DOP853",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"MEGNO 参照积分失败：{sol.message}")
    return _pack_result(sol, sol.t)


def _cr3bp_megno_eom(mu: float):
    def eom(t: float, y: np.ndarray) -> np.ndarray:
        x, y_, z, vx, vy, vz = y[:6]
        r1 = np.hypot(x + mu, np.hypot(y_, z))
        r2 = np.hypot(x - 1.0 + mu, np.hypot(y_, z))
        ax = 2.0 * vy + x - (1.0 - mu) * (x + mu) / r1**3 - mu * (x - 1.0 + mu) / r2**3
        ay = -2.0 * vx + y_ - (1.0 - mu) * y_ / r1**3 - mu * y_ / r2**3
        az = -(1.0 - mu) * z / r1**3 - mu * z / r2**3
        i3 = 1.0 / r1**3
        i5 = i3 / (r1 * r1)
        j3 = 1.0 / r2**3
        j5 = j3 / (r2 * r2)
        xm = 1.0 - mu
        d1, d2 = x + mu, x - 1.0 + mu
        uxx = 1.0 - xm * (i3 - 3.0 * d1 * d1 * i5) - mu * (j3 - 3.0 * d2 * d2 * j5)
        uyy = 1.0 - xm * (i3 - 3.0 * y_ * y_ * i5) - mu * (j3 - 3.0 * y_ * y_ * j5)
        uzz = -xm * (i3 - 3.0 * z * z * i5) - mu * (j3 - 3.0 * z * z * j5)
        uxy = 3.0 * xm * d1 * y_ * i5 + 3.0 * mu * d2 * y_ * j5
        uxz = 3.0 * xm * d1 * z * i5 + 3.0 * mu * d2 * z * j5
        uyz = 3.0 * xm * y_ * z * i5 + 3.0 * mu * y_ * z * j5
        dr, dv = y[6:9], y[9:12]
        ddv = np.array(
            [
                uxx * dr[0] + uxy * dr[1] + uxz * dr[2] + 2.0 * dv[1],
                uxy * dr[0] + uyy * dr[1] + uyz * dr[2] - 2.0 * dv[0],
                uxz * dr[0] + uyz * dr[1] + uzz * dr[2],
            ]
        )
        delta_sq = float(dr @ dr + dv @ dv)
        delta_dot = float(dr @ dv + dv @ ddv)
        i1d = t * delta_dot / delta_sq if delta_sq > 0.0 else 0.0
        i2d = 2.0 * y[12] / t if t > 0.0 else 0.0
        return np.concatenate([[vx, vy, vz, ax, ay, az], dv, ddv, [i1d, i2d]])

    return eom


def propagate_cr3bp_megno(
    mu: float,
    t_span: tuple[float, float],
    t_eval: np.ndarray,
    initial_state: np.ndarray,
    *,
    initial_delta: np.ndarray | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-10,
    backend: Literal["rust", "python"] = "rust",
) -> dict[str, Any]:
    """CR3BP MEGNO 传播（显式 backend；rust = 生产路径，python = 参照）。"""
    if backend == "rust":
        require_rust_extension("propagate_cr3bp_megno_py")
        delta = None if initial_delta is None else [float(v) for v in initial_delta]
        return _rust_cr3bp_megno(
            mu,
            tuple(t_span),
            [float(t) for t in t_eval],
            [float(v) for v in initial_state],
            rtol,
            atol,
            delta,
        )
    return megno_reference(
        _cr3bp_megno_eom(mu),
        t_span,
        np.asarray(t_eval, dtype=float),
        initial_state,
        initial_delta,
        rtol=rtol,
        atol=atol,
    )


def propagate_bcr4bp_megno(
    mu: float,
    mu_sun: float,
    sun_distance: float,
    sun_angular_rate: float,
    sun_phase0: float,
    t_span: tuple[float, float],
    t_eval: np.ndarray,
    initial_state: np.ndarray,
    *,
    initial_delta: np.ndarray | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-10,
    backend: Literal["rust", "python"] = "rust",
) -> dict[str, Any]:
    """BCR4BP MEGNO 传播（太阳参数语义同 ``propagate_bcr4bp_py``）。"""
    if backend == "rust":
        require_rust_extension("propagate_bcr4bp_megno_py")
        delta = None if initial_delta is None else [float(v) for v in initial_delta]
        return _rust_bcr4bp_megno(
            mu,
            mu_sun,
            sun_distance,
            sun_angular_rate,
            sun_phase0,
            tuple(t_span),
            [float(t) for t in t_eval],
            [float(v) for v in initial_state],
            rtol,
            atol,
            delta,
        )
    raise NotImplementedError(
        "BCR4BP 的 python 参照实现未提供（mu_sun=0 退化为 CR3BP，用 CR3BP 参照对照）"
    )
