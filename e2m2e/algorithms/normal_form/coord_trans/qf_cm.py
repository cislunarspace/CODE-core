"""QF ↔ CM：quasi-Floquet 坐标 ↔ 中心流形坐标（高阶 Lie 级数）。

迁移自 qiao ``Subfunction/coord_trans/qpQF2qpCM.py`` /
``qpCM2qpQF.py``——变换链中最复杂的一段。对应切片 #173 中心流形化简
（Code10/Code11）的**坐标层面**应用：把 quasi-Floquet 坐标通过生成函数
``W`` 的高阶 Lie 级数映射到中心流形坐标。

算法（qiao ``CONTEXT.md`` §三"化简到中心流形"）：

1. **实→复基底变换** ``Re2Im``：用对角化矩阵 ``D⁻¹`` 把实 QF 坐标
   ``(q, p)`` 映到复坐标。``D`` 把双曲/中心方向拆成纯实/复分量：
   双曲方向 ``(q1, p1)`` 不变，平面/垂直中心方向
   ``(q2, p2)``/``(q3, p3)`` 各组合成 ``±i`` 模式（``√2`` 归一）。
2. **逐阶 Lie 流**：对每个阶 ``order ≥ 2``（qiao ``W_series{3..N}``），
   用生成函数 ``W_order(q,p)`` 的 Hamilton 流
   ``dX/dt = J·∇W_order`` 从 ``t=0`` 积到 ``t=1``（一步近恒等变换）。
   正向（QF→CM）取 ``W`` 系数取反；反向（CM→QF）不取反、阶序倒序。
3. **复→实基底变换** ``Im2Re``：用 ``D`` 映回实坐标，取实部。

Hamilton 流的右端 ``dX/dt = J·∇W`` 用向量化实现
（:func:`_hamilton_flow_rhs`），与 qiao ``_dynfunc_wtrans_vec`` 逐位一致；
``0^0=1`` 边界用降幂写法（``qp^(n-1)``）天然正确，避免除法写法的
``0/0=nan``。

与 qiao 的差异：

- qiao 在 ``globalparam.data_array`` 上对每个 ``W_series`` 系数做
  Catmull-Rom 插值（``preinterp_coeffs`` 路径）；本仓库的
  :class:`CenterManifoldResult.W_series` 已按 ``{step: {order: {pow:
  coef_array}}}`` 组织，本模块接收**已插值为标量**的系数表
  ``W_series_at_t``（``{order: {pow: complex_scalar}}``），把插值决策上浮
  到 :class:`LibrationCatalogTransformer`。
- qiao 依赖 ``globalparam.odeoptions``（``scipy.solve_ivp`` 选项）；本模块
  显式传 ``DOP853`` 默认容差（``rtol=1e-11``、``atol=1e-13``），不引入
  全局可变状态。
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# 实/复基底变换（迁移自 qiao _complex_basis.py / Re2Im.py / Im2Re.py）
# ---------------------------------------------------------------------------

# 6×6 复对角化矩阵 D（与 qiao _complex_basis._D 逐元素一致）。
# 双曲方向 q1/p1（位 0/3）保持实；平面中心对 q2/p2（位 1/4）、垂直中心
# 对 q3/p3（位 2/5）通过 (1/√2, ±i/√2) 组合拆成 ±i 模式。
_D: npt.NDArray[np.complex128] = np.zeros((6, 6), dtype=complex)
_D[0, 0] = 1.0
_D[1, 1] = 1.0 / np.sqrt(2.0)
_D[1, 4] = 1j / np.sqrt(2.0)
_D[2, 2] = 1.0 / np.sqrt(2.0)
_D[2, 5] = 1j / np.sqrt(2.0)
_D[3, 3] = 1.0
_D[4, 4] = 1.0 / np.sqrt(2.0)
_D[4, 1] = 1j / np.sqrt(2.0)
_D[5, 5] = 1.0 / np.sqrt(2.0)
_D[5, 2] = 1j / np.sqrt(2.0)

#: ``D`` 的逆（预计算，供 Re2Im 复用）。
_D_INV: npt.NDArray[np.complex128] = cast("npt.NDArray[np.complex128]", np.linalg.inv(_D))


def _re_to_im(X_real: npt.NDArray[np.floating]) -> npt.NDArray[np.complex128]:
    """实数基 → 复数基：``X_im = D⁻¹ · X_real``（qiao ``Re2Im``）。"""
    return _D_INV @ np.asarray(X_real, dtype=float)


def _im_to_re(X_im: npt.NDArray[np.complex128]) -> npt.NDArray[np.floating]:
    """复数基 → 实数基：``X_real = D · X_im``，取实部（qiao ``Im2Re``）。

    合法输入下 ``D·X_im`` 应为实数；取实部丢弃数值虚部残余。
    """
    return np.real(_D @ np.asarray(X_im, dtype=complex))


# ---------------------------------------------------------------------------
# Hamilton 流右端 dX/dt = J·∇W（迁移自 qiao _dynfunc_wtrans_vec）
# ---------------------------------------------------------------------------

# J·∇W 结构：输出方向 out_j 由 (求导坐标 coord, 符号 sign) 决定。
# dq_i/dt = +∂W/∂p_i（coord=3,4,5，符号 +1），
# dp_i/dt = -∂W/∂q_i（coord=0,1,2，符号 -1）。
_DERIV: tuple[tuple[int, float], ...] = (
    (3, 1.0),
    (4, 1.0),
    (5, 1.0),
    (0, -1.0),
    (1, -1.0),
    (2, -1.0),
)
_OTHERS: tuple[tuple[int, ...], ...] = tuple(
    tuple(k for k in range(6) if k != c) for c, _ in _DERIV
)


def _pack_wpoly(
    W_poly: dict[tuple[int, ...], complex],
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.complex128]]:
    """把 ``W`` 多项式字典 ``{pow_tuple: scalar}`` 打包成 ``(exps, coefs)``。

    迁移自 qiao ``_pack_wpoly``。``exps`` 形状 ``(N, 6)``，``coefs`` 形状
    ``(N,)``（复标量）。
    """
    if not W_poly:
        return (
            np.zeros((0, 6), dtype=int),
            np.zeros(0, dtype=complex),
        )
    pows = list(W_poly.keys())
    exps = np.array(pows, dtype=int).reshape(-1, 6)
    coefs = np.array([W_poly[p] for p in pows], dtype=complex)
    return exps, coefs


def _hamilton_flow_rhs(
    X: npt.NDArray[np.complex128],
    exps: npt.NDArray[np.int_],
    coefs: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Hamilton 流右端 ``dX/dt = J·∇W(X)``，向量化。

    迁移自 qiao ``_dynfunc_wtrans_vec``：降幂写法不做除法，在
    ``qp_j=0, n_j=1`` 处 ``qp^(n-1)=0^0=1``（numpy 天然），与逐项版
    整数幂语义逐位一致；除法写法会在该点产生 ``0/0=nan``。
    """
    B = X**exps  # (N, 6)，X^exp
    dX = np.zeros(6, dtype=complex)
    for out_j, (coord, sign) in enumerate(_DERIV):
        e_col = exps[:, coord]
        prod_excl = np.prod(B[:, _OTHERS[out_j]], axis=1)  # Π_{k≠coord} qp_k^n_k
        qp_red = X[coord] ** np.maximum(e_col - 1, 0)  # qp_coord^(n_coord-1)
        dX[out_j] = sign * (coefs * e_col * prod_excl * qp_red).sum()
    return dX


# ---------------------------------------------------------------------------
# Lie 级数应用（逐阶 ODE 积分）
# ---------------------------------------------------------------------------


def _apply_lie_series(
    X0: npt.NDArray[np.complex128],
    W_series_at_t: dict[int, dict[tuple[int, ...], complex]],
    *,
    forward: bool,
    rtol: float = 1e-11,
    atol: float = 1e-13,
) -> npt.NDArray[np.complex128]:
    """逐阶应用 Hamilton 流 ``dX/dt = J·∇W_order``，从 ``t=0`` 积到 ``t=1``。

    迁移自 qiao ``qpQF2qpCM`` / ``qpCM2qpQF`` 的逐阶循环：

    - ``forward=True``（QF→CM）：``W`` 系数取反，阶序升序 ``2..N``；
    - ``forward=False``（CM→QF）：``W`` 系数不取反，阶序降序 ``N..2``。

    每阶一次 ``scipy.solve_ivp``（DOP853），末态作为下阶初值。

    Args:
        X0: ``(6,)`` 复初值。
        W_series_at_t: ``{order: {pow_tuple: complex_scalar}}``，已插值。
        forward: 方向标志（``True`` 为 QF→CM，``False`` 为 CM→QF，见上方 docstring）。
        rtol, atol: ODE 容差。

    Returns:
        ``(6,)`` 复终值。
    """
    from scipy.integrate import solve_ivp

    orders = sorted(W_series_at_t.keys(), reverse=not forward)
    X = np.array(X0, dtype=complex)
    for order in orders:
        if order < 2:  # qiao：Python key 2 = MATLAB W_series{3}（第一个非空阶）
            continue
        W_order = W_series_at_t[order]
        if not W_order:
            continue
        if forward:
            W_order = {k: -v for k, v in W_order.items()}  # 正向取反
        exps, coefs = _pack_wpoly(W_order)
        if coefs.size == 0:
            continue

        def rhs(_t, X_local, exps=exps, coefs=coefs):
            return _hamilton_flow_rhs(np.asarray(X_local, dtype=complex), exps, coefs)

        sol = solve_ivp(
            rhs,
            (0.0, 1.0),
            X,
            method="DOP853",
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"QF↔CM Lie 级数 ODE 积分失败（order={order}）：{sol.message}")
        X = np.asarray(sol.y[:, -1], dtype=complex)
    return X


# ---------------------------------------------------------------------------
# 公开变换
# ---------------------------------------------------------------------------


def qf_to_cm(
    X_qf: npt.ArrayLike,
    W_series_at_t: dict[int, dict[tuple[int, ...], complex]],
) -> npt.NDArray[np.floating]:
    """quasi-Floquet 坐标 → 中心流形坐标（高阶 Lie 级数）。

    对应 qiao ``qpQF2qpCM``：实→复基底 → 逐阶 Lie 流（``W`` 取反、升序）
    → 复→实基底。

    Args:
        X_qf: ``(6,)`` QF 状态 ``[Q_qf, P_qf]``，无量纲实数。
        W_series_at_t: ``{order: {pow_tuple: complex_scalar}}``——在时刻
            ``t`` 插值后的 :class:`CenterManifoldResult.W_series`（复值
            系数，跨 ``invariant``/``center`` 两步合并）。

    Returns:
        ``(6,)`` CM 状态 ``[Q_cm, P_cm]``，无量纲实数。
    """
    X = np.asarray(X_qf, dtype=float).ravel()
    X_im = _re_to_im(X)
    X_im = _apply_lie_series(X_im, W_series_at_t, forward=True)
    return _im_to_re(X_im)


def cm_to_qf(
    X_cm: npt.ArrayLike,
    W_series_at_t: dict[int, dict[tuple[int, ...], complex]],
) -> npt.NDArray[np.floating]:
    """中心流形坐标 → quasi-Floquet 坐标（高阶 Lie 级数，反向）。

    对应 qiao ``qpCM2qpQF``：实→复基底 → 逐阶 Lie 流（``W`` 不取反、
    降序）→ 复→实基底。是 :func:`qf_to_cm` 的精确逆。

    Args:
        X_cm: ``(6,)`` CM 状态 ``[Q_cm, P_cm]``，无量纲实数。
        W_series_at_t: 同 :func:`qf_to_cm`。

    Returns:
        ``(6,)`` QF 状态 ``[Q_qf, P_qf]``，无量纲实数。
    """
    X = np.asarray(X_cm, dtype=float).ravel()
    X_im = _re_to_im(X)
    X_im = _apply_lie_series(X_im, W_series_at_t, forward=False)
    return _im_to_re(X_im)


__all__ = ["cm_to_qf", "qf_to_cm"]
