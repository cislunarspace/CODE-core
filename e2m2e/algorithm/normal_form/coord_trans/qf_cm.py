"""QF ↔ CM：quasi-Floquet 坐标 ↔ 中心流形坐标（高阶 Lie 级数）。

迁移自 qiao ``Subfunction/coord_trans/qpQF2qpCM.py`` /
``qpCM2qpQF.py``——变换链中最复杂的一段。是中心流形化简
（Code10/Code11）的**坐标层面** 应用：把 quasi-Floquet 坐标通过生成函数
``W`` 的高阶 Lie 级数映射到中心流形坐标。

算法（qiao ``CONTEXT.md`` §三"化简到中心流形"）：

1. **实→复基底变换** ``Re2Im``：用对角化矩阵 ``D⁻¹`` 把实 QF 坐标
   ``(q, p)`` 映到复坐标。``D`` 把双曲/中心方向拆成纯实/复分量：
   双曲方向 ``(q1, p1)`` 不变，平面/垂直中心方向
   ``(q2, p2)``/``(q3, p3)`` 各组合成 ``±i`` 模式（``√2`` 归一）。
2. **逐阶 Lie 流**：对每个阶 ``order ≥ 2`` （qiao ``W_series{3..N}``），
   用生成函数 ``W_order(q,p)`` 的 Hamilton 流
   ``dX/dt = J·∇W_order`` 从 ``t=0`` 积到 ``t=1`` （一步近恒等变换）。
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
  coef_array}}}`` 组织，本模块接收**已插值为标量** 的系数表
  ``W_series_at_t`` （``{order: {pow: complex_scalar}}``），把插值决策上浮
  到 :class:`LibrationCatalogTransformer`。
- qiao 依赖 ``globalparam.odeoptions`` （``scipy.solve_ivp`` 选项）；本模块
  显式传 ``DOP853`` 默认容差（``rtol=1e-11``、``atol=1e-13``），不引入
  全局可变状态。

默认后端为 Rust：复值 Lie 流用 12 实维分裂
（``[Re X, Im X]``）走 ``e2m2e-integrators`` 的 DOP853，与 scipy 对复
``y0`` 的内部分裂数学等价。``backend="python"`` 仅作显式对照，禁止 auto
静默降级（ADR 0020）。
"""

from __future__ import annotations

from typing import Literal, cast

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
    """实数基 → 复数基：``X_im = D⁻¹ · X_real`` （qiao ``Re2Im``）。"""
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
    ``(N,)`` （复标量）。
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
    ``qp_j=0, n_j=1`` 处 ``qp^(n-1)=0^0=1`` （numpy 天然），与逐项版
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
# Lie 级数应用（逐阶 ODE 积分）——Python 参照后端
# ---------------------------------------------------------------------------


def _apply_lie_series_python(
    X0: npt.NDArray[np.complex128],
    W_series_at_t: dict[int, dict[tuple[int, ...], complex]],
    *,
    forward: bool,
    rtol: float = 1e-11,
    atol: float = 1e-13,
) -> npt.NDArray[np.complex128]:
    """Python/scipy 参照：逐阶 Hamilton 流，从 ``t=0`` 积到 ``t=1``。

    仅由显式 ``backend="python"`` 调用，不作默认路径。
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


def _pack_w_series_for_rust(
    W_series_at_t: dict[int, dict[tuple[int, ...], complex]],
) -> list[tuple[int, list[list[int]], list[float], list[float]]]:
    """把 ``W_series_at_t`` 打包为 Rust FFI 元组列表。

    每项 ``(order, exps, coefs_re, coefs_im)``；空阶跳过。
    """
    packed: list[tuple[int, list[list[int]], list[float], list[float]]] = []
    for order, W_order in W_series_at_t.items():
        if order < 2 or not W_order:
            continue
        pows = list(W_order.keys())
        if not pows:
            continue
        exps = [[int(p) for p in pow_t] for pow_t in pows]
        cre = [float(np.real(W_order[p])) for p in pows]
        cim = [float(np.imag(W_order[p])) for p in pows]
        packed.append((int(order), exps, cre, cim))
    return packed


# ---------------------------------------------------------------------------
# 公开变换
# ---------------------------------------------------------------------------


def qf_to_cm(
    X_qf: npt.ArrayLike,
    W_series_at_t: dict[int, dict[tuple[int, ...], complex]],
    *,
    backend: Literal["rust", "python"] = "rust",
    rtol: float = 1e-11,
    atol: float = 1e-13,
) -> npt.NDArray[np.floating]:
    """quasi-Floquet 坐标 → 中心流形坐标（高阶 Lie 级数）。

    对应 qiao ``qpQF2qpCM``：实→复基底 → 逐阶 Lie 流（``W`` 取反、升序）
    → 复→实基底。

    Args:
        X_qf: ``(6,)`` QF 状态 ``[Q_qf, P_qf]``，无量纲实数。
        W_series_at_t: ``{order: {pow_tuple: complex_scalar}}``——在时刻
            ``t`` 插值后的 :class:`CenterManifoldResult.W_series` （复值
            系数，跨 ``invariant``/``center`` 两步合并）。
        backend: ``"rust"``（默认）或显式对照 ``"python"``。禁止 auto。
        rtol, atol: ODE 容差（与 Python 参考实现默认一致）。

    Returns:
        ``(6,)`` CM 状态 ``[Q_cm, P_cm]``，无量纲实数。
    """
    if backend not in ("rust", "python"):
        raise ValueError(f"backend 须为 'rust' 或 'python'，得到 {backend!r}")

    X = np.asarray(X_qf, dtype=float).ravel()
    if X.size != 6:
        raise ValueError(f"X_qf 须为 6 维，得到 {X.size}")

    if backend == "python":
        X_im = _re_to_im(X)
        X_im = _apply_lie_series_python(X_im, W_series_at_t, forward=True, rtol=rtol, atol=atol)
        return _im_to_re(X_im)

    from e2m2e.integrators import qf_to_cm_py, require_rust_extension

    require_rust_extension("qf_to_cm_py")
    packed = _pack_w_series_for_rust(W_series_at_t)
    out = qf_to_cm_py(X.tolist(), packed, float(rtol), float(atol))
    return np.asarray(out, dtype=float)


def cm_to_qf(
    X_cm: npt.ArrayLike,
    W_series_at_t: dict[int, dict[tuple[int, ...], complex]],
    *,
    backend: Literal["rust", "python"] = "rust",
    rtol: float = 1e-11,
    atol: float = 1e-13,
) -> npt.NDArray[np.floating]:
    """中心流形坐标 → quasi-Floquet 坐标（高阶 Lie 级数，反向）。

    对应 qiao ``qpCM2qpQF``：实→复基底 → 逐阶 Lie 流（``W`` 不取反、
    降序）→ 复→实基底。是 :func:`qf_to_cm` 的精确逆。

    Args:
        X_cm: ``(6,)`` CM 状态 ``[Q_cm, P_cm]``，无量纲实数。
        W_series_at_t: 同 :func:`qf_to_cm`。
        backend: ``"rust"``（默认）或显式对照 ``"python"``。禁止 auto。
        rtol, atol: ODE 容差。

    Returns:
        ``(6,)`` QF 状态 ``[Q_qf, P_qf]``，无量纲实数。
    """
    if backend not in ("rust", "python"):
        raise ValueError(f"backend 须为 'rust' 或 'python'，得到 {backend!r}")

    X = np.asarray(X_cm, dtype=float).ravel()
    if X.size != 6:
        raise ValueError(f"X_cm 须为 6 维，得到 {X.size}")

    if backend == "python":
        X_im = _re_to_im(X)
        X_im = _apply_lie_series_python(X_im, W_series_at_t, forward=False, rtol=rtol, atol=atol)
        return _im_to_re(X_im)

    from e2m2e.integrators import cm_to_qf_py, require_rust_extension

    require_rust_extension("cm_to_qf_py")
    packed = _pack_w_series_for_rust(W_series_at_t)
    out = cm_to_qf_py(X.tolist(), packed, float(rtol), float(atol))
    return np.asarray(out, dtype=float)


__all__ = ["cm_to_qf", "qf_to_cm"]
