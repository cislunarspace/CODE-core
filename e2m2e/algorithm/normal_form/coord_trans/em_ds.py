"""EM ↔ DS：星历模型 Hamilton 坐标 ↔ 动力学替代坐标。

迁移自 qiao ``Subfunction/coord_trans/qpDS2qpEM.py`` /
``qpEM2qpDS.py``。对应变换链第二段：EM 坐标 ``(q, p)`` 与动力学
替代坐标 ``(Q, P)`` 之间的**平移**关系。

数学关系（qiao ``CONTEXT.md`` §三"动态替换"）：

    DS = W(t)        （生成函数，6 维 ``[A, B]`` 时间序列）
    A = DS[:3]
    B = DS[3:]
    q = Q + B        （正向 EM = DS + W 平移）
    p = P − A
    Q = q − B        （反向 DS = EM − W 平移）
    P = p + A

即 DS 是 EM 减去替代轨道的平移 ``W(t) = [A(t), B(t)]``——把坐标原点从
平动点挪到动力学替代轨道上。本段是纯仿射变换，无量纲、无单位换算。

与 qiao 的差异：

- qiao 通过 ``list_interp`` 在全局 ``data_array`` 上 Catmull-Rom 插值
  ``W_poly``；本仓库的 ``DynamicalSubstituteResult`` 已在每个采样点上
  存好 ``W_poly``（``{pow_tuple: coef_array}``，6 个线性项），这里只做
  **线性插值**到时刻 ``t`` 即可（与 :class:`QuasiFloquetResult.B_at` 的
  策略一致）。Catmull-Rom 在等距光滑数据上与线性插值差别极小，且
  本仓库不引入 qiao 的二进制系数表。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

#: 6 个线性幂次，对应 ``W_poly`` 的键序（与 ``dynamical_substitution`` 一致）。
_LINEAR_POWS: tuple[tuple[int, ...], ...] = (
    (1, 0, 0, 0, 0, 0),
    (0, 1, 0, 0, 0, 0),
    (0, 0, 1, 0, 0, 0),
    (0, 0, 0, 1, 0, 0),
    (0, 0, 0, 0, 1, 0),
    (0, 0, 0, 0, 0, 1),
)


def _interp_W_at(
    W_poly: dict[tuple[int, ...], npt.NDArray[np.floating]],
    tlist: npt.NDArray[np.floating],
    t: float,
) -> npt.NDArray[np.floating]:
    """在时刻 ``t`` 线性插值 ``W_poly`` 的 6 个线性项 → ``(6,)`` 平移向量。

    ``W_poly`` 是 ``DynamicalSubstituteResult.W_poly``：键为 6 个线性
    幂次元组，值为与 ``tlist`` 等长的系数时间序列。缺失键按 0 处理。
    """
    t_arr = np.asarray(tlist, dtype=float).ravel()
    out = np.zeros(6, dtype=float)
    if t_arr.size == 0:
        return out
    for k, pow_tuple in enumerate(_LINEAR_POWS):
        coef = W_poly.get(pow_tuple)
        if coef is None:
            continue
        coef_arr = np.asarray(coef, dtype=float).ravel()
        if coef_arr.size == 1:
            out[k] = float(coef_arr[0])
        elif coef_arr.size == t_arr.size:
            out[k] = float(np.interp(t, t_arr, coef_arr))
        else:
            # 形状不一致：用第一个值兜底（不应在正常路径出现）
            out[k] = float(coef_arr[0])
    return out


def em_to_ds(
    X_em: npt.ArrayLike,
    W_poly_at_t: npt.ArrayLike,
) -> npt.NDArray[np.floating]:
    """EM 坐标 → 动力学替代坐标（平移）。

    ``Q = q − B``、``P = p + A``，其中 ``[A, B] = W(t)``。对应 qiao
    ``qpEM2qpDS``。

    Args:
        X_em: ``(6,)`` EM 状态 ``[q, p]``，无量纲。
        W_poly_at_t: ``(6,)`` 在时刻 ``t`` 插值后的 ``W(t) = [A, B]``。

    Returns:
        ``(6,)`` DS 状态 ``[Q, P]``，无量纲。
    """
    X = np.asarray(X_em, dtype=float).ravel()
    W = np.asarray(W_poly_at_t, dtype=float).ravel()
    A = W[:3]
    B = W[3:]
    q = X[:3]
    p = X[3:]
    Q = q - B
    P = p + A
    return np.concatenate([Q, P])


def ds_to_em(
    X_ds: npt.ArrayLike,
    W_poly_at_t: npt.ArrayLike,
) -> npt.NDArray[np.floating]:
    """动力学替代坐标 → EM 坐标（平移）。

    ``q = Q + B``、``p = P − A``，其中 ``[A, B] = W(t)``。对应 qiao
    ``qpDS2qpEM``。是 :func:`em_to_ds` 的精确逆。

    Args:
        X_ds: ``(6,)`` DS 状态 ``[Q, P]``，无量纲。
        W_poly_at_t: ``(6,)`` 在时刻 ``t`` 插值后的 ``W(t) = [A, B]``。

    Returns:
        ``(6,)`` EM 状态 ``[q, p]``，无量纲。
    """
    X = np.asarray(X_ds, dtype=float).ravel()
    W = np.asarray(W_poly_at_t, dtype=float).ravel()
    A = W[:3]
    B = W[3:]
    Q = X[:3]
    P = X[3:]
    q = Q + B
    p = P - A
    return np.concatenate([q, p])


__all__ = ["ds_to_em", "em_to_ds"]
