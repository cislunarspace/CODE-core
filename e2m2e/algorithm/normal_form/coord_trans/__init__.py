"""函数式坐标变换链：``rho ↔ EM ↔ DS ↔ QF ↔ CM ↔ param``。

本子包把 qiao ``rho2param`` / ``param2rho`` 的完整变换链拆成 5 段函数式
接口（对应 qiao ``Subfunction/coord_trans/`` 各子模块），并在本
``__init__`` 里组合成端到端的 ``rho_to_param`` / ``param_to_rho``。

变换链（全部无量纲，qiao 归一化单位）：

    rho → EM → DS → QF → CM → param

- **rho ↔ EM** (:mod:`.rho_em`)：纯动量耦合 ``p = ρ̇ − Cdot_dimᵀ·C·ρ``；
- **EM ↔ DS** (:mod:`.em_ds`)：平移 ``W(t) = [A, B]``（动力学替代）；
- **DS ↔ QF** (:mod:`.ds_qf`)：矩阵变换 ``X_QF = B(t)⁻¹·X_DS``；
- **QF ↔ CM** (:mod:`.qf_cm`)：高阶 Lie 级数（生成函数 ``W_series``）；
- **CM ↔ param** (:mod:`.cm_param`)：复→极坐标（作用量-角变量）。

端到端链式函数 :func:`rho_to_param` / :func:`param_to_rho` 依赖三个预计算
结果句柄（切片 #171–#173 交付）：

- :class:`~e2m2e.algorithms.normal_form.dynamical_substitution.DynamicalSubstituteResult`
  提供 ``W_poly``（DS 平移，``{pow: coef_array}``）；
- :class:`~e2m2e.algorithms.normal_form.quasi_floquet.QuasiFloquetResult`
  提供 ``B_at(t)``（QF 矩阵插值访问器）；
- :class:`~e2m2e.algorithms.normal_form.center_manifold.CenterManifoldResult`
  提供 ``W_series``（CM 高阶 Lie 级数，``{step: {order: {pow: coef_array}}}``）。

调用方负责把这三个结果聚合（:class:`~e2m2e.algorithms.normal_form.catalog.LibrationCatalogData`
是开箱即用的聚合器）；本模块只做纯函数，不在内部缓存。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .cm_param import cm_to_param, param_to_cm
from .ds_qf import ds_to_qf, qf_to_ds
from .em_ds import _interp_W_at, ds_to_em, em_to_ds
from .qf_cm import cm_to_qf, qf_to_cm
from .rho_em import em_to_rho, rho_to_em

if TYPE_CHECKING:
    from ..center_manifold import CenterManifoldResult
    from ..context import NormalFormContext
    from ..dynamical_substitution import DynamicalSubstituteResult
    from ..quasi_floquet import QuasiFloquetResult

__all__ = [
    # 端到端链
    "rho_to_param",
    "param_to_rho",
    # 叶子函数（供单段测试 / 高级用法）
    "rho_to_em",
    "em_to_rho",
    "em_to_ds",
    "ds_to_em",
    "ds_to_qf",
    "qf_to_ds",
    "qf_to_cm",
    "cm_to_qf",
    "cm_to_param",
    "param_to_cm",
]


# ---------------------------------------------------------------------------
# 内部：在时刻 t 聚合 W_series（跨 invariant/center 两步合并）
# ---------------------------------------------------------------------------


def _interp_W_series_at_t(
    cm_result: CenterManifoldResult,
    qf_result: QuasiFloquetResult,
    t: float,
) -> dict[int, dict[tuple[int, ...], complex]]:
    """把 ``CenterManifoldResult.W_series`` 在时刻 ``t`` 插值为标量系数。

    ``W_series`` 结构 ``{step: {order: {pow: coef_array}}}``；本函数把
    ``invariant`` / ``center`` 两步的 ``W`` 合并到统一的 ``{order: {pow:
    complex_scalar}}`` 表。两步的阶域一般不重叠（``invariant`` 处理双曲
    交叉项、``center`` 处理中心耦合），合并等价于 qiao 单 ``W_series``
    逐阶应用。系数按 qiao ``rho2param`` 的 ``preinterp_coeffs`` 格式
    （``vr + 1j·vi``）取复值。

    时间网格取 ``qf_result.tlist``：``CenterManifoldReducer.reduce`` 用
    ``qf_result.tlist`` 作时间轴生成 W_series 系数数组（数组长度 =
    ``qf_result.tlist.size``），但 :class:`CenterManifoldResult` 本身不
    存 tlist，故插值网格须由调用方从配套的 ``qf_result`` 传入。这正是
    端到端 :func:`rho_to_param` / :func:`param_to_rho` 同时持有
    ``qf_result`` 与 ``cm_result`` 的原因。
    """
    t_arr = np.asarray(qf_result.tlist, dtype=float).ravel()
    out: dict[int, dict[tuple[int, ...], complex]] = {}
    for step_data in cm_result.W_series.values():
        for order, poly in step_data.items():
            if not poly:
                continue
            out.setdefault(order, {})
            for pow_tuple, coef_arr in poly.items():
                arr = np.asarray(coef_arr, dtype=complex).ravel()
                if arr.size == 0:
                    continue
                if arr.size == 1:
                    val = complex(arr[0])
                else:
                    # 线性插值实/虚部，网格用 qf_result.tlist
                    re = float(np.interp(t, t_arr, arr.real))
                    im = float(np.interp(t, t_arr, arr.imag))
                    val = complex(re, im)
                # 两步同阶/同幂合并（理论上不重叠；累加兜底）
                out[order][pow_tuple] = out[order].get(pow_tuple, 0.0 + 0.0j) + val
    return out


# ---------------------------------------------------------------------------
# 端到端链式组合
# ---------------------------------------------------------------------------


def rho_to_param(
    X_rho: npt.ArrayLike,
    t: float,
    context: NormalFormContext,
    ds_result: DynamicalSubstituteResult,
    qf_result: QuasiFloquetResult,
    cm_result: CenterManifoldResult,
) -> npt.NDArray[np.floating]:
    """完整逆链 ``rho → EM → DS → QF → CM → param``。

    对应 qiao ``rho2param``。输入 rho 坐标状态，输出表征参数
    ``[q1, p1, I2, θ2, I3, θ3]``。

    Args:
        X_rho: ``(6,)`` rho 状态 ``[ρ, ρ̇]``，无量纲。
        t: 归一化时间 TU。
        context: 归一化上下文（提供 LU/TU/平动点/历元）。
        ds_result: 动力学替代结果（提供 ``W_poly`` 与 ``tlist``）。
        qf_result: quasi-Floquet 结果（提供 ``B_at(t)``）。
        cm_result: 中心流形化简结果（提供 ``W_series``）。

    Returns:
        ``(6,)`` 表征参数 ``[q1, p1, I2, θ2, I3, θ3]``，无量纲。
    """
    # rho → EM
    X_em = rho_to_em(X_rho, t, context)
    # EM → DS
    W_at_t = _interp_W_at(ds_result.W_poly, np.asarray(ds_result.tlist, dtype=float).ravel(), t)
    X_ds = em_to_ds(X_em, W_at_t)
    # DS → QF
    B_at_t = qf_result.B(t)
    X_qf = ds_to_qf(X_ds, B_at_t)
    # QF → CM
    W_series_at_t = _interp_W_series_at_t(cm_result, qf_result, t)
    X_cm = qf_to_cm(X_qf, W_series_at_t)
    # CM → param
    return cm_to_param(X_cm)


def param_to_rho(
    X_param: npt.ArrayLike,
    t: float,
    context: NormalFormContext,
    ds_result: DynamicalSubstituteResult,
    qf_result: QuasiFloquetResult,
    cm_result: CenterManifoldResult,
) -> npt.NDArray[np.floating]:
    """完整正链 ``param → CM → QF → DS → EM → rho``。

    对应 qiao ``param2rho``。输入表征参数，输出 rho 坐标状态。
    是 :func:`rho_to_param` 的精确逆。

    Args:
        X_param: ``(6,)`` 表征参数 ``[q1, p1, I2, θ2, I3, θ3]``，无量纲。
        t: 归一化时间 TU。
        context: 归一化上下文。
        ds_result: 动力学替代结果。
        qf_result: quasi-Floquet 结果。
        cm_result: 中心流形化简结果。

    Returns:
        ``(6,)`` rho 状态 ``[ρ, ρ̇]``，无量纲。
    """
    # param → CM
    X_cm = param_to_cm(X_param)
    # CM → QF
    W_series_at_t = _interp_W_series_at_t(cm_result, qf_result, t)
    X_qf = cm_to_qf(X_cm, W_series_at_t)
    # QF → DS
    B_at_t = qf_result.B(t)
    X_ds = qf_to_ds(X_qf, B_at_t)
    # DS → EM
    W_at_t = _interp_W_at(ds_result.W_poly, np.asarray(ds_result.tlist, dtype=float).ravel(), t)
    X_em = ds_to_em(X_ds, W_at_t)
    # EM → rho
    return em_to_rho(X_em, t, context)
