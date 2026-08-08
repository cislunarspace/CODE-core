"""rho ↔ EM：平动点相对坐标 ↔ 星历模型 Hamilton 坐标 (q, p)。

迁移自 qiao ``Subfunction/coord_trans/rho2qpEM.py`` /
``qpEM2rho.py``。本模块对应变换链的第一段：rho（无量纲平动点相对
坐标 ``[ρ, ρ̇]``）与星历模型 Hamilton 坐标 ``(q, p)`` 之间的**纯动量
耦合**层。

数学关系（qiao ``CONTEXT.md`` §二）：

    q   = ρ
    p   = ρ̇ − Cdot_dimᵀ · C · ρ

其中 ``C`` 是 EMR 旋转矩阵（月球瞬时姿态），``Cdot_dim = Cdot · TU``
是其无量纲时间导数。该耦合把会合系下的速度 ``ρ̇`` 投影到非惯性
Hamilton 系的共轭动量 ``p``，**不做物理量换算**——``q``、``p`` 仍在
无量纲 rho 单位下。

与 qiao 的差异：

- qiao 的 ``rho2qpEM`` / ``qpEM2rho`` 同时做单位换算（输出 km/km/s）并
  平移平动点位置；本仓库按 PRD 约定**全程无量纲**，单位换算交给
  :mod:`e2m2e.algorithms.normal_form.units` 的调用方；平动点平移在
  DS 段（``em_ds``）由生成函数 ``W`` 完成。
- ``C``、``Cdot`` 不通过 qiao 的 ``globalparam.data_array`` 二进制表
  暴露，而由 :class:`NormalFormContext` + :mod:`._ephemeris` 在请求时刻
  重新解析求值（SPICE 不可用时退化到纯 CR3BP 常值旋转）。

退化（纯 CR3BP）：``C`` 取会合系单位旋转 ``[[0,1,0],[-1,0,0],[0,0,0]]``、
``Cdot = 0``（无量纲时间下为常数）。此时动量耦合项消失，
``p = ρ̇``——这正是切片 0–4 测试一直使用的纯 CR3BP 退路，往返误差
应在机器精度内。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..context import NormalFormContext

# ---------------------------------------------------------------------------
# 默认（纯 CR3BP 退路）姿态矩阵
# ---------------------------------------------------------------------------

#: 纯 CR3BP 退路用的常值 EMR 旋转矩阵 ``C``（无量纲会合系）。
_CR3BP_C: npt.NDArray[np.floating] = np.array(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float
)


def _resolve_C_Cdot(
    context: NormalFormContext, t: float
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """在时刻 ``t``（归一化 TU）求值 EMR 旋转矩阵 ``C`` 与 ``Cdot_dim``。

    返回 ``C`` 与无量纲化时间导数 ``Cdot_dim = Cdot · TU``——这是
    动量耦合公式 ``p = ρ̇ − Cdot_dimᵀ · C · ρ`` 直接需要的形式。

    - ``context.force_cr3bp=True``：**显式纯 CR3BP**——``C`` 取会合系常值
      旋转、``Cdot_dim = 0``，不探 SPICE。这是 CR3BP 中心流形约化的正路
      （动量耦合项消失、``p = ρ̇``），不是降级。
    - 默认：通过 :mod:`._ephemeris` 取月球瞬时姿态（SPICE）；SPICE 不可用时
      退化到上述 CR3BP 常值（保留向后兼容）。

    与 :mod:`._ephemeris.eval_params` 一致：``Cpq = Cdot_dimᵀ · C``。
    """
    if context.force_cr3bp:
        return _CR3BP_C.copy(), np.zeros((3, 3), dtype=float)
    tu_days = float(context.TU) / 86400.0
    jd = float(context.epoch) + float(t) * tu_days
    try:
        from .._ephemeris import _derive_moon_param, _ephemeris_states

        r_em, v_em, r_es, v_es = _ephemeris_states(jd)
        # _derive_moon_param 返回 (C, Cdot, Cdotdot, A_em)（见 _ephemeris.py）
        C, cdot, _, _ = _derive_moon_param(
            r_em,
            v_em,
            r_es,
            v_es,
            mu_e=float(context.mu_e),
            mu_m=float(context.mu_m),
            mu_s=float(context.mu_s),
            lu=float(context.LU),
            tu=float(context.TU),
        )
        cdot_dim = np.asarray(cdot, dtype=float) * float(context.TU)
        return np.asarray(C, dtype=float), cdot_dim
    except Exception:
        logger.warning("EM 参数计算失败，退化到纯 CR3BP（C=常值, Cdot=0）", exc_info=True)
        return _CR3BP_C.copy(), np.zeros((3, 3), dtype=float)


# ---------------------------------------------------------------------------
# 公开变换
# ---------------------------------------------------------------------------


def rho_to_em(
    X_rho: npt.ArrayLike,
    t: float,
    context: NormalFormContext,
) -> npt.NDArray[np.floating]:
    """rho 状态 → EM 坐标 ``(q, p)``（无量纲）。

    对应 qiao ``rho2param`` 第一段：``q = ρ``，
    ``p = ρ̇ − Cdot_dimᵀ · C · ρ``。

    Args:
        X_rho: ``(6,)`` rho 状态 ``[ρ_x, ρ_y, ρ_z, ρ̇_x, ρ̇_y, ρ̇_z]``，
            无量纲。
        t: 归一化时间 TU。
        context: 归一化上下文。

    Returns:
        ``(6,)`` EM 状态 ``[q_x, q_y, q_z, p_x, p_y, p_z]``，无量纲。
    """
    X = np.asarray(X_rho, dtype=float).ravel()
    if X.shape != (6,):
        raise ValueError(f"X_rho 必须是 (6,)，得到 {X.shape}")
    rho = X[:3]
    rhodot = X[3:]
    C, Cdot_dim = _resolve_C_Cdot(context, t)
    q = rho
    p = rhodot - Cdot_dim.T @ C @ rho
    return np.concatenate([q, p])


def em_to_rho(
    X_em: npt.ArrayLike,
    t: float,
    context: NormalFormContext,
) -> npt.NDArray[np.floating]:
    """EM 坐标 ``(q, p)`` → rho 状态（无量纲）。

    对应 qiao ``param2rho`` 末段（逆动量耦合）：``ρ = q``，
    ``ρ̇ = p + Cdot_dimᵀ · C · q``。是 :func:`rho_to_em` 的精确逆。

    Args:
        X_em: ``(6,)`` EM 状态 ``[q, p]``，无量纲。
        t: 归一化时间 TU。
        context: 归一化上下文。

    Returns:
        ``(6,)`` rho 状态 ``[ρ, ρ̇]``，无量纲。
    """
    X = np.asarray(X_em, dtype=float).ravel()
    if X.shape != (6,):
        raise ValueError(f"X_em 必须是 (6,)，得到 {X.shape}")
    q = X[:3]
    p = X[3:]
    C, Cdot_dim = _resolve_C_Cdot(context, t)
    rho = q
    rhodot = p + Cdot_dim.T @ C @ q
    return np.concatenate([rho, rhodot])


__all__ = ["em_to_rho", "rho_to_em"]
