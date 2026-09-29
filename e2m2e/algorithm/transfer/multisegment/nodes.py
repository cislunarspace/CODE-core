"""节点约束组装：flyby 节点的等模等式与转角不等式（#726 自 SF 搬移）。

逻辑逐行来自 ``SimsFlanaganMultiLegProblem._flyby_constraints``（#741），
节点参数（``mu_km3_s2``/``r_p_min_km``）从 ``SimsFlanaganNode`` 改为显式
关键字参数，供 SF 与 MGA 精化两个框架实例共用。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..mga import flyby_turn_angle

__all__ = ["FlybyConstraintEval", "flyby_node_constraints"]

#: 方向/转角梯度退化的模长阈值（km/s）：与 Sims-Flanagan 的 û 阈值同值
#: （1e-12），低于该值的方向向量与转角梯度按退化处理。
_UHAT_EPS_KM_S = 1e-12


@dataclass(frozen=True)
class FlybyConstraintEval:
    """单 flyby 节点约束的全部评估产物（值 + 相对 [v_in; v_out] 的局部梯度）。"""

    v_in: npt.NDArray[np.floating]
    v_out: npt.NDArray[np.floating]
    n_in: float
    n_out: float
    delta: float
    delta_max: float
    v_eff: float
    eq_val: float
    eq_grad: npt.NDArray[np.floating]
    ineq_val: float
    ineq_grad: npt.NDArray[np.floating]


def flyby_node_constraints(
    v_in: npt.ArrayLike,
    v_out: npt.ArrayLike,
    *,
    r_p_min_km: float,
    mu_km3_s2: float,
    vel_scale: float,
) -> FlybyConstraintEval:
    """flyby 节点的等模等式与转角不等式（值 + 相对 ``[v_in; v_out]`` 的梯度）。

    等模 ``(|v_in|² − |v_out|²)/vel_scale²``；转角 ``g = δ_max − δ``，
    ``δ = acos(clip(v̂_in·v̂_out))``，``δ_max`` 复用
    :func:`.mga.flyby_turn_angle` 在 ``v_eff = (|v_in|+|v_out|)/2`` 处闭式。
    退化守卫：任一模低于 ``_UHAT_EPS_KM_S`` 时 δ 及其梯度取 0；``v_eff``
    低于阈值时 ``g ≡ 1.0``（常数可行、梯度 0）。``∂δ/∂v_in =
    −(v̂_out − cosδ·v̂_in)/(n_in·sinδ)``（``sinδ`` 低于阈值取 0）；
    ``dδ_max/dv_eff = −4√k/((1+k·v²)·√(2+k·v²))``（``k = r_p_min/μ``，
    ``1−w²`` 的稳定重排，``k → 0`` 时无 catastrophic cancellation），再链
    ``∂v_eff/∂v_in = ½·v̂_in``。

    Args:
        v_in: 进入渐近线速度 ``(3,)``（km/s）。
        v_out: 离开渐近线速度 ``(3,)``（km/s）。
        r_p_min_km: 最小近心点半径（km），正有限。
        mu_km3_s2: 飞越天体引力常数（km³/s²），正有限。
        vel_scale: 速度归一化尺度（km/s），正有限。

    Returns:
        :class:`FlybyConstraintEval`（``eq_val``/``ineq_val`` 为约束值，
        ``eq_grad``/``ineq_grad`` 为相对 ``[v_in; v_out]`` 六列的梯度）。

    Raises:
        ValueError: 向量形状非 ``(3,)``、含非有限分量，或标量参数非正有限。
    """
    v_in_arr = np.asarray(v_in, dtype=float)
    v_out_arr = np.asarray(v_out, dtype=float)
    if v_in_arr.shape != (3,) or v_out_arr.shape != (3,):
        raise ValueError(f"v_in/v_out 必须为 (3,) 向量，得到 {v_in_arr.shape} 与 {v_out_arr.shape}")
    if not (np.all(np.isfinite(v_in_arr)) and np.all(np.isfinite(v_out_arr))):
        raise ValueError("v_in/v_out 含非有限分量")
    scalars = (("r_p_min_km", r_p_min_km), ("mu_km3_s2", mu_km3_s2), ("vel_scale", vel_scale))
    for name, value in scalars:
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} 必须为正的有限数，得到 {value!r}")

    eps = _UHAT_EPS_KM_S
    vel2 = float(vel_scale) ** 2
    n_in = float(np.linalg.norm(v_in_arr))
    n_out = float(np.linalg.norm(v_out_arr))
    eq_val = (n_in**2 - n_out**2) / vel2
    eq_grad = np.concatenate([2.0 * v_in_arr, -2.0 * v_out_arr]) / vel2

    u_in: npt.NDArray[np.floating] = np.zeros(3)
    u_out: npt.NDArray[np.floating] = np.zeros(3)
    dd_in: npt.NDArray[np.floating] = np.zeros(3)
    dd_out: npt.NDArray[np.floating] = np.zeros(3)
    delta = 0.0
    if n_in >= eps and n_out >= eps:
        u_in = v_in_arr / n_in
        u_out = v_out_arr / n_out
        cos_d = float(np.clip(np.dot(u_in, u_out), -1.0, 1.0))
        delta = float(np.arccos(cos_d))
        sin_d = float(np.sin(delta))
        if sin_d >= eps:
            dd_in = -(u_out - cos_d * u_in) / (n_in * sin_d)
            dd_out = -(u_in - cos_d * u_out) / (n_out * sin_d)

    v_eff = 0.5 * (n_in + n_out)
    ineq_grad = np.zeros(6)
    delta_max = 0.0
    if v_eff <= eps:
        ineq_val = 1.0  # 常数可行（零退化极限），梯度 0
    else:
        delta_max = flyby_turn_angle(r_p_min_km, v_eff, mu_km3_s2)
        ineq_val = delta_max - delta
        k = r_p_min_km / mu_km3_s2
        kv2 = k * v_eff**2
        dd_max = -4.0 * np.sqrt(k) / ((1.0 + kv2) * np.sqrt(2.0 + kv2))
        ineq_grad = np.concatenate([0.5 * dd_max * u_in, 0.5 * dd_max * u_out]) - np.concatenate(
            [dd_in, dd_out]
        )
    return FlybyConstraintEval(
        v_in=v_in_arr,
        v_out=v_out_arr,
        n_in=n_in,
        n_out=n_out,
        delta=delta,
        delta_max=delta_max,
        v_eff=v_eff,
        eq_val=float(eq_val),
        eq_grad=eq_grad,
        ineq_val=float(ineq_val),
        ineq_grad=ineq_grad,
    )
