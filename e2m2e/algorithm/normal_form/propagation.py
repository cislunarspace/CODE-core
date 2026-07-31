"""M0 流积分器：在中心流形上传播表征参数。

移植 qiao ``propagation.py`` 的 ``propagate_parametric``，用降幂写法计算
Hamilton 正则方程右端项（避免 qiao 原版的除法 0/0）。

表征参数链（M0 流）：

    rho 初值 → param（经 catalog_transformer）→ Hamilton 正则方程积分
    （钳制双曲方向 q1/p1=0，留 4 维中心流形）→ param → rho

中心流形约化后，Hamilton 量 ``H`` 不含双曲-中心耦合；在中心流形上（q1=p1=0）
积分 ``H`` 的正则方程即得 Lissajous 轨道。双曲方向被钳制，保证不发散。

参考：
- qiao ``propagate_parametric`` / ``_eval_hamiltonian_rhs``
- Gómez vol III 2.7：约化 H⁰ 在 ``(ξ, η)`` 中心坐标上的流
- ``run_nrho_normal_form.py`` 的约定注释（CM 坐标、降幂写法、双曲钳制）
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .context import NormalFormContext
    from .types import NormalFormResult

__all__ = ["propagate_parametric"]


def _eval_hamiltonian_rhs(
    t: float,
    X: npt.ArrayLike,
    hamiltonian_terms: dict[tuple[int, ...], npt.NDArray[np.floating]],
    tlist: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Hamilton 正则方程右端项（CM 坐标 ``[q1,q2,q3,p1,p2,p3]``）。

    .. math::
        H = \\sum c(t) \\cdot q_1^{n_1} q_2^{n_2} q_3^{n_3} p_1^{n_4} p_2^{n_5} p_3^{n_6}

        \\dot q_i = \\partial H/\\partial p_i, \\quad \\dot p_i = -\\partial H/\\partial q_i

    偏导用**降幂写法**：对单项式先算完整幂 ``m = c·∏x_k^{n_k}``，再对方向 k
    乘 ``n_k`` 并降一次幂（``x_k^{n_k} = x_k^{n_k-1} \\cdot x_k``，故
    ``∂/∂x_k (c·x_k^{n_k}·...) = c·n_k·x_k^{n_k-1}·...``）。实现上先算
    ``m``，再乘 ``n_k / x_k`` 等价于乘 ``n_k`` 并把 ``x_k`` 的幂降一次——
    但直接 ``m / x_k`` 在 ``x_k=0`` 时除零。改用：先算不含 ``x_k`` 的部分
    ``m_{\\bar k} = m / x_k^{n_k}``（解析地，不真的除），再 ``∂/∂x_k = c·n_k·x_k^{n_k-1}·m_{\\bar k}``。

    本实现用「逐项累积 + 降幂」：对每项先算 ``m``（完整幂积），再对每个
    非零指数方向，贡献 ``n_k · m / x_k``——但为避免除零，改写成
    ``n_k · (m 不含 x_k 的部分) · x_k^{n_k-1}``，即重新算一次降幂后的幂积。

    双曲方向钳制：``dX[0]=dX[3]=0``（q1、p1 方向冻结），留中心流形。

    Args:
        t: 归一化时间 TU。
        X: ``(6,)`` CM 坐标状态 ``[q1,q2,q3,p1,p2,p3]``。
        hamiltonian_terms: ``{pow_tuple: coef_array}``，系数为时间序列。
        tlist: 系数时间序列的网格（``np.interp`` 用）。

    Returns:
        ``(6,)`` 右端项 ``[q̇1,q̇2,q̇3,ṗ1,ṗ2,ṗ3]``。
    """
    X_arr = np.asarray(X, dtype=float).ravel()
    dX = np.zeros(6, dtype=float)
    t_arr = np.asarray(tlist, dtype=float).ravel()

    for (n1, n2, n3, n4, n5, n6), coef_arr in hamiltonian_terms.items():
        arr = np.asarray(coef_arr, dtype=float).ravel()
        if arr.size == 0:
            continue
        val = float(np.interp(t, t_arr, arr)) if arr.size > 1 else float(arr[0])
        if val == 0.0:
            continue

        # q̇_i = ∂H/∂p_i：对 p_i 降幂。p_i 指数对应 n4(p1), n5(p2), n6(p3)。
        # m = val · q1^n1 · q2^n2 · q3^n3 · p1^n4 · p2^n5 · p3^n6
        # ∂H/∂p_i = val · n_{p_i} · (其余不变) · p_i^{n_{p_i}-1}
        # 用降幂：先算 m，再 ∂H/∂p_i = m · n_{p_i} / p_i。
        # 为避免除零，直接按幂重组：对 p_i 方向，幂积用 p_i^(n-1) 而非 p_i^n / p_i。
        q1, q2, q3, p1, p2, p3 = X_arr

        # 预计算各变量的幂（降幂形式：x^n 与 x^(n-1)）
        # 用 np.power 处理 0^0=1 的约定（0^0 在幂积中视为 1，即该项不含该变量）
        def _pow(base: float, exp: int) -> float:
            if exp == 0:
                return 1.0
            if base == 0.0:
                return 0.0  # 0^n (n>0) = 0
            return base ** exp

        # q̇_i = ∂H/∂p_i：p_i 指数降 1，乘 n_{p_i}
        if n4 > 0:  # ∂/∂p1
            dX[0] += val * n4 * (
                _pow(q1, n1) * _pow(q2, n2) * _pow(q3, n3)
                * _pow(p1, n4 - 1) * _pow(p2, n5) * _pow(p3, n6)
            )
        if n5 > 0:  # ∂/∂p2
            dX[1] += val * n5 * (
                _pow(q1, n1) * _pow(q2, n2) * _pow(q3, n3)
                * _pow(p1, n4) * _pow(p2, n5 - 1) * _pow(p3, n6)
            )
        if n6 > 0:  # ∂/∂p3
            dX[2] += val * n6 * (
                _pow(q1, n1) * _pow(q2, n2) * _pow(q3, n3)
                * _pow(p1, n4) * _pow(p2, n5) * _pow(p3, n6 - 1)
            )

        # ṗ_i = -∂H/∂q_i：q_i 指数降 1，乘 -n_{q_i}
        if n1 > 0:  # -∂/∂q1
            dX[3] -= val * n1 * (
                _pow(q1, n1 - 1) * _pow(q2, n2) * _pow(q3, n3)
                * _pow(p1, n4) * _pow(p2, n5) * _pow(p3, n6)
            )
        if n2 > 0:  # -∂/∂q2
            dX[4] -= val * n2 * (
                _pow(q1, n1) * _pow(q2, n2 - 1) * _pow(q3, n3)
                * _pow(p1, n4) * _pow(p2, n5) * _pow(p3, n6)
            )
        if n3 > 0:  # -∂/∂q3
            dX[5] -= val * n3 * (
                _pow(q1, n1) * _pow(q2, n2) * _pow(q3, n3 - 1)
                * _pow(p1, n4) * _pow(p2, n5) * _pow(p3, n6)
            )

    # 钳制双曲方向（q1/p1），留中心流形。对应 Gómez 2.7 的 q1=p1=0 约化。
    dX[0] = 0.0
    dX[3] = 0.0
    return dX


def propagate_parametric(
    rho_state: npt.ArrayLike,
    t_span: npt.ArrayLike,
    nf_result: NormalFormResult,
    context: NormalFormContext,
    *,
    truth_rho: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray | None]:
    """表征参数链外推（M0 流：Hamilton 正则方程，DOP853 积分器）。

    在中心流形上积分 Hamilton 正则方程（钳制双曲方向 ``q1/p1=0``），逐时刻
    转回 rho。这是约化 Hamilton ``H⁰`` 的流——Lissajous 轨道。

    **坐标约定**：积分在 **CM 笛卡尔坐标** ``[q1,q2,q3,p1,p2,p3]`` 上进行
    （:func:`_eval_hamiltonian_rhs` 的正则方程），输入输出端经
    ``param_to_cm``/``cm_to_param`` 与表征参数 ``(q1,p1,I2,θ2,I3,θ3)``
    互转。不要直接把 param 当笛卡尔积分——作用量-角变量不是正则坐标。

    Args:
        rho_state: ``(6,)`` 初始 rho 状态（无量纲）。
        t_span: ``(K,)`` 时间序列（无量纲 TU），单调递增。
        nf_result: 标准形结果（提供 catalog_transformer 与 hamiltonian_terms）。
        context: 标准形上下文。
        truth_rho: 可选真值 rho 轨迹 ``(K, 6)``，与 t_span 同网格。

    Returns:
        ``(t_out, rho_out, pos_err_km)`` 元组。
        - ``t_out``: ``(M,)`` 输出时间（TU）。
        - ``rho_out``: ``(M, 6)`` rho 状态。
        - ``pos_err_km``: ``(M,)`` 位置误差（km），无真值时为 ``None``。
    """
    from scipy.integrate import solve_ivp

    from .coord_trans.cm_param import cm_to_param, param_to_cm

    if nf_result.catalog_transformer is None:
        raise ValueError(
            "nf_result.catalog_transformer 为 None，无法做 rho↔param 变换。"
        )
    if nf_result.cm_result is None:
        raise ValueError("nf_result.cm_result 为 None，缺少 Hamiltonian 系数。")

    rho_state = np.asarray(rho_state, dtype=float)
    t_arr = np.asarray(t_span, dtype=float).ravel()
    if t_arr.size < 2:
        return t_arr, rho_state.reshape(1, -1), None

    transformer = nf_result.catalog_transformer
    hamiltonian_terms = nf_result.cm_result.hamiltonian_terms
    tlist = np.asarray(nf_result.qf_result.tlist, dtype=float).ravel()

    # rho → param → CM 笛卡尔（作用量-角变量 → 正则坐标）初值
    X0_param = transformer.rho_to_param(rho_state, t_arr[0])
    if np.any(np.isnan(X0_param)):
        return np.array([]), np.empty((0, 6)), None
    X0_cm = param_to_cm(X0_param)

    # 积分 Hamilton 正则方程（CM 笛卡尔坐标）
    span = t_arr[-1] - t_arr[0]
    max_step = 0.1 * abs(span)
    sol = solve_ivp(
        fun=lambda t, X: _eval_hamiltonian_rhs(t, X, hamiltonian_terms, tlist),
        t_span=[t_arr[0], t_arr[-1]],
        y0=X0_cm,
        method="DOP853",
        t_eval=t_arr,
        rtol=1e-12,
        atol=1e-14,
        max_step=max_step,
    )
    if not sol.success:
        return np.array([]), np.empty((0, 6)), None

    X_cm_list = sol.y.T
    t_out = sol.t

    # 逐时刻 CM 笛卡尔 → param → rho
    rho_list = []
    t_valid = []
    for i in range(len(t_out)):
        if np.all(np.isfinite(X_cm_list[i])):
            X_param_i = cm_to_param(X_cm_list[i])
            rho_i = transformer.param_to_rho(X_param_i, t_out[i])
            if np.all(np.isfinite(rho_i)):
                rho_list.append(rho_i)
                t_valid.append(t_out[i])

    if len(rho_list) < 2:
        return np.array([]), np.empty((0, 6)), None

    t_out = np.array(t_valid)
    rho_out = np.array(rho_list)

    pos_err_km = None
    if truth_rho is not None:
        truth = np.asarray(truth_rho, dtype=float)
        if truth.shape[0] == rho_out.shape[0]:
            pos_err_km = (
                np.linalg.norm(rho_out[:, :3] - truth[:, :3], axis=1) * context.LU
            )

    return t_out, rho_out, pos_err_km
