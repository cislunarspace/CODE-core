"""``normal_form`` 包内部：从 SPICE 求 ``Eval_expr`` 所需星历参数。

实现相当于 qiao ``Eval_expr.py`` 的功能，但通过 e2m2e 自带的
``SPICEManager`` 获取天体状态，避免上层 ``Hamilton`` 模块直接依赖
``spiceypy`` / DE 内核路径。

输出 dict 与 qiao ``Eval_expr`` 完全一致（键名一致、形状一致），便于
``Hamilton.evaluated_coefficients`` 与 qiao fixture 对齐。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .context import NormalFormContext


# 共线平动点 γ 值——与 ``constants._COLLINEAR_GAMMAS`` 保持一致；
# 在内部模块复制一份避免与顶层常量循环导入（常量子模块可被本文件
# 反过来引用，故独立定义）。
_COLLINEAR_GAMMAS: dict[int, float] = {
    1: 0.150934288618019,
    2: 0.167832751054508,
    3: 0.992912060200654,
}


def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )


def _derive_moon_param(
    r_em: np.ndarray,
    v_em: np.ndarray,
    r_es: np.ndarray,
    v_es: np.ndarray,
    mu_e: float,
    mu_m: float,
    mu_s: float,
    lu: float,
    tu: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """推导 EMR 旋转矩阵 C(t)、Cdot、Cdotdot 与 A_EM。

    与 qiao ``Calc_MoonParam._derive_moon_param`` 等价。
    本函数刻意保持与 qiao 源 1:1 对齐（包括形式上相同的中间变量名），
    以便后续切片能够简单叉验差异。
    """
    gm_e = mu_e * lu**3 / tu**2
    gm_m = mu_m * lu**3 / tu**2
    gm_s = mu_s * lu**3 / tu**2

    r_sm = r_em - r_es
    v_sm = v_em - v_es

    r_em_n = float(np.linalg.norm(r_em))
    r_es_n = float(np.linalg.norm(r_es))
    r_sm_n = float(np.linalg.norm(r_sm))

    r_em2 = r_em_n * r_em_n
    r_es2 = r_es_n * r_es_n
    r_sm2 = r_sm_n * r_sm_n

    # A_EM = -(GM_e+GM_m) * R_EM / r_em^3 - GM_s * (R_SM/r_sm^3 + R_ES/r_es^3)
    a_em = (
        -(gm_e + gm_m) / (r_em2 * r_em_n) * r_em
        - gm_s / (r_sm2 * r_sm_n) * r_sm
        - gm_s / (r_es2 * r_es_n) * r_es
    )

    # J_EM = -Σ GM * (V*|R|^2 - 3*<R,V>*R) / |R|^5
    rv_em = float(np.dot(r_em, v_em))
    j_em = -(gm_e + gm_m) / (r_em2 * r_em2 * r_em_n) * (r_em2 * v_em - 3 * rv_em * r_em)
    rv_sm = float(np.dot(r_sm, v_sm))
    j_em = j_em - gm_s / (r_sm2 * r_sm2 * r_sm_n) * (r_sm2 * v_sm - 3 * rv_sm * r_sm)
    rv_es = float(np.dot(r_es, v_es))
    j_em = j_em - gm_s / (r_es2 * r_es2 * r_es_n) * (r_es2 * v_es - 3 * rv_es * r_es)

    # Rotation matrix C(t)
    em_r = r_em_n
    ldot = rv_em / em_r
    ldotdot = (float(np.dot(v_em, v_em) + np.dot(r_em, a_em)) * em_r - ldot * rv_em) / em_r**2

    h = _cross3(r_em, v_em)
    hn = float(np.linalg.norm(h))
    hdot = _cross3(r_em, a_em)
    hndot = float(np.dot(hdot, h)) / hn
    hdotdot = _cross3(r_em, j_em) + _cross3(v_em, a_em)
    hndotdot = (float(np.dot(hdotdot, h)) + float(np.dot(hdot, hdot))) / hn - float(
        np.dot(h, hdot)
    ) * hndot / (hn * hn)

    x_hat = r_em / em_r
    z_hat = h / hn
    y_hat = _cross3(z_hat, x_hat)
    c = np.column_stack([x_hat, y_hat, z_hat])

    x_hat_dot = v_em / em_r - (ldot / em_r) * x_hat
    z_hat_dot = hdot / hn - (hndot / hn) * z_hat
    y_hat_dot = _cross3(z_hat_dot, x_hat) + _cross3(z_hat, x_hat_dot)
    cdot = np.column_stack([x_hat_dot, y_hat_dot, z_hat_dot])

    x_hat_dotdot = (
        a_em / em_r
        - (ldot / (em_r * em_r)) * v_em
        - (ldot / em_r) * x_hat_dot
        - (ldotdot / em_r - (ldot * ldot) / (em_r * em_r)) * x_hat
    )
    z_hat_dotdot = (
        hdotdot / hn
        - (hndot / (hn * hn)) * hdot
        - (hndot / hn) * z_hat_dot
        - (hndotdot / hn - (hndot * hndot) / (hn * hn)) * z_hat
    )
    y_hat_dotdot = (
        _cross3(z_hat_dotdot, x_hat)
        + 2 * _cross3(z_hat_dot, x_hat_dot)
        + _cross3(z_hat, x_hat_dotdot)
    )
    cdotdot = np.column_stack([x_hat_dotdot, y_hat_dotdot, z_hat_dotdot])

    return c, cdot, cdotdot, a_em


def _lp_state(
    r_em: np.ndarray,
    v_em: np.ndarray,
    a_em: np.ndarray,
    libr: int,
    c: np.ndarray,
    cdot: np.ndarray,
    cdotdot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """平动点位置、速度、加速度（J2000 系，km、km/s、km/s²）。

    与 qiao ``Calc_LPstate`` 等价。``c/cdot/cdotdot`` 仅 L4/L5 使用。
    """
    if libr in (1, 2, 3):
        gamma = _COLLINEAR_GAMMAS[libr]
        if libr == 1:
            return (1 - gamma) * r_em, (1 - gamma) * v_em, (1 - gamma) * a_em
        if libr == 2:
            return (1 + gamma) * r_em, (1 + gamma) * v_em, (1 + gamma) * a_em
        # libr == 3
        return -gamma * r_em, -gamma * v_em, -gamma * a_em

    ang = -np.pi / 3 if libr == 4 else np.pi / 3
    r_mat = np.array(
        [
            [np.cos(ang), np.sin(ang), 0.0],
            [-np.sin(ang), np.cos(ang), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    m = c @ r_mat @ c.T
    r_lp = m @ r_em
    v_lp = c @ r_mat @ c.T @ v_em + c @ r_mat @ cdot.T @ r_em + cdot @ r_mat @ c.T @ r_em
    a_lp = (
        c @ r_mat @ c.T @ a_em
        + 2 * c @ r_mat @ cdot.T @ v_em
        + 2 * cdot @ r_mat @ c.T @ v_em
        + c @ r_mat @ cdotdot.T @ r_em
        + 2 * cdot @ r_mat @ cdot.T @ r_em
        + cdotdot @ r_mat @ c.T @ r_em
    )
    return r_lp, v_lp, a_lp


def _ephemeris_states(jd_tdb: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 (R_EM, V_EM, R_ES, V_ES)（J2000 系, km/km/s）。

    复用 :class:`e2m2e.core.SPICEManager`；其内部已自动加载闰秒
    内核，因此本函数无需关心 ``de430.bsp`` 等星历内核路径。
    """
    from e2m2e.core import SPICEManager
    from e2m2e.core._spice_loader import get_spiceypy

    mgr = SPICEManager()
    spice = get_spiceypy()
    et = float(spice.str2et(f"{jd_tdb:.20f} JDTDB"))
    moon_state = mgr.get_body_state("MOON", et, "J2000", "EARTH")
    sun_state = mgr.get_body_state("SUN", et, "J2000", "EARTH")
    return (
        np.asarray(moon_state[:3], dtype=float),
        np.asarray(moon_state[3:6], dtype=float),
        np.asarray(sun_state[:3], dtype=float),
        np.asarray(sun_state[3:6], dtype=float),
    )


def eval_params(jd_tdb: float, context: NormalFormContext) -> dict[str, float]:
    """与 qiao ``Eval_expr`` 等价。

    Args:
        jd_tdb: 历书时（TDB）儒略日。
        context: :class:`NormalFormContext`，负责取 LU/TU/mu_e/mu_m/mu_s
            与 libr（平动点编号）。

    Returns:
        dict：键名与 qiao ``Eval_expr`` 完全一致——``Cpq1..Cpq9``、
        ``Cqq1..Cqq9``、``f1..f3``、``rex, rey, rez, re0``、``rmx, rmy,
        rmz, rm0``、``rsx, rsy, rsz, rs0``、``mu_e, mu_m, mu_s``。
    """
    tu = float(context.TU)
    lu = float(context.LU)
    mu_e = float(context.mu_e)
    mu_m = float(context.mu_m)
    mu_s = float(context.mu_s)
    libr = int(context.libration_point.value)

    r_em, v_em, r_es, v_es = _ephemeris_states(jd_tdb)
    c, cdot, cdotdot, a_em = _derive_moon_param(
        r_em,
        v_em,
        r_es,
        v_es,
        mu_e=mu_e,
        mu_m=mu_m,
        mu_s=mu_s,
        lu=lu,
        tu=tu,
    )

    # 平动点（J2000 系，km / km/s / km/s²）；再转无量纲 EMR 坐标。
    r_lp, v_lp, a_lp = _lp_state(r_em, v_em, a_em, libr, c, cdot, cdotdot)
    cdot_nd = cdot * tu
    cdotdot_nd = cdotdot * tu**2
    v_u = lu / tu
    a_u = lu / tu**2

    r0 = c.T @ (r_lp / lu)
    r0dot = cdot_nd.T @ (r_lp / lu) + c.T @ (v_lp / v_u)
    r0dotdot = cdotdot_nd.T @ (r_lp / lu) + 2 * cdot_nd.T @ (v_lp / v_u) + c.T @ (a_lp / a_u)

    rm_nd = c.T @ (r_em / lu)
    rs_nd = c.T @ (r_es / lu)

    cpq = cdot_nd.T @ c
    cqq = c.T @ cdot_nd @ cdot_nd.T @ c + cdotdot_nd.T @ c

    force = (
        mu_m * rm_nd / float(np.linalg.norm(rm_nd)) ** 3
        + mu_s * rs_nd / float(np.linalg.norm(rs_nd)) ** 3
        + r0dotdot
        + 2 * c.T @ cdot_nd @ r0dot
        + c.T @ cdotdot_nd @ r0
    )

    rtemp = -r0
    rtemp_m = -r0 + rm_nd
    rtemp_s = -r0 + rs_nd

    params: dict[str, float] = {}
    for i in range(3):
        for j in range(3):
            params[f"Cpq{i * 3 + j + 1}"] = float(cpq[i, j])
            params[f"Cqq{i * 3 + j + 1}"] = float(cqq[i, j])
    params["f1"] = float(force[0])
    params["f2"] = float(force[1])
    params["f3"] = float(force[2])

    params["rex"] = float(rtemp[0])
    params["rey"] = float(rtemp[1])
    params["rez"] = float(rtemp[2])
    params["re0"] = float(np.linalg.norm(rtemp))

    params["rmx"] = float(rtemp_m[0])
    params["rmy"] = float(rtemp_m[1])
    params["rmz"] = float(rtemp_m[2])
    params["rm0"] = float(np.linalg.norm(rtemp_m))

    params["rsx"] = float(rtemp_s[0])
    params["rsy"] = float(rtemp_s[1])
    params["rsz"] = float(rtemp_s[2])
    params["rs0"] = float(np.linalg.norm(rtemp_s))

    params["mu_e"] = mu_e
    params["mu_m"] = mu_m
    params["mu_s"] = mu_s

    return params


__all__ = ["eval_params"]
