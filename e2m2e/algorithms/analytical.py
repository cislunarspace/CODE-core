"""
Halo轨道解析近似模块

提供Richardson三阶近似等解析方法用于生成Halo轨道的初始猜测。
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple

import numpy.typing as npt


def compute_halo_coefficients(mu: float, L: int) -> Dict[str, float]:
    """计算Halo轨道三阶近似的系数

    参数：
        mu: 质量比
        L: 拉格朗日点 (1=L1, 2=L2)

    返回：
        dict: 包含所有系数的字典

    参考：Richardson, D. L. (1980). Analytic construction of periodic orbits
          about the collinear points. Celestial Mechanics.
    """
    if L not in [1, 2]:
        raise ValueError(f"L必须是1或2，当前为{L}")

    gamma_dict = {1: 0.012149, 2: -0.012149}
    gamma = gamma_dict[L]

    c1 = 1.0 - mu - (1 - 2 * mu) * gamma**3 / (1 - gamma) ** 3
    c2 = 2 * mu * (1 - mu)
    c3 = 3 * mu * (2 - mu)

    if L == 1:
        a21 = 1.0 / (2 * gamma)
        a22 = (3 * gamma + 1) / (4 * gamma**2)
        a23 = -(3 * gamma + 1) / (8 * gamma**3)
        a24 = -(3 * gamma - 1) / (8 * gamma**3)
        a31 = 1.0 / (8 * gamma**2)

        b21 = (3 * gamma + 2) / (4 * gamma)
        b22 = (3 * gamma - 1) / (4 * gamma)
        b31 = 1.0 / (16 * gamma**2)

        d21 = (3 * gamma + 1) / (4 * gamma**2)
        d31 = (3 * gamma + 2) / (32 * gamma**3)
        d32 = (3 * gamma - 1) / (32 * gamma**3)

        k = 1.0
        delta = -1.0

    else:
        a21 = -1.0 / (2 * gamma)
        a22 = (3 * gamma - 1) / (4 * gamma**2)
        a23 = (3 * gamma - 1) / (8 * gamma**3)
        a24 = (3 * gamma + 1) / (8 * gamma**3)
        a31 = 1.0 / (8 * gamma**2)

        b21 = -(3 * gamma - 2) / (4 * gamma)
        b22 = -(3 * gamma + 1) / (4 * gamma)
        b31 = 1.0 / (16 * gamma**2)

        d21 = (3 * gamma - 1) / (4 * gamma**2)
        d31 = (3 * gamma - 2) / (32 * gamma**3)
        d32 = (3 * gamma + 1) / (32 * gamma**3)

        k = -1.0
        delta = 1.0

    l1 = -1.0 / (2 * gamma)
    l2 = (3 * gamma**2 + 3 * gamma + 1) / (4 * gamma**2)
    l3 = (3 * gamma**2 + 9 * gamma + 4) / (32 * gamma**3)

    kappa1 = (3 * gamma**2 + 3 * gamma + 1) / (4 * gamma**2)
    kappa2 = (3 * gamma**2 + 9 * gamma + 4) / (32 * gamma**3)

    return {
        "gamma": gamma,
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "a21": a21,
        "a22": a22,
        "a23": a23,
        "a24": a24,
        "a31": a31,
        "b21": b21,
        "b22": b22,
        "b31": b31,
        "d21": d21,
        "d31": d31,
        "d32": d32,
        "k": k,
        "delta": delta,
        "l1": l1,
        "l2": l2,
        "l3": l3,
        "kappa1": kappa1,
        "kappa2": kappa2,
    }


def halo_third_order_approximation(
    mu: float,
    Au: float,
    Aw: float,
    phi: float,
    L: int,
    tf: float,
    N: int,
    halo_class: int = 0,
) -> Tuple[npt.NDArray, npt.NDArray, float]:
    """计算Halo轨道三阶解析近似

    参数：
        mu: 质量比
        Au: U方向振幅
        Aw: W方向振幅
        phi: 相位偏移
        L: 拉格朗日点 (1=L1, 2=L2)
        tf: 终止时间
        N: 点数
        halo_class: 0=Class I (北), 1=Class II (南)

    返回：
        SV_uvw: 状态向量序列 (N, 6)，[u, v, w, u_dot, v_dot, w_dot]
        t: 时间序列
        T: 周期

    参考：Richardson, D. L. (1980). Analytic construction of periodic orbits
          about the collinear points. Celestial Mechanics.
    """
    if L not in [1, 2]:
        raise ValueError(f"L必须是1或2，当前为{L}")
    if N < 2:
        raise ValueError(f"N必须大于等于2，当前为{N}")
    if tf <= 0:
        raise ValueError(f"tf必须为正数，当前为{tf}")
    if halo_class not in [0, 1]:
        raise ValueError(f"halo_class必须是0或1，当前为{halo_class}")

    coeffs = compute_halo_coefficients(mu, L)
    gamma = coeffs["gamma"]
    L_position = 1 - mu - gamma if L == 1 else 1 - mu + gamma

    a21 = coeffs["a21"]
    a22 = coeffs["a22"]
    a23 = coeffs["a23"]
    a24 = coeffs["a24"]
    a31 = coeffs["a31"]
    b21 = coeffs["b21"]
    b22 = coeffs["b22"]
    b31 = coeffs["b31"]
    d21 = coeffs["d21"]
    d31 = coeffs["d31"]
    d32 = coeffs["d32"]
    k = coeffs["k"]
    delta = coeffs["delta"]
    kappa1 = coeffs["kappa1"]
    kappa2 = coeffs["kappa2"]

    if halo_class == 1:
        delta = -delta
        phi = phi + np.pi

    T = 2 * np.pi * (1 + kappa1 * Au**2 + kappa2 * Aw**2)
    tau = np.linspace(0, 2 * np.pi, N)
    t = np.linspace(0, tf, N)

    u = (
        a21 * Au**2
        + a22 * Aw**2
        - Au * np.cos(tau + phi)
        + (a23 * Au**2 - a24 * Aw**2) * np.cos(2 * (tau + phi))
        + a31 * Au**3 * np.cos(3 * (tau + phi))
    )

    v = (
        k * Au * np.sin(tau + phi)
        + (b21 * Au**2 - b22 * Aw**2) * np.sin(2 * (tau + phi))
        + b31 * Au**3 * np.sin(3 * (tau + phi))
    )

    w = delta * (
        Aw * np.cos(tau + phi)
        + d21 * Au * Aw * (np.cos(2 * (tau + phi)) - 3)
        + (d32 * Aw * Au**2 - d31 * Aw**3) * np.cos(3 * (tau + phi))
    )

    u_dot = Au * np.sin(tau + phi) + 2 * (a23 * Au**2 - a24 * Aw**2) * np.sin(2 * (tau + phi))
    v_dot = k * Au * np.cos(tau + phi) + 2 * (b21 * Au**2 - b22 * Aw**2) * np.cos(2 * (tau + phi))
    w_dot = -Aw * np.sin(tau + phi) - 2 * d21 * Au * Aw * np.sin(2 * (tau + phi))

    x = L_position + u
    y = v
    z = w

    x_dot = u_dot
    y_dot = v_dot
    z_dot = w_dot

    SV_uvw = np.column_stack([x, y, z, x_dot, y_dot, z_dot])

    return SV_uvw, t, T


def compute_halo_initial_guess(
    mu: float,
    z_amplitude: float,
    L: int = 1,
    halo_class: int = 0,
) -> Dict[str, float]:
    """计算Halo轨道初始猜测参数

    用于生成高质量的初始猜测，配合微分修正器使用。

    参数：
        mu: 质量比
        z_amplitude: Z方向振幅
        L: 拉格朗日点 (1=L1, 2=L2)
        halo_class: 0=北Halo, 1=南Halo

    返回：
        dict: 包含初始猜测参数的字典
            - x0: 初始x坐标
            - y0: 初始y坐标 (0)
            - z0: 初始z坐标
            - vx0: 初始vx (0)
            - vy0: 初始vy
            - vz0: 初始vz (0)
            - T_half: 半周期
    """
    if z_amplitude <= 0:
        raise ValueError(f"z_amplitude必须为正数，当前为{z_amplitude}")

    gamma_dict = {1: 0.012149, 2: -0.012149}
    gamma = gamma_dict[L]

    coeffs = compute_halo_coefficients(mu, L)
    kappa1 = coeffs["kappa1"]

    Aw = z_amplitude
    Au = np.sqrt(-kappa1 * z_amplitude**2 / coeffs["l1"])

    L_position = 1 - mu - gamma if L == 1 else 1 - mu + gamma

    if L == 1:
        x0 = L_position - coeffs["a21"] * Au**2 - coeffs["a22"] * Aw**2 + Au
    else:
        x0 = L_position - coeffs["a21"] * Au**2 - coeffs["a22"] * Aw**2 - Au

    vy0 = -coeffs["k"] * Au * (1 + coeffs["l1"] * Au**2 + coeffs["l2"] * Aw**2)

    T_half = np.pi * (1 + kappa1 * Au**2)

    return {
        "x0": x0,
        "y0": 0.0,
        "z0": 0.0,
        "vx0": 0.0,
        "vy0": vy0,
        "vz0": 0.0,
        "T_half": T_half,
        "Au": Au,
        "Aw": Aw,
    }


__all__ = [
    "compute_halo_coefficients",
    "halo_third_order_approximation",
    "compute_halo_initial_guess",
]
