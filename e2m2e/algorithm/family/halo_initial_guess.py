"""
Halo轨道初始猜测模块

提供 Richardson 三阶近似方法，用于生成 Halo 轨道的初始猜测参数。
配合微分修正器使用，将解析近似结果精化为精确的周期轨道。

包含：
- Lagrange 点距离参数 gamma 的求解
- 面内振荡频率 omega_p 的计算
- Richardson 三阶近似系数
- Halo 轨道三阶解析近似
- 初始猜测参数生成

References:
    Richardson, D. L. (1980). Analytic construction of periodic orbits
    about the collinear points. Celestial Mechanics, 22(3), 303-320.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq


def _compute_gamma(mu: float, L: int) -> float:
    """求解平动点到次天体的距离参数 gamma。

    通过求解共线平动点位置的五次方程（由引力与离心力平衡条件导出），
    使用 Brent 方法得到精确的 gamma 值。gamma 定义为次天体到平动点的距离，
    以两个主天体间的距离（=1，归一化单位）为参考。

    对于 L1（位于两天体之间）和 L2（位于次天体外侧），五次方程的系数不同。

    Args:
        mu: 三体系统质量比 mu = m2 / (m1 + m2)，m2 为次天体质量。
        L: 拉格朗日点编号（1=L1, 2=L2）。

    Returns:
        float: gamma 值（始终为正数），表示次天体到平动点的归一化距离。

    Raises:
        ValueError: 当 L 不为 1 或 2 时。

    References:
        Richardson, D. L. (1980). Analytic construction of periodic orbits
        about the collinear points. Celestial Mechanics, 22(3), 303-320.
    """
    if L == 1:
        # L1 五次方程：由 L1 处引力加速度与旋转坐标系离心加速度平衡条件导出
        # gamma^5 - (3-mu)*gamma^4 + (3-2mu)*gamma^3 - mu*gamma^2 + 2*mu*gamma - mu = 0
        def eq(g: float) -> float:
            return g**5 - (3 - mu) * g**4 + (3 - 2 * mu) * g**3 - mu * g**2 + 2 * mu * g - mu
    else:
        # L2 五次方程：由 L2 处引力加速度与旋转坐标系离心加速度平衡条件导出
        # gamma^5 + (3-mu)*gamma^4 + (3-2mu)*gamma^3 - mu*gamma^2 - 2*mu*gamma - mu = 0
        def eq(g: float) -> float:
            return g**5 + (3 - mu) * g**4 + (3 - 2 * mu) * g**3 - mu * g**2 - 2 * mu * g - mu

    # Hill 球近似 (mu/3)^{1/3} 作为初始猜测，用于确定 Brent 方法的搜索区间
    g0 = (mu / 3) ** (1 / 3)
    # Brent 方法在 [g0/2, 2*g0] 区间上求解，保证收敛到物理上有意义的根
    return brentq(eq, g0 * 0.5, g0 * 2.0)


def _compute_omega_p(gamma: float, mu: float, L: int) -> float:
    """计算平动点处的面内振荡频率 omega_p。

    在平动点附近线性化 CR3BP 方程后，面内运动 (x-y) 的特征方程为：
    s^4 + (2 - c2)*s^2 + (1 + 2*c2)*(1 - c2) = 0
    其中 c2 是有效引力势在平动点处的二阶偏导数（Legendre 系数）。
    该特征方程有两对纯虚根 s = ±i*omega_p 和 s = ±i*omega_v，
    omega_p 对应面内振荡（较小的频率），omega_v 对应纵向振荡（较大的频率）。
    这里取较小的频率 omega_p 作为 Halo 轨道三阶近似中的基本频率。

    Args:
        gamma: 距离参数（次天体到平动点的归一化距离，始终取正值）。
        mu: 质量比。
        L: 拉格朗日点编号（1=L1, 2=L2）。

    Returns:
        float: omega_p，面内振荡频率（归一化角速度单位）。
    """
    # Legendre 系数 c2：来自有效引力势 U 的二阶展开
    # c2 = (1-mu)/d1^3 + mu/d2^3，其中 d1、d2 分别为主天体和次天体到平动点的距离
    if L == 1:
        c2 = (1 - mu) / abs(1 - gamma) ** 3 + mu / gamma**3
    else:
        c2 = (1 - mu) / abs(1 + gamma) ** 3 + mu / gamma**3

    # 面内特征方程: s^4 + (2-c2)*s^2 + (1+2*c2)*(1-c2) = 0
    # 令 s^2 = S，化为二次方程: S^2 + a*S + b = 0
    a = 2 - c2
    b = (1 + 2 * c2) * (1 - c2)
    disc = a**2 - 4 * b
    # 取负根 S_minus = (-a - sqrt(disc))/2 < 0，对应纯虚特征值
    # 因为 S_minus < 0，所以 s = ±sqrt(-S_minus) 为纯虚数，对应面内振荡
    s2_minus = (-a - np.sqrt(disc)) / 2
    return np.sqrt(-s2_minus)


def compute_halo_coefficients(mu: float, L: int) -> dict[str, float]:
    """计算 Halo 轨道 Richardson 三阶近似所需的全部系数。

    根据 Richardson (1980) 的三阶解析构造方法，在共线平动点附近将 CR3BP 运动方程
    展开为非线性扰动级数。三阶近似将轨道位移分解为面内（u-v）和面外（w）分量，
    用 Fourier 级数表示，包含基频 omega_p 的各阶谐波。

    核心系数包括：
    - a_ij: u 方向（沿主天体连线）的振幅修正系数
    - b_ij: v 方向（面内垂直方向）的振幅修正系数
    - d_ij: w 方向（面外方向）的振幅修正系数
    - k, delta: 与平动点位置相关的符号因子
    - kappa1, kappa2: 频率修正系数（用于计算非线性周期）

    Args:
        mu: 质量比 mu = m2 / (m1 + m2)。
        L: 拉格朗日点编号（1=L1, 2=L2）。

    Returns:
        Dict[str, float]: 包含所有 Richardson 三阶近似系数的字典，键包括：
            - gamma: 次天体到平动点的距离（L2 时取负值）
            - omega_p: 面内振荡基频
            - c1, c2, c3: Legendre 系数（有效势展开的前三阶）
            - a21~a31, b21~b31, d21~d32: 各方向振幅修正系数
            - k, delta: 符号因子
            - l1~l3, kappa1, kappa2: 频率和周期修正系数

    Raises:
        ValueError: 当 L 不为 1 或 2 时。

    References:
        Richardson, D. L. (1980). Analytic construction of periodic orbits
        about the collinear points. Celestial Mechanics, 22(3), 303-320.
    """
    if L not in [1, 2]:
        raise ValueError(f"L必须是1或2，当前为{L}")

    # 精确求解 gamma（次天体到平动点的距离），通过五次方程的 Brent 法求解
    gamma = _compute_gamma(mu, L)

    # L2 使用负值约定以匹配 Richardson 公式中的符号约定
    # （L1 的 gamma > 0，L2 的 gamma < 0，使得 x_L = 1 - mu - gamma 统一表达）
    if L == 2:
        gamma = -gamma

    # 计算面内振荡频率 omega_p（使用 gamma 的绝对值）
    omega_p = _compute_omega_p(abs(gamma), mu, L)

    abs_gamma = abs(gamma)

    # Legendre 系数：来自有效引力势在平动点处的 Taylor 展开
    # c_n = (1/gamma^3) * [(-1)^n * (1-mu) * (gamma/(1∓gamma))^{n+1} + mu]
    # 其中 ∓ 对应 L1/L2
    c1 = 1.0 - mu - (1 - 2 * mu) * abs_gamma**3 / (1 - abs_gamma) ** 3
    if L == 1:
        c2_c = (1 - mu) / (1 - abs_gamma) ** 3 + mu / abs_gamma**3
    else:
        c2_c = (1 - mu) / (1 + abs_gamma) ** 3 + mu / abs_gamma**3
    c3 = 3 * mu * (2 - mu)

    # Richardson 三阶近似系数（L1 和 L2 的公式不同，符号和分母略有差异）
    # 下标含义：第一个数字表示阶数，第二个数字表示谐波阶次
    # 例如 a21 = 二阶修正 × 一次谐波，a31 = 三阶修正 × 一次谐波
    if L == 1:
        # u 方向修正系数（沿 x 轴，主天体连线方向）
        a21 = 1.0 / (2 * gamma)
        a22 = (3 * gamma + 1) / (4 * gamma**2)
        a23 = -(3 * gamma + 1) / (8 * gamma**3)
        a24 = -(3 * gamma - 1) / (8 * gamma**3)
        a31 = 1.0 / (8 * gamma**2)

        # v 方向修正系数（面内垂直于 x 轴方向）
        b21 = (3 * gamma + 2) / (4 * gamma)
        b22 = (3 * gamma - 1) / (4 * gamma)
        b31 = 1.0 / (16 * gamma**2)

        # w 方向修正系数（面外方向，z 轴）
        d21 = (3 * gamma + 1) / (4 * gamma**2)
        d31 = (3 * gamma + 2) / (32 * gamma**3)
        d32 = (3 * gamma - 1) / (32 * gamma**3)

        # 符号因子：k 控制面内运动方向，delta 控制面外运动方向
        k = 1.0
        delta = -1.0

    else:
        # L2 的系数公式（gamma 为负值，导致分母和分子中的符号变化）
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

    # 频率修正系数：用于计算非线性效应对轨道周期的修正
    # 周期公式 T = 2π / (omega_p + kappa1*Au^2 + kappa2*Aw^2)
    l1 = -1.0 / (2 * gamma)
    l2 = (3 * gamma**2 + 3 * gamma + 1) / (4 * gamma**2)
    l3 = (3 * gamma**2 + 9 * gamma + 4) / (32 * gamma**3)

    kappa1 = (3 * gamma**2 + 3 * gamma + 1) / (4 * gamma**2)
    kappa2 = (3 * gamma**2 + 9 * gamma + 4) / (32 * gamma**3)

    return {
        "gamma": gamma,
        "omega_p": omega_p,
        "c1": c1,
        "c2": c2_c,
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
) -> tuple[npt.NDArray, npt.NDArray, float]:
    """计算Halo轨道三阶解析近似

    Args:
        mu: 质量比
        Au: U方向振幅
        Aw: W方向振幅
        phi: 相位偏移
        L: 拉格朗日点 (1=L1, 2=L2)
        tf: 终止时间
        N: 点数
        halo_class: 0=Class I (北), 1=Class II (南)

    Returns:
        SV_uvw: 状态向量序列 (N, 6)，[u, v, w, u_dot, v_dot, w_dot]
        t: 时间序列
        T: 周期

    Reference:
        Richardson, D. L. (1980). Analytic construction of periodic orbits
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
    L_position = 1 - mu - gamma  # L1: gamma>0, L2: gamma<0

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
        delta = -delta  # 轨道轨迹的 z 分量需要翻转 delta 以生成南族轨道

    omega_p = coeffs["omega_p"]

    # 周期公式：T = 2π/(omega_p + 频率修正)
    T = 2 * np.pi / (omega_p + kappa1 * Au**2 + kappa2 * Aw**2)
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

    # 使用 sin 参数化使 z(0)=0（轨道起始点在赤道面穿越处）
    w = delta * (
        Aw * np.sin(tau + phi)
        + d21 * Au * Aw * np.sin(2 * (tau + phi))
        + (d32 * Aw * Au**2 - d31 * Aw**3) * np.sin(3 * (tau + phi))
    )

    u_dot = Au * np.sin(tau + phi) + 2 * (a23 * Au**2 - a24 * Aw**2) * np.sin(2 * (tau + phi))
    v_dot = k * Au * np.cos(tau + phi) + 2 * (b21 * Au**2 - b22 * Aw**2) * np.cos(2 * (tau + phi))
    w_dot = delta * (
        Aw * np.cos(tau + phi)
        + 2 * d21 * Au * Aw * np.cos(2 * (tau + phi))
        + 3 * (d32 * Aw * Au**2 - d31 * Aw**3) * np.cos(3 * (tau + phi))
    )

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
) -> dict[str, float]:
    """计算Halo轨道初始猜测参数

    使用 Richardson 三阶近似系数生成初始猜测，配合微分修正器使用。
    初始状态位于 XZ 平面穿越点（y=0），赤道面穿越处（z=0）。

    Args:
        mu: 质量比
        z_amplitude: Z方向振幅
        L: 拉格朗日点 (1=L1, 2=L2)
        halo_class: 0=北Halo, 1=南Halo（当前不影响返回值，z 方向由调用方处理）

    Returns:
        包含初始猜测参数的字典:
        - x0: 初始x坐标
        - y0: 初始y坐标 (0)
        - z0: 初始z坐标 (0)
        - vx0: 初始vx (0)
        - vy0: 初始vy
        - vz0: 初始vz (0)
        - T_half: 半周期
        - Au: U方向振幅
        - Aw: W方向振幅
    """
    if z_amplitude <= 0:
        raise ValueError(f"z_amplitude必须为正数，当前为{z_amplitude}")

    coeffs = compute_halo_coefficients(mu, L)
    gamma = coeffs["gamma"]
    omega_p = coeffs["omega_p"]
    k = coeffs["k"]
    delta = coeffs["delta"]

    # 平动点位置
    L_position = 1 - mu - gamma  # L1: gamma>0, L2: gamma<0

    # 振幅关系：Au ∝ sqrt(Aw)（Richardson 三阶非线性耦合）
    Au = np.sqrt(z_amplitude) * 0.5
    Aw = z_amplitude

    # CR3BP z→-z 对称性：北族 (x0,0,+z0,0,vy0,0) 与南族 (x0,0,-z0,0,vy0,0)
    # 共享相同的 x0 和 vy0。不翻转 delta，z 方向由调用方通过 initial_z 处理。
    x0 = L_position + delta * z_amplitude * 0.05

    # vy0：基于频率和振幅的速度估计
    vy0 = k * Au * omega_p

    # 半周期（小振幅近似，微分修正器会进一步调整）
    T_half = np.pi / omega_p

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

