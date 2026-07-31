"""L4/L5 三角平动点邻域的一阶解析初猜。

L4/L5 在纯 CR3BP 下是椭圆型平衡点（地月 μ=0.0122 < μ_Routh≈0.0385），
线性化矩阵只有纯虚特征值，没有双曲方向。邻域运动是三模态拟周期叠加：

- 面内两个模态：短周期（频率 ω_s，高频）与长周期（频率 ω_l，低频）；
- 面外一个模态：垂直方向独立振动（频率 ω_v）。

两频率满足特征方程 ω⁴ − ω² + 27μ(1−μ)/4 = 0（无量纲），比值不可约，
因此 L4/L5 邻域本质是拟周期轨道，不强求周期闭合。本模块用线性化特征
向量构造一阶初猜，面内振幅默认均分给短/长两模态，后续按 DFH golden
样本标定拆分比例。初猜直接作 patch points 采样基准，交给下游星历
多重打靶精化。

References:
    Gómez, G., et al. (2001). Dynamics and mission design near libration
    points, Vol. I. 三角点邻域的线性化模态分解。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..dynamics import CR3BP_System, LibrationPoint

#: L4/L5 编号 → LibrationPoint 枚举
_TRIANGULAR = {4: LibrationPoint.L4, 5: LibrationPoint.L5}


def _triangular_modes(
    system: CR3BP_System, point: int
) -> tuple[
    float,
    npt.NDArray[np.complexfloating],
    float,
    npt.NDArray[np.complexfloating],
    float,
    npt.NDArray[np.complexfloating],
    npt.NDArray[np.floating],
]:
    """L4/L5 线性化：返回 (ω_s, v_s, ω_l, v_l, ω_v, v_z, x_L)。

    对 6×6 线性化矩阵 A 做特征分解，按特征向量方向识别三个纯虚模态：
    - 垂直模态：位置部分 z 占优（判据同共线点 Lissajous 模块）；
    - 面内两模态：其余两个纯虚特征向量，按频率大小分短周期（高）与
      长周期（低）。
    锚点 ``x_L`` 是 L4/L5 的完整 3 维位置（等边三角形顶点）。
    """
    if point not in _TRIANGULAR:
        raise ValueError(f"point 必须为 4（L4）或 5（L5），当前 {point}")
    lp = _TRIANGULAR[point]
    if not system.has_L_points:
        system.compute_libration_points()
    ret = system.compute_stability_index(lp)
    A = np.asarray(ret["linear_matrix"], dtype=float)
    # 用 np.linalg.eig 一次拿特征值与特征向量，保证二者顺序一致
    eigvals, vecs = np.linalg.eig(A)
    x_L = np.asarray(system.get_libration_point(lp), dtype=float)

    v_s: npt.NDArray[np.complexfloating] | None = None
    v_l: npt.NDArray[np.complexfloating] | None = None
    v_z: npt.NDArray[np.complexfloating] | None = None
    omega_s = omega_l = omega_v = 0.0
    seen: set[int] = set()
    for i, ev in enumerate(eigvals):
        im = abs(np.imag(ev))
        if im < 1e-9 or abs(np.real(ev)) > 1e-9:
            # 椭圆点不应有实部非零的特征值；若有（星历模型下）则跳过
            continue
        key = round(im, 6)
        if key in seen:  # ±iω 共轭对只取一个
            continue
        seen.add(key)
        pos = vecs[:3, i]
        if abs(pos[2]) > np.linalg.norm(pos[:2]) + 1e-12:
            v_z = vecs[:, i]
            omega_v = im
        elif v_s is None or im > omega_s:
            v_s = vecs[:, i]
            omega_s = im
        else:
            v_l = vecs[:, i]
            omega_l = im

    if v_s is None or v_l is None or v_z is None:
        raise RuntimeError(
            f"L{point} 未找到三个纯虚模态（短/长周期 + 垂直），特征值 {eigvals}"
        )
    return omega_s, v_s, omega_l, v_l, omega_v, v_z, x_L


def compute_triangular_initial_guess(
    system: CR3BP_System,
    point: int,
    amplitude_in_km: float,
    amplitude_out_km: float,
    phase_in: float,
    phase_out: float,
) -> tuple[npt.NDArray[np.floating], float]:
    """构造 L4/L5 邻域一阶初猜状态。

    .. math::
        \\vec x(t) = \\vec x_L
            + \\alpha_s\\, \\mathrm{Re}[\\vec v_s\\, e^{i(\\omega_s t + \\phi)}]
            + \\alpha_l\\, \\mathrm{Re}[\\vec v_l\\, e^{i(\\omega_l t + \\phi)}]
            + \\beta\\, \\mathrm{Re}[\\vec v_z\\, e^{i(\\omega_v t + \\psi)}]

    面内振幅 ``amplitude_in_km`` 默认均分给短/长两模态（按 golden 标定
    前的约定）；面外振幅给垂直模态。相位 ``phase_in`` 同时作用于两个面内
    模态，``phase_out`` 作用于垂直模态。

    Args:
        system: CR3BP 系统（含已计算平动点）。
        point: 4（L4）或 5（L5）。
        amplitude_in_km: 面内振幅（km）。
        amplitude_out_km: 面外振幅（km）。
        phase_in: 面内初始相位（0~1，映射到 0~2π）。
        phase_out: 面外初始相位（0~1，映射到 0~2π）。

    Returns:
        (state0, nominal_period)：t=0 的 6 维状态（无量纲 synodic）与
        面内短周期名义周期 2π/ω_s（作 patch points 采样基准）。
    """
    omega_s, v_s, omega_l, v_l, _omega_v, v_z, x_L = _triangular_modes(system, point)
    l_c = system.characteristic_length

    # 振幅归一化：位置偏移最大 km → 无量纲系数
    alpha_s = (0.5 * amplitude_in_km / l_c) / np.linalg.norm(v_s[:3])
    alpha_l = (0.5 * amplitude_in_km / l_c) / np.linalg.norm(v_l[:3])
    beta = (amplitude_out_km / l_c) / np.linalg.norm(v_z[:3])
    phi = float(phase_in) * 2.0 * np.pi
    psi = float(phase_out) * 2.0 * np.pi

    def _mode_contrib(
        v: npt.NDArray[np.complexfloating], amp: float, phase: float
    ) -> npt.NDArray[np.floating]:
        return amp * (np.real(v) * np.cos(phase) - np.imag(v) * np.sin(phase))

    state0 = np.zeros(6)
    state0[:3] = x_L
    state0 += _mode_contrib(v_s, alpha_s, phi)
    state0 += _mode_contrib(v_l, alpha_l, phi)
    state0 += _mode_contrib(v_z, beta, psi)

    nominal_period = 2.0 * np.pi / omega_s
    return state0, nominal_period

