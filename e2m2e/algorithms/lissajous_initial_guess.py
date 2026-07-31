"""Lissajous 轨道一阶解析初猜（共线平动点 L1/L2/L3）。

Lissajous 是围绕共线点的**准周期**轨道：面内运动（x-y）以频率 ω_xy、
面外运动（z）以频率 ω_z 独立振荡，两频率不可约（不共振）。Halo 轨道是
其面内/面外锁相（1:1 共振）的周期特例。

本模块用共线点线性化矩阵的特征值/特征向量构造一阶初猜，统一处理
L1/L2/L3（不依赖 Richardson 的 gamma 五次方程近似，直接用精确平动点
位置 + 线性化频率）。一阶初猜对小振幅（L1/L2 ≤7600 km）精度足够；
更大振幅由下游星历修正（多重打靶）精化。

面内/面外的区分**按特征向量方向**（z 占优=面外，x/y 占优=面内），不按
频率大小——共线点面内频率通常较大，但不可靠，必须看方向。

References:
    Gómez, G., et al. (2001). Dynamics and mission design near libration
    points, Vol. I, Sec. 7. Lissajous 作为双频率 Lindstedt-Poincaré 展开
    的一阶近似。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..core.cr3bp_system import CR3BP_System, LibrationPoint

#: 共线点编号 → LibrationPoint 枚举
_COLLINEAR = {1: LibrationPoint.L1, 2: LibrationPoint.L2, 3: LibrationPoint.L3}


def _linear_modes(
    system: CR3BP_System, collinear_point: int
) -> tuple[float, npt.NDArray[np.complexfloating], float, npt.NDArray[np.complexfloating], float]:
    """共线点线性化：返回 (ω_xy, v_xy, ω_z, v_z, x_L)。

    从 6×6 线性化矩阵 A 的特征分解提取两个纯虚振荡模态：
    - 面内模态（x/y 占优）：频率 ω_xy + 完整 6 维复特征向量 v_xy
    - 面外模态（z 占优）：频率 ω_z + 完整 6 维复特征向量 v_z

    特征向量含位置+速度分量，线性化轨道 = Re(v · e^{iωt}) 给出完整的
    6 维状态演化（位置和速度都正确），无需手动求导。

    x_L 是共线点 x 坐标（位置锚点）。
    """
    if collinear_point not in _COLLINEAR:
        raise ValueError(f"collinear_point 必须为 1/2/3，当前 {collinear_point}")
    ret = system.compute_stability_index(_COLLINEAR[collinear_point])
    eigvals = np.asarray(ret["eigenvalues"], dtype=complex)
    A = np.asarray(ret["linear_matrix"], dtype=float)
    _, vecs = np.linalg.eig(A)
    point = system.get_libration_point(_COLLINEAR[collinear_point])
    x_L = float(point[0])

    # 两对纯虚特征值（±iω），按特征向量方向分面内/面外
    v_xy: npt.NDArray[np.complexfloating] | None = None
    v_z: npt.NDArray[np.complexfloating] | None = None
    omega_xy = 0.0
    omega_z = 0.0
    seen: set[int] = set()
    for i, ev in enumerate(eigvals):
        im = abs(np.imag(ev))
        if im < 1e-9 or abs(np.real(ev)) > 1e-9:
            continue
        key = round(im, 6)
        if key in seen:  # ±iω 共轭对只取一个
            continue
        seen.add(key)
        pos = vecs[:3, i]  # 复位置部分
        if abs(pos[2]) > np.linalg.norm(pos[:2]) + 1e-12:
            v_z = vecs[:, i]
            omega_z = im
        else:
            v_xy = vecs[:, i]
            omega_xy = im

    if v_xy is None or v_z is None:
        raise RuntimeError(f"共线点 L{collinear_point} 未找到面内/面外两个模态，特征值={eigvals}")
    return omega_xy, v_xy, omega_z, v_z, x_L


def compute_lissajous_initial_guess(
    system: CR3BP_System,
    collinear_point: int,
    amplitude_in_km: float,
    amplitude_out_km: float,
    phase_in: float,
    phase_out: float,
) -> tuple[npt.NDArray[np.floating], float]:
    """构造 Lissajous 轨道一阶初猜状态。

    在共线点邻域，面内运动为椭圆（频率 ω_xy），面外为正弦（频率 ω_z）：

    .. math::
        \\vec x(t) = \\vec x_L
            + \\alpha\\, \\mathrm{Re}[\\vec v_{xy}\\, e^{i(\\omega_{xy} t + \\phi)}]
            + \\beta\\, \\mathrm{Re}[\\vec v_z\\, e^{i(\\omega_z t + \\psi)}]

    其中 α、β 为面内/面外振幅（无量纲），φ、ψ 为相位，v_xy/v_z 为完整 6 维
    复特征向量（含位置+速度，保证位置与速度的相位关系自洽）。

    Args:
        system: CR3BP 系统（含已计算的平动点）。
        collinear_point: 共线点编号 1/2/3。
        amplitude_in_km: 面内振幅（km，物理单位）。
        amplitude_out_km: 面外振幅（km，物理单位）。
        phase_in: 面内初始相位（0~1，映射到 0~2π）。
        phase_out: 面外初始相位（0~1，映射到 0~2π）。

    Returns:
        (state0, nominal_period)：t=0 的 6 维状态（无量纲 synodic），
        与面内标称周期 2π/ω_xy（作 patch points 采样基准）。
    """
    if not system.has_L_points:
        system.compute_libration_points()

    omega_xy, v_xy, omega_z, v_z, x_L = _linear_modes(system, collinear_point)
    l_c = system.characteristic_length

    # 振幅归一化：α、β 是位置振幅，但特征向量 v 的位置部分模长不一定是 1。
    # 把振幅解释为"面内/面外位置偏移的最大 km"，则 α = (km/l_c) / |v_pos|。
    alpha = (amplitude_in_km / l_c) / np.linalg.norm(v_xy[:3])
    beta = (amplitude_out_km / l_c) / np.linalg.norm(v_z[:3])
    phi = float(phase_in) * 2.0 * np.pi
    psi = float(phase_out) * 2.0 * np.pi

    # t=0：Re[v · e^{iφ}] = Re[v]·cosφ − Im[v]·sinφ（完整 6 维，位置+速度）
    def _mode_contrib(
        v: npt.NDArray[np.complexfloating], amp: float, phase: float
    ) -> npt.NDArray[np.floating]:
        return amp * (np.real(v) * np.cos(phase) - np.imag(v) * np.sin(phase))

    state0 = np.zeros(6)
    state0[0] = x_L  # 共线点锚点
    state0 += _mode_contrib(v_xy, alpha, phi)
    state0 += _mode_contrib(v_z, beta, psi)

    nominal_period = 2.0 * np.pi / omega_xy  # 面内周期（TU）
    return state0, nominal_period
