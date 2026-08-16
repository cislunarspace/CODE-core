"""Lissajous 轨道一阶解析初猜（共线平动点 L1/L2/L3）。

Lissajous 是围绕共线点的**准周期** 轨道：面内运动（x-y）以频率 ω_xy、
面外运动（z）以频率 ω_z 独立振荡，两频率不可约（不共振）。Halo 轨道是
其面内/面外锁相（1:1 共振）的周期特例。

本模块用共线点线性化矩阵的特征值/特征向量构造一阶初猜，统一处理
L1/L2/L3（不依赖 Richardson 的 gamma 五次方程近似，直接用精确平动点
位置 + 线性化频率）。一阶初猜对小振幅（L1/L2 ≤7600 km）精度足够；
更大振幅由下游星历修正（多重打靶）精化。

面内/面外的区分**按特征向量方向** （z 占优=面外，x/y 占优=面内），不按
频率大小——共线点面内频率通常较大，但不可靠，必须看方向。

References:
    Gómez, G., et al. (2001). Dynamics and mission design near libration
    points, Vol. I, Sec. 7. Lissajous 作为双频率 Lindstedt-Poincaré 展开
    的一阶近似。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ...data.templates import ConvergenceState
from ...data.templates.enums import LibrationPoint
from ..dynamics import CR3BP_System

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
    # 调用约定保证系统已初始化特征尺度（模块内只算平动点不设尺度）
    assert l_c is not None

    # 振幅归一化：α、β 是位置振幅，但特征向量 v 的位置部分模长不一定是 1。
    # 把振幅解释为"面内/面外位置偏移的最大 km"，则 α = (km/l_c) / |v_pos|。
    alpha = float((amplitude_in_km / l_c) / np.linalg.norm(v_xy[:3]))
    beta = float((amplitude_out_km / l_c) / np.linalg.norm(v_z[:3]))
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


# =============================================================================
# 非线性 CR3BP 下的有界 Lissajous 轨迹（中心流形约化）
# =============================================================================
#
# 一阶线性初猜精确落在线性中心流形上（不稳定/稳定分量为机器零），但在
# **含不稳定方向的完整 CR3BP** 里传播必然发散：非线性中心流形在 O(α²) 处
# 弯曲，残余的双曲-中心耦合经 e^(λt)（L2 λ≈2.16/TU）放大，3 周期放大
# ~3e9，面内偏移长到百万 km。任何有限阶近似都逃不开这条不稳定方向。
#
# 唯一可靠的有界轨迹来源是 **约化流**：用辛 Lie 变换消去 Hamiltonian 里
# 全部 k₁≠k₂ 的双曲-中心耦合项（Jorba-Masdemont 1999 §2；Gómez 2001
# Vol.III），在双曲方向冻结（q₁=p₁）的 4D 中心流形上积分约化 Hamiltonian
# （``propagate_parametric``），逐时刻转回物理坐标。这给出真正有界的准周期
# Lissajous 轨迹（面内 ~2× 振幅），代价是每个 (μ, 平动点, 阶数) 跑一次
# 中心流形约化 + 一次参数化传播。

#: 中心流形约化阶数。order 5 已足以消去主导双曲耦合、压住发散（实测面内
#: 偏移 ~2× 振幅）；6 留余量。``reduce`` 耗时 ~O(order²)：5≈0.5s, 6≈0.7s,
#: 8≈1.4s, 12≈20s。
_LISSAJOUS_NF_ORDER = 6

#: 默认轨迹覆盖的名义周期数（"初猜有效时段"）。足够 two_level 默认 1 圈
#: patch-point 采样 + 下游可视化；segmented 长弧由 ``design_orbit`` 按需
#: 重建更长轨迹。
_LISSAJOUS_DEFAULT_PERIODS = 3

#: 轨迹每周期采样点数（可视化平滑 + patch-point 插值精度）。
_LISSAJOUS_POINTS_PER_PERIOD = 60


def _rho_from_synodic(
    state_syn: npt.NDArray[np.floating], mu: float, libration_position: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """synodic 质心系状态 → normal_form 的 rho 状态（相对平动点）。

    normal_form 的 ``libration_position`` 取地心锚定约定（L2 = 1+γ），故
    rho[0] = (质心 x + μ) − libration_position[0] = 质心 x − 平动点(质心)；
    y/z 与速度是平移不变量，直接复用。
    """
    rho = np.array(state_syn, dtype=float)
    rho[0] = (float(state_syn[0]) + mu) - float(libration_position[0])
    return rho


def compute_lissajous_bounded_trajectory(
    system: CR3BP_System,
    collinear_point: int,
    amplitude_in_km: float,
    amplitude_out_km: float,
    phase_in: float,
    phase_out: float,
    *,
    n_periods: int = _LISSAJOUS_DEFAULT_PERIODS,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], float]:
    """生成非线性 CR3BP 下有界的 Lissajous 轨迹（中心流形约化流）。

    流程：一阶线性初猜（``compute_lissajous_initial_guess``）→ 转 rho →
    中心流形约化（``NormalFormPipeline.reduce``，消去双曲-中心耦合）→
    在约化 Hamiltonian 上传播（``propagate_parametric``，双曲方向冻结）→
    逐点转回 synodic 质心系。产物在完整 CR3BP 下**保持有界** （面内 ~2×
    振幅），供 ``design_lissajous`` 返回多点轨迹、供下游 patch-point 采样
    与可视化。

    约化流是任何有限阶近似里**唯一** 绕开不稳定方向的有界轨迹来源：把初猜
    状态或其中心流形投影直接喂回原生 CR3BP 传播，残余 O(αⁿ⁺¹) 双曲耦合
    仍经 e^(λt) 放大而发散（实测 3 周期偏移 ~70 万 km）。

    Args:
        system: CR3BP 系统（含已计算的平动点）。
        collinear_point: 共线点编号 1/2/3。
        amplitude_in_km: 面内振幅（km）。
        amplitude_out_km: 面外振幅（km）。
        phase_in / phase_out: 面内/面外初始相位（0~1）。
        n_periods: 轨迹覆盖的名义周期数。

    Returns:
        (states, times, nominal_period)：``states`` 形状 (M, 6) synodic 质心
        系无量纲状态，``times`` 形状 (M,) 对应无量纲时间（TU = T/(2π)，与
        :class:`CR3BP_Dynamics` 一致），``nominal_period`` 面内名义周期。

    Raises:
        RuntimeError: 中心流形约化或参数化传播失败（非典型参数下可能发生）。
    """
    # 惰性导入：normal_form 依赖较重，仅在真正生成有界轨迹时引入。
    from ..normal_form.constants import JD0_J2000
    from ..normal_form.context import NormalFormContext
    from ..normal_form.pipeline import NormalFormPipeline
    from ..normal_form.propagation import propagate_parametric

    if not system.has_L_points:
        system.compute_libration_points()

    state0, nominal_period = compute_lissajous_initial_guess(
        system, collinear_point, amplitude_in_km, amplitude_out_km, phase_in, phase_out
    )

    ctx = NormalFormContext(
        system=system,
        libration_point=LibrationPoint(collinear_point),
        epoch=JD0_J2000,
        order=_LISSAJOUS_NF_ORDER,
        force_cr3bp=True,
    )
    libration_position = np.asarray(ctx.libration_position, dtype=float)
    mu = float(system.mu)

    rho0 = _rho_from_synodic(state0, mu, libration_position)

    # fast 流水线参数（与 tests/algorithm/normal_form 一致）。CR3BP 模式由
    # ctx.force_cr3bp 声明：整条约化路径（DS rhs、Bdot2A 的 C_pq、rho↔EM 旋转
    # 矩阵）一律直接用 CR3BP 常量、不探 SPICE 星历——design_orbit 会全局加载
    # SPICE 内核，否则星历几何进入约化会使 quasi-Floquet↔CM Lie ODE 失稳。
    pipeline = NormalFormPipeline(
        context=ctx,
        # CR3BP 模型（force_cr3bp=True）下 M(t) 常数矩阵，QF 显式选 constant
        # 方法（ADR 0020 决策 4：显式选择，不静默降）。
        quasi_floquet_method="constant",
        center_max_order=_LISSAJOUS_NF_ORDER,
        center_steps=("invariant", "center"),
        dynamical_kwargs={
            "t_total": 4.0,
            "node_step": 0.8,
            "dense_step": 0.2,
            "max_iter": 3,
            "tolerance": 1e-6,
            "prefer": "fft",
        },
    )
    nf_result = pipeline.reduce(rho0)
    if nf_result.status is not ConvergenceState.CONVERGED or nf_result.catalog_transformer is None:
        raise RuntimeError(f"共线点 L{collinear_point} 中心流形约化失败：{nf_result.message}")

    n_points = max(_LISSAJOUS_POINTS_PER_PERIOD * n_periods, 30)
    t_span = np.linspace(0.0, n_periods * nominal_period, n_points)
    t_out, rho_out, _ = propagate_parametric(rho0, t_span, nf_result, ctx)
    if rho_out.shape[0] < 2:
        raise RuntimeError(f"共线点 L{collinear_point} Lissajous 参数化传播失败（积分未产出轨迹）")

    # 逐点 rho → synodic 质心系（仅 x 分量平移，y/z/速度不变）。
    states = np.array(rho_out, dtype=float)
    states[:, 0] = rho_out[:, 0] + libration_position[0] - mu
    return states, np.asarray(t_out, dtype=float), nominal_period
