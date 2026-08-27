"""Lissajous 轨道一阶解析初猜（共线平动点 L1/L2/L3）。

Lissajous 是围绕共线点的**准周期** 轨道：面内运动（x-y）以频率 ω_xy、
面外运动（z）以频率 ω_z 独立振荡，两频率不可约（不共振）。Halo 轨道是
其面内/面外锁相（1:1 共振）的周期特例。

本模块通过 Rust 数值内核求解共线点和线性中心模态，统一处理 L1/L2/L3。
一阶初猜对小振幅（L1/L2 ≤7600 km）精度足够；更大振幅由下游星历修正
（多重打靶）精化。

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
from ...integrators import lissajous_bounded_trajectory_py
from ..dynamics import CR3BP_System


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
    characteristic_length = system.characteristic_length
    if characteristic_length is None:
        raise ValueError("CR3BP system 尚未设置特征长度")
    states, _, nominal_period = lissajous_bounded_trajectory_py(
        float(system.mu),
        collinear_point,
        float(characteristic_length),
        amplitude_in_km,
        amplitude_out_km,
        phase_in,
        phase_out,
        n_periods=1,
        points_per_period=2,
    )
    return np.asarray(states[0], dtype=float), float(nominal_period)


#: 中心流形约化阶数。
_LISSAJOUS_NF_ORDER = 6

#: 默认轨迹覆盖的名义周期数。
_LISSAJOUS_DEFAULT_PERIODS = 3

#: 轨迹每周期采样点数。
_LISSAJOUS_POINTS_PER_PERIOD = 60


def _rho_from_synodic(
    state_syn: npt.NDArray[np.floating], mu: float, libration_position: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """synodic 质心系状态转为 normal_form 的平动点相对状态。"""
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

    该单轨设计入口保留原有高阶 normal-form 语义；统一轨道族参数
    采样由独立 Rust family_generation 模块完成，不经过本函数。

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
        RuntimeError: 中心流形约化或参数化传播失败。
    """
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
    pipeline = NormalFormPipeline(
        context=ctx,
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
    states = np.array(rho_out, dtype=float)
    states[:, 0] = rho_out[:, 0] + libration_position[0] - mu
    return states, np.asarray(t_out, dtype=float), nominal_period
