"""Lissajous 拟周期族：共线点邻域的中心流形有界轨迹。

面内/面外频率不可约，不做周期闭合；返回中心流形约化流上的有界
多点轨迹，供下游可视化与 patch-point 采样直接使用。
"""

from __future__ import annotations

import warnings

import numpy as np

from ....data.types.orbit import Orbit
from ...dynamics import CR3BP_Dynamics
from ...results import FamilyGenerationResult
from ..lissajous_initial_guess import (
    compute_lissajous_bounded_trajectory,
    compute_lissajous_initial_guess,
)
from .walk import _generate_rust_family, earth_moon_system


def design_lissajous(
    collinear_point: int,
    amplitude_in_km: float,
    amplitude_out_km: float,
    phase_in: float,
    phase_out: float,
    *,
    dynamics: CR3BP_Dynamics | None = None,
    n_periods: int = 3,
) -> Orbit:
    """生成指定共线点（L1/L2/L3）的 Lissajous 拟周期轨道。

    Lissajous 面内/面外频率不可约，是准周期轨道，不做周期闭合。一阶线性
    初猜在非线性 CR3BP 下必发散（不稳定方向泄漏，见
    :func:`compute_lissajous_bounded_trajectory`），故本函数返回**中心流形
    约化流的有界多点轨迹**（覆盖 ``n_periods`` 个名义周期），供下游可视化
    与 patch-point 采样直接使用——原生 CR3BP 重传播会重新激发不稳定方向
    而发散，下游不得用 ``states[0]`` 重传播。

    ``period`` 取面内名义周期 2π/ω_xy；``states[0]`` 为历元参考状态。

    Args:
        collinear_point: 共线点编号 1/2/3。
        amplitude_in_km / amplitude_out_km: 面内/面外振幅（km）。
        phase_in / phase_out: 面内/面外初始相位（0~1）。
        dynamics: CR3BP 动力学；缺省构造标准地月系统。
        n_periods: 返回轨迹覆盖的名义周期数（默认 3）。

    Returns:
        含多点有界轨迹的 :class:`Orbit`。中心流形约化失败时退回一阶线性
        单点初猜并发出 :class:`RuntimeWarning` （保下游不崩，但失去有界性）。
    """
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())

    try:
        states, times, nominal_period = compute_lissajous_bounded_trajectory(
            dynamics.system,
            collinear_point,
            amplitude_in_km,
            amplitude_out_km,
            phase_in,
            phase_out,
            n_periods=n_periods,
        )
    except RuntimeError as exc:
        # 非典型参数下中心流形约化可能失败：退回线性单点初猜，明确告警。
        # 下游此时不应依赖有界性（patch-point 采样仍会发散）。
        warnings.warn(
            f"中心流形约化失败，退回一阶线性 Lissajous 初猜（无有界性保证）：{exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        state0, nominal_period = compute_lissajous_initial_guess(
            dynamics.system,
            collinear_point,
            amplitude_in_km,
            amplitude_out_km,
            phase_in,
            phase_out,
        )
        states = state0.reshape(1, -1)
        times = np.array([0.0])

    orbit = Orbit(states=states, times=times, system=dynamics.system)
    orbit.period = nominal_period
    orbit.family_type = "lissajous"
    orbit.parameters = {
        "collinear_point": collinear_point,
        "amplitude_in_km": amplitude_in_km,
        "amplitude_out_km": amplitude_out_km,
    }
    return orbit


def design_lissajous_family(
    libration_point: int,
    amplitude_in_km: float,
    amplitude_out_km: float,
    phase_in: float,
    phase_out: float,
    *,
    n_orbits: int = 50,
    sampling_mode: str = "linear-amplitudes",
    dynamics: CR3BP_Dynamics | None = None,
    n_periods: int = 3,
) -> FamilyGenerationResult:
    """生成 Lissajous 拟周期轨迹采样族。

    Lissajous 面内/面外频率不可约，不是周期族；族生成是参数采样而
    非延拓：面内/面外振幅从请求值的 1/n 到 1 线性插值取
    ``n_orbits`` 个样本，两相位固定。每个成员是 Rust 非线性中心约化流上的
    有界多点轨迹（语义同 ``design_lissajous``），无周期闭合，不得
    按严格周期族消费——族上显式标注
    ``metadata["periodicity"] = "quasi-periodic"``。

    Args:
        libration_point: 共线点编号（1/2/3）。
        amplitude_in_km / amplitude_out_km: 面内/面外振幅上限（km）。
        phase_in / phase_out: 面内/面外初始相位（0~1，采样中固定）。
        n_orbits: 采样成员数。
        dynamics: CR3BP 动力学；缺省构造标准地月系统。
        n_periods: 每条轨迹覆盖的名义周期数（默认 3）。

    Returns:
        :class:`FamilyGenerationResult`；``family`` 是拟周期成员组成的
        ``OrbitFamily``（``family_type="lissajous"``，
        ``is_quasi_periodic`` 为真）。
    """
    if libration_point not in (1, 2, 3):
        raise ValueError(f"libration_point 必须为 1、2 或 3，当前为 {libration_point}")
    if n_orbits < 1:
        raise ValueError(f"n_orbits 必须大于 0，当前为 {n_orbits}")
    if n_periods < 1:
        raise ValueError(f"n_periods 必须大于 0，当前为 {n_periods}")
    if amplitude_in_km <= 0.0 or amplitude_out_km <= 0.0:
        raise ValueError("Lissajous 振幅必须为正数")
    if not 0.0 <= phase_in <= 1.0 or not 0.0 <= phase_out <= 1.0:
        raise ValueError("Lissajous 相位必须在 [0, 1] 内")
    if sampling_mode != "linear-amplitudes":
        raise ValueError("Lissajous sampling_mode 仅支持 'linear-amplitudes'")
    return _generate_rust_family(
        "lissajous",
        libration_point,
        n_orbits,
        dynamics,
        amplitude_in_km=amplitude_in_km,
        amplitude_out_km=amplitude_out_km,
        phase_in=phase_in,
        phase_out=phase_out,
        n_periods=n_periods,
    )
