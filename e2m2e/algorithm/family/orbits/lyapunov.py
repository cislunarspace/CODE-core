"""Lyapunov 族：共线点邻域的平面周期轨道。

以 ``max|y|``（km）为振幅定义。小振幅区用固定 ``x₀`` 和独立线性
模态种子匹配；越过种子振幅后改用固定半周期自然延拓，避免同一
``x₀`` 上的 Lyapunov/DRO 多解跳支。
"""

from __future__ import annotations

import numpy as np

from ....data.types.orbit import Orbit
from ....integrators import collinear_center_modes_py
from ...dynamics import CR3BP_Dynamics
from ..axial_initial_guess import (
    _correct_lyapunov_fixed_t,
    _correct_lyapunov_fixed_x0,
)
from .walk import Cr3bpOrbitError, _walk_family, _y_amplitude_max, earth_moon_system

_LYAPUNOV_SEED_DX = 0.01


def design_lyapunov(
    collinear_point: int,
    amplitude_km: float,
    *,
    dynamics: CR3BP_Dynamics | None = None,
    tol_km: float = 20.0,
) -> Orbit:
    """生成指定共线点和面内振幅的平面 Lyapunov 周期轨道。

    振幅定义为一个周期内 ``max|y|``（km）。支持 L1/L2。小振幅区用固定
    ``x₀`` 和独立线性模态种子匹配；越过种子振幅后改用固定半周期自然延拓，
    避免同一 ``x₀`` 上的 Lyapunov/DRO 多解跳支。
    """
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())
    system = dynamics.system
    du = system.characteristic_length
    assert du is not None
    target_du = amplitude_km / du

    x_l, omega_xy, _, y_ratio = collinear_center_modes_py(float(system.mu), collinear_point)
    t_lin = 2.0 * np.pi / omega_xy
    vy_factor = -omega_xy * y_ratio
    seed_x0 = x_l - _LYAPUNOV_SEED_DX
    seed = _correct_lyapunov_fixed_x0(
        dynamics,
        seed_x0,
        None,
        x_L=x_l,
        x_factor=1.0,
        vy_factor=vy_factor,
        T_lin=t_lin,
    )
    if seed is None:
        raise Cr3bpOrbitError(f"L{collinear_point} Lyapunov 种子修正失败 (x0={seed_x0:.4f})")
    amp_seed = _y_amplitude_max(dynamics, seed)

    if target_du <= amp_seed:

        def correct_at_x0(x0: float, _guess: Orbit | None) -> Orbit:
            orbit = _correct_lyapunov_fixed_x0(
                dynamics,
                x0,
                None,
                x_L=x_l,
                x_factor=1.0,
                vy_factor=vy_factor,
                T_lin=t_lin,
            )
            if orbit is None:
                raise Cr3bpOrbitError(f"Lyapunov(L{collinear_point}, x0={x0:.6f}) 修正未收敛")
            return orbit

        return _walk_family(
            correct_at=correct_at_x0,
            measure=lambda orbit: _y_amplitude_max(dynamics, orbit),
            target=target_du,
            p_seed=seed_x0,
            dp_init=0.002,
            max_step=0.004,
            tol=tol_km / du,
            seed_orbit=seed,
        )

    assert seed.period is not None

    def correct_at_period(period: float, guess: Orbit | None) -> Orbit:
        assert guess is not None
        orbit = _correct_lyapunov_fixed_t(dynamics, period / 2.0, guess)
        if orbit is None:
            raise Cr3bpOrbitError(f"Lyapunov(L{collinear_point}, T={period:.6f}) 修正未收敛")
        if abs(float(orbit.states[0, 4])) < 1e-9:
            raise Cr3bpOrbitError(
                f"Lyapunov(L{collinear_point}, T={period:.6f}) 收敛到平动点平凡解"
            )
        jacobi_jump = abs(
            float(system.get_jacobi_constant(orbit.states[0]))
            - float(system.get_jacobi_constant(guess.states[0]))
        )
        if jacobi_jump > 0.02:
            raise Cr3bpOrbitError(f"Lyapunov(L{collinear_point}, T={period:.6f}) 修正跳支")
        return orbit

    return _walk_family(
        correct_at=correct_at_period,
        measure=lambda orbit: _y_amplitude_max(dynamics, orbit),
        target=target_du,
        p_seed=float(seed.period),
        dp_init=0.005,
        max_step=0.005,
        tol=tol_km / du,
        seed_orbit=seed,
        max_iter=200,
    )
