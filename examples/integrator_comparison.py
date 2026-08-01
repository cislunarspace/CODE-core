#!/usr/bin/env python3
"""
积分器对比示例

对比 PD45、PD78、RK89 三种 Runge-Kutta 方法在同一轨道传播上的精度和步数。
"""

import numpy as np

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.data.types.orbit import Orbit
from e2m2e.integrators import RkMethod, rk_step


def propagate_orbit(dynamics, orbit, method, tol):
    """用指定积分器和容差传播一个完整周期"""
    y0 = orbit.states[0].copy()
    T = orbit.period
    t, y, h = 0.0, y0.copy(), T / 100
    n_steps = 0

    while t < T:
        if t + h > T:
            h = T - t
        result = rk_step(method, t, y, h, tol, dynamics.equations_of_motion)
        y = np.asarray(result.y_new, dtype=float)
        t += h
        h = min(result.h_next, T - t)
        n_steps += 1

    # 周期性误差
    periodic_error = np.linalg.norm(y - y0)
    return y, periodic_error, n_steps


def main():
    """主函数"""
    print("=" * 60)
    print("积分器对比示例")
    print("=" * 60)

    # 1. 创建系统和参考轨道
    system = CR3BP_System(mu=0.0121506683, primary="Earth", secondary="Moon")._with_default_scales()
    system.set_characteristic_scales(384400, 27.32 * 86400)
    system.compute_libration_points()
    dynamics = CR3BP_Dynamics(system)

    # DRO 轨道
    from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection

    initial_state = [0.79188556619742, 0.0, 0.0, 0.0, 0.53682, 0.0]
    seed_orbit = Orbit(states=[initial_state], times=[0], system=system)
    seed_orbit.period = 3.0

    corrector = DifferentialCorrection(dynamics)
    corrector.setup_2D_symmetric_x_fixed_x0(x0=initial_state[0])
    orbit = corrector.iterate_correction(initial_guess=seed_orbit, verbose=False)

    if orbit is None:
        print("轨道修正失败")
        return

    print(f"\n参考轨道: T = {orbit.period:.6f}")

    # 2. 对比三种积分器
    methods = [
        ("PD45 (5阶)", RkMethod.PD45),
        ("PD78 (8阶)", RkMethod.PD78),
        ("RK89 (9阶)", RkMethod.RK89),
    ]
    tolerances = [1e-8, 1e-10, 1e-12]

    print(f"\n{'方法':>14s}  {'容差':>10s}  {'周期误差':>12s}  {'步数':>6s}")
    print("-" * 52)

    for name, method in methods:
        for tol in tolerances:
            _, error, steps = propagate_orbit(dynamics, orbit, method, tol)
            print(f"{name:>14s}  {tol:10.0e}  {error:12.2e}  {steps:6d}")

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
