#!/usr/bin/env python3
"""测量低推力解析雅可比相对数值差分的求解耗时。

用法:
    uv run python scripts/benchmark_lowthrust_analytic_jacobian.py
"""

from __future__ import annotations

from time import perf_counter
from types import SimpleNamespace

import numpy as np

from e2m2e.algorithm.forces import PointMassGravity
from e2m2e.algorithm.transfer import EngineConfig, LowThrustShooting


def make_shooter() -> LowThrustShooting:
    """构造不依赖 SPICE 的纯二体低推力打靶问题。"""
    mu = 398600.435507
    radius = 7000.0
    velocity = np.sqrt(mu / radius)
    initial_state = np.array([radius, 0.0, 0.0, 0.0, velocity, 0.0])
    return LowThrustShooting(
        SimpleNamespace(origin="EARTH"),
        [PointMassGravity("EARTH", mu=mu)],
        EngineConfig(t_max=0.5, isp=3000.0),
        initial_state,
        initial_mass=1000.0,
        target_state=initial_state.copy(),
        t0=0.0,
        tf=1200.0,
    )


def solve_with(shooter: LowThrustShooting, use_analytic_jac: bool) -> tuple[float, float]:
    """运行固定问题，返回耗时与燃料消耗。"""
    segments = 4
    initial_guess = shooter._default_x0(segments)
    initial_guess[0::3] = 0.5
    start = perf_counter()
    solution = shooter.solve(
        segments,
        x0=initial_guess,
        use_analytic_jac=use_analytic_jac,
        maxiter=30,
    )
    return perf_counter() - start, solution.fuel_consumed


def main() -> None:
    """输出两种雅可比路径的耗时、加速比和解差异。"""
    analytic_time, analytic_fuel = solve_with(make_shooter(), use_analytic_jac=True)
    numeric_time, numeric_fuel = solve_with(make_shooter(), use_analytic_jac=False)

    print(f"解析雅可比: {analytic_time:.3f} s, 燃料消耗 {analytic_fuel:.6f}")
    print(f"数值差分: {numeric_time:.3f} s, 燃料消耗 {numeric_fuel:.6f}")
    print(f"加速比: {numeric_time / analytic_time:.2f}x")
    print(f"燃料差: {abs(analytic_fuel - numeric_fuel):.6e}")


if __name__ == "__main__":
    main()
