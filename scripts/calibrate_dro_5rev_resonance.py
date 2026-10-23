"""标定月心 DRO 族的五周期共振点与分岔支起链（#820）。

复算 issue #820 报告的共振几何：沿平面 DRO 族跟踪非平凡复数乘子对的
辐角，二分定位辐角 72 度（2π/5）的成员，报告该处单值矩阵五次幂的乘子
偏差与共振特征向量，随后沿共振平面 deflection 起链并行走分岔支若干成
员。全部数值在库内定义性复算（闭合、Jacobi 漂移、单圈不闭合），不使
用文献振幅做断言。

用法（不要用会触发项目重建的 uv run）::

    .venv/bin/python scripts/calibrate_dro_5rev_resonance.py

与 issue 报告值的差异说明：#820 正文给出的 x 约 0.879、vy 约 0.4705、
临界乘子 1.0000214 出自报告人自己的 scipy 实现。库内标定得 x0 约
0.9007、vy0 约 0.4775；两者是相邻而不同的成员参数化。报告的
1.0000214 恰为平凡乘子对数值分裂（约 1±4.3e-6）的五次幂，不是共振
乘子本身，库内判据因此显式跟踪非平凡共振对（见
``e2m2e.algorithm.family.resonant_branch`` 模块说明）。
"""

from __future__ import annotations

import numpy as np

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family.orbits.walk import earth_moon_system
from e2m2e.algorithm.family.resonant_branch import (
    generate_resonant_branch_family,
    locate_dro_revolution_resonance,
    per_revolution_closure,
)

DAYS_PER_TU = 4.342479848294297


def main() -> None:
    dynamics = CR3BP_Dynamics(earth_moon_system())

    resonance = locate_dro_revolution_resonance(dynamics, revolutions=5)
    print("五周期共振成员（库内标定）")
    print(f"  x0    = {resonance.state[0]:.6f}")
    print(f"  vy0   = {resonance.state[4]:.6f}")
    print(f"  T     = {resonance.period:.6f} TU ({resonance.period * DAYS_PER_TU:.4f} d)")
    print(f"  共振乘子辐角 = {resonance.phase_deg:.4f} 度（目标 72）")
    print(f"  五次幂乘子与 1 的偏差 = {resonance.multiplier_gap:.3e}")
    print(f"  特征向量实部 = {np.round(np.real(resonance.eigenvector), 6)}")
    print(f"  特征向量虚部 = {np.round(np.imag(resonance.eigenvector), 6)}")

    result = generate_resonant_branch_family(
        resonance,
        n_orbits=6,
        step_size=0.005,
        initial_direction="increase-x0",
        dynamics=dynamics,
    )
    print(f"分岔支行走（{result.status}）：{result.message}")
    for orbit in result.family.orbits:
        state = orbit.states[0]
        single = per_revolution_closure(dynamics, orbit, 5)
        print(
            f"  x0={state[0]:.6f} y0={state[1]:.2e} vy0={state[4]:.6f}"
            f" T5={float(orbit.period):.4f} TU ({float(orbit.period) * DAYS_PER_TU:.2f} d)"
            f" closure={float(orbit.closure_error):.1e} 单圈偏差={single:.2e}"
        )

    reverse = generate_resonant_branch_family(
        resonance,
        n_orbits=6,
        step_size=0.005,
        initial_direction="decrease-x0",
        dynamics=dynamics,
    )
    print(f"反方向（{reverse.status}）：{reverse.message}，成员 {len(reverse.family.orbits)} 条")


if __name__ == "__main__":
    main()
