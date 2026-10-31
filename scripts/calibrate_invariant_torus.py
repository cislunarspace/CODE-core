"""不变环面构造的振幅包络标定（#842 第一阶段）。

人工诊断脚本，不进默认测试套件。直接使用 .venv/bin/python 运行：

    .venv/bin/python scripts/calibrate_invariant_torus.py

内容：
1. L2 Halo（面外振幅 3000 km，单中心对形态）按振幅与网格匹配的口径
   复算定义级残差：17 点网格到 1.1e-2，33 点网格到论文法向振幅 0.031；
2. 平面 DRO 种子（双中心对形态）探针：未规范的面内振幅方向被丢弃后
   ρ 不再漂移，但振幅塌缩与带边垃圾仍在，作为第二阶段的记录证据。

论文对照值（第二周期 36.5 天与 58.3 天、法向振幅 0.031）属结果性数值
（ADR 0055），只作人工对照，不进断言；本脚本的退出码只反映构造本身的
收敛状态。
"""

from __future__ import annotations

import sys
import time

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family import design_halo
from e2m2e.algorithm.family.orbits.dro import _correct_dro
from e2m2e.algorithm.family.orbits.walk import earth_moon_system
from e2m2e.algorithm.torus import construct_invariant_torus


def _run(label: str, orbit, dynamics: CR3BP_Dynamics, amplitude: float, n_points: int) -> bool:
    started = time.perf_counter()
    result = construct_invariant_torus(orbit, amplitude, n_points=n_points, dynamics=dynamics)
    elapsed = time.perf_counter() - started
    print(
        f"{label}: A={amplitude:g} n={n_points} -> {result.status.value} "
        f"inv={result.invariance_residual:.2e} conj={result.conjugacy_residual:.2e} "
        f"amp={result.amplitude:.6f} rho={result.rotation_number:.6f} "
        f"T2={result.second_period_days:.1f}d {elapsed:.0f}s"
    )
    if result.status.value != "converged":
        print(f"  {result.message}")
    return result.status.value == "converged"


def main() -> int:
    dynamics = CR3BP_Dynamics(earth_moon_system())
    halo = design_halo(2, 3000.0, dynamics=dynamics)
    print(f"halo period={float(halo.period):.6f} TU")
    ok = True
    for amplitude in (1e-3, 1.1e-2):
        ok &= _run("halo", halo, dynamics, amplitude, 17)
    ok &= _run("halo", halo, dynamics, 0.031, 33)

    dro = _correct_dro(dynamics, 0.79188556619742, None)
    print(f"dro period={float(dro.period):.6f} TU（双中心对形态，见 ADR 0063 决策 8）")
    _run("dro", dro, dynamics, 1e-4, 17)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
