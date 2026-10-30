"""L2 南 Halo 族乘子过 +1 的临界成员定位与 butterfly 分岔支标定（#825）。

沿 L2 南 Halo 族逐成员计算单值矩阵，按特征向量投影判别平凡乘子对，
连续跟踪非平凡乘子对的稳定性指数 ν，定位并精化乘子过 +1 与过 −1 的
族成员，再从首个可起链的 +1 合并构造共振点，用三维全周期 PAL 沿分岔
支双向行走。

用法（构建产物齐备的开发环境）：

    .venv/bin/python scripts/calibrate_l2_butterfly_bifurcation.py

脚本输出是人工复核用的记录，不写入仓库数据文件。Zhang 2022 Fig. 12
给出的 butterfly 面外振幅量级（z 到 −0.2）只作人工对照，不作断言
（ADR 0055 的结果性数值口径）。

命中 +1 合并并生成分岔支时退出码为 0，未命中或起链不可行时退出码为
1，并打印状态三元组、扫描区间、非平凡乘子对到 ±1 的最小偏差与 PAL
终止情况。
"""

from __future__ import annotations

import sys

import numpy as np

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family.orbits.walk import Cr3bpOrbitError, earth_moon_system
from e2m2e.algorithm.family.resonant_branch import (
    HaloMergeScan,
    MultiplierMerge,
    generate_resonant_branch_family,
    locate_halo_multiplier_merges,
)
from e2m2e.algorithm.results import ContinuationResult

DAYS_PER_TU = 4.342479848294297

BRANCH_DIRECTIONS = ("increase-x0", "decrease-x0")
BRANCH_ORBITS = 8
BRANCH_STEP = 0.005


def _print_scan(scan: HaloMergeScan) -> None:
    low, high = scan.parameter_interval
    period_low, period_high = scan.period_interval
    print("=" * 72)
    print("乘子合并扫描")
    print(f"  成员数 {scan.members_scanned}，x0 区间 [{low:.9f}, {high:.9f}]")
    print(
        f"  周期区间 [{period_low:.6f}, {period_high:.6f}] TU "
        f"[{period_low * DAYS_PER_TU:.3f}, {period_high * DAYS_PER_TU:.3f}] 天"
    )
    print(f"  状态 {scan.status.value}，原因 {scan.cause.value}")
    print(f"  消息 {scan.message}")
    print(
        f"  事件 {len(scan.events)} 个："
        f"+1 合并 {sum(event.kind == '+1' for event in scan.events)} 个，"
        f"−1 合并 {sum(event.kind == '-1' for event in scan.events)} 个"
    )
    for index, event in enumerate(scan.events):
        _print_event(index, event)
    if scan.resonance is not None:
        resonance = scan.resonance
        print("共振点")
        print(
            f"  x0={resonance.x0:.9f} T={resonance.period:.9f} TU "
            f"（{resonance.period * DAYS_PER_TU:.3f} 天）"
        )
        print(f"  乘子偏差 {resonance.multiplier_gap:.3e}，圈数 {resonance.revolutions}")
        print(f"  横截特征向量 {np.array2string(resonance.eigenvector, precision=6)}")


def _print_event(index: int, event: MultiplierMerge) -> None:
    print(f"  事件 {index}：{event.kind} 合并")
    print(
        f"    x0={event.parameter:.9f} T={event.period:.9f} TU "
        f"（{event.period * DAYS_PER_TU:.3f} 天）成员下标 {event.member_index}"
    )
    print(
        f"    乘子 {event.multiplier.real:.12e}{event.multiplier.imag:+.3e}i，"
        f"偏差 {event.multiplier_gap:.3e}，ν={event.nu:.12f}"
    )
    print(f"    横截特征向量 {np.array2string(event.eigenvector, precision=6)}")
    print(f"    流方向投影 {event.flow_projection:.6f}，族切向投影 {event.tangent_projection:.6f}")
    print(
        f"    精化区间 [{event.bracket[0]:.9f}, {event.bracket[1]:.9f}]，"
        f"精化{'成功' if event.refined else '未达容差'}"
    )


def _print_trace(scan: HaloMergeScan) -> None:
    print("逐成员乘子轨迹（每 10 个成员取一个）")
    for index, trace in enumerate(scan.trace):
        if index % 10 != 0 and index != len(scan.trace) - 1:
            continue
        nus = "，".join(f"{nu:+.6f}" for nu in trace.nus)
        print(f"  m{index:4d} x0={trace.parameter:.9f} T={trace.period:.6f} ν=[{nus}]")


def _print_branch(result: ContinuationResult, direction: str) -> None:
    print(f"分岔支方向 {direction}")
    print(f"  状态 {result.status.value}，原因 {result.cause.value}")
    print(f"  消息 {result.message}")
    if result.family is None:
        return
    for index, orbit in enumerate(result.family.orbits):
        state = orbit.states[0]
        asymmetry = max(abs(float(state[1])), abs(float(state[3])), abs(float(state[5])))
        print(
            f"  m{index} x0={float(state[0]):.9f} y0={float(state[1]):+.3e} "
            f"z0={float(state[2]):.9f} vx0={float(state[3]):+.3e} "
            f"vy0={float(state[4]):.9f} vz0={float(state[5]):+.3e} "
            f"T={float(orbit.period):.9f} 闭合 {float(orbit.closure_error):.2e} "
            f"对称破缺 {asymmetry:.3e} "
            f"Jacobi 漂移 {float(orbit.parameters.get('jacobi_drift', float('nan'))):.2e}"
        )


def main() -> int:
    dynamics = CR3BP_Dynamics(earth_moon_system())
    scan = locate_halo_multiplier_merges(dynamics)
    _print_scan(scan)
    _print_trace(scan)
    if scan.resonance is None:
        print("未构造共振点：横截方向不可用或无 +1 合并，无法起链")
        return 1
    failures = 0
    for direction in BRANCH_DIRECTIONS:
        try:
            result = generate_resonant_branch_family(
                scan.resonance,
                n_orbits=BRANCH_ORBITS,
                step_size=BRANCH_STEP,
                initial_direction=direction,
                dynamics=dynamics,
                family_type="butterfly",
                spatial=True,
            )
        except Cr3bpOrbitError as error:
            print(f"分岔支方向 {direction} 起链失败：{error}")
            failures += 1
            continue
        _print_branch(result, direction)
    return 0 if failures < len(BRANCH_DIRECTIONS) else 1


if __name__ == "__main__":
    sys.exit(main())
