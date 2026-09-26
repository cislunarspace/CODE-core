#!/usr/bin/env python
"""平面 DRO 族垂直临界点标定（#689 三维 DRO 族的种子来源）。

用法（诊断/基线脚本约定：直接用虚拟环境解释器，不要走 `uv run`）：

    .venv/bin/python scripts/calibrate_dro3d_vertical_critical.py

流程：

1. 从平面 DRO 标准种子出发，沿 x0 双向链式延拓铺族（种子本身已是族上的大
   振幅成员，垂直临界点可能落在振幅增大侧或减小侧，两侧都扫）；
2. 用 ``StabilityAnalysis.detect_bifurcation_in_family``（#688）在 x0 上跟踪
   Floquet 乘子的稳定性指数 ν，取 SADDLE_NODE 类型的穿越候选；
3. 对每个候选点用库内 ``_vertical_trace`` 复算 z-vz 块半迹
   ``vt = 0.5·(M[2,2] + M[5,5])``，打印 ``|vt − 1|``（定义级交叉验证：ν 跨
   +2 ⟺ vt 过 +1），并以 ``|vt − 1| ≤ _VT_TOL`` 加门控；未过门控的
   SADDLE_NODE 候选标注为「疑似面内对，非垂直临界」，不计入结论；
4. 取族曲线上距种子最近的垂直临界点（即从种子出发的第一个），打印可直接粘贴
   进 ``crates/e2m2e-integrators/src/family_generation/dro3d.rs`` 的常量块。

单方向延拓在修正失败时退回上一成功成员并把步长减半重试（下限与生产族行走
``_walk_family`` 同为 1e-4），并记录该方向的停止原因（域边界 / 修正失败 / 成员
预算耗尽）。域内无垂直临界穿越时以退出码 1 结束：仅当两侧延拓均走到族参数域
边界、扫描无失败点且无跳支区间时才判为物理阻塞；否则说明延拓被中断、结论不可
得出，并列出停止原因与失败/跳支计数。任一种情况都须把证据发到 #689 请求裁决，
不得静默改用其他分岔点。
"""

from __future__ import annotations

import argparse
import sys
from typing import NamedTuple

import numpy as np

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.algorithm.family.axial_initial_guess import _vertical_trace
from e2m2e.algorithm.family.cr3bp_orbits import (
    _correct_dro,
    _moon_distance_minmax,
    _z_amplitude_max,
)
from e2m2e.algorithm.stability import (
    BifurcationType,
    FamilyBifurcationPoint,
    StabilityAnalysis,
    _member_nus,
)
from e2m2e.data.constants import Datum
from e2m2e.data.templates.seed import _DRO_SEED_X0
from e2m2e.data.types.orbit import Orbit

#: 垂直临界判据 ``|vt − 1|`` 容差；与
#: ``e2m2e/algorithm/family/axial_initial_guess.py`` 的
#: ``abs(vt_mid - 1.0) < 1e-4`` 同口径
_VT_TOL = 1e-4

#: 延拓步长下限；与 ``e2m2e/algorithm/family/cr3bp_orbits.py::_walk_family``
#: 的 ``step *= 0.5; if step < 1e-4`` 退半步重试下限一致
_MIN_WALK_STEP = 1e-4


def _period(orbit: Orbit) -> float:
    """周期（周期轨道必有 period；运行时守卫）。"""
    assert orbit.period is not None
    return float(orbit.period)


def _char_length(dynamics: CR3BP_Dynamics) -> float:
    """特征长度（DU→km；系统必有缩放）。"""
    length = dynamics.system.characteristic_length
    assert length is not None
    return float(length)


def _moon_amplitude_km(dynamics: CR3BP_Dynamics, orbit: Orbit) -> float:
    """一个周期内距月心距离 min/max 均值（km），与 ``design_dro`` 同定义。"""
    minimum, maximum = _moon_distance_minmax(dynamics, orbit)
    return 0.5 * (minimum + maximum) * _char_length(dynamics)


def _z_amplitude_km(dynamics: CR3BP_Dynamics, orbit: Orbit) -> float:
    """一个周期内 ``max|z|``（km）；采样点数用库函数默认值（1000）。"""
    return _z_amplitude_max(dynamics, orbit) * _char_length(dynamics)


def _nus(dynamics: CR3BP_Dynamics, orbit: Orbit) -> list[complex] | None:
    """三条倒数对的稳定性指数 ν（平凡对 ν=2）；乘子未配成 3 对时返回 None。"""
    eigenvalues = StabilityAnalysis(orbit, dynamics).compute_floquet_multipliers()
    try:
        return _member_nus(np.asarray(eigenvalues))
    except ValueError:
        return None


class _DroWalker:
    """链式初猜的平面 DRO 族修正器（最近已知成员作 Newton 初猜）。"""

    def __init__(self, dynamics: CR3BP_Dynamics, seed_x0: float) -> None:
        self.dynamics = dynamics
        self.known: list[tuple[float, Orbit]] = [(seed_x0, _correct_dro(dynamics, seed_x0, None))]

    def __call__(self, x0: float) -> Orbit:
        guess = min(self.known, key=lambda item: abs(item[0] - x0))[1]
        orbit = _correct_dro(self.dynamics, x0, guess)
        self.known.append((x0, orbit))
        return orbit

    def seed(self, seed_x0: float) -> Orbit:
        return next(orbit for x0, orbit in self.known if x0 == seed_x0)


class _WalkResult(NamedTuple):
    """单方向延拓结果：成员序列（含种子）、停止原因、是否走到族参数域边界。"""

    members: list[tuple[float, Orbit]]
    stop_reason: str
    at_domain_edge: bool


def _walk(
    walker: _DroWalker,
    step: float,
    count: int,
    dynamics: CR3BP_Dynamics,
) -> _WalkResult:
    """自种子沿 ``step``（带符号）定步延拓，返回成员序列与停止原因。

    修正失败时退回上一成功成员并把步长减半重试（下限 ``_MIN_WALK_STEP``，
    与生产族行走 ``_walk_family`` 同口径）——从失败点原步长继续会在域边界处
    一路失败；``x0`` 越出族参数域、步长减到下限仍失败、成员预算耗尽三者任一
    都停止该方向并记录原因。
    """
    x0 = _DRO_SEED_X0
    members: list[tuple[float, Orbit]] = [(x0, walker(x0))]
    for _ in range(count - 1):
        while True:
            x_try = x0 + step
            if not 0.0 < x_try < 1.0 - dynamics.system.mu:
                return _WalkResult(
                    members,
                    f"x0={x_try:.6f} 越出族参数域（0 < x0 < {1.0 - dynamics.system.mu:.6f}），"
                    f"止于 x0={x0:.6f}（{len(members)} 条成员）",
                    True,
                )
            try:
                orbit = walker(x_try)
            except Exception as exc:  # noqa: BLE001 - 诊断脚本：失败即退半步重试
                step *= 0.5
                if abs(step) < _MIN_WALK_STEP:
                    return _WalkResult(
                        members,
                        f"在 x0={x_try:.6f} 修正失败（{type(exc).__name__}: {exc}），"
                        f"步长已退到下限 {_MIN_WALK_STEP:.1e} 以下，"
                        f"止于 x0={x0:.6f}（{len(members)} 条成员）",
                        False,
                    )
                continue
            x0 = x_try
            members.append((x0, orbit))
            break
    return _WalkResult(
        members,
        f"成员预算耗尽（{len(members)} 条），止于 x0={x0:.6f}",
        False,
    )


def _print_trace(label: str, members: list[tuple[float, Orbit]], dynamics: CR3BP_Dynamics) -> None:
    amplitudes = [_moon_amplitude_km(dynamics, orbit) for _, orbit in members]
    print(
        f"# {label}：x0 ∈ [{members[0][0]:.6f}, {members[-1][0]:.6f}]、{len(members)} 条成员、"
        f"振幅 {amplitudes[0]:.1f} → {amplitudes[-1]:.1f} km、"
        f"vt {_vertical_trace(dynamics, members[0][1]):+.6f} → "
        f"{_vertical_trace(dynamics, members[-1][1]):+.6f}"
    )
    print("#   x0          振幅(km)  max|z|(km)      vt       |vt-1|   ν（三条）")
    for (x0, orbit), amplitude in zip(members, amplitudes, strict=True):
        vt = float(_vertical_trace(dynamics, orbit))
        nus = _nus(dynamics, orbit)
        # 乘子未配成 3 个倒数对时显式标注，不得静默少打一条
        nus_text = (
            " ".join(f"{nu.real:+.6f}{nu.imag:+.6f}i" for nu in nus)
            if nus is not None
            else "<倒数对配对失败>"
        )
        print(
            f"    {x0:.6f}  {amplitude:10.1f}  {_z_amplitude_km(dynamics, orbit):9.3e}"
            f"  {vt:+.6f}  {abs(vt - 1.0):.3e}  {nus_text}"
        )


def _rust_block(orbit: Orbit, amplitude_km: float) -> str:
    state = np.asarray(orbit.states[0], dtype=float)
    lines = [
        "pub(crate) const SEED_DRO3D_STATE: [f64; 6] = [",
        *(f"    {value:.17g}," for value in state),
        "];",
        f"pub(crate) const SEED_DRO3D_PERIOD: f64 = {_period(orbit):.17g};",
        f"pub(crate) const SEED_DRO3D_AMPLITUDE_KM: f64 = {amplitude_km:.1f};",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="平面 DRO 族垂直临界点标定（#689）")
    parser.add_argument("--step", type=float, default=0.005, help="x0 延拓步长（无量纲）")
    parser.add_argument("--max-members", type=int, default=90, help="单方向延拓成员上限（含种子）")
    args = parser.parse_args(argv)

    system = CR3BP_System(
        mu=Datum.DE421.mu, primary="Earth", secondary="Moon"
    )._with_default_scales()
    dynamics = CR3BP_Dynamics(system)
    print(
        f"# μ = {system.mu!r}（DE421 地月）、DU = {_char_length(dynamics):.1f} km、"
        f"月心 x = {1.0 - system.mu:.6f}"
    )

    walker = _DroWalker(dynamics, _DRO_SEED_X0)
    seed = walker.seed(_DRO_SEED_X0)
    print(
        f"# 种子 x0={_DRO_SEED_X0:.6f}：振幅 {_moon_amplitude_km(dynamics, seed):.1f} km、"
        f"max|z| {_z_amplitude_km(dynamics, seed):.3e} km、"
        f"vt {_vertical_trace(dynamics, seed):+.6f}、T {_period(seed):.6f}"
    )

    step = abs(args.step)
    decreasing = _walk(walker, -step, args.max_members, dynamics)
    _print_trace("x0 减小侧（振幅增大侧）", decreasing.members, dynamics)
    print(f"# x0 减小侧延拓停止：{decreasing.stop_reason}")
    increasing = _walk(walker, +step, args.max_members, dynamics)
    _print_trace("x0 增大侧（振幅减小侧）", increasing.members, dynamics)
    print(f"# x0 增大侧延拓停止：{increasing.stop_reason}")
    members = decreasing.members + increasing.members

    # 扫描：parameters 须严格递增，按 x0 升序送入；x0 相同的成员（种子两侧
    # 延拓各出现一次）合流
    combined = dict(members)
    ordered = sorted(combined.items())
    scan = StabilityAnalysis.detect_bifurcation_in_family(
        [orbit for _, orbit in ordered],
        [x0 for x0, _ in ordered],
        parameter_name="x0",
        member_at=walker,
        dynamics=dynamics,
    )
    print(
        f"# 扫描（{len(ordered)} 成员）：{len(scan.points)} 个候选穿越、"
        f"{len(scan.branch_jumps)} 个跳支区间、{len(scan.failures)} 个失败点"
    )
    for failure in scan.failures:
        print(f"#   失败 x0={failure.parameter:.6f}: {failure.message}")
    for jump in scan.branch_jumps:
        print(
            f"#   跳支 [{jump.parameter_lo:.6f}, {jump.parameter_hi:.6f}]"
            f" Δν_max={jump.max_displacement:.6f}"
        )

    vertical: list[tuple[FamilyBifurcationPoint, float, float]] = []  # (点, 振幅, vt)
    for point in scan.points:
        amplitude = _moon_amplitude_km(dynamics, point.orbit)
        vt = float(_vertical_trace(dynamics, point.orbit))
        print(
            f"# 候选 {point.type.value:14s} x0={point.parameter:.9f}"
            f" 振幅={amplitude:10.1f} km max|z|={_z_amplitude_km(dynamics, point.orbit):.3e} km"
            f" 残差={point.residual:.3e} |vt-1|={abs(vt - 1.0):.3e} vt={vt:+.9f}"
            f" T={_period(point.orbit):.9f}"
        )
        if point.type is not BifurcationType.SADDLE_NODE:
            continue
        if abs(vt - 1.0) <= _VT_TOL:
            vertical.append((point, amplitude, vt))
        else:
            # ν 跨 +2 但 vt 不过 1：疑似面内对（非垂直临界），不得计入结论
            print(
                f"#   ↑ ν=+2 候选（疑似面内对，非垂直临界）："
                f"|vt-1|={abs(vt - 1.0):.3e} > {_VT_TOL:.1e}"
            )

    if not vertical:
        print("# 未在可行走的平面 DRO 族参数域内发现 ν 跨 +2（垂直临界）穿越。", file=sys.stderr)
        if (
            decreasing.at_domain_edge
            and increasing.at_domain_edge
            and not scan.failures
            and not scan.branch_jumps
        ):
            print(
                "# 两侧延拓均走到族参数域边界、扫描无失败点与跳支区间：属物理阻塞，"
                "须把本输出发到 #689 请求裁决。",
                file=sys.stderr,
            )
        else:
            print(
                "# 延拓被中断，结论不可得出；须把本输出发到 #689 请求裁决。",
                file=sys.stderr,
            )
            print(f"#   x0 减小侧停止原因：{decreasing.stop_reason}", file=sys.stderr)
            print(f"#   x0 增大侧停止原因：{increasing.stop_reason}", file=sys.stderr)
            print(
                f"#   扫描失败点 {len(scan.failures)} 个、跳支区间 {len(scan.branch_jumps)} 个",
                file=sys.stderr,
            )
        gated_out = sum(1 for point in scan.points if point.type is BifurcationType.SADDLE_NODE)
        if gated_out:
            print(
                f"#   另有 {gated_out} 个 ν=+2 候选未通过 |vt-1| ≤ {_VT_TOL:.1e} 门控"
                "（疑似面内对）",
                file=sys.stderr,
            )
        return 1

    # 距种子最近者即"从种子出发沿族曲线遇到的第一个垂直临界点"
    point, amplitude, vt = min(vertical, key=lambda item: abs(item[0].parameter - _DRO_SEED_X0))
    on_growing_side = point.parameter < _DRO_SEED_X0
    side = "x0 减小侧（振幅增大侧）" if on_growing_side else "x0 增大侧（振幅减小侧）"
    print(
        f"\n# 选定垂直临界点（{side}）：x0={point.parameter:.9f}、振幅={amplitude:.1f} km、"
        f"vt={vt:+.9f}（|vt-1|={abs(vt - 1.0):.3e}）、残差={point.residual:.3e}、"
        f"T={_period(point.orbit):.9f}"
    )
    print("\n# 粘贴进 crates/e2m2e-integrators/src/family_generation/dro3d.rs：")
    print(_rust_block(point.orbit, amplitude))
    return 0


if __name__ == "__main__":
    sys.exit(main())
