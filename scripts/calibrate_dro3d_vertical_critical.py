#!/usr/bin/env python
"""平面 DRO 族垂直临界点标定（#689 三维 DRO 族的种子来源）。

用法（诊断/基线脚本约定：直接用虚拟环境解释器，不要走 `uv run`）：

    .venv/bin/python scripts/calibrate_dro3d_vertical_critical.py

流程：

1. 从平面 DRO 标准种子出发，沿 x0 双向链式延拓铺族（种子本身已是族上的大
   振幅成员，垂直临界点可能落在振幅增大侧或减小侧，两侧都扫）；
2. 用 ``StabilityAnalysis.detect_bifurcation_in_family``（#688）在 x0 上跟踪
   Floquet 乘子的稳定性指数 ν，取 SADDLE_NODE 类型的穿越候选；
3. 对每个候选点独立复算 z-vz 块半迹 ``vt = 0.5·(M[2,2] + M[5,5])``，打印
   ``|vt − 1|``——定义级交叉验证（ν 跨 +2 ⟺ vt 过 +1），不复制扫描实现；
4. 取族曲线上距种子最近的垂直临界点（即从种子出发的第一个），打印可直接粘贴
   进 ``crates/e2m2e-integrators/src/family_generation/dro3d.rs`` 的常量块。

域内无 ν 跨 +2（垂直临界）穿越时打印诊断并以退出码 1 结束——这属于物理
阻塞，须把证据发到 #689 请求裁决，不得静默改用其他分岔点。
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.algorithm.family.cr3bp_orbits import _correct_dro
from e2m2e.algorithm.stability import (
    BifurcationType,
    FamilyBifurcationPoint,
    StabilityAnalysis,
    _pair_stability_index,
    _reciprocal_pairs,
)
from e2m2e.data.constants import Datum
from e2m2e.data.templates.seed import _DRO_SEED_X0
from e2m2e.data.types.orbit import Orbit
from e2m2e.integrators import orbit_family_metric_py

#: 振幅（距月心距离 min/max 均值）测量采样点数，与 Rust 族度量同口径
_AMPLITUDE_SAMPLES = 4000


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
    minimum, maximum = orbit_family_metric_py(
        float(dynamics.system.mu),
        "moon-distance",
        0,
        np.asarray(orbit.states[0], dtype=float).tolist(),
        _period(orbit),
        sample_count=_AMPLITUDE_SAMPLES,
    )
    return 0.5 * (minimum + maximum) * _char_length(dynamics)


def _z_amplitude_km(dynamics: CR3BP_Dynamics, orbit: Orbit) -> float:
    """一个周期内 ``max|z|``（km）。"""
    _, maximum = orbit_family_metric_py(
        float(dynamics.system.mu),
        "z-amplitude",
        0,
        np.asarray(orbit.states[0], dtype=float).tolist(),
        _period(orbit),
        sample_count=1000,
    )
    return maximum * _char_length(dynamics)


def _half_trace_vt(dynamics: CR3BP_Dynamics, orbit: Orbit) -> float:
    """z-vz 块半迹 ``vt = 0.5·(M[2,2] + M[5,5])``（垂直临界 ⟺ vt = 1）。"""
    monodromy = dynamics.compute_state_transition_matrix(orbit.states[0], _period(orbit))
    return 0.5 * float(monodromy[2, 2] + monodromy[5, 5])


def _nus(dynamics: CR3BP_Dynamics, orbit: Orbit) -> list[complex]:
    """三条倒数对的稳定性指数 ν（平凡对 ν=2）。"""
    eigenvalues = StabilityAnalysis(orbit, dynamics).compute_floquet_multipliers()
    return [_pair_stability_index(pair) for pair in _reciprocal_pairs(np.asarray(eigenvalues))]


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


def _walk(
    walker: _DroWalker,
    step: float,
    count: int,
    dynamics: CR3BP_Dynamics,
) -> list[tuple[float, Orbit]]:
    """自种子沿 ``step`` 定步延拓，返回 (x0, orbit) 序列（含种子）。"""
    x0 = _DRO_SEED_X0
    members: list[tuple[float, Orbit]] = [(x0, walker(x0))]
    for _ in range(count - 1):
        x0 += step
        if not 0.0 < x0 < 1.0 - dynamics.system.mu:
            print(f"#   x0={x0:.6f} 越出族参数域，停止延拓")
            break
        try:
            orbit = walker(x0)
        except Exception as exc:  # noqa: BLE001 - 诊断脚本：失败即停并打印
            print(f"#   x0={x0:.6f} 修正失败（{type(exc).__name__}: {exc}），停止延拓")
            break
        members.append((x0, orbit))
    return members


def _print_trace(label: str, members: list[tuple[float, Orbit]], dynamics: CR3BP_Dynamics) -> None:
    amplitudes = [_moon_amplitude_km(dynamics, orbit) for _, orbit in members]
    print(
        f"# {label}：x0 ∈ [{members[0][0]:.6f}, {members[-1][0]:.6f}]、{len(members)} 条成员、"
        f"振幅 {amplitudes[0]:.1f} → {amplitudes[-1]:.1f} km、"
        f"vt {_half_trace_vt(dynamics, members[0][1]):+.6f} → "
        f"{_half_trace_vt(dynamics, members[-1][1]):+.6f}"
    )
    print("#   x0          振幅(km)  max|z|(km)      vt       |vt-1|   ν（三条）")
    for (x0, orbit), amplitude in zip(members, amplitudes, strict=True):
        vt = _half_trace_vt(dynamics, orbit)
        nus = " ".join(f"{nu.real:+.6f}{nu.imag:+.6f}i" for nu in _nus(dynamics, orbit))
        print(
            f"    {x0:.6f}  {amplitude:10.1f}  {_z_amplitude_km(dynamics, orbit):9.3e}"
            f"  {vt:+.6f}  {abs(vt - 1.0):.3e}  {nus}"
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
        f"vt {_half_trace_vt(dynamics, seed):+.6f}、T {_period(seed):.6f}"
    )

    step = abs(args.step)
    decreasing = _walk(walker, -step, args.max_members, dynamics)
    _print_trace("x0 减小侧（振幅增大侧）", decreasing, dynamics)
    increasing = _walk(walker, +step, args.max_members, dynamics)
    _print_trace("x0 增大侧（振幅减小侧）", increasing, dynamics)
    members = decreasing + increasing

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
        vt = _half_trace_vt(dynamics, point.orbit)
        print(
            f"# 候选 {point.type.value:14s} x0={point.parameter:.9f}"
            f" 振幅={amplitude:10.1f} km max|z|={_z_amplitude_km(dynamics, point.orbit):.3e} km"
            f" 残差={point.residual:.3e} |vt-1|={abs(vt - 1.0):.3e} vt={vt:+.9f}"
            f" T={_period(point.orbit):.9f}"
        )
        if point.type is BifurcationType.SADDLE_NODE:
            vertical.append((point, amplitude, vt))

    if not vertical:
        print(
            "# 未在可行走的平面 DRO 族参数域内发现 ν 跨 +2（垂直临界）穿越；"
            "属物理阻塞，须把本输出发到 #689 请求裁决。",
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
