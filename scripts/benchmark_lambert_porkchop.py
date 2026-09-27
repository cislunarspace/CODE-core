"""基准：porkchop 纯 Lambert+ΔV 热路径（``porkchop_grid_states_py``）耗时。

为 issue #715（Lambert 内核迁移 pykep-core，ADR 0052）提供量化依据：迁移前以
FAT 移植内核、迁移后以 pykep-core 适配层各跑一次，比对 revs=0 与 revs=1 两档
中位 wall-time，回归超过 2× 则按 issue 终止条件 #1 否决迁移。

网格为确定性合成状态（解析圆轨道采样，无 SPICE、无 CR3BP 传播——终端状态由
本脚本直接构造，``porkchop_grid_states_py`` 只做 Lambert + ΔV）：

- 地心二体（``mu_central = 398600.4418`` km³/s²）。
- 出发：r_dep = 7000 km 圆轨道等角采样 2400 点（状态 = 位置 + 圆速度）。
- 到达：r_arr = 30000 km 圆轨道，角度由 ``t_dep[i] + tof[j]`` 反推。注意
  ``porkchop_grid_states_py`` 的到达状态与 (i, j) 一一绑定（``n*m*6`` 行优先），
  无独立到达维度，故 72 000 格 = 2400 出发 × 30 tof。
- tof：5–25 h 线性 30 档（revs=1 在高档位大量可行，覆盖多圈双分支路径）。

规模：2400×30 = 72 000 格/次；``long_way=False``、``parallel=False``（串行
消除调度噪声）；重复 ≥5 次取中位 wall-time。

运行（勿用 ``uv run``，直接虚拟环境解释器，Windows 为
``.venv\\Scripts\\python.exe``，Linux 为 ``.venv/bin/python``）::

    .venv\\Scripts\\python.exe scripts/benchmark_lambert_porkchop.py \\
        --stage 迁移前 [--reps 5]

``--stage``（迁移前/迁移后）决定结果写入报告的哪一节，两节齐全时在文末生成
对照表。报告：``docs/plans/lambert-pykep-core-benchmark.md``。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import math
import platform
import re
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = REPO_ROOT / "docs" / "plans" / "lambert-pykep-core-benchmark.md"

MU_EARTH = 398600.4418  # km³/s²
R_DEP = 7000.0  # km
R_ARR = 30000.0  # km
N_DEP = 2400
N_TOF = 30

STAGES = ("迁移前", "迁移后")


def _circular_states(radius: float, angles: np.ndarray) -> np.ndarray:
    """圆轨道状态 (x, y, z, vx, vy, 0)，z 分量置零（平面几何）。"""
    n = angles.shape[0]
    states = np.zeros((n, 6))
    speed = math.sqrt(MU_EARTH / radius)
    states[:, 0] = radius * np.cos(angles)
    states[:, 1] = radius * np.sin(angles)
    states[:, 3] = -speed * np.sin(angles)
    states[:, 4] = speed * np.cos(angles)
    return states


def build_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造 dep_states (n,6)、arr_states (n*m,6)、tofs (m,)。"""
    dep_angles = np.linspace(0.0, 40.0 * np.pi, N_DEP)  # 20 圈弧段等角采样
    dep_states = _circular_states(R_DEP, dep_angles)
    tofs = np.linspace(5.0 * 3600.0, 25.0 * 3600.0, N_TOF)
    omega_dep = math.sqrt(MU_EARTH / R_DEP**3)
    omega_arr = math.sqrt(MU_EARTH / R_ARR**3)
    # t_dep[i] + tof[j]，行优先展平与 (i*m+j) 约定一致
    grid_t = dep_angles[:, None] / omega_dep + tofs[None, :]
    arr_angles = grid_t.ravel() * omega_arr
    arr_states = _circular_states(R_ARR, arr_angles)
    return dep_states, arr_states, tofs


def run_once(dep_states: np.ndarray, arr_states: np.ndarray, tofs: np.ndarray, revs: int) -> float:
    """跑一次网格扫描，返回 wall-time（秒）。"""
    from e2m2e._integrators import porkchop_grid_states_py

    t0 = time.perf_counter()
    porkchop_grid_states_py(
        dep_states.ravel().tolist(),
        arr_states.ravel().tolist(),
        tofs.tolist(),
        MU_EARTH,
        False,
        revs,
        parallel=False,
    )
    return time.perf_counter() - t0


def measure(reps: int) -> dict[int, float]:
    dep_states, arr_states, tofs = build_grid()
    medians: dict[int, float] = {}
    for revs in (0, 1):
        times = []
        for _ in range(reps):
            times.append(run_once(dep_states, arr_states, tofs, revs))
        medians[revs] = float(np.median(times))
        print(f"revs={revs}: 各次 {[f'{t:.3f}' for t in times]} → 中位 {medians[revs]:.3f} s")
    return medians


def render_section(stage: str, medians: dict[int, float], reps: int) -> str:
    now = _dt.datetime.now().isoformat(timespec="seconds")
    cpu = platform.processor() or platform.machine()
    lines = [
        f"### {stage}",
        "",
        f"- 时间：{now}",
        f"- 平台：{platform.platform()}｜CPU：{cpu}",
        f"- 规模：{N_DEP}×{N_TOF} = {N_DEP * N_TOF:,} 格/次，重复 {reps} 次取中位",
        "",
        "| 档位 | 中位 wall-time (s) |",
        "| --- | --- |",
        f"| revs=0 | {medians[0]:.3f} |",
        f"| revs=1 | {medians[1]:.3f} |",
        "",
    ]
    return "\n".join(lines)


def write_report(stage: str, medians: dict[int, float], reps: int) -> None:
    header = (
        "# Lambert 内核迁移 pykep-core 基准（issue #715 / ADR 0052）",
        "",
        "porkchop 纯 Lambert+ΔV 热路径（`porkchop_grid_states_py`，串行）",
        "在 FAT 移植内核与 pykep-core 适配层下的耗时对照。",
        "构造与复现方式见 `scripts/benchmark_lambert_porkchop.py` docstring。",
        "",
    )
    section = render_section(stage, medians, reps)
    text = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else "\n".join(header)
    pattern = rf"### {stage}\n.*?(?=\n### |\Z)"
    if re.search(pattern, text, flags=re.DOTALL):
        text = re.sub(pattern, section.rstrip("\n"), text, flags=re.DOTALL)
    else:
        text = text.rstrip("\n") + "\n\n" + section
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(f"报告已写入：{REPORT_PATH}")


def write_comparison() -> None:
    if not REPORT_PATH.exists():
        return
    text = REPORT_PATH.read_text(encoding="utf-8")
    values: dict[str, tuple[float, float]] = {}
    for stage in STAGES:
        m = re.search(
            rf"### {stage}\n.*?\| revs=0 \| ([0-9.]+) \|.*?\| revs=1 \| ([0-9.]+) \|",
            text,
            flags=re.DOTALL,
        )
        if m:
            values[stage] = (float(m.group(1)), float(m.group(2)))
    if len(values) < len(STAGES):
        return
    lines = [
        "## 对照（回归 > 2× 否决迁移）",
        "",
        "| 档位 | 迁移前 (s) | 迁移后 (s) | 比值 |",
        "| --- | --- | --- | --- |",
    ]
    for i, rev in enumerate((0, 1)):
        before = values["迁移前"][i]
        after = values["迁移后"][i]
        lines.append(f"| revs={rev} | {before:.3f} | {after:.3f} | {after / before:.2f}× |")
    block = "\n".join(lines)
    if "## 对照" in text:
        text = re.sub(r"## 对照.*\Z", block, text, flags=re.DOTALL)
    else:
        text = text.rstrip("\n") + "\n\n" + block + "\n"
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    medians = measure(args.reps)
    write_report(args.stage, medians, args.reps)
    write_comparison()


if __name__ == "__main__":
    main()
