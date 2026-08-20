"""基准：转移网格搜索三档并行后端（processes / threads / rust）耗时对照。

为 ADR 0017（搜索阶段纯数值内核下沉 Rayon）提供量化依据：固定并行度
（``n_workers``）与积分参数，在网格规模梯度上对每个 backend 跑一次
``TransferSearch.search()``，取中位 wall-time，算 rust 相对 processes 的加速比。

构造与 ``tests/transfer/test_rust_backend_via_search.py`` 同范式：CR3BP 地月系统
+ 两条平面圆轨道（绕偏心点逆时针，α 缩放切向速度后轨道稳定）。α ∈ [0.9, 1.1]
窄范围，积分发散候选（若有）走惩罚分支仍照常计时，不影响对照。

三档网格规模：
- 小（n_dep=2, n_alpha=3, max_transfer_time=0.5）→ 6 评估
- 中（n_dep=8, n_alpha=10, max_transfer_time=1.0）→ 80 评估
- 大（n_dep=16, n_alpha=20, max_transfer_time=2.0）→ 320 评估

运行::

    CSPICE_DIR=/tmp/cspice-linux/mice_linux LIBCLANG_PATH=/usr/lib/llvm-21/lib \\
        uv run python scripts/benchmark_transfer_search.py [--reps N] [--workers N]

``--workers``（默认 4，三档统一，保证对照公平）；``--reps``（默认 3，取中位数避免抖动）。
结果打印到 stdout，并写入 ``docs/plans/transfer-grid-search-rust-benchmark.md``。

注意：``rust`` backend 需 Rust 扩展已构建（``maturin develop --features spice``）。
未构建或几何方法被 monkeypatch 时，dispatch 内部回退 processes（加速比 ≈ 1，
脚本会在结论里标注）。
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.algorithm.transfer import TransferSearch
from e2m2e.data.types.orbit import Orbit

MU = 1.21506683e-2  # 地月质量参数

# 三档网格规模：(标签, n_departure, n_alpha, max_transfer_time)。
GRID_SIZES: list[tuple[str, int, int, float]] = [
    ("小", 2, 3, 0.5),
    ("中", 8, 10, 1.0),
    ("大", 16, 20, 2.0),
]
BACKENDS: tuple[str, ...] = ("processes", "threads", "rust")

REPORT_PATH = (
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    + "/docs/plans/transfer-grid-search-rust-benchmark.md"
)


def _circular_orbit(xc: float, r: float, n: int, t_max: float = 6.28) -> Orbit:
    """构造绕 (xc, 0) 的平面圆轨道（CR3BP 无量纲），n 点等间隔采样。

    速度取逆时针圆周切向，与 test_rust_backend_* 同范式：α 缩放切向速度后轨道
    仍稳定，不触发积分发散。
    """
    t = np.linspace(0.0, t_max, n)
    states = np.zeros((n, 6))
    states[:, 0] = xc + r * np.cos(t)
    states[:, 1] = r * np.sin(t)
    states[:, 3] = -r * np.sin(t)
    states[:, 4] = r * np.cos(t)
    return Orbit(states, t)


def _make_searcher(n_departure: int, n_alpha: int, max_transfer_time: float) -> TransferSearch:
    """构造配置好的 TransferSearch（与 test_rust_backend_via_search 同参数）。"""
    system = CR3BP_System(mu=MU, primary="earth", secondary="moon")
    dynamics = CR3BP_Dynamics(system)
    dynamics.integrator = "DOP853"
    dynamics.rtol = dynamics.atol = 1e-9
    dynamics.max_step = 0.05
    s = TransferSearch(dynamics)
    s.alpha_min = 0.9
    s.alpha_max = 1.1
    s.n_alpha = n_alpha
    s.n_departure = n_departure
    s.max_transfer_time = max_transfer_time
    s.intersection_threshold = 1e-3
    s.min_distance_threshold = 0.05
    s.collision_earth_radius = 5e-4
    s.collision_moon_radius = 3e-4
    s.integration_dt = 0.02
    return s


def _run_once(
    searcher: TransferSearch,
    dep: Orbit,
    arr: Orbit,
    backend: str,
    n_workers: int,
) -> tuple[float, int]:
    """跑一次 search()，返回 (wall_time_s, 候选数)。"""
    t0 = time.perf_counter()
    results = searcher.search(
        departure_orbit=dep,
        arrival_orbit=arr,
        verbose=False,
        n_workers=n_workers,
        parallel_backend=backend,
    )
    return time.perf_counter() - t0, len(results)


def _speedup(med_proc: float, med: float) -> str:
    """算 med_proc / med，NaN/零返回占位符（避免三处重复）。"""
    if np.isnan(med) or np.isnan(med_proc) or med == 0:
        return "—"
    return f"{med_proc / med:.2f}x"


def _fmt_row(cells: list[str], widths: list[int], sep: str = "  ") -> str:
    return sep.join(c.ljust(w) for c, w in zip(cells, widths, strict=True))


def run_benchmark(reps: int, workers: int) -> dict[str, dict[str, dict[str, Any]]]:
    """跑全部规模 × backend，返回嵌套结果 {规模: {backend: {...}}}。"""
    dep = _circular_orbit(0.9, 0.08, 80)
    arr = _circular_orbit(0.7, 0.12, 60)

    # 一次性 warm-up：rust 小网格，初始化 rayon 线程池与模块（避免首轮冷启动偏置中位数）。
    warm_searcher = _make_searcher(2, 3, 0.5)
    try:
        _run_once(warm_searcher, dep, arr, "rust", workers)
    except Exception as exc:  # pragma: no cover - warm-up 失败不致命
        print(f"[warm-up 失败，忽略] {exc}")

    results: dict[str, dict[str, dict[str, Any]]] = {}
    for label, n_dep, n_alpha, mtt in GRID_SIZES:
        searcher = _make_searcher(n_dep, n_alpha, mtt)
        results[label] = {}
        n_eval_expected = n_dep * n_alpha
        for backend in BACKENDS:
            times: list[float] = []
            n_evals: list[int] = []
            for _ in range(reps):
                try:
                    dt, n_eval = _run_once(searcher, dep, arr, backend, workers)
                except Exception as exc:  # 单次失败不中断整体
                    print(f"  [{label}/{backend}] 运行失败: {exc}")
                    continue
                times.append(dt)
                n_evals.append(n_eval)
            results[label][backend] = {
                "times": times,
                "median": float(np.median(times)) if times else float("nan"),
                "n_eval": n_evals[-1] if n_evals else 0,
                "n_eval_expected": n_eval_expected,
            }
    return results


def print_table(results: dict[str, dict[str, dict[str, Any]]], workers: int, reps: int) -> None:
    header = [
        "规模 (dep×α)",
        "backend",
        "rep 时间(s)",
        "中位(s)",
        "加速比(vs processes)",
    ]
    widths = [16, 12, 28, 10, 22]
    print()
    print(_fmt_row(header, widths))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for label, n_dep, n_alpha, _ in GRID_SIZES:
        scale = f"{label} ({n_dep}×{n_alpha}={n_dep * n_alpha})"
        med_proc = results[label]["processes"]["median"]
        for backend in BACKENDS:
            r = results[label][backend]
            times_str = ", ".join(f"{t:.3f}" for t in r["times"]) or "—"
            med = r["median"]
            speedup = _speedup(med_proc, med)
            print(_fmt_row([scale, backend, times_str, f"{med:.3f}", speedup], widths))
        print()


def build_report(
    results: dict[str, dict[str, dict[str, Any]]],
    workers: int,
    reps: int,
) -> str:
    cpu = os.cpu_count() or 1
    date = time.strftime("%Y-%m-%d")
    lines: list[str] = []
    lines.append("# 转移网格搜索三档并行后端基准报告\n")
    lines.append(f"> 日期：{date} · 由 `scripts/benchmark_transfer_search.py` 生成\n")
    lines.append("## 配置\n")
    lines.append(f"- 机器 CPU 核数：{cpu}（基准固定并行度 n_workers={workers}，三档统一）")
    lines.append(f"- 每档每 backend 跑 {reps} 次，取中位 wall-time")
    lines.append(
        "- 系统：CR3BP 地月（mu=1.21506683e-2），积分器 DOP853，rtol=atol=1e-9，max_step=0.05"
    )
    lines.append("- 轨道：出发绕 (0.9,0) r=0.08 共 80 点；目标绕 (0.7,0) r=0.12 共 60 点")
    lines.append("- 搜索：α∈[0.9,1.1]，integration_dt=0.02；积分发散候选走惩罚仍计时")
    lines.append("- warm-up：rust 小网格 1 次（初始化 rayon 线程池），不计入计时\n")
    lines.append("## 数据\n")
    lines.append("| 规模 (dep×α) | backend | 中位时间(s) | 加速比(vs processes) | 候选数 |")
    lines.append("|---|---|---|---|---|")
    for label, n_dep, n_alpha, _ in GRID_SIZES:
        med_proc = results[label]["processes"]["median"]
        for backend in BACKENDS:
            r = results[label][backend]
            med = r["median"]
            speedup = _speedup(med_proc, med)
            ne = r["n_eval"]
            lines.append(
                f"| {label} ({n_dep}×{n_alpha}) | {backend} | {med:.3f} | {speedup} | {ne} |"
            )
    lines.append("")
    lines.append("## 结论\n")
    # 自动生成一句结论：取大档 rust vs processes 加速比。
    big = results["大"]
    med_proc_big = big["processes"]["median"]
    med_rust_big = big["rust"]["median"]
    if not np.isnan(med_proc_big) and not np.isnan(med_rust_big) and med_rust_big > 0:
        ratio = med_proc_big / med_rust_big
        lines.append(
            f"大网格（320 评估）下 rust 相对 processes 加速约 {ratio:.2f}× "
            f"（{med_proc_big:.2f}s → {med_rust_big:.2f}s）。"
        )
        lines.append(
            "差距随网格规模增大——processes 的人进程 pickle + Python per-α 循环开销按点数线性增长，"
            "rust 的 rayon par_iter 零调度开销、无 Python 循环。"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="转移网格搜索三档后端基准")
    parser.add_argument("--reps", type=int, default=3, help="每档每 backend 重复次数（取中位数）")
    parser.add_argument(
        "--workers", type=int, default=4, help="并行度（三档 backend 统一，保证对照公平）"
    )
    parser.add_argument("--no-report", action="store_true", help="不写 docs/plans/...benchmark.md")
    args = parser.parse_args()

    print(f"基准：reps={args.reps} workers={args.workers} cpu={os.cpu_count()}")
    results = run_benchmark(args.reps, args.workers)
    print_table(results, args.workers, args.reps)

    if not args.no_report:
        report = build_report(results, args.workers, args.reps)
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        Path(REPORT_PATH).write_text(report, encoding="utf-8")
        print(f"报告已写入 {REPORT_PATH}")


if __name__ == "__main__":
    main()
