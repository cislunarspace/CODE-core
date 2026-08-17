#!/usr/bin/env python3
"""NRHO 星历修正对照矩阵（开发期反馈回路，#463）。

扫近月高 × 采样策略 × 第 1 步圈/段，打印收敛与墙钟。不进默认 pytest；
生产默认策略由 ``tests/algorithm/design/test_nrho_ephemeris_correction.py``
短弧回归锁定。

依赖：已 ``make dev`` / ``make dev-release``，``SPICE_KERNEL_DIR`` 指向内核目录
（或仓库 ``kernels/``）。

示例（对齐 issue #463 反馈回路）：

```bash
SPICE_KERNEL_DIR=kernels python scripts/nrho_ephemeris_correction_matrix.py \\
  --perilune 5000 --sampling uniform,drop_near \\
  --revs-per-group 1,3 --duration-days 30.4375 --max-iter 30
```
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from collections.abc import Callable

import numpy as np

from e2m2e.algorithm.solver.multiple_shooting import (
    sample_patch_points_drop_near_perilune,
    sample_patch_points_perilune_clustered,
)


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_sampling_list(text: str) -> list[str]:
    allowed = {"uniform", "clustered", "drop_near"}
    items = [x.strip() for x in text.split(",") if x.strip()]
    bad = [x for x in items if x not in allowed]
    if bad:
        raise SystemExit(f"未知采样策略 {bad}；允许 {sorted(allowed)}")
    return items


def _make_sampler(mode: str) -> Callable:
    """返回替换 ``_sample_patch_points`` 的采样函数。"""
    m = importlib.import_module("e2m2e.algorithm.design.design_orbit")

    def _sample(dynamics, state0, period, n_revolutions, *, sampling=None):
        dense = m._dense_orbit(dynamics, state0, period)
        if mode == "clustered":
            t_rel, states = sample_patch_points_perilune_clustered(dense, dynamics)
        elif mode == "drop_near":
            t_rel, states = sample_patch_points_drop_near_perilune(
                dense, dynamics, n_points=m._POINTS_PER_REV
            )
        else:  # uniform
            t_rel = np.linspace(0.0, period, m._POINTS_PER_REV, endpoint=False)
            states = np.empty((len(t_rel), 6))
            for i in range(6):
                states[:, i] = np.interp(t_rel, dense.times, dense.states[:, i])
        t_patch = np.concatenate([t_rel + k * period for k in range(n_revolutions)])
        return t_patch, np.tile(states, (n_revolutions, 1))

    return _sample


def _run_one(
    *,
    perilune: float,
    duration_days: float,
    sampling: str,
    revs_per_group: int,
    max_iter: int,
    collinear_point: int,
    north_south: int,
) -> tuple[str, float, str]:
    """跑一条 design_orbit，返回 (状态, 秒, 摘要)。"""
    from types import SimpleNamespace

    from e2m2e.algorithm.design import design_orbit

    m = importlib.import_module("e2m2e.algorithm.design.design_orbit")
    orig_sample = m._sample_patch_points
    orig_seg = m._design_apolune_segmented

    def _seg_wrapped(*args, **kwargs):
        args_list = list(args)
        # (forces, observer, t, s, revs_per_group, points_per_rev, ...)
        if len(args_list) >= 5:
            args_list[4] = revs_per_group
        else:
            kwargs["revs_per_group"] = revs_per_group
        kwargs.setdefault("max_iter", max_iter)
        return orig_seg(*args_list, **kwargs)

    m._sample_patch_points = _make_sampler(sampling)
    m._design_apolune_segmented = _seg_wrapped
    t0 = time.perf_counter()
    try:
        req = SimpleNamespace(
            orbit_type="NRHO",
            amplitude=None,
            phase=0.0,
            collinear_point=collinear_point,
            north_south=north_south,
            amplitude_in=None,
            amplitude_out=None,
            phase_in=None,
            phase_out=None,
            perilune_height=perilune,
            inclination=None,
            arg_of_pericenter=None,
            semi_major_axis=None,
            epoch=(2024, 1, 1, 0, 0, 0.0),
            duration=duration_days * 86400.0,
            output_step=3600.0,
            perturbation=None,
            dyb=None,
            earth_degree=10,
            moon_degree=10,
            correction_method="segmented",
            correction_revolutions=1,
        )
        res = design_orbit(req)
        dt = time.perf_counter() - t0
        resid = res.correction.max_residual if res.correction is not None else float("nan")
        n = len(res.ephemeris) if res.ephemeris is not None else 0
        return "OK", dt, f"resid={resid:.3e} n={n}"
    except Exception as exc:  # noqa: BLE001 — 诊断脚本要吞一切失败并制表
        dt = time.perf_counter() - t0
        msg = str(exc).split("\n")[0][:100]
        return "FAIL", dt, f"{type(exc).__name__}: {msg}"
    finally:
        m._sample_patch_points = orig_sample
        m._design_apolune_segmented = orig_seg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NRHO 星历修正对照矩阵（#463）")
    parser.add_argument(
        "--perilune",
        default="5000",
        help="近月点高度 km，逗号分隔（默认 5000）",
    )
    parser.add_argument(
        "--sampling",
        default="uniform,drop_near,clustered",
        help="采样策略：uniform,drop_near,clustered",
    )
    parser.add_argument(
        "--revs-per-group",
        default="1,3",
        help="第 1 步每段圈数，逗号分隔（默认 1,3）",
    )
    parser.add_argument(
        "--duration-days",
        type=float,
        default=30.4375,
        help="弧长（天，默认 30.4375 ≈ 1 月）",
    )
    parser.add_argument("--max-iter", type=int, default=30, help="每段最大迭代（默认 30）")
    parser.add_argument("--collinear-point", type=int, default=2, choices=(1, 2))
    parser.add_argument("--north-south", type=int, default=2, choices=(1, 2), help="1=北 2=南")
    args = parser.parse_args(argv)

    if not os.environ.get("SPICE_KERNEL_DIR") and os.path.isdir("kernels"):
        os.environ["SPICE_KERNEL_DIR"] = "kernels"

    perilunes = _parse_float_list(args.perilune)
    samplings = _parse_sampling_list(args.sampling)
    revs_list = _parse_int_list(args.revs_per_group)

    print(
        f"NRHO 星历修正矩阵  L{args.collinear_point}"
        f"{'南' if args.north_south == 2 else '北'}  "
        f"duration={args.duration_days}d  max_iter={args.max_iter}"
    )
    print(f"{'perilune':>8} {'sampling':>10} {'revs':>4} {'status':>5} {'sec':>7}  detail")
    print("-" * 72)

    n_fail = 0
    for perilune in perilunes:
        for sampling in samplings:
            for revs in revs_list:
                status, dt, detail = _run_one(
                    perilune=perilune,
                    duration_days=args.duration_days,
                    sampling=sampling,
                    revs_per_group=revs,
                    max_iter=args.max_iter,
                    collinear_point=args.collinear_point,
                    north_south=args.north_south,
                )
                if status != "OK":
                    n_fail += 1
                print(f"{perilune:8.0f} {sampling:>10} {revs:4d} {status:>5} {dt:7.1f}  {detail}")

    print("-" * 72)
    print(f"完成：失败 {n_fail} 条")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
