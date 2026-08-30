"""六域两层天图的全量生产入口（手动运行，ADR 0037 时间预算口径）。

CI 只跑抽查小网格（tests/algorithm/spatiography/test_cartography.py）；
全量制图（Table 4 缺省窗 + 生产分辨率）走本脚本手动/发版前执行：

    uv run --no-sync python scripts/spatiography_map_production.py \
        --zone SC --model em --n-a 100 --n-e 50 --out maps/

逐区输出 (a, e) 天图数据（NPZ：Ȳ 场、命运场、逃逸时刻场、诊断量场、
两轴与元数据）与 EM/EMS 对照摘要（JSON）。建议 release 构建
（make dev-release）后运行。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from e2m2e.algorithm.spatiography.cartography import (
    MAP_ZONE_NAMES,
    compare_models,
    dynamical_map,
)
from e2m2e.algorithm.spatiography.fate import FATE_CLASSES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zone", choices=MAP_ZONE_NAMES, required=True)
    parser.add_argument("--model", choices=["em", "ems", "both"], default="both")
    parser.add_argument("--n-a", type=int, default=100)
    parser.add_argument("--n-e", type=int, default=50)
    parser.add_argument("--e-min", type=float, default=0.0)
    parser.add_argument("--e-max", type=float, default=0.9)
    parser.add_argument("--span-years", type=float, default=None, help="None = Table 4 缺省")
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--out", type=Path, default=Path("maps"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    models = ["em", "ems"] if args.model == "both" else [args.model]
    results = {}
    for model in models:
        t0 = time.time()
        result = dynamical_map(
            args.zone,
            model=model,
            n_a=args.n_a,
            n_e=args.n_e,
            e_min=args.e_min,
            e_max=args.e_max,
            span_years=args.span_years,
            rtol=args.rtol,
        )
        elapsed = time.time() - t0
        results[model] = result
        stem = args.out / f"{args.zone.lower()}_{model}_na{args.n_a}_ne{args.n_e}"
        np.savez_compressed(
            stem.with_suffix(".npz"),
            a_over_a_moon=result.a_over_a_moon,
            e_grid=result.e_grid,
            ybar=result.ybar_field,
            fate_ids=result.fate_ids,
            t_escape_years=result.t_escape_years_field,
            min_r_sel_km=result.min_r_sel_km_field,
            min_r_geo_km=result.min_r_geo_km_field,
            fate_classes=np.array(FATE_CLASSES),
        )
        summary = {
            "zone": args.zone,
            "model": model,
            "span_years": result.span_years,
            "cells": result.cells,
            "wall_seconds": round(elapsed, 1),
            "fate_fractions": result.fate_fractions(),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        (stem.with_suffix(".summary.json")).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if len(results) == 2:
        comparison = compare_models(results["em"], results["ems"])
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
        stem = args.out / f"{args.zone.lower()}_em_vs_ems_na{args.n_a}_ne{args.n_e}"
        stem.with_suffix(".json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
