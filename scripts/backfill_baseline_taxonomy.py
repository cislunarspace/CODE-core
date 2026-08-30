"""给随包 baseline 数据集回填分类学标签（issue #581 / ADR 0042）。

对 ``e2m2e/data/catalog_baseline/`` 每个族记录：按成员初值 + 周期实测
分类（与生产 ingest 同一条打标路径 ``stamp_taxonomy_labels``），把
``classification.taxonomy_labels``（记录级去重集合）与
``members[].taxonomy_label``（成员级 primary）写回 JSON。NPZ 数组不动。

期望集为空的族（lissajous 拟周期、horseshoe 不在分类学内）直接置空
标签。设计侧族名与实测不符的成员（如 #587 修复前的 dpo 逆行成员）
会触发冲突告警并按实测值入库——脚本输出可见。

用法（直接用 venv 解释器，同 generate_catalog_baseline）::

    .venv/bin/python scripts/backfill_baseline_taxonomy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from e2m2e.algorithm.dynamics.cr3bp_system import CR3BP_System
from e2m2e.api.catalog_ingest import stamp_taxonomy_labels
from e2m2e.data.types.orbit import Orbit

#: 基线数据目录（与 generate_catalog_baseline.py 同源）
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "e2m2e" / "data" / "catalog_baseline"


def backfill_family(json_path: Path) -> tuple[str, list[str]]:
    """回填一个族记录，返回（族名，记录级标签列表）。"""
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    bundle = np.load(json_path.with_suffix(".npz"))
    mu = meta["scalars"].get("mu")
    if mu is None:
        return json_path.stem, []
    system = CR3BP_System(mu=float(mu), primary="earth", secondary="moon")
    orbits = []
    for i, member in enumerate(meta["members"]):
        state = bundle[f"cr3bp/members/{i:04d}/states"][0]
        orbit = Orbit(states=state.reshape(1, -1), times=np.array([0.0]), system=system)
        orbit.period = member.get("period")
        orbit.parameters = dict(member.get("parameters", {}))
        orbits.append(orbit)
    labels, member_labels = stamp_taxonomy_labels(
        meta["classification"]["orbit_family"],
        orbits,
        periodicity=meta["scalars"].get("periodicity", "periodic"),
        context=json_path.stem,
    )
    meta["classification"]["taxonomy_labels"] = labels
    for member, label in zip(meta["members"], member_labels, strict=True):
        member["taxonomy_label"] = label
    # 与 record.meta_to_json 同格式（indent=2、不转义中文、无行尾换行）
    json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path.stem, labels


def main() -> int:
    count = 0
    for json_path in sorted(OUTPUT_DIR.glob("baseline-*.json")):
        name, labels = backfill_family(json_path)
        print(f"{name}: taxonomy_labels = {labels}")
        count += 1
    print(f"共回填 {count} 个族记录")
    return 0


if __name__ == "__main__":
    sys.exit(main())
