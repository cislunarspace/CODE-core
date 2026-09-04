"""基线分发包（ADR 0045 决策 8）：v1 族束传输格式 → v2 成员记录展开。

包内 ``catalog_baseline/`` 的族束是分发与压缩的传输单元，不是轨道
记录：JSON 元数据 + ``members`` 成员表 + ``cr3bp/members/`` 段数组，
冻结在 v1 布局。首用导入（``baseline.py``）把每束展开为逐成员的
v2 轨道记录——库内只有一种记录形态。

束的 ``record_id``（如 ``baseline-halo-l2``）即展开成员的 ``family_id``
（生成批次标识）；成员 ``record_id`` 确定性命名
（``{family_id}-m{index:04d}``），同版本重复导入零副作用。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .record import (
    geometric_amplitude_km,
    member_array_key,
    point_interval,
)

__all__ = ["expand_bundle"]


def expand_bundle(
    bundle_meta: dict[str, Any], bundle_arrays: dict[str, np.ndarray]
) -> list[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    """把一个族束展开为成员记录列表 ``(meta, arrays)``（v2，一轨一记录）。

    共享的运行级溯源（请求快照、requested/generated、mu、特征长度、
    基线版本）随每条成员记录走——同一次生成写下的同一来源，不会漂移。
    成员分类：jacobi 取成员值（单点包络），主振幅按成员存储段几何
    重算（km，多数基线族成员只存初态时为 0，参数振幅在
    ``scalars.amplitude_km``），实测标签取成员 primary。
    """
    family_id = bundle_meta["record_id"]
    classification = bundle_meta["classification"]
    shared_scalars = {
        key: value for key, value in bundle_meta.get("scalars", {}).items() if key != "member_count"
    }
    char_length_km = shared_scalars.get("char_length_km")

    records: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []
    for member in bundle_meta.get("members", []):
        index = int(member["index"])
        states = bundle_arrays[member_array_key(index, "states")]
        times = bundle_arrays[member_array_key(index, "times")]
        jacobi = member.get("jacobi")
        label = member.get("taxonomy_label")
        member_meta: dict[str, Any] = {
            "record_id": f"{family_id}-m{index:04d}",
            "source_tool": bundle_meta["source_tool"],
            "source_record_id": bundle_meta.get("source_record_id"),
            "family_id": family_id,
            "member_index": index,
            "classification": {
                "orbit_family": classification["orbit_family"],
                "libration_point": classification["libration_point"],
                "jacobi": point_interval(jacobi),
                "amplitude": point_interval(geometric_amplitude_km(states, char_length_km)),
                "has_cr3bp": True,
                "has_ephemeris": False,
                "taxonomy_labels": [label] if label else None,
            },
            # 状态三元组继承束记录：成员的来历（含软失败的族）不粉饰。
            "status": bundle_meta["status"],
            "cause": bundle_meta["cause"],
            "message": bundle_meta["message"],
            "scalars": {
                **shared_scalars,
                "period": member.get("period"),
                "closure_error": member.get("closure_error"),
                "amplitude_km": member.get("amplitude_km"),
                "amplitudes": member.get("amplitudes", {}),
                "parameters": member.get("parameters", {}),
            },
            "request": bundle_meta["request"],
            "tags": list(bundle_meta.get("tags", [])),
            "note": bundle_meta.get("note", ""),
        }
        member_arrays = {
            "cr3bp/states": np.asarray(states, dtype=float),
            "cr3bp/times": np.asarray(times, dtype=float),
        }
        records.append((member_meta, member_arrays))
    return records
