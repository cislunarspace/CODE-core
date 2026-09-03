"""分类学打标接线测试：入库打标、冲突告警、索引列、响应富化、成员提升继承。

族对象由随包 baseline 真实成员组装（单状态 + 周期，与生产 ingest 输入
同构），不依赖设计管线的重型生成。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from e2m2e.algorithm.dynamics.cr3bp_system import CR3BP_System
from e2m2e.api.catalog import _family_generation_payload
from e2m2e.api.catalog_ingest import build_family_record
from e2m2e.api.models import FamilyGenerationRequest
from e2m2e.data.catalog import CatalogFilter, CatalogStore
from e2m2e.data.constants import Datum
from e2m2e.data.templates import ConvergenceState, FailureCause
from e2m2e.data.types.orbit import Orbit, OrbitFamily

pytestmark = pytest.mark.interface

BASELINE_DIR = Path(__file__).resolve().parents[2] / "e2m2e" / "data" / "catalog_baseline"


def _family_from_baseline(
    name: str, family_type: str, max_members: int = 3
) -> tuple[OrbitFamily, FamilyGenerationRequest]:
    """用 baseline 成员组装族对象与对应请求模型。"""
    meta = json.loads((BASELINE_DIR / f"baseline-{name}.json").read_text(encoding="utf-8"))
    bundle = np.load(BASELINE_DIR / f"baseline-{name}.npz")
    system = CR3BP_System(mu=Datum.DE421.mu, primary="earth", secondary="moon")
    orbits: list[Orbit] = []
    for i, member in enumerate(meta["members"][:max_members]):
        state = bundle[f"cr3bp/members/{i:04d}/states"][0]
        orbit = Orbit(states=state.reshape(1, -1), times=np.array([0.0]), system=system)
        orbit.period = member["period"]
        orbit.parameters = dict(member.get("parameters", {}))
        orbits.append(orbit)
    family = OrbitFamily(orbits=orbits, family_type=family_type, system=system)
    family.metadata = {"periodicity": meta["scalars"].get("periodicity", "periodic")}
    return family, FamilyGenerationRequest(orbit_type=family_type.upper())


def _record(family: OrbitFamily, request: FamilyGenerationRequest):
    built = build_family_record(
        request,
        family=family,
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="ok",
        requested_members=len(family),
        generated_members=len(family),
    )
    assert built is not None
    return built


def test_family_record_stamps_dro():
    family, request = _family_from_baseline("dro", "dro")
    meta, _ = _record(family, request)
    assert meta["classification"]["taxonomy_labels"] == ["distant_retrograde"]
    assert all(m["taxonomy_label"] == "distant_retrograde" for m in meta["members"])


def test_family_record_quasi_periodic_family_gets_empty_labels():
    family, request = _family_from_baseline("lissajous-l1", "lissajous")
    meta, _ = _record(family, request)
    assert meta["classification"]["taxonomy_labels"] == []
    assert all(m["taxonomy_label"] is None for m in meta["members"])


def test_family_record_conflict_warns_and_keeps_measured(caplog):
    """设计侧族与实测不符：记 warning，按实测值入库（两边都保留）。"""
    family, request = _family_from_baseline("halo-l1", "dro")  # 谎报族：期望 distant_retrograde
    with caplog.at_level(logging.WARNING, logger="e2m2e.api.catalog_ingest"):
        meta, _ = _record(family, request)
    assert meta["classification"]["taxonomy_labels"] == ["halo_l1_northern"]
    assert any("分类学冲突" in record.message for record in caplog.records)


def test_family_payload_enrichment():
    family, _ = _family_from_baseline("spo-l4", "spo")
    response = _family_generation_payload(family)
    assert response.taxonomy_labels == ["shortperiod_l4"]


def test_store_roundtrip_and_promote(tmp_path):
    """入库 → 索引摘要带标签；成员提升继承成员级标签。"""
    family, request = _family_from_baseline("dro", "dro")
    meta, arrays = _record(family, request)
    store = CatalogStore(tmp_path / "catalog")
    record_id = store.put(meta, arrays)
    summaries = store.query(CatalogFilter())
    assert len(summaries) == 1
    assert summaries[0]["classification"]["taxonomy_labels"] == ["distant_retrograde"]

    promoted = store.promote_member(record_id, 0)
    assert promoted.meta["classification"]["taxonomy_labels"] == ["distant_retrograde"]
    store.close()


def test_summary_without_taxonomy_key_reads_as_none(tmp_path):
    """未打标的旧记录（classification 缺 taxonomy_labels 键）可读且摘要为 None。"""
    family, request = _family_from_baseline("dro", "dro")
    meta, arrays = _record(family, request)
    del meta["classification"]["taxonomy_labels"]  # 模拟旧 schema 记录
    store = CatalogStore(tmp_path / "catalog")
    store.put(meta, arrays)
    summaries = store.query(CatalogFilter())
    assert summaries[0]["classification"]["taxonomy_labels"] is None
    store.close()
