"""基线数据集（ADR 0036/0045）：分发包展开导入与生成脚本校验断言。

不真跑九族全量生成：导入测试用合成族束充当基线源目录（v1 传输格式
json+npz，Release 资产解压后的形态，ADR 0047），校验断言直接喂构造
的束元数据。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from e2m2e.data.catalog import BASELINE_TAG, CatalogFilter, CatalogStore, import_baseline

pytestmark = pytest.mark.data


def _write_baseline_source(source, *, version: str = "1.0.0", n_members: int = 2) -> None:
    """往扁平源目录写一个合成族束（v1 传输格式，手写不做 v2 校验）。"""
    members = [
        {
            "index": i,
            "period": 6.0 + i,
            "jacobi": 3.15 - 0.01 * i,
            "taxonomy_label": "halo_l2_northern",
        }
        for i in range(n_members)
    ]
    bundle = {
        "record_id": "baseline-halo-l2",
        "schema_version": 1,
        "source_tool": "orbit_family_generation",
        "source_record_id": None,
        "classification": {
            "orbit_family": "halo",
            "libration_point": 2,
            "jacobi": [3.14, 3.15],
            "amplitude": [0.0, 0.0],
            "has_cr3bp": True,
            "has_ephemeris": False,
            "taxonomy_labels": ["halo_l2_northern"],
        },
        "status": "converged",
        "cause": "none",
        "message": "轨道族生成完成",
        "scalars": {
            "member_count": n_members,
            "requested_members": 100,
            "generated_members": n_members,
            "mu": 0.01215,
            "char_length_km": 384400.0,
            "baseline_version": version,
        },
        "request": {"orbit_type": "HALO", "libration_point": 2},
        "members": members,
        "tags": [BASELINE_TAG],
        "note": "",
        "arrays": {"cr3bp/members/0000/states": {"shape": [1, 6], "dtype": "float64"}},
    }
    arrays = {
        f"cr3bp/members/{i:04d}/states": np.full((1, 6), float(i)) for i in range(n_members)
    } | {f"cr3bp/members/{i:04d}/times": np.zeros(1) for i in range(n_members)}
    (source / "baseline-halo-l2.json").write_text(
        json.dumps(bundle, ensure_ascii=False), encoding="utf-8"
    )
    np.savez(source / "baseline-halo-l2.npz", **arrays)


@pytest.fixture
def baseline_source(tmp_path):
    source = tmp_path / "package_baseline"
    source.mkdir()
    _write_baseline_source(source)
    return source


@pytest.fixture
def store(tmp_path):
    return CatalogStore(tmp_path / "catalog")


class TestImportBaseline:
    def test_empty_catalog_expands_bundle_into_member_records(self, store, baseline_source):
        """首用导入：束展开为逐成员 v2 记录（ADR 0045 决策 8）。"""
        assert import_baseline(store, baseline_source) == 2
        summaries = store.query(CatalogFilter(family_id="baseline-halo-l2"))
        assert [s["member_index"] for s in summaries] == [0, 1]
        record = store.get("baseline-halo-l2-m0000")
        assert record.meta["scalars"]["baseline_version"] == "1.0.0"
        assert record.meta["classification"]["taxonomy_labels"] == ["halo_l2_northern"]
        assert "cr3bp/states" in record.arrays
        # 库内不落束形态：无 members 键、无打捆段
        assert "members" not in record.meta

    def test_same_version_not_reimported(self, store, baseline_source):
        """同版本幂等：跳过——用户对基线成员的删除得到尊重（不回补）。"""
        import_baseline(store, baseline_source)
        store.delete("baseline-halo-l2-m0000")
        assert import_baseline(store, baseline_source) == 0
        summaries = store.query(CatalogFilter(family_id="baseline-halo-l2"))
        assert [s["member_index"] for s in summaries] == [1]

    def test_version_mismatch_replaces_family(self, store, tmp_path):
        """版本不一致：先删旧成员再全量展开，不留陈旧尾巴。"""
        source_v1 = tmp_path / "v1"
        source_v1.mkdir()
        _write_baseline_source(source_v1, version="1.0.0", n_members=3)
        import_baseline(store, source_v1)
        assert len(store.query(CatalogFilter(family_id="baseline-halo-l2"))) == 3

        source_v2 = tmp_path / "v2"
        source_v2.mkdir()
        _write_baseline_source(source_v2, version="2.0.0", n_members=2)
        assert import_baseline(store, source_v2) == 2
        summaries = store.query(CatalogFilter(family_id="baseline-halo-l2"))
        assert [s["member_index"] for s in summaries] == [0, 1]
        assert store.get_meta("baseline-halo-l2-m0000")["scalars"]["baseline_version"] == "2.0.0"
        # 旧版本多出的成员（m0002）不残留
        assert not (store.records_dir / "baseline-halo-l2-m0002.json").exists()

    def test_missing_npz_raises(self, store, baseline_source):
        (baseline_source / "baseline-halo-l2.npz").unlink()
        with pytest.raises(FileNotFoundError, match="数据集不完整"):
            import_baseline(store, baseline_source)

    def test_empty_source_dir_is_noop(self, store, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert import_baseline(store, empty) == 0

    def test_missing_source_dir_raises(self, store, tmp_path):
        """源目录必填（ADR 0047）：路径不存在报错，不静默当空源跳过。"""
        with pytest.raises(FileNotFoundError, match="基线源目录不存在"):
            import_baseline(store, tmp_path / "no-such-release-asset")
