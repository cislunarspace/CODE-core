"""基线数据集（ADR 0036）：首用导入逻辑与生成脚本校验断言。

不真跑九族全量生成：导入测试用合成记录充当包内基线源目录（扁平
json+npz），校验断言直接喂构造的记录元数据。
"""

from __future__ import annotations

import pytest

from data.catalog.conftest import make_record
from e2m2e.data.catalog import BASELINE_TAG, CatalogFilter, CatalogStore, import_baseline

pytestmark = pytest.mark.data


def _write_baseline_source(source, *, version: str = "1.0.0") -> None:
    """往扁平源目录写一条合成基线记录（复用存储引擎做序列化）。"""
    stage = CatalogStore(source.parent / "stage")
    meta, arrays = make_record(
        orbit_family="halo",
        libration_point=2,
        with_ephemeris=False,
        members=[{"index": 0, "period": 6.0}, {"index": 1, "period": 6.5}],
        tags=[BASELINE_TAG],
    )
    meta["record_id"] = "baseline-halo-l2"
    meta["scalars"]["baseline_version"] = version
    record_id = stage.put(meta, arrays)
    for suffix in (".json", ".npz"):
        content = (stage.records_dir / f"{record_id}{suffix}").read_bytes()
        (source / f"baseline-halo-l2{suffix}").write_bytes(content)


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
    def test_empty_catalog_imports_baseline(self, store, baseline_source):
        assert import_baseline(store, baseline_source) == 1
        summaries = store.query(CatalogFilter(tags=(BASELINE_TAG,)))
        assert [s["record_id"] for s in summaries] == ["baseline-halo-l2"]
        record = store.get("baseline-halo-l2")
        assert record.meta["scalars"]["baseline_version"] == "1.0.0"
        assert "cr3bp/members/0000/states" in record.arrays

    def test_same_version_not_reimported(self, store, baseline_source):
        import_baseline(store, baseline_source)
        before = (store.records_dir / "baseline-halo-l2.json").read_bytes()
        assert import_baseline(store, baseline_source) == 0
        assert (store.records_dir / "baseline-halo-l2.json").read_bytes() == before

    def test_version_mismatch_reimports(self, store, tmp_path):
        old_source = tmp_path / "old"
        old_source.mkdir()
        _write_baseline_source(old_source, version="1.0.0")
        import_baseline(store, old_source)

        upgraded = tmp_path / "upgraded"
        upgraded.mkdir()
        _write_baseline_source(upgraded, version="2.0.0")
        assert import_baseline(store, upgraded) == 1
        assert store.get("baseline-halo-l2").meta["scalars"]["baseline_version"] == "2.0.0"

    def test_missing_source_dir_is_noop(self, store, tmp_path):
        assert import_baseline(store, tmp_path / "no-such-dir") == 0
        assert store.query(CatalogFilter(tags=(BASELINE_TAG,))) == []


class TestValidateBaselineRecord:
    @staticmethod
    def _meta(**overrides):
        meta, _arrays = make_record(
            members=[{"index": 0, "period": 6.0}], with_ephemeris=False, tags=[BASELINE_TAG]
        )
        meta["record_id"] = "baseline-test"
        meta["scalars"].update(
            member_count=1,
            requested_members=100,
            generated_members=1,
            baseline_version="1.0.0",
            amplitude_envelope_km=[1000.0, 50000.0],
        )
        meta.update(overrides)
        return meta

    def test_valid_record_passes(self):
        from scripts.generate_catalog_baseline import validate_baseline_record

        validate_baseline_record(self._meta())

    @pytest.mark.parametrize(
        ("overrides", "reason"),
        [
            ({"status": "failed"}, "status"),
            ({"members": []}, "零成员"),
            ({"tags": []}, "baseline 标签"),
            ({"message": ""}, "message"),
            (
                {"classification": {**make_record()[0]["classification"], "amplitude": None}},
                "振幅",
            ),
        ],
    )
    def test_invalid_record_raises(self, overrides, reason):
        from scripts.generate_catalog_baseline import validate_baseline_record

        with pytest.raises(ValueError, match=reason):
            validate_baseline_record(self._meta(**overrides))

    def test_missing_baseline_version_raises(self):
        from scripts.generate_catalog_baseline import validate_baseline_record

        meta = self._meta()
        del meta["scalars"]["baseline_version"]
        with pytest.raises(ValueError, match="基线版本号"):
            validate_baseline_record(meta)

    @pytest.mark.parametrize("envelope", [None, [0.0, 0.0], [50000.0, 1000.0]])
    def test_degenerate_amplitude_envelope_raises(self, envelope):
        from scripts.generate_catalog_baseline import validate_baseline_record

        meta = self._meta()
        if envelope is None:
            del meta["scalars"]["amplitude_envelope_km"]
        else:
            meta["scalars"]["amplitude_envelope_km"] = envelope
        with pytest.raises(ValueError, match="参数振幅"):
            validate_baseline_record(meta)
