"""轨道库存储引擎测试：读写往返、双段、族标签、索引重建、删除、标注、导出。

只断外部行为（库中出现了什么记录、查询返回什么），不断实现细节
（SQLite 表结构、JSON 内部布局、索引列）。
"""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest

from data.catalog.conftest import make_record
from e2m2e.data.catalog import (
    CatalogError,
    CatalogFilter,
    CatalogStore,
    RecordNotFoundError,
)

pytestmark = pytest.mark.data


@pytest.fixture
def store(tmp_path):
    return CatalogStore(tmp_path / "catalog")


class TestRoundTrip:
    def test_put_then_get_returns_same_record(self, store):
        meta, arrays = make_record()
        record_id = store.put(meta, arrays)

        record = store.get(record_id)
        assert record.meta["record_id"] == record_id
        assert record.meta["classification"]["orbit_family"] == "nrho"
        assert record.meta["classification"]["jacobi"] == [3.05, 3.05]
        np.testing.assert_array_equal(record.arrays["cr3bp/states"], arrays["cr3bp/states"])
        np.testing.assert_array_equal(record.arrays["eph/position_km"], arrays["eph/position_km"])

    def test_schema_version_is_two(self, store):
        """schema v2（ADR 0045：一轨一记录）。"""
        record_id = store.put(*make_record())
        assert store.get(record_id).meta["schema_version"] == 2

    def test_record_files_exist_on_disk(self, store):
        record_id = store.put(*make_record())
        assert (store.records_dir / f"{record_id}.json").exists()
        assert (store.records_dir / f"{record_id}.npz").exists()

    def test_get_unknown_record_raises_structured_error(self, store):
        with pytest.raises(RecordNotFoundError):
            store.get("no-such-record")

    @pytest.mark.parametrize("bad_id", ["../../etc/passwd", "../x", "a/b", "..", ".hidden"])
    def test_path_traversal_record_id_is_rejected(self, store, bad_id):
        """record_id 直接拼文件路径：非法形态一律按记录不存在拒绝。"""
        with pytest.raises(RecordNotFoundError):
            store.get_meta(bad_id)
        with pytest.raises(RecordNotFoundError):
            store.delete(bad_id)
        with pytest.raises(RecordNotFoundError):
            store.tag(bad_id, ["t"])

    def test_missing_npz_with_array_pointer_raises(self, store):
        """元数据声称有数组段而 NPZ 缺失：记录损坏，抛错而非返回半成品。"""
        record_id = store.put(*make_record())
        (store.records_dir / f"{record_id}.npz").unlink()
        with pytest.raises(CatalogError, match="数组段文件缺失"):
            store.get(record_id)


class TestSegments:
    def test_design_record_keeps_both_segments(self, store):
        record_id = store.put(*make_record(with_cr3bp=True, with_ephemeris=True))
        record = store.get(record_id)
        assert record.meta["classification"]["has_cr3bp"] is True
        assert record.meta["classification"]["has_ephemeris"] is True
        assert "cr3bp/states" in record.arrays
        assert "eph/position_km" in record.arrays

    def test_control_record_keeps_ephemeris_only(self, store):
        record_id = store.put(
            *make_record(
                with_cr3bp=False,
                with_ephemeris=True,
                jacobi=None,
                amplitude=None,
                source_tool="control_orbit",
            )
        )
        record = store.get(record_id)
        assert record.meta["classification"]["has_cr3bp"] is False
        assert record.meta["classification"]["has_ephemeris"] is True
        assert not any(key.startswith("cr3bp/") for key in record.arrays)
        assert "eph/position_km" in record.arrays


class TestFamilyLabels:
    """族是标签而非容器（ADR 0045 决策 2）：成员各自成记录，整族按
    family_id 过滤查询。"""

    def test_family_members_are_individual_records(self, store):
        for index in range(2):
            store.put(
                *make_record(
                    orbit_family="halo",
                    jacobi=(3.1 - 0.1 * index, 3.1 - 0.1 * index),
                    with_ephemeris=False,
                    source_tool="orbit_family_generation",
                    family_id="run-42",
                    member_index=index,
                )
            )

        summaries = store.query(CatalogFilter(family_id="run-42"))
        assert len(summaries) == 2
        assert sorted(s["member_index"] for s in summaries) == [0, 1]
        assert all(s["classification"]["orbit_family"] == "halo" for s in summaries)
        # 族标签不溢出到其他维度：不过滤时也只这两条，无打捆记录
        assert len(store.query(CatalogFilter())) == 2

    def test_single_records_carry_no_family_labels(self, store):
        record_id = store.put(*make_record())
        summary = store.query(CatalogFilter())[0]
        assert summary["family_id"] is None
        assert summary["member_index"] is None
        assert store.get(record_id).meta["family_id"] is None


class TestQuery:
    @pytest.fixture
    def populated(self, store):
        store.put(
            *make_record(
                orbit_family="nrho",
                libration_point=2,
                jacobi=(3.05, 3.05),
                amplitude=(65000.0, 65000.0),
            )
        )
        store.put(
            *make_record(
                orbit_family="halo",
                libration_point=1,
                jacobi=(3.10, 3.10),
                amplitude=(1000.0, 3000.0),
                with_ephemeris=False,
                source_tool="orbit_family_generation",
                family_id="run-1",
                member_index=0,
            )
        )
        store.put(
            *make_record(
                orbit_family=None,
                libration_point=None,
                jacobi=None,
                amplitude=None,
                with_cr3bp=False,
                source_tool="control_orbit",
                status="failed",
                cause="unknown",
                message="全部样本失败",
            )
        )
        return store

    def test_filter_by_orbit_family(self, populated):
        summaries = populated.query(CatalogFilter(orbit_family="nrho"))
        assert len(summaries) == 1
        assert summaries[0]["classification"]["orbit_family"] == "nrho"

    def test_filter_by_libration_point(self, populated):
        summaries = populated.query(CatalogFilter(libration_point=1))
        assert len(summaries) == 1
        assert summaries[0]["classification"]["orbit_family"] == "halo"

    def test_filter_by_jacobi_range_overlap(self, populated):
        assert len(populated.query(CatalogFilter(jacobi_min=3.0, jacobi_max=3.12))) == 2
        # v2 成员 jacobi 是单点包络：边界等值命中（ADR 0045）
        assert len(populated.query(CatalogFilter(jacobi_min=3.10, jacobi_max=3.10))) == 1
        assert len(populated.query(CatalogFilter(jacobi_min=2.0, jacobi_max=2.5))) == 0

    def test_filter_by_amplitude_range_overlap(self, populated):
        assert len(populated.query(CatalogFilter(amplitude_min_km=60000.0))) == 1
        assert len(populated.query(CatalogFilter(amplitude_max_km=5000.0))) == 1

    def test_filter_by_segment_presence(self, populated):
        assert len(populated.query(CatalogFilter(has_ephemeris=True))) == 2
        assert len(populated.query(CatalogFilter(has_cr3bp=True))) == 2
        assert len(populated.query(CatalogFilter(has_cr3bp=False))) == 1

    def test_filter_by_status(self, populated):
        summaries = populated.query(CatalogFilter(status="failed"))
        assert len(summaries) == 1
        assert summaries[0]["source_tool"] == "control_orbit"

    def test_combined_filter(self, populated):
        summaries = populated.query(
            CatalogFilter(
                orbit_family="halo",
                libration_point=1,
                jacobi_min=3.0,
                jacobi_max=3.1,
                has_cr3bp=True,
                has_ephemeris=False,
            )
        )
        assert len(summaries) == 1

    def test_filter_by_tags(self, store):
        record_id = store.put(*make_record())
        store.put(*make_record())
        store.tag(record_id, ["第3周", "nrho-案例"])

        summaries = store.query(CatalogFilter(tags=["第3周"]))
        assert len(summaries) == 1
        assert summaries[0]["record_id"] == record_id

    def test_summary_carries_no_arrays(self, populated):
        summaries = populated.query(CatalogFilter())
        assert len(summaries) == 3
        for summary in summaries:
            assert "arrays" not in summary
            assert "request" not in summary


class TestIndexRebuild:
    def test_rebuild_after_db_deletion_preserves_query_results(self, store):
        store.put(*make_record(orbit_family="nrho", libration_point=2))
        store.put(*make_record(orbit_family="halo", libration_point=1, with_ephemeris=False))
        before = store.query(CatalogFilter())

        store.close()
        store.db_path.unlink()
        rebuilt = CatalogStore(store.root)
        after = rebuilt.query(CatalogFilter())

        assert [s["record_id"] for s in after] == [s["record_id"] for s in before]
        assert (
            rebuilt.query(CatalogFilter(orbit_family="halo"))[0]["record_id"]
            == (before[1]["record_id"])
        )

    def test_legacy_index_schema_is_rebuilt_on_open(self, store):
        """存量库索引表缺列（如 #574 前的旧 schema）时，打开即废弃重建。

        记录文件是事实来源：索引表结构演进不丢记录、查询结果不变。"""
        import sqlite3

        store.put(*make_record(orbit_family="nrho", libration_point=2))
        record_id = store.put(*make_record(orbit_family="halo", libration_point=1))
        before = store.query(CatalogFilter())

        store.close()
        conn = sqlite3.connect(store.db_path)
        conn.execute("DROP TABLE records")
        # 伪造旧 schema：只有 #574 之前的列
        conn.execute(
            "CREATE TABLE records ("
            "record_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, source_tool TEXT NOT NULL)"
        )
        conn.commit()
        conn.close()

        reopened = CatalogStore(store.root)
        after = reopened.query(CatalogFilter())

        assert [s["record_id"] for s in after] == [s["record_id"] for s in before]
        assert reopened.query(CatalogFilter(orbit_family="halo"))[0]["record_id"] == record_id

    def test_stale_v1_record_error_names_file_and_remedy(self, store):
        """v1 旧记录（ADR 0045 前的库产物）打开即失败，报错带文件名与
        删除重算指引（ADR 0045 实现注承诺的措辞）。"""
        store.put(*make_record())
        stale = {
            "schema_version": 1,
            "record_id": "stale-v1",
            "classification": {"orbit_family": "halo", "libration_point": 2},
        }
        (store.records_dir / "stale-v1.json").write_text(
            json.dumps(stale, ensure_ascii=False), encoding="utf-8"
        )
        store.close()
        store.db_path.unlink()

        with pytest.raises(CatalogError) as excinfo:
            CatalogStore(store.root)

        message = str(excinfo.value)
        assert "stale-v1.json" in message
        assert "删除后重算" in message

    def test_corrupted_db_raises_structured_catalog_error(self, tmp_path):
        root = tmp_path / "catalog"
        (root / "records").mkdir(parents=True)
        (root / "catalog.db").write_bytes(b"not a sqlite database")
        with pytest.raises(CatalogError):
            CatalogStore(root)


class TestDelete:
    def test_delete_removes_files_and_index_entry(self, store):
        record_id = store.put(*make_record())
        store.delete(record_id)

        assert not (store.records_dir / f"{record_id}.json").exists()
        assert not (store.records_dir / f"{record_id}.npz").exists()
        assert store.query(CatalogFilter()) == []
        with pytest.raises(RecordNotFoundError):
            store.get(record_id)

    def test_delete_unknown_record_raises(self, store):
        with pytest.raises(RecordNotFoundError):
            store.delete("no-such-record")


class TestTag:
    def test_tag_is_written_to_record_json_not_only_index(self, store):
        record_id = store.put(*make_record())
        store.tag(record_id, ["期中案例"], note="注意近月点高度")

        with open(store.records_dir / f"{record_id}.json", encoding="utf-8") as f:
            on_disk = json.load(f)
        assert on_disk["tags"] == ["期中案例"]
        assert on_disk["note"] == "注意近月点高度"

        record = store.get(record_id)
        assert record.meta["tags"] == ["期中案例"]
        assert record.meta["note"] == "注意近月点高度"

    def test_tag_none_note_keeps_existing(self, store):
        record_id = store.put(*make_record(note="旧注释"))
        store.tag(record_id, ["t"])
        assert store.get(record_id).meta["note"] == "旧注释"

    def test_tag_unknown_record_raises(self, store):
        with pytest.raises(RecordNotFoundError):
            store.tag("no-such-record", ["t"])


class TestExport:
    def test_export_subset_package_is_self_contained(self, store, tmp_path):
        kept_id = store.put(*make_record(orbit_family="nrho", tags=["教学"]))
        store.put(*make_record(orbit_family="halo"))
        dest = tmp_path / "export"

        exported = store.export(CatalogFilter(tags=["教学"]), dest)

        assert exported == [kept_id]
        assert (dest / "records" / f"{kept_id}.json").exists()
        assert (dest / "records" / f"{kept_id}.npz").exists()
        # 标注随包走
        with open(dest / "records" / f"{kept_id}.json", encoding="utf-8") as f:
            assert json.load(f)["tags"] == ["教学"]
        # 导出包可直接作为库打开（索引派生重建）
        reopened = CatalogStore(dest)
        assert [s["record_id"] for s in reopened.query(CatalogFilter())] == [kept_id]


class TestThreadSafety:
    """跨线程复用（#559）：mcp-serve 逐 tools/call 换线程池线程，索引连接
    由首用线程创建、后续调用线程复用——不得报 SQLite 跨线程错误。"""

    def test_query_from_other_thread_after_first_touch(self, store):
        store.put(*make_record())

        results: list[int] = []
        errors: list[Exception] = []

        def query_from_other_thread() -> None:
            try:
                results.append(len(store.query(CatalogFilter())))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        thread = threading.Thread(target=query_from_other_thread)
        thread.start()
        thread.join()

        assert errors == []
        assert results == [1]

    def test_concurrent_puts_are_serialized(self, store):
        """并发 put：RLock 串行化索引访问，全部记录可见。"""
        threads = [
            threading.Thread(target=store.put, args=make_record(tags=[f"t{i}"])) for i in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(store.query(CatalogFilter())) == 8
