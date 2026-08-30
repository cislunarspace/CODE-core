"""catalog SQLite 派生索引：只存过滤维度与文件指针，删除后可全量重建。

记录文件（records/*.json）是事实来源；本索引只是查询加速（ADR 0031
决策 5）。索引行与记录一一对应，表结构是实现细节，不作为外部契约。
"""

from __future__ import annotations

import functools
import json
import sqlite3
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .record import CatalogError, CatalogFilter, numeric_or_none
from .record import member_count as _member_count

__all__ = ["CatalogIndex"]

_SCHEMA = """
CREATE TABLE records (
    record_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    source_tool TEXT NOT NULL,
    source_record_id TEXT,
    orbit_family TEXT,
    libration_point INTEGER,
    jacobi_min REAL,
    jacobi_max REAL,
    amplitude_min REAL,
    amplitude_max REAL,
    has_cr3bp INTEGER NOT NULL,
    has_ephemeris INTEGER NOT NULL,
    taxonomy_labels TEXT,
    transfer_type TEXT,
    delta_v REAL,
    tli_epoch REAL,
    status TEXT NOT NULL,
    cause TEXT NOT NULL,
    message TEXT NOT NULL,
    member_count INTEGER NOT NULL,
    tags TEXT NOT NULL,
    note TEXT NOT NULL
)
"""

#: 表结构完备性检查列集：存量库缺任一列（如 #574 新增 transfer 维度、
#: #581 新增分类学标签维度）时整个表废弃重建——索引是派生物（ADR 0031
#: 决策 5），重建由 ``CatalogStore`` 依据 :attr:`CatalogIndex.schema_reset`
#: 标记触发。
_REQUIRED_COLUMNS = frozenset(
    {
        "record_id",
        "created_at",
        "source_tool",
        "source_record_id",
        "orbit_family",
        "libration_point",
        "jacobi_min",
        "jacobi_max",
        "amplitude_min",
        "amplitude_max",
        "has_cr3bp",
        "has_ephemeris",
        "taxonomy_labels",
        "transfer_type",
        "delta_v",
        "tli_epoch",
        "status",
        "cause",
        "message",
        "member_count",
        "tags",
        "note",
    }
)


def _guard(method: Any) -> Any:
    """把 sqlite3 错误翻译为 :class:`CatalogError`（错误在数据层边界结构化）。"""

    @functools.wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return method(*args, **kwargs)
        except sqlite3.Error as exc:
            raise CatalogError(f"索引操作失败：{exc}") from exc

    return wrapper


class CatalogIndex:
    """catalog.db 的封装： upsert / delete / query / rebuild。

    线程契约：mcp-serve 把每个 ``tools/call`` 丢进 anyio 线程池，调用之间
    不保证同线程，而连接由 CatalogStore 惰性创建、跨调用复用（#559）——
    因此连接不绑定创建线程（``check_same_thread=False``），全部访问由
    ``RLock`` 串行化（SQLite C 层本身线程安全，外部串行即可；锁须可重入，
    ``rebuild`` 内部复调 ``upsert``）。retain 线程池并发收益：长计算与快速
    查询可在不同线程重叠，只有落到本索引的操作互斥。
    """

    @_guard
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.schema_reset = False
        if not self._has_table():
            self._conn.execute(_SCHEMA)
            self._conn.commit()
        elif self._missing_columns():
            self._conn.execute("DROP TABLE records")
            self._conn.execute(_SCHEMA)
            self._conn.commit()
            self.schema_reset = True

    def _missing_columns(self) -> list[str]:
        """现有表缺失的必备列（存量库 schema 演进检测）。"""
        rows = self._conn.execute("PRAGMA table_info(records)").fetchall()
        names = {row["name"] for row in rows}
        return sorted(_REQUIRED_COLUMNS - names)

    def _has_table(self) -> bool:
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='records'"
        ).fetchone()
        return row is not None

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @_guard
    def upsert(self, meta: dict[str, Any]) -> None:
        """按记录元数据写入或更新索引行。"""
        classification = meta["classification"]
        jacobi = classification["jacobi"]
        amplitude = classification["amplitude"]
        member_count = _member_count(meta)
        # transfer 维度（#574）取自 scalars；数值历元才入区间列
        scalars = meta.get("scalars", {})
        transfer_type = scalars.get("transfer_type")
        delta_v = numeric_or_none(scalars.get("delta_v_km_s"))
        tli_epoch = numeric_or_none(scalars.get("tli_epoch"))
        taxonomy_labels = classification.get("taxonomy_labels")
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO records (
                    record_id, created_at, source_tool, source_record_id,
                    orbit_family, libration_point, jacobi_min, jacobi_max,
                    amplitude_min, amplitude_max, has_cr3bp, has_ephemeris,
                    taxonomy_labels, transfer_type, delta_v, tli_epoch,
                    status, cause, message, member_count, tags, note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    meta["record_id"],
                    meta["created_at"],
                    meta["source_tool"],
                    meta["source_record_id"],
                    classification["orbit_family"],
                    classification["libration_point"],
                    None if jacobi is None else float(jacobi[0]),
                    None if jacobi is None else float(jacobi[1]),
                    None if amplitude is None else float(amplitude[0]),
                    None if amplitude is None else float(amplitude[1]),
                    1 if classification["has_cr3bp"] else 0,
                    1 if classification["has_ephemeris"] else 0,
                    None if not taxonomy_labels else ",".join(taxonomy_labels),
                    transfer_type,
                    delta_v,
                    tli_epoch,
                    meta["status"],
                    meta["cause"],
                    meta["message"],
                    member_count,
                    json.dumps(list(meta["tags"]), ensure_ascii=False),
                    meta["note"],
                ),
            )
            self._conn.commit()

    @_guard
    def delete(self, record_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM records WHERE record_id = ?", (record_id,))
            self._conn.commit()

    @_guard
    def rebuild(self, metas: Iterable[dict[str, Any]]) -> None:
        """清空并全量重建索引。"""
        with self._lock:
            self._conn.execute("DROP TABLE IF EXISTS records")
            self._conn.execute(_SCHEMA)
            for meta in metas:
                self.upsert(meta)
            self._conn.commit()

    @_guard
    def query(self, catalog_filter: CatalogFilter) -> list[dict[str, Any]]:
        """按多维过滤返回摘要列表（不含数组与请求快照）。

        区间维度做相交匹配；``tags`` 维度命中任一即匹配（在 Python 侧
        过滤，tags 在索引中以 JSON 文本存储）。
        """
        clauses: list[str] = []
        params: list[Any] = []
        if catalog_filter.orbit_family is not None:
            clauses.append("orbit_family = ?")
            params.append(catalog_filter.orbit_family)
        if catalog_filter.libration_point is not None:
            clauses.append("libration_point = ?")
            params.append(catalog_filter.libration_point)
        if catalog_filter.jacobi_min is not None:
            clauses.append("jacobi_max >= ?")
            params.append(catalog_filter.jacobi_min)
        if catalog_filter.jacobi_max is not None:
            clauses.append("jacobi_min <= ?")
            params.append(catalog_filter.jacobi_max)
        if catalog_filter.amplitude_min_km is not None:
            clauses.append("amplitude_max >= ?")
            params.append(catalog_filter.amplitude_min_km)
        if catalog_filter.amplitude_max_km is not None:
            clauses.append("amplitude_min <= ?")
            params.append(catalog_filter.amplitude_max_km)
        if catalog_filter.has_cr3bp is not None:
            clauses.append("has_cr3bp = ?")
            params.append(1 if catalog_filter.has_cr3bp else 0)
        if catalog_filter.has_ephemeris is not None:
            clauses.append("has_ephemeris = ?")
            params.append(1 if catalog_filter.has_ephemeris else 0)
        if catalog_filter.transfer_type is not None:
            clauses.append("transfer_type = ?")
            params.append(catalog_filter.transfer_type)
        if catalog_filter.delta_v_min_km_s is not None:
            clauses.append("delta_v >= ?")
            params.append(catalog_filter.delta_v_min_km_s)
        if catalog_filter.delta_v_max_km_s is not None:
            clauses.append("delta_v <= ?")
            params.append(catalog_filter.delta_v_max_km_s)
        if catalog_filter.tli_epoch_min is not None:
            clauses.append("tli_epoch >= ?")
            params.append(catalog_filter.tli_epoch_min)
        if catalog_filter.tli_epoch_max is not None:
            clauses.append("tli_epoch <= ?")
            params.append(catalog_filter.tli_epoch_max)
        if catalog_filter.status is not None:
            clauses.append("status = ?")
            params.append(catalog_filter.status)

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM records{where} ORDER BY created_at, record_id", params
            ).fetchall()

        summaries = [self._row_to_summary(row) for row in rows]
        if catalog_filter.tags:
            wanted = set(catalog_filter.tags)
            summaries = [s for s in summaries if wanted & set(s["tags"])]
        return summaries

    @staticmethod
    def _row_to_summary(row: sqlite3.Row) -> dict[str, Any]:
        jacobi = None if row["jacobi_min"] is None else [row["jacobi_min"], row["jacobi_max"]]
        amplitude = (
            None if row["amplitude_min"] is None else [row["amplitude_min"], row["amplitude_max"]]
        )
        taxonomy_text = row["taxonomy_labels"]
        return {
            "record_id": row["record_id"],
            "created_at": row["created_at"],
            "source_tool": row["source_tool"],
            "source_record_id": row["source_record_id"],
            "classification": {
                "orbit_family": row["orbit_family"],
                "libration_point": row["libration_point"],
                "jacobi": jacobi,
                "amplitude": amplitude,
                "has_cr3bp": bool(row["has_cr3bp"]),
                "has_ephemeris": bool(row["has_ephemeris"]),
                "taxonomy_labels": None if not taxonomy_text else taxonomy_text.split(","),
            },
            "transfer_type": row["transfer_type"],
            "delta_v_km_s": row["delta_v"],
            "tli_epoch": row["tli_epoch"],
            "status": row["status"],
            "cause": row["cause"],
            "message": row["message"],
            "member_count": row["member_count"],
            "tags": json.loads(row["tags"]),
            "note": row["note"],
        }
