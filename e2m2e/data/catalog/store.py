"""轨道库存储引擎：记录文件读写、删除、标注、导出、成员提升与索引重建。

存储布局（ADR 0031 决策 5）::

    catalog/
    ├── records/<record_id>.json + <record_id>.npz   # 事实来源
    └── catalog.db                                    # SQLite 索引，派生物

库目录经 Config 注入。catalog.db 缺失时（新库、被删除、导出包首次打开）
扫描 records/ 全量重建。
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np

from .index import CatalogIndex
from .record import (
    SCHEMA_VERSION,
    CatalogError,
    CatalogFilter,
    CatalogRecord,
    RecordNotFoundError,
    cr3bp_segment_arrays,
    geometric_amplitude_km,
    member_array_key,
    meta_from_json,
    meta_to_json,
    new_record_id,
    point_interval,
    validate_meta,
    validate_record_id,
)

__all__ = ["CatalogStore"]


class CatalogStore:
    """一个库目录的读写入口。"""

    def __init__(self, root: str | Path) -> None:
        """打开（必要时创建）库目录；catalog.db 缺失时全量重建索引。"""
        self.root = Path(root)
        self.records_dir = self.root / "records"
        self.db_path = self.root / "catalog.db"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        rebuild = not self.db_path.exists()
        self._index = CatalogIndex(self.db_path)
        if rebuild:
            self.rebuild_index()

    # ---- 写入 ----

    def put(self, meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> str:
        """写入一条记录（JSON + NPZ + 索引行），返回 record_id。

        ``schema_version`` / ``record_id`` / ``created_at`` 由引擎填写；
        ``arrays`` 的数据指针（键、形状、dtype）写入元数据 ``arrays`` 键。
        """
        meta = dict(meta)
        meta.setdefault("schema_version", SCHEMA_VERSION)
        meta.setdefault("record_id", new_record_id())
        try:
            validate_record_id(meta["record_id"])
        except RecordNotFoundError as exc:
            raise CatalogError(f"非法 record_id：{meta['record_id']!r}") from exc
        meta.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        meta["arrays"] = {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        }
        validate_meta(meta)
        record_id = meta["record_id"]

        npz_tmp = self.records_dir / f".{record_id}.tmp.npz"
        # savez 桩签名混有 allow_pickle: bool 与 **kwds: ArrayLike，**dict 展开
        # 需经 cast 消除与 bool 形参的冲突。
        np.savez(npz_tmp, **cast(dict[str, Any], arrays))
        os.replace(npz_tmp, self.records_dir / f"{record_id}.npz")

        json_tmp = self.records_dir / f".{record_id}.json.tmp"
        with open(json_tmp, "w", encoding="utf-8") as handle:
            handle.write(meta_to_json(meta))
        os.replace(json_tmp, self.records_dir / f"{record_id}.json")

        self._index.upsert(meta)
        return record_id

    # ---- 读取 ----

    def get(self, record_id: str) -> CatalogRecord:
        """按 record_id 取完整记录（含数组段）。

        元数据声称有数组段而 NPZ 缺失时抛 :class:`CatalogError`（记录
        损坏），不静默返回半成品。
        """
        meta = self.get_meta(record_id)
        npz_path = self.records_dir / f"{record_id}.npz"
        arrays: dict[str, np.ndarray] = {}
        if npz_path.exists():
            with np.load(npz_path) as bundle:
                arrays = {key: bundle[key] for key in bundle.files}
        elif meta["arrays"]:
            raise CatalogError(f"记录 {record_id} 的数组段文件缺失：{npz_path.name}")
        return CatalogRecord(meta=meta, arrays=arrays)

    def get_meta(self, record_id: str) -> dict[str, Any]:
        """按 record_id 取记录元数据（不读数组）。"""
        json_path = self.records_dir / f"{validate_record_id(record_id)}.json"
        if not json_path.exists():
            raise RecordNotFoundError(f"记录不存在：{record_id}")
        return meta_from_json(json_path.read_text(encoding="utf-8"))

    def query(self, catalog_filter: CatalogFilter) -> list[dict[str, Any]]:
        """多维过滤，返回摘要列表（不含数组段与请求快照）。"""
        return self._index.query(catalog_filter)

    # ---- 删除 ----

    def delete(self, record_id: str) -> None:
        """删除记录文件与索引条目。"""
        json_path = self.records_dir / f"{validate_record_id(record_id)}.json"
        if not json_path.exists():
            raise RecordNotFoundError(f"记录不存在：{record_id}")
        json_path.unlink()
        npz_path = self.records_dir / f"{record_id}.npz"
        if npz_path.exists():
            npz_path.unlink()
        self._index.delete(record_id)

    # ---- 标注 ----

    def tag(self, record_id: str, tags: list[str], note: str | None = None) -> dict[str, Any]:
        """写标注入 JSON 记录并更新索引；``note=None`` 保留原注释。"""
        meta = self.get_meta(record_id)
        meta["tags"] = list(tags)
        if note is not None:
            meta["note"] = note
        json_path = self.records_dir / f"{record_id}.json"
        json_path.write_text(meta_to_json(meta), encoding="utf-8")
        self._index.upsert(meta)
        return meta

    # ---- 族成员提升 ----

    def promote_member(self, record_id: str, member_index: int) -> CatalogRecord:
        """把族成员提升为独立记录（``source_record_id`` 指向所属族）。

        提升记录只含该成员的 CR3BP 段；分类继承族记录，jacobi 取成员
        值，主振幅按成员状态与族特征长度重算（km）。
        """
        family = self.get(record_id)
        members = family.meta["members"]
        if member_index < 0 or member_index >= len(members):
            raise RecordNotFoundError(
                f"族记录 {record_id} 没有成员 {member_index}（共 {len(members)} 个）"
            )
        member = members[member_index]
        try:
            states = family.arrays[member_array_key(member_index, "states")]
            times = family.arrays[member_array_key(member_index, "times")]
        except KeyError as exc:
            raise CatalogError(
                f"族记录 {record_id} 缺少成员 {member_index} 的数组段：{exc}"
            ) from exc

        char_length = family.meta["scalars"].get("char_length_km")
        jacobi = member.get("jacobi")
        family_classification = family.meta["classification"]
        classification = {
            "orbit_family": family_classification["orbit_family"],
            "libration_point": family_classification["libration_point"],
            "jacobi": None if jacobi is None else [float(jacobi), float(jacobi)],
            "amplitude": point_interval(geometric_amplitude_km(states, char_length)),
            "has_cr3bp": True,
            "has_ephemeris": False,
        }
        meta: dict[str, Any] = {
            "source_tool": "catalog_promote",
            "source_record_id": record_id,
            "classification": classification,
            # 状态三元组继承族记录：成员的来历（含软失败的族）不粉饰。
            "status": family.meta["status"],
            "cause": family.meta["cause"],
            "message": family.meta["message"],
            "scalars": {
                "member_count": 1,
                "member_index": member_index,
                "family_record_id": record_id,
                "char_length_km": char_length,
            },
            "request": {
                "source": "catalog_promote",
                "family_record_id": record_id,
                "member_index": member_index,
                "parameters": member.get("parameters", {}),
            },
            "members": [],
            "tags": [],
            "note": "",
        }
        arrays = cr3bp_segment_arrays(states, times)
        new_id = self.put(meta, arrays)
        return self.get(new_id)

    # ---- 导出 ----

    def export(self, catalog_filter: CatalogFilter, dest: str | Path) -> list[str]:
        """把查询子集打包导出（标注随文件走），返回导出的 record_id 列表。

        ``dest`` 以 ``.zip`` 结尾时产出 zip 包，否则产出目录。包布局与库
        目录一致（records/ + manifest.json），可直接作为库打开（索引派生
        重建）。
        """
        summaries = self.query(catalog_filter)
        record_ids = [summary["record_id"] for summary in summaries]
        dest = Path(dest)
        if dest.suffix == ".zip":
            with tempfile.TemporaryDirectory() as staging:
                staged = self._write_package(Path(staging), record_ids)
                with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as bundle:
                    for path in sorted(staged.rglob("*")):
                        bundle.write(path, path.relative_to(staged))
        else:
            self._write_package(dest, record_ids)
        return record_ids

    def _write_package(self, dest: Path, record_ids: list[str]) -> Path:
        records_dir = dest / "records"
        records_dir.mkdir(parents=True, exist_ok=True)
        for record_id in record_ids:
            shutil.copy2(self.records_dir / f"{record_id}.json", records_dir)
            npz_path = self.records_dir / f"{record_id}.npz"
            if npz_path.exists():
                shutil.copy2(npz_path, records_dir)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "record_ids": record_ids,
            "exported_count": len(record_ids),
        }
        (dest / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return dest

    # ---- 维护 ----

    def rebuild_index(self) -> None:
        """扫描记录文件全量重建索引；重建后查询结果不变。"""
        metas = []
        for json_path in sorted(self.records_dir.glob("*.json")):
            metas.append(meta_from_json(json_path.read_text(encoding="utf-8")))
        self._index.rebuild(metas)

    def close(self) -> None:
        self._index.close()
