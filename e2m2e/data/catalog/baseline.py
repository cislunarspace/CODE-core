"""CR3BP 基线数据集首用导入（ADR 0036 决策 5）。

包内 ``e2m2e/data/catalog_baseline/`` 是一批预生成的 ADR 0031 catalog
记录（一族一条，``tags=["baseline"]``，``scalars.baseline_version`` 标
基线版本，record_id 确定性命名如 ``baseline-halo-l2``）。用户库缺记录
或基线版本不一致时，把包内记录文件复制进库目录并重建索引；存储引擎
布局零改动，之后一切走既有 ``catalog_query``。
"""

from __future__ import annotations

from importlib.resources.abc import Traversable

from .record import meta_from_json, validate_record_id
from .store import CatalogStore

__all__ = ["BASELINE_TAG", "baseline_source_dir", "import_baseline"]

#: 基线记录统一携带的标签
BASELINE_TAG = "baseline"


def baseline_source_dir() -> Traversable:
    """包内基线数据目录（importlib.resources 定位，不猜安装路径）。"""
    from importlib.resources import files

    return files("e2m2e.data").joinpath("catalog_baseline")


def import_baseline(store: CatalogStore, source_dir: Traversable | None = None) -> int:
    """把包内基线记录导入用户库，返回实际写入（覆盖）的记录数。

    逐条对位 ``record_id``：库中无该记录、或库中基线版本与包内不一致
    （含版本落后于包内）时复制 JSON + NPZ 进 ``records/``；全部对位后
    重建索引（ADR 0031 决策 5：索引是派生物，可全量重建）。包内无基线
    数据（如开发环境未生成）时静默跳过。``source_dir`` 供测试注入合成
    基线源。
    """
    src = source_dir if source_dir is not None else baseline_source_dir()
    if not src.is_dir():
        return 0
    imported = 0
    for item in src.iterdir():
        if not item.name.endswith(".json"):
            continue
        meta = meta_from_json(item.read_text(encoding="utf-8"))
        record_id = validate_record_id(meta["record_id"])
        target_json = store.records_dir / f"{record_id}.json"
        if target_json.exists():
            existing = meta_from_json(target_json.read_text(encoding="utf-8"))
            if _baseline_version(existing) == _baseline_version(meta):
                continue
        target_json.write_bytes(item.read_bytes())
        npz_src = src / f"{record_id}.npz"
        if npz_src.is_file():
            (store.records_dir / f"{record_id}.npz").write_bytes(npz_src.read_bytes())
        imported += 1
    if imported:
        store.rebuild_index()
    return imported


def _baseline_version(meta: dict) -> str | None:
    version = meta.get("scalars", {}).get("baseline_version")
    return str(version) if version is not None else None
