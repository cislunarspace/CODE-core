"""轨道库 catalog：记录格式、存储引擎与派生索引（ADR 0031）。

- ``record.py``：记录格式（schema 版本、段数组键约定、校验、段序列化）。
- ``store.py``：存储引擎（写入/读取/删除/标注/导出/成员提升/索引重建）。
- ``index.py``：SQLite 派生索引（只存过滤维度与文件指针，可全量重建）。
- ``baseline.py``：随包基线数据集的首用导入（ADR 0036）。

记录文件（records/*.json + .npz）是事实来源；catalog.db 是派生物。
"""

from .baseline import BASELINE_TAG, baseline_source_dir, import_baseline
from .record import (
    SCHEMA_VERSION,
    CatalogError,
    CatalogFilter,
    CatalogRecord,
    RecordNotFoundError,
    cr3bp_segment_arrays,
    ephemeris_from_arrays,
    ephemeris_segment_arrays,
    geometric_amplitude_km,
    member_array_key,
    member_count,
    new_record_id,
    point_interval,
    validate_meta,
)
from .store import CatalogStore

__all__ = [
    "BASELINE_TAG",
    "SCHEMA_VERSION",
    "CatalogError",
    "CatalogFilter",
    "CatalogRecord",
    "CatalogStore",
    "RecordNotFoundError",
    "baseline_source_dir",
    "cr3bp_segment_arrays",
    "ephemeris_from_arrays",
    "ephemeris_segment_arrays",
    "geometric_amplitude_km",
    "import_baseline",
    "member_array_key",
    "member_count",
    "new_record_id",
    "point_interval",
    "validate_meta",
]
