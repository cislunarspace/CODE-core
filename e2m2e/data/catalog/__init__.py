"""轨道库 catalog：记录格式、分发包、存储引擎与派生索引（ADR 0031/0045）。

- ``record.py``：记录格式（schema v2：一轨一记录；段数组键约定、校验、
  段序列化）。
- ``bundle.py``：基线分发包（v1 族束传输格式）→ v2 成员记录的展开。
- ``store.py``：存储引擎（写入/读取/删除/标注/导出/索引重建）。
- ``baseline.py``：基线数据集的显式导入（Release 资产源，ADR 0047）。
- ``terminology.py``：术语清单（42 标签表 + 族名/转移类型闭值集，
  ADR 0044）。

记录文件（records/*.json + .npz）是事实来源；catalog.db 是派生物。
"""

from .baseline import BASELINE_TAG, import_baseline
from .bundle import expand_bundle
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
    new_record_id,
    numeric_or_none,
    point_interval,
    transfer_segment_arrays,
    validate_meta,
)
from .store import CatalogStore
from .terminology import (
    RECORD_ORBIT_FAMILIES,
    TAXONOMY,
    TAXONOMY_BY_CANONICAL,
    TRANSFER_TYPES,
    Hemisphere,
    TaxonomyCategory,
    TaxonomyLabel,
    label_legend,
    parse_taxonomy_label,
)

__all__ = [
    "BASELINE_TAG",
    "SCHEMA_VERSION",
    "CatalogError",
    "CatalogFilter",
    "CatalogRecord",
    "CatalogStore",
    "RecordNotFoundError",
    "cr3bp_segment_arrays",
    "ephemeris_from_arrays",
    "ephemeris_segment_arrays",
    "expand_bundle",
    "geometric_amplitude_km",
    "import_baseline",
    "label_legend",
    "member_array_key",
    "new_record_id",
    "numeric_or_none",
    "parse_taxonomy_label",
    "point_interval",
    "RECORD_ORBIT_FAMILIES",
    "TAXONOMY",
    "TAXONOMY_BY_CANONICAL",
    "TRANSFER_TYPES",
    "Hemisphere",
    "TaxonomyCategory",
    "TaxonomyLabel",
    "transfer_segment_arrays",
    "validate_meta",
]
