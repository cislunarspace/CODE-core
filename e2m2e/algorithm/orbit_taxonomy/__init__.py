"""轨道分类学内核（orbit taxonomy）：42 标签分类器（词表在数据层）。

词汇与判据的权威记录是 ADR 0042；本包只做实现。分类是"给一条轨迹
推断族标签"，与设计侧的 ``OrbitFamilyType``（生成时已知族）是两个
方向的问题，两者经 ADR 0042 映射表对齐。42 标签表自 ADR 0044 起是
数据层静态参考数据（``e2m2e.data.catalog.terminology``，随
``catalog_terminology`` 出库）；本包经 ``__init__`` 转出既有公共名，
算法层内用法不变。
"""

from ...data.catalog.terminology import (
    TAXONOMY,
    TAXONOMY_BY_CANONICAL,
    Hemisphere,
    TaxonomyCategory,
    TaxonomyLabel,
    label_legend,
    parse_taxonomy_label,
)
from .classify import TaxonomyResult, classify_orbit

__all__ = [
    "TAXONOMY",
    "TAXONOMY_BY_CANONICAL",
    "Hemisphere",
    "TaxonomyCategory",
    "TaxonomyLabel",
    "TaxonomyResult",
    "classify_orbit",
    "label_legend",
    "parse_taxonomy_label",
]
