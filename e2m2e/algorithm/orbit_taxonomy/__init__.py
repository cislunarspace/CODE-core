"""轨道分类学内核（orbit taxonomy）：42 标签词汇与周期轨道分类器。

词汇与判据的权威记录是 ADR 0042；本包只做实现。分类是"给一条轨迹
推断族标签"，与设计侧的 ``OrbitFamilyType``（生成时已知族）是两个
方向的问题，两者经 ADR 0042 映射表对齐。
"""

from .classify import TaxonomyResult, classify_orbit
from .labels import (
    TAXONOMY,
    TAXONOMY_BY_CANONICAL,
    Hemisphere,
    TaxonomyCategory,
    TaxonomyLabel,
    label_legend,
    parse_taxonomy_label,
)

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
