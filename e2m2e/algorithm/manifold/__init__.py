"""不变流形与庞加莱截面。

流形种子生成与批量传播数值在 Rust；Python 侧保留参数校验、领域对象组装
与庞加莱截面定义（``sections.py`` 有意留 Python）。
"""

from __future__ import annotations

from .manifolds import InvariantManifold, ManifoldKind, ManifoldTube
from .sections import PoincareSection, SectionCrossings, detect_crossings

__all__ = [
    "InvariantManifold",
    "ManifoldKind",
    "ManifoldTube",
    "PoincareSection",
    "SectionCrossings",
    "detect_crossings",
]
