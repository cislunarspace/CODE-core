"""不变流形与庞加莱截面。

流形种子/截面留 Python（领域知识，ADR 0011 迁移，源：
``algorithms/manifolds.py`` + ``algorithms/sections.py``）：Floquet 模选取、
种子生成、庞加莱截面定义。传播走 Rust STM。
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
