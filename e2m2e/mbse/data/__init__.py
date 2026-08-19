"""数据模型子包

提供基于 Pydantic 的统一数据结构。领域枚举见 ``e2m2e.data.templates.enums``。
"""

from .core_models import (
    OrbitProperties,
)

__all__ = [
    "OrbitProperties",
]
