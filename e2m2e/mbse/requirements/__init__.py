"""需求模型子包

提供形式化的需求定义、注册和追溯能力。
"""

from .base import Requirement, RequirementCategory, RequirementPriority, RequirementRegistry

__all__ = [
    "Requirement",
    "RequirementCategory",
    "RequirementPriority",
    "RequirementRegistry",
]
