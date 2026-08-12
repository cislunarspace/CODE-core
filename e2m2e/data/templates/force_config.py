"""力模型配置 schema（纯数据）。

配置 JSON 是数据（可存盘、可版本化）：``{"version": 1, "forces": [...]}``，每条
``{name, type, enabled, params}``。构建逻辑（``ForceModel.from_config/to_config``）
在 ``algorithm/forces/`` （ADR 0011，ADR 0004 的 schema 不变，只拆"schema 数据"
与"构建逻辑"）。

实现状态：骨架。schema 定义待从 ``core/forces/force_config.py`` 迁入。
"""

from __future__ import annotations

from typing import Any, TypedDict

__all__ = ["ForceConfig", "ForceEntry", "CONFIG_VERSION"]

CONFIG_VERSION = 1


class ForceEntry(TypedDict):
    """单条力配置。"""

    name: str
    type: str
    enabled: bool
    params: dict[str, Any]


class ForceConfig(TypedDict):
    """力模型配置字典。"""

    version: int
    forces: list[ForceEntry]
