"""已废弃：``SearchConfig`` 已合并到 :mod:`e2m2e.transfer.config` 的 :class:`TransferConfig`。

原先两个 dataclass（``TransferConfig`` 描述优化阶段、``SearchConfig`` 描述搜索阶段）
描述同一件事，现已合并为带 ``search_*`` / ``nlp_*`` 前缀的单一 :class:`TransferConfig`。

本模块仅保留 ``SearchConfig = TransferConfig`` 别名用于向后兼容导入；新代码请直接
使用 :class:`e2m2e.transfer.config.TransferConfig`。
"""

from __future__ import annotations

from .config import TransferConfig

# 向后兼容别名。注意：原 SearchConfig 的字段名（alpha_min、n_alpha 等）与
# TransferConfig 的前缀字段名不同，此别名仅供导入兼容，不应直接按旧字段名构造。
SearchConfig = TransferConfig
