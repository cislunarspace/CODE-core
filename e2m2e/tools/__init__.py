"""工具层：可视化、日志。

第 5 层，辅助层（ADR 0011 迁移，源：``visualization/``）：核心库不 import
tools/。可视化可选依赖（``[viz]`` extra）；日志为标准 logging + 键值对、
零新依赖。

仓库全貌与一条任务链的走读见 README 的仓库怎么读一节。
"""

from .logging import KeyValueFormatter, configure_logging

__all__ = ["configure_logging", "KeyValueFormatter"]
