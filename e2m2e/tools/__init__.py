"""工具层：可视化、日志。

第 5 层，辅助层（ADR 0011）：核心库不 import tools/。可视化可选依赖
（``[viz]`` extra）。

- ``viz/``：可视化（待从 visualization/ 迁入）。
- ``logging/``：结构化日志（标准 logging + 键值对，零新依赖）。

实现状态：骨架。
"""

__all__: list[str] = []
