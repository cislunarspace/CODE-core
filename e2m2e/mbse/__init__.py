"""MBSE (Model-Based Systems Engineering) 模型层

提供 SysML 风格的系统建模基础设施，包括：

- **需求模型** (requirements/): 形式化需求定义与追溯矩阵
- **架构模型** (architecture/): 组件登记与依赖关系
- **数据模型** (data/): Pydantic 统一数据结构
- **图表生成** (diagrams/): Mermaid 图表自动生成

所有模型均为 living artifacts，随实现代码同步更新。
"""

from . import architecture, data, diagrams, requirements

__all__ = ["architecture", "data", "diagrams", "requirements"]
