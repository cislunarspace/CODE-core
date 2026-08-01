"""微分修正策略的基础配置数据类。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CorrectionConfig:
    """微分修正策略的不可变配置。

    将原本散布在 DifferentialCorrection 各 setup_* 方法中的
    修正参数封装为单一数据对象。

    Attributes:
        setup_type: 修正配置类型标识符。
        symmetry_condition: 修正所利用的对称性（如 'x_axis'）。
        fixed_parameters: 修正过程中保持不变的参数值。
        free_variables: 牛顿求解器可调整的变量名列表。
        free_variable_indices: 自由变量在状态向量中对应的索引。
        target_conditions: 约束名称到目标值的映射。
        constraint_indices: 约束在状态向量中的求值索引。
        constraint_weights: 各约束的雅可比加权因子。
        constraint_types: 各约束的分类（如 'equality'）。
    """

    setup_type: str
    symmetry_condition: str
    fixed_parameters: dict[str, float] = field(default_factory=dict)
    free_variables: list[str] = field(default_factory=list)
    free_variable_indices: list[int] = field(default_factory=list)
    target_conditions: dict[str, float] = field(default_factory=dict)
    constraint_indices: list[int] = field(default_factory=list)
    constraint_weights: dict[str, float] = field(default_factory=dict)
    constraint_types: dict[str, str] = field(default_factory=dict)
