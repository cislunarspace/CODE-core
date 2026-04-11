"""Algorithms 层组件注册

将 algorithms/ 层的所有组件注册到 MBSE 组件注册表中。
"""

from .components import Component

ALGORITHMS_COMPONENTS = [
    Component(
        name="DifferentialCorrection",
        module_path="e2m2e.algorithms.differential_correction",
        protocols=["CorrectorStrategy"],
        dependencies=["CR3BP_Dynamics"],
        layer="algorithms",
        description="微分修正 Newton 迭代求解器",
    ),
    Component(
        name="CorrectionConfig",
        module_path="e2m2e.algorithms.strategies.base",
        protocols=[],
        dependencies=[],
        layer="algorithms",
        description="修正策略不可变配置",
    ),
    Component(
        name="Symmetric2DFixedX0",
        module_path="e2m2e.algorithms.strategies.symmetric_2d",
        protocols=["CorrectorStrategy"],
        dependencies=["CorrectionConfig"],
        layer="algorithms",
        description="2D 对称固定 X0 策略",
    ),
    Component(
        name="SymmetricXZFixedZ0",
        module_path="e2m2e.algorithms.strategies.halo",
        protocols=["CorrectorStrategy"],
        dependencies=["CorrectionConfig"],
        layer="algorithms",
        description="Halo 固定 Z0 策略",
    ),
    Component(
        name="Continuation",
        module_path="e2m2e.algorithms.continuation",
        protocols=[],
        dependencies=["DifferentialCorrection"],
        layer="algorithms",
        description="轨道族延拓（自然参数 + 伪弧长）",
    ),
    Component(
        name="StabilityAnalysis",
        module_path="e2m2e.algorithms.stability",
        protocols=[],
        dependencies=["CR3BP_Dynamics"],
        layer="algorithms",
        description="Floquet 稳定性分析",
    ),
    Component(
        name="MultipleShooting",
        module_path="e2m2e.algorithms.multiple_shooting",
        protocols=[],
        dependencies=["CR3BP_Dynamics"],
        layer="algorithms",
        description="多点射击法并行传播",
    ),
    Component(
        name="compute_F_and_dF",
        module_path="e2m2e.algorithms.continuation",
        protocols=[],
        dependencies=["CR3BP_Dynamics"],
        layer="algorithms",
        description="XZ 对称约束函数与 Jacobian",
    ),
]
