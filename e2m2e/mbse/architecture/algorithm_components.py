"""算法层组件目录。"""

from .components import Component

ALGORITHM_COMPONENTS = [
    Component(
        name="CR3BP_System",
        module_path="e2m2e.algorithm.dynamics.cr3bp_system",
        layer="algorithm",
        description="CR3BP 系统定义（质量参数、平动点、Jacobi 常数）",
    ),
    Component(
        name="Dynamics",
        module_path="e2m2e.algorithm.dynamics.dynamics",
        layer="algorithm",
        description="通用动力学基类",
    ),
    Component(
        name="CR3BP_Dynamics",
        module_path="e2m2e.algorithm.dynamics.dynamics",
        dependencies=["Dynamics", "CR3BP_System"],
        layer="algorithm",
        description="CR3BP 动力学方程与 STM 计算",
    ),
    Component(
        name="EphemerisSystem",
        module_path="e2m2e.algorithm.dynamics.ephemeris_system",
        dependencies=["SPICEManager"],
        layer="algorithm",
        description="星历系统配置",
    ),
    Component(
        name="EphemerisDynamics",
        module_path="e2m2e.algorithm.dynamics.ephemeris_dynamics",
        dependencies=["Dynamics", "EphemerisSystem"],
        layer="algorithm",
        description="星历 N 体动力学（遗留，仅供多点射击内部使用）",
    ),
    Component(
        name="DifferentialCorrection",
        module_path="e2m2e.algorithm.solver.differential_correction",
        dependencies=["CR3BP_Dynamics"],
        layer="algorithm",
        description="微分修正问题构造入口",
    ),
    Component(
        name="CorrectionConfig",
        module_path="e2m2e.algorithm.family.strategies.base",
        layer="algorithm",
        description="修正策略不可变配置",
    ),
    Component(
        name="symmetric_2d_fixed_x0",
        module_path="e2m2e.algorithm.family.strategies.symmetric_2d",
        dependencies=["CorrectionConfig"],
        layer="algorithm",
        description="二维对称固定 x0 修正策略",
    ),
    Component(
        name="symmetric_xz_fixed_z0",
        module_path="e2m2e.algorithm.family.strategies.symmetric_3d",
        dependencies=["CorrectionConfig"],
        layer="algorithm",
        description="XZ 对称固定 z0 修正策略",
    ),
    Component(
        name="Continuation",
        module_path="e2m2e.algorithm.solver.continuation",
        dependencies=["DifferentialCorrection"],
        layer="algorithm",
        description="轨道族延拓",
    ),
    Component(
        name="StabilityAnalysis",
        module_path="e2m2e.algorithm.stability",
        dependencies=["CR3BP_Dynamics"],
        layer="algorithm",
        description="Floquet 稳定性分析",
    ),
    Component(
        name="MultipleShooting",
        module_path="e2m2e.algorithm.solver.multiple_shooting",
        dependencies=["CR3BP_Dynamics"],
        layer="algorithm",
        description="多点射击法问题构造入口",
    ),
    Component(
        name="compute_F_and_dF_symmetric_xz_plane",
        module_path="e2m2e.algorithm.solver.continuation",
        dependencies=["CR3BP_Dynamics"],
        layer="algorithm",
        description="XZ 对称约束函数与 Jacobian",
    ),
]
