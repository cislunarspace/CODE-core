"""Core 层组件注册

将 core/ 层的所有组件登记到 MBSE 组件注册表，
供生成 BDD 与依赖图使用。
"""

from .components import Component

CORE_COMPONENTS = [
    Component(
        name="CR3BP_System",
        module_path="e2m2e.core.cr3bp_system",
        protocols=[],
        dependencies=[],
        layer="core",
        description="CR3BP 系统定义（质量参数、平动点、Jacobi 常数）",
    ),
    Component(
        name="Dynamics",
        module_path="e2m2e.core.dynamics",
        protocols=[],
        dependencies=["CR3BP_System"],
        layer="core",
        description="通用动力学基类（Template Method）",
    ),
    Component(
        name="CR3BP_Dynamics",
        module_path="e2m2e.core.dynamics",
        protocols=[],
        dependencies=["Dynamics", "CR3BP_System"],
        layer="core",
        description="CR3BP 动力学方程与 STM 计算",
    ),
    Component(
        name="EphemerisDynamics",
        module_path="e2m2e.core.ephemeris_dynamics",
        protocols=[],
        dependencies=["Dynamics", "EphemerisSystem"],
        layer="core",
        description="星历 N 体动力学",
    ),
    Component(
        name="Orbit",
        module_path="e2m2e.core.orbit",
        protocols=[],
        dependencies=["CR3BP_System", "CR3BP_Dynamics"],
        layer="core",
        description="轨道数据容器（组合模式）",
    ),
    Component(
        name="OrbitFamily",
        module_path="e2m2e.core.orbit",
        protocols=[],
        dependencies=["Orbit"],
        layer="core",
        description="轨道族容器",
    ),
    Component(
        name="SPICEManager",
        module_path="e2m2e.core.spice",
        protocols=[],
        dependencies=[],
        layer="core",
        description="SPICE 内核管理",
    ),
    Component(
        name="EphemerisSystem",
        module_path="e2m2e.core.ephemeris_system",
        protocols=[],
        dependencies=["SPICEManager"],
        layer="core",
        description="星历系统配置",
    ),
]
