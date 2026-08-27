"""数据层组件目录。"""

from .components import Component

DATA_COMPONENTS = [
    Component(
        name="Orbit",
        module_path="e2m2e.data.types.orbit",
        layer="data",
        description="Orbit data container",
    ),
    Component(
        name="OrbitFamily",
        module_path="e2m2e.data.types.orbit",
        dependencies=["Orbit"],
        layer="data",
        description="Orbit family container",
    ),
    Component(
        name="SPICEManager",
        module_path="e2m2e.data.kernels.manager",
        layer="data",
        description="SPICE kernel management",
    ),
]
