"""数值层组件目录。"""

from .components import Component

NUMERICAL_COMPONENTS = [
    Component(
        name="Integrators",
        module_path="e2m2e.integrators",
        layer="numerical",
        description="Python facade for Rust numerical integration and solvers",
    ),
]
