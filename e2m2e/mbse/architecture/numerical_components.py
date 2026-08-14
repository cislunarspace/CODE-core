"""数值层组件目录。"""

from .components import Component

NUMERICAL_COMPONENTS = [
    Component(
        name="Integrators",
        module_path="e2m2e.integrators",
        layer="numerical",
        description="Rust 数值积分与求解器的 Python 门面",
    ),
]
