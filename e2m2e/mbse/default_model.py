"""默认 MBSE 模型装配。"""

from __future__ import annotations

from .architecture.algorithm_components import ALGORITHM_COMPONENTS
from .architecture.api_components import API_COMPONENTS
from .architecture.components import ComponentRegistry
from .architecture.data_components import DATA_COMPONENTS
from .architecture.numerical_components import NUMERICAL_COMPONENTS
from .architecture.tools_components import TOOLS_COMPONENTS
from .requirements.algorithms_requirements import ALGORITHMS_REQUIREMENTS
from .requirements.base import RequirementRegistry
from .requirements.core_requirements import CORE_REQUIREMENTS


def register_default_model(
    requirements: RequirementRegistry,
    components: ComponentRegistry,
) -> None:
    """向调用方提供的注册表登记官方 MBSE 目录。

    调用方负责注册表生命周期；本函数不会清空既有登记。
    """
    requirements.register_many(CORE_REQUIREMENTS + ALGORITHMS_REQUIREMENTS)
    components.register_many(
        DATA_COMPONENTS
        + NUMERICAL_COMPONENTS
        + ALGORITHM_COMPONENTS
        + API_COMPONENTS
        + TOOLS_COMPONENTS
    )
