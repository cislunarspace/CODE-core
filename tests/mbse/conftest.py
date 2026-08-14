"""MBSE 测试共享 fixture。"""

import pytest

from e2m2e.mbse import register_default_model
from e2m2e.mbse.architecture import ComponentRegistry
from e2m2e.mbse.requirements import RequirementRegistry


@pytest.fixture
def mbse_model():
    """提供装入官方目录且隔离的 MBSE 注册表。"""
    requirements = RequirementRegistry()
    components = ComponentRegistry()
    requirements.clear()
    components.clear()
    register_default_model(requirements, components)

    yield requirements, components

    requirements.clear()
    components.clear()
