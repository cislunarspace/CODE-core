"""默认 MBSE 模型的完整性测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from e2m2e.mbse.architecture.components import ARCHITECTURE_LAYERS
from e2m2e.mbse.requirements.base import REQUIREMENT_ID_PATTERN, VALID_VERIFICATION_METHODS

pytestmark = pytest.mark.aux

PROJECT_ROOT = Path(__file__).parents[2]


def test_default_model_has_complete_requirement_traceability(mbse_model):
    """官方需求目录的追溯元数据完整。"""
    requirements, _ = mbse_model

    report = requirements.coverage_report()

    assert report["total"] > 0
    assert report["coverage_rate"] == 1.0
    assert report["uncovered"] == 0
    assert report["uncovered_ids"] == []


def test_default_requirements_follow_the_catalog_contract(mbse_model):
    """官方需求具有稳定标识、验证方法和无重复的追溯链接。"""
    requirements, _ = mbse_model

    for requirement in requirements:
        assert REQUIREMENT_ID_PATTERN.fullmatch(requirement.id)
        assert requirement.description
        assert requirement.verification_method in VALID_VERIFICATION_METHODS
        assert requirement.linked_code
        assert requirement.linked_tests
        assert len(requirement.linked_code) == len(set(requirement.linked_code))
        assert len(requirement.linked_tests) == len(set(requirement.linked_tests))


def test_default_requirement_links_resolve_to_current_code_and_tests(mbse_model):
    """追溯矩阵中的路径指向现行模块和存在的测试文件。"""
    requirements, _ = mbse_model

    unresolved_modules = [
        module
        for requirement in requirements
        for module in requirement.linked_code
        if importlib.util.find_spec(module) is None
    ]
    missing_tests = [
        test_path
        for requirement in requirements
        for test_path in requirement.linked_tests
        if not (PROJECT_ROOT / test_path).is_file()
    ]

    assert unresolved_modules == []
    assert missing_tests == []


def test_default_components_match_the_current_architecture(mbse_model):
    """组件目录只表达现行架构、实际模块和已登记依赖。"""
    _, components = mbse_model
    component_names = {component.name for component in components}

    assert component_names
    for component in components:
        assert component.layer in ARCHITECTURE_LAYERS
        assert importlib.util.find_spec(component.module_path) is not None
        assert set(component.dependencies) <= component_names
        assert component.protocols == []
