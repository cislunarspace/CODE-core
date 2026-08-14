"""RequirementRegistry 需求登记与追溯测试。

覆盖注册、查重、分类过滤、追溯矩阵与覆盖率报告。
"""

from __future__ import annotations

import pytest

from e2m2e.mbse.requirements.base import (
    Requirement,
    RequirementCategory,
    RequirementRegistry,
)

pytestmark = pytest.mark.aux


def make_requirement(
    req_id: str,
    *,
    category: RequirementCategory = RequirementCategory.FUNCTIONAL,
    parent: str | None = None,
    linked_tests: list[str] | None = None,
) -> Requirement:
    """构造合成需求对象，用于注册表行为测试。"""
    return Requirement(
        id=req_id,
        title=f"Requirement {req_id}",
        category=category,
        description="Synthetic requirement for registry behavior tests.",
        parent=parent,
        linked_code=["e2m2e/example.py"],
        linked_tests=[] if linked_tests is None else linked_tests,
    )


@pytest.fixture
def registry():
    """提供已清空的 RequirementRegistry 实例。"""
    reg = RequirementRegistry()
    reg.clear()
    yield reg
    reg.clear()


def test_register_rejects_duplicate_ids_and_supports_lookup(registry):
    """注册拒绝重复 ID，并支持按 ID 查找。"""
    requirement = make_requirement("REQ-001")

    registry.register(requirement)

    assert registry.get("REQ-001") is requirement
    assert "REQ-001" in registry
    assert list(registry) == [requirement]
    with pytest.raises(ValueError, match="REQ-001"):
        registry.register(requirement)
    with pytest.raises(KeyError, match="REQ-999"):
        registry.get("REQ-999")


def test_filters_by_category_and_recursive_children(registry):
    """按分类过滤与递归获取子需求。"""
    parent = make_requirement("REQ-001", category=RequirementCategory.FUNCTIONAL)
    child = make_requirement(
        "REQ-002",
        category=RequirementCategory.INTERFACE,
        parent="REQ-001",
    )
    grandchild = make_requirement(
        "REQ-003",
        category=RequirementCategory.INTERFACE,
        parent="REQ-002",
    )
    registry.register_many([parent, child, grandchild])

    assert registry.by_category(RequirementCategory.INTERFACE) == [child, grandchild]
    assert registry.by_parent("REQ-001") == [child]
    assert registry.children_of("REQ-001") == [child, grandchild]


def test_traceability_matrix_marks_requirements_with_linked_tests(registry):
    """追溯矩阵标记已关联测试的需求。"""
    covered = make_requirement("REQ-001", linked_tests=["tests/example_test.py"])
    uncovered = make_requirement("REQ-002")
    registry.register_many([covered, uncovered])

    matrix = registry.traceability_matrix()

    assert matrix["REQ-001"]["requirement"] is covered
    assert matrix["REQ-001"]["code"] == ["e2m2e/example.py"]
    assert matrix["REQ-001"]["tests"] == ["tests/example_test.py"]
    assert matrix["REQ-001"]["has_coverage"] is True
    assert matrix["REQ-002"]["has_coverage"] is False


def test_coverage_report_handles_empty_and_partially_covered_registry(registry):
    """覆盖率报告处理空注册表与部分覆盖场景。"""
    assert registry.coverage_report() == {
        "total": 0,
        "covered": 0,
        "uncovered": 0,
        "coverage_rate": 0.0,
    }

    registry.register_many(
        [
            make_requirement("REQ-001", linked_tests=["tests/example_test.py"]),
            make_requirement("REQ-002"),
        ]
    )

    assert registry.coverage_report() == {
        "total": 2,
        "covered": 1,
        "uncovered": 1,
        "coverage_rate": 0.5,
        "uncovered_ids": ["REQ-002"],
    }


def test_requirement_rejects_invalid_id_and_verification_method():
    """需求只接受统一 ID 格式和既定验证方法。"""
    with pytest.raises(ValueError, match="REQ-001"):
        make_requirement("requirement-1")
    with pytest.raises(ValueError, match="verification_method"):
        Requirement(
            id="REQ-001",
            title="验证方法",
            category=RequirementCategory.FUNCTIONAL,
            description="验证方法必须受限。",
            verification_method="manual",
        )
