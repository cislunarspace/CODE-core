"""Test that all MBSE requirements have linked code and tests."""

import re

import pytest

from e2m2e.mbse.requirements.base import RequirementCategory, RequirementPriority, RequirementRegistry
from e2m2e.mbse.requirements.core_requirements import CORE_REQUIREMENTS
from e2m2e.mbse.requirements.algorithms_requirements import ALGORITHMS_REQUIREMENTS

ALL_REQUIREMENTS = CORE_REQUIREMENTS + ALGORITHMS_REQUIREMENTS


def test_all_requirements_have_ids():
    """Every requirement must have a unique ID."""
    ids = [r.id for r in ALL_REQUIREMENTS]
    duplicates = list({x for x in ids if ids.count(x) > 1})
    assert len(ids) == len(set(ids)), f"Duplicate requirement IDs: {duplicates}"


def test_all_requirements_have_linked_code():
    """Every requirement must link to at least one source code module."""
    for req in ALL_REQUIREMENTS:
        assert req.linked_code, f"{req.id} has no linked code"


def test_all_requirements_have_linked_tests():
    """Every requirement must link to at least one test file."""
    for req in ALL_REQUIREMENTS:
        assert req.linked_tests, f"{req.id} has no linked tests"


def test_all_requirements_have_verification_method():
    """Every requirement must declare a valid verification method."""
    valid_methods = ("test", "analysis", "inspection")
    for req in ALL_REQUIREMENTS:
        assert req.verification_method in valid_methods, (
            f"{req.id} has invalid verification method: {req.verification_method}"
        )


def test_all_requirements_have_valid_category():
    """Every requirement must have a valid RequirementCategory."""
    valid_categories = {c.value for c in RequirementCategory}
    for req in ALL_REQUIREMENTS:
        assert req.category.value in valid_categories, (
            f"{req.id} has invalid category: {req.category}"
        )


def test_all_requirements_have_valid_priority():
    """Every requirement must have a valid RequirementPriority."""
    valid_priorities = {p.value for p in RequirementPriority}
    for req in ALL_REQUIREMENTS:
        assert req.priority.value in valid_priorities, (
            f"{req.id} has invalid priority: {req.priority}"
        )


def test_requirement_ids_follow_convention():
    """Requirement IDs must follow the REQ-NNN naming convention."""
    pattern = re.compile(r"^REQ-\d{3}$")
    for req in ALL_REQUIREMENTS:
        assert pattern.match(req.id), f"{req.id} does not follow REQ-NNN convention"


def test_core_requirements_in_correct_range():
    """Core requirements must have IDs in the 000-099 range."""
    for req in CORE_REQUIREMENTS:
        num = int(req.id.split("-")[1])
        assert 0 <= num < 100, f"{req.id} is a core requirement but ID is outside 000-099 range"


def test_algorithms_requirements_in_correct_range():
    """Algorithms requirements must have IDs in the 100-199 range."""
    for req in ALGORITHMS_REQUIREMENTS:
        num = int(req.id.split("-")[1])
        assert 100 <= num < 200, f"{req.id} is an algorithms requirement but ID is outside 100-199 range"


def test_requirement_registry_coverage():
    """RequirementRegistry must report 100% coverage for all registered requirements."""
    reg = RequirementRegistry()
    reg.clear()
    reg.register_many(CORE_REQUIREMENTS)
    reg.register_many(ALGORITHMS_REQUIREMENTS)

    report = reg.coverage_report()
    assert report["coverage_rate"] == 1.0, (
        f"Expected 100% coverage, got {report['coverage_rate']:.1%}. "
        f"Uncovered: {report.get('uncovered_ids', [])}"
    )


def test_requirement_registry_traceability():
    """Traceability matrix must contain all registered requirements."""
    reg = RequirementRegistry()
    reg.clear()
    reg.register_many(CORE_REQUIREMENTS)
    reg.register_many(ALGORITHMS_REQUIREMENTS)

    matrix = reg.traceability_matrix()
    assert len(matrix) == len(ALL_REQUIREMENTS)

    for req in ALL_REQUIREMENTS:
        assert req.id in matrix, f"{req.id} not in traceability matrix"
        entry = matrix[req.id]
        assert entry["has_coverage"], f"{req.id} has no test coverage in matrix"
