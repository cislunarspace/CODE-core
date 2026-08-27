"""MBSE 图表与受管文档产物测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from e2m2e.mbse.architecture import ComponentRegistry
from e2m2e.mbse.diagrams import DiagramGenerator
from e2m2e.mbse.requirements import RequirementRegistry

pytestmark = pytest.mark.aux

PROJECT_ROOT = Path(__file__).parents[2]
GENERATED_DOCUMENTS = {
    "bdd-data.md",
    "bdd-numerical.md",
    "bdd-algorithm.md",
    "bdd-api.md",
    "bdd-tools.md",
    "requirements.md",
    "traceability-matrix.md",
}


def test_generator_preserves_explicit_empty_registries(tmp_path):
    """显式传入的空模型不会被默认注册表替换。"""
    requirements = RequirementRegistry()
    components = ComponentRegistry()

    generator = DiagramGenerator(requirements=requirements, components=components)

    assert generator.requirements is requirements
    assert generator.components is components
    assert generator.generate_all(str(tmp_path)) == []


def test_default_model_generates_documented_artifacts(mbse_model, tmp_path):
    """默认模型生成带标题的 Mermaid 图表和追溯矩阵。"""
    requirements, components = mbse_model
    generator = DiagramGenerator(requirements=requirements, components=components)

    generated = {Path(path).name for path in generator.generate_all(str(tmp_path))}

    assert generated == GENERATED_DOCUMENTS
    for filename in generated:
        content = (tmp_path / filename).read_text(encoding="utf-8")
        assert content.startswith("---\ntitle: ")
        assert "\n# " in content
    assert "classDiagram" in (tmp_path / "bdd-data.md").read_text(encoding="utf-8")
    assert "requirementDiagram" in (tmp_path / "requirements.md").read_text(encoding="utf-8")
    assert "| Requirement ID |" in (tmp_path / "traceability-matrix.md").read_text(encoding="utf-8")


def test_committed_generated_documents_match_the_default_model(mbse_model, tmp_path):
    """提交的 MBSE 生成文档与默认模型保持同步。"""
    requirements, components = mbse_model
    generator = DiagramGenerator(requirements=requirements, components=components)
    generator.generate_all(str(tmp_path))

    committed_dir = PROJECT_ROOT / "docs" / "reference" / "mbse" / "generated"
    assert {path.name for path in committed_dir.iterdir()} == GENERATED_DOCUMENTS
    for filename in GENERATED_DOCUMENTS:
        assert (committed_dir / filename).read_text(encoding="utf-8") == (
            tmp_path / filename
        ).read_text(encoding="utf-8")


def test_mbse_index_references_only_generated_model_artifacts():
    """文档入口引用受管生成产物而非旧架构快照。"""
    index = (PROJECT_ROOT / "docs" / "reference" / "mbse" / "index.md").read_text(encoding="utf-8")

    for filename in GENERATED_DOCUMENTS:
        assert f"generated/{filename.removesuffix('.md')}" in index
    for obsolete_document in (
        "bdd-core",
        "bdd-algorithms",
        "[功能需求](requirements)",
        "[追溯矩阵](traceability-matrix)",
    ):
        assert obsolete_document not in index
