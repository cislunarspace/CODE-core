#!/usr/bin/env python3
"""生成默认 MBSE 模型的受管参考文档。

用法:
    python scripts/generate_mbse_diagrams.py
"""

from __future__ import annotations

from pathlib import Path

from e2m2e.mbse import register_default_model
from e2m2e.mbse.architecture import ComponentRegistry
from e2m2e.mbse.diagrams import DiagramGenerator
from e2m2e.mbse.requirements import RequirementRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "docs" / "reference" / "mbse" / "generated"


def main() -> list[str]:
    """装配默认模型并生成受管图表和追溯矩阵。"""
    requirements = RequirementRegistry()
    components = ComponentRegistry()
    requirements.clear()
    components.clear()
    register_default_model(requirements, components)
    component_count = len(components)

    try:
        generator = DiagramGenerator(requirements=requirements, components=components)
        generated = generator.generate_all(str(OUTPUT_DIR))
        report = requirements.coverage_report()
    finally:
        requirements.clear()
        components.clear()

    print(f"已注册 {report['total']} 条需求")
    print(f"已注册 {component_count} 个组件")
    print(f"追溯元数据完整率: {report['coverage_rate']:.1%}")
    for path in generated:
        print(f"已生成 {path}")
    return generated


if __name__ == "__main__":
    main()
