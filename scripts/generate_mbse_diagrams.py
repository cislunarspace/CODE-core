#!/usr/bin/env python3
"""MBSE 图表生成脚本

注册所有需求和组件，生成 SysML 风格的 Mermaid 图表到 docs/mbse/ 目录。

用法:
    python scripts/generate_mbse_diagrams.py
"""

import sys
import os

# 确保项目根目录在 sys.path 中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "e2m2e"))

from mbse.requirements.base import RequirementRegistry
from mbse.architecture.components import ComponentRegistry
from mbse.diagrams.generator import DiagramGenerator


def register_core_requirements(registry: RequirementRegistry) -> None:
    """注册 Core 层需求"""
    from mbse.requirements.core_requirements import CORE_REQUIREMENTS
    registry.register_many(CORE_REQUIREMENTS)


def register_algorithms_requirements(registry: RequirementRegistry) -> None:
    """注册 Algorithms 层需求"""
    from mbse.requirements.algorithms_requirements import ALGORITHMS_REQUIREMENTS
    registry.register_many(ALGORITHMS_REQUIREMENTS)


def register_core_components(registry: ComponentRegistry) -> None:
    """注册 Core 层组件"""
    from mbse.architecture.core_components import CORE_COMPONENTS
    registry.register_many(CORE_COMPONENTS)


def register_algorithms_components(registry: ComponentRegistry) -> None:
    """注册 Algorithms 层组件"""
    from mbse.architecture.algorithms_components import ALGORITHMS_COMPONENTS
    registry.register_many(ALGORITHMS_COMPONENTS)


def main():
    req_registry = RequirementRegistry()
    comp_registry = ComponentRegistry()

    # 清空已有注册（单例模式，重新运行时需要）
    req_registry.clear()
    comp_registry.clear()

    # 注册
    register_core_requirements(req_registry)
    register_algorithms_requirements(req_registry)
    register_core_components(comp_registry)
    register_algorithms_components(comp_registry)

    print(f"已注册 {len(req_registry)} 条需求")
    print(f"已注册 {len(comp_registry)} 个组件")

    # 生成图表
    generator = DiagramGenerator(requirements=req_registry, components=comp_registry)
    output_dir = os.path.join(project_root, "docs", "mbse")
    generated = generator.generate_all(output_dir)

    # 生成额外的图表
    # 状态机：收敛状态
    convergence_states = generator.generate_state_machine(
        "Convergence",
        ["iterating", "converged", "diverged", "max_iterations"],
        [
            ("iterating", "converged", "error < tol"),
            ("iterating", "diverged", "error > max"),
            ("iterating", "max_iterations", "iter >= max_iter"),
        ],
    )
    path = os.path.join(output_dir, "state-convergence.md")
    generator.write_diagram(convergence_states, path)
    generated.append(path)

    # 状态机：轨道生命周期
    orbit_lifecycle = generator.generate_state_machine(
        "OrbitLifecycle",
        ["created", "properties_computed", "stability_computed", "serialized"],
        [
            ("created", "properties_computed", "compute_basic_properties()"),
            ("properties_computed", "stability_computed", "compute_stability()"),
            ("stability_computed", "serialized", "save_to_file()"),
        ],
    )
    path = os.path.join(output_dir, "state-orbit-lifecycle.md")
    generator.write_diagram(orbit_lifecycle, path)
    generated.append(path)

    # 活动图：轨道设计工作流
    orbit_design_activity = generator.generate_activity(
        "OrbitDesign",
        [
            {"id": "start", "label": "开始", "type": "start"},
            {"id": "sys", "label": "创建 CR3BP_System", "type": "process"},
            {"id": "dyn", "label": "创建 CR3BP_Dynamics", "type": "process"},
            {"id": "prop", "label": "传播初始猜测", "type": "process"},
            {"id": "correct", "label": "微分修正", "type": "process"},
            {"id": "cont", "label": "轨道延续", "type": "process"},
            {"id": "end", "label": "获得轨道族", "type": "end"},
        ],
    )
    path = os.path.join(output_dir, "activity-orbit-design.md")
    generator.write_diagram(orbit_design_activity, path)
    generated.append(path)

    # 序列图：传播交互
    propagation_sequence = generator.generate_sequence(
        ["Client", "Dynamics", "solve_ivp"],
        [
            ("Client", "Dynamics", "propagate(state, t_span, with_stm=True)"),
            ("Dynamics", "Dynamics", "_get_eom_func(with_stm=True)"),
            ("Dynamics", "solve_ivp", "integrate 42-dim augmented state"),
            ("solve_ivp", "Dynamics", "result (42 x n_points)"),
            ("Dynamics", "Dynamics", "extract states (n, 6) + STM (n, 6, 6)"),
            ("Dynamics", "Client", "dict{time, states, stm}"),
        ],
    )
    path = os.path.join(output_dir, "sequence-propagation.md")
    generator.write_diagram(propagation_sequence, path)
    generated.append(path)

    # ---- Algorithms 层额外图表 ----

    # 活动图：微分修正 Newton 迭代流程
    correction_activity = generator.generate_activity(
        "DifferentialCorrection",
        [
            {"id": "start", "label": "开始修正", "type": "start"},
            {"id": "config", "label": "加载 CorrectionConfig 策略", "type": "process"},
            {"id": "propagate", "label": "传播半周期 (with_stm=True)", "type": "process"},
            {"id": "error", "label": "计算约束误差向量", "type": "process"},
            {"id": "check", "label": "收敛?", "type": "process"},
            {"id": "update", "label": "Newton 更新自由变量", "type": "process"},
            {"id": "end", "label": "返回收敛轨道", "type": "end"},
        ],
        decision_nodes=[
            {
                "id": "check",
                "label": "收敛?",
                "branches": [
                    {"target": "end", "condition": "是 (error < tol)"},
                    {"target": "update", "condition": "否"},
                ],
            },
        ],
    )
    path = os.path.join(output_dir, "activity-differential-correction.md")
    generator.write_diagram(correction_activity, path)
    generated.append(path)

    # 序列图：微分修正交互
    correction_sequence = generator.generate_sequence(
        ["Client", "DiffCorrection", "Strategy", "Dynamics"],
        [
            ("Client", "DiffCorrection", "correct(initial_state, period)"),
            ("DiffCorrection", "Strategy", "get_free_variable_indices()"),
            ("DiffCorrection", "Dynamics", "propagate(state, T/2, with_stm=True)"),
            ("Dynamics", "DiffCorrection", "states, stm"),
            ("DiffCorrection", "Strategy", "compute_error(orbit, dynamics)"),
            ("Strategy", "DiffCorrection", "error_vector"),
            ("DiffCorrection", "DiffCorrection", "Newton update: dx = -J_inv * error"),
            ("DiffCorrection", "Client", "corrected Orbit"),
        ],
    )
    path = os.path.join(output_dir, "sequence-correction.md")
    generator.write_diagram(correction_sequence, path)
    generated.append(path)

    # 覆盖率报告
    report = req_registry.coverage_report()
    print(f"\n需求覆盖率: {report['coverage_rate']:.1%}")
    print(f"  已覆盖: {report['covered']}/{report['total']}")
    if report.get("uncovered_ids"):
        print(f"  未覆盖: {report['uncovered_ids']}")

    print(f"\n已生成 {len(generated)} 个图表文件:")
    for path in generated:
        print(f"  {path}")

    return generated


if __name__ == "__main__":
    main()
