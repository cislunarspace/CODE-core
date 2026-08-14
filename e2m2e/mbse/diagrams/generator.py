"""Mermaid 图表统一生成编排器

从 MBSE 模型定义自动生成 SysML 风格的 Mermaid 图表，嵌入 MkDocs 文档。

支持的图表类型：
- BDD (Block Definition Diagram) → classDiagram
- IBD (Internal Block Diagram) → graph
- Activity Diagram → flowchart
- Sequence Diagram → sequenceDiagram
- State Machine → stateDiagram-v2
- Requirement Diagram → requirementDiagram
"""

from __future__ import annotations

import os

from ..architecture.components import ARCHITECTURE_LAYERS, ComponentRegistry
from ..requirements.base import RequirementCategory, RequirementRegistry


class DiagramGenerator:
    """MBSE 图表生成编排器

    从 RequirementRegistry 和 ComponentRegistry 生成 Mermaid 图表。
    """

    def __init__(
        self,
        requirements: RequirementRegistry | None = None,
        components: ComponentRegistry | None = None,
    ):
        """初始化图表生成器

        Args:
            requirements: 需求注册表实例，默认使用全局单例
            components: 组件注册表实例，默认使用全局单例
        """
        self.requirements = requirements if requirements is not None else RequirementRegistry()
        self.components = components if components is not None else ComponentRegistry()

    def generate_bdd(self, layer: str | None = None) -> str:
        """生成 BDD (Block Definition Diagram) — Mermaid classDiagram

        Args:
            layer: 可选，仅生成指定层的组件图

        Returns:
            Mermaid classDiagram 语法字符串
        """
        components = self.components.by_layer(layer) if layer else self.components.all()

        lines = ["classDiagram"]

        for comp in components:
            # 组件定义
            lines.append(f"    class {comp.name} {{")
            lines.append(f"        &lt;&lt;{comp.layer}&gt;&gt;")
            lines.append(f"        {comp.description}" if comp.description else "")
            lines.append("    }")

            # ADR 0001 后 protocols 默认为空；仅保留对历史元数据的兼容渲染
            for proto in comp.protocols:
                lines.append(f"    {comp.name} ..|> {proto} : implements")

            # 依赖关系
            for dep in comp.dependencies:
                lines.append(f"    {comp.name} --> {dep} : uses")

        return "\n".join(lines)

    def generate_requirement_diagram(self) -> str:
        """生成 Requirement Diagram — Mermaid requirementDiagram

        Returns:
            Mermaid requirementDiagram 语法字符串
        """
        lines = ["requirementDiagram"]

        for req in self.requirements:
            # 将 SysML 需求分类映射为 Mermaid requirementDiagram 的 type 字段
            # 默认 "functional" 兜底未匹配的分类
            type_label = {
                RequirementCategory.FUNCTIONAL: "functional",
                RequirementCategory.PERFORMANCE: "performance",
                RequirementCategory.INTERFACE: "interface",
                RequirementCategory.VERIFICATION: "verification",
                RequirementCategory.CONSTRAINT: "constraint",
            }.get(req.category, "functional")

            lines.append(f"    requirement {req.id.replace('-', '_')} {{")
            lines.append(f"        title: {req.title}")
            lines.append(f"        type: {type_label}")
            lines.append(f"        risk: {req.priority.value}")
            lines.append("    }")

            # 父需求关系
            if req.parent:
                lines.append(
                    f"    {req.parent.replace('-', '_')} -traces-> {req.id.replace('-', '_')}"
                )

            # 代码追溯
            for code_path in req.linked_code:
                safe_name = code_path.replace(".", "_").replace("/", "_")
                lines.append(f"    {safe_name} -satisfies-> {req.id.replace('-', '_')}")

        return "\n".join(lines)

    def generate_traceability_matrix(self) -> str:
        """生成需求到代码和测试的 Markdown 追溯矩阵。"""
        lines = [
            "| 需求 ID | 标题 | 类别 | 优先级 | 验证方法 | 关联代码 | 关联测试 |",
            "|---------|------|------|--------|----------|----------|----------|",
        ]
        for requirement in self.requirements:
            code = "<br>".join(requirement.linked_code)
            tests = "<br>".join(requirement.linked_tests)
            lines.append(
                "| "
                f"{requirement.id} | {requirement.title} | {requirement.category.value} | "
                f"{requirement.priority.value} | {requirement.verification_method} | "
                f"{code} | {tests} |"
            )
        return "\n".join(lines)

    def generate_state_machine(
        self, name: str, states: list[str], transitions: list[tuple[str, str, str]]
    ) -> str:
        """生成 State Machine Diagram — Mermaid stateDiagram-v2

        Args:
            name: 状态机名称
            states: 状态列表
            transitions: 转换列表 [(from_state, to_state, trigger)]

        Returns:
            Mermaid stateDiagram-v2 语法字符串
        """
        lines = ["stateDiagram-v2"]
        lines.append(f"    [*] --> {states[0]}")

        for from_state, to_state, trigger in transitions:
            lines.append(f"    {from_state} --> {to_state} : {trigger}")

        # 终止状态集合：覆盖数值求解的典型终态（收敛/发散）和通用终态
        terminal_states = {"complete", "converged", "failed", "diverged"}
        for state in states:
            if state.lower() in terminal_states:
                lines.append(f"    {state} --> [*]")

        return "\n".join(lines)

    def generate_activity(
        self, name: str, steps: list[dict], decision_nodes: list[dict] | None = None
    ) -> str:
        """生成 Activity Diagram — Mermaid flowchart

        Args:
            name: 活动名称
            steps: 步骤列表 [{"id": str, "label": str, "type": "process|io|start|end"}]
            decision_nodes: 决策节点
                [{"id": str, "label": str, "branches": [{"target": str, "condition": str}]}]

        Returns:
            Mermaid flowchart 语法字符串
        """
        lines = ["flowchart TD"]

        for step in steps:
            step_id = step["id"]
            label = step["label"]
            step_type = step.get("type", "process")

            if step_type == "start" or step_type == "end":
                lines.append(f"    {step_id}([{label}])")
            elif step_type == "io":
                lines.append(f"    {step_id}[/{label}/]")
            else:
                lines.append(f"    {step_id}[{label}]")

        # 顺序连接
        for i in range(len(steps) - 1):
            lines.append(f"    {steps[i]['id']} --> {steps[i + 1]['id']}")

        # 决策节点
        if decision_nodes:
            for decision in decision_nodes:
                lines.append(f"    {decision['id']}{{{decision['label']}}}")
                for branch in decision["branches"]:
                    lines.append(
                        f"    {decision['id']} -->|{branch['condition']}| {branch['target']}"
                    )

        return "\n".join(lines)

    def generate_sequence(
        self, participants: list[str], interactions: list[tuple[str, str, str]]
    ) -> str:
        """生成 Sequence Diagram — Mermaid sequenceDiagram

        Args:
            participants: 参与者列表
            interactions: 交互列表 [(from, to, message)]

        Returns:
            Mermaid sequenceDiagram 语法字符串
        """
        lines = ["sequenceDiagram"]

        for p in participants:
            lines.append(f"    participant {p}")

        for from_p, to_p, message in interactions:
            lines.append(f"    {from_p}->>{to_p}: {message}")

        return "\n".join(lines)

    def generate_ibd(self, component_name: str, attributes: list[dict]) -> str:
        """生成 IBD (Internal Block Diagram) — Mermaid graph

        Args:
            component_name: 组件名称
            attributes: 属性列表 [{"name": str, "type": str, "direction": "input|output|internal"}]

        Returns:
            Mermaid graph 语法字符串
        """
        lines = ["graph LR"]

        lines.append(f"    subgraph {component_name}")
        for attr in attributes:
            direction = attr.get("direction", "internal")
            name = attr["name"]
            type_str = attr.get("type", "any")
            if direction == "input":
                lines.append(f"        in_{name}([{name}: {type_str}])")
                lines.append(f"        in_{name} --> {component_name}_core")
            elif direction == "output":
                lines.append(f"        out_{name}([{name}: {type_str}])")
                lines.append(f"        {component_name}_core --> out_{name}")
            else:
                lines.append(f"        {name}[{name}: {type_str}]")
        lines.append("    end")

        return "\n".join(lines)

    def write_document(self, title: str, content: str, output_path: str) -> None:
        """将受管 Markdown 文档写入指定位置。"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(f"---\ntitle: {title}\n---\n\n# {title}\n\n{content}\n")

    def write_diagram(self, title: str, content: str, output_path: str) -> None:
        """将 Mermaid 图表作为受管 Markdown 文档写入指定位置。"""
        self.write_document(title, f"```mermaid\n{content}\n```", output_path)

    def generate_all(self, output_dir: str) -> list[str]:
        """生成默认模型的受管图表和追溯文档。

        Args:
            output_dir: 输出目录路径

        Returns:
            生成的文件路径列表
        """
        generated = []
        layer_titles = {
            "data": "数据",
            "numerical": "数值",
            "algorithm": "算法",
            "api": "接口",
            "tools": "工具",
        }

        for layer in ARCHITECTURE_LAYERS:
            if layer == "mbse":
                continue
            bdd_content = self.generate_bdd(layer)
            if bdd_content == "classDiagram":
                continue
            path = os.path.join(output_dir, f"bdd-{layer}.md")
            self.write_diagram(f"BDD：{layer_titles[layer]}层", bdd_content, path)
            generated.append(path)

        if len(self.requirements) > 0:
            requirement_path = os.path.join(output_dir, "requirements.md")
            self.write_diagram("功能需求", self.generate_requirement_diagram(), requirement_path)
            generated.append(requirement_path)

            matrix_path = os.path.join(output_dir, "traceability-matrix.md")
            self.write_document("需求追溯矩阵", self.generate_traceability_matrix(), matrix_path)
            generated.append(matrix_path)

        return generated
