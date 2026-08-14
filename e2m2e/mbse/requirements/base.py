"""需求模型基础类

提供形式化的需求定义和注册机制，支持 SysML 需求图的自动生成。

每个需求具有唯一 ID、分类、优先级、验证方法，以及与代码和测试的追溯链接。
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field

REQUIREMENT_ID_PATTERN = re.compile(r"REQ-\d{3}")
VALID_VERIFICATION_METHODS = frozenset({"test", "analysis", "inspection"})


class RequirementCategory(enum.Enum):
    """需求分类（对应 SysML 需求类型）"""

    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    INTERFACE = "interface"
    VERIFICATION = "verification"
    CONSTRAINT = "constraint"


class RequirementPriority(enum.Enum):
    """需求优先级（对应 SysML <<shall>>, <<should>>, <<may>>）"""

    SHALL = "shall"
    SHOULD = "should"
    MAY = "may"


@dataclass(frozen=True)
class Requirement:
    """形式化需求定义

    对应 SysML Requirement Diagram 中的需求元素。

    Attributes:
        id: 唯一标识符，如 "REQ-001"
        title: 简短标题
        category: 需求分类
        description: 详细描述
        priority: 优先级（shall/should/may）
        verification_method: 验证方法（test/analysis/inspection）
        parent: 父需求 ID（用于需求分解层次）
        linked_code: 关联的源代码模块路径
        linked_tests: 关联的测试文件路径
    """

    id: str
    title: str
    category: RequirementCategory
    description: str
    priority: RequirementPriority = RequirementPriority.SHALL
    verification_method: str = "test"
    parent: str | None = None
    linked_code: list[str] = field(default_factory=list)
    linked_tests: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """校验需求目录的稳定标识与验证方法。"""
        if REQUIREMENT_ID_PATTERN.fullmatch(self.id) is None:
            raise ValueError("需求 ID 必须匹配 REQ-001 格式")
        if self.verification_method not in VALID_VERIFICATION_METHODS:
            raise ValueError("verification_method 必须是 test、analysis 或 inspection 之一")


class RequirementRegistry:
    """需求注册表。

    调用方持有注册表生命周期，支持按分类、层次和追溯关系查询。
    """

    def __init__(self) -> None:
        self._requirements: dict[str, Requirement] = {}

    def register(self, requirement: Requirement) -> None:
        """注册一个需求"""
        if requirement.id in self._requirements:
            raise ValueError(f"需求 {requirement.id} 已存在")
        self._requirements[requirement.id] = requirement

    def register_many(self, requirements: list[Requirement]) -> None:
        """批量注册需求"""
        for req in requirements:
            self.register(req)

    def get(self, req_id: str) -> Requirement:
        """按 ID 获取需求"""
        if req_id not in self._requirements:
            raise KeyError(f"需求 {req_id} 未注册")
        return self._requirements[req_id]

    def all(self) -> list[Requirement]:
        """获取所有已注册需求"""
        return list(self._requirements.values())

    def by_category(self, category: RequirementCategory) -> list[Requirement]:
        """按分类筛选需求"""
        return [r for r in self._requirements.values() if r.category == category]

    def by_parent(self, parent_id: str) -> list[Requirement]:
        """获取指定父需求的子需求"""
        return [r for r in self._requirements.values() if r.parent == parent_id]

    def children_of(self, req_id: str) -> list[Requirement]:
        """获取指定需求的所有子需求（递归）"""
        direct = self.by_parent(req_id)
        all_children = list(direct)
        for child in direct:
            all_children.extend(self.children_of(child.id))
        return all_children

    def traceability_matrix(self) -> dict[str, dict]:
        """构建需求追溯矩阵

        Returns:
            {req_id: {"requirement": Requirement, "code": [...],
                       "tests": [...], "has_coverage": bool}}
        """
        matrix = {}
        for req in self._requirements.values():
            matrix[req.id] = {
                "requirement": req,
                "code": req.linked_code,
                "tests": req.linked_tests,
                "has_coverage": len(req.linked_tests) > 0,
            }
        return matrix

    def coverage_report(self) -> dict:
        """生成需求覆盖率报告

        Returns:
            包含 total/covered/uncovered/coverage_rate/uncovered_ids 的字典
        """
        all_reqs = self.all()
        total = len(all_reqs)
        if total == 0:
            return {"total": 0, "covered": 0, "uncovered": 0, "coverage_rate": 0.0}

        covered = sum(1 for r in all_reqs if r.linked_tests)
        uncovered = total - covered
        return {
            "total": total,
            "covered": covered,
            "uncovered": uncovered,
            "coverage_rate": covered / total,
            "uncovered_ids": [r.id for r in all_reqs if not r.linked_tests],
        }

    def clear(self) -> None:
        """清空所有已注册需求（仅用于测试）"""
        self._requirements.clear()

    def __len__(self) -> int:
        """返回已注册需求数量"""
        return len(self._requirements)

    def __contains__(self, req_id: str) -> bool:
        """检查需求是否已注册"""
        return req_id in self._requirements

    def __iter__(self):
        """遍历所有已注册需求"""
        return iter(self._requirements.values())
