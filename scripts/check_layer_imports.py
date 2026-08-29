"""五层依赖方向检查（ADR 0012，经 ADR 0039 修订）。

硬规则：
- ``api/`` 不 import ``tools/``、``mbse/``（工具层辅助；mbse 为非运行时文档产物）
- ``algorithm/`` 不 import ``api/``、``tools/``、``mbse/``
- ``data/`` 不 import ``algorithm/``、``api/``、``tools/``、``mbse/``，也不得
  穿数值层门面 ``integrators``（SPICE 桥接走共享内核叶 ``spice_ext``）
- 包根共享内核叶（``exceptions``/``status``/``spice_ext``）与数值层门面
  ``integrators`` 自身不 import 任何层（``__init__`` 组合根与构建产物
  ``_rust_abi`` 豁免；新增包根模块必须在此登记）

相对 import（``from ..x import y``、``from . import z``）按文件所在包
解析为绝对路径后检查；``from e2m2e import <层>`` 的别名形态同样检查。
"""

from __future__ import annotations

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1] / "e2m2e"

# 受检层 → 禁止 import 的目标
FORBIDDEN: dict[str, set[str]] = {
    "data": {"algorithm", "api", "tools", "integrators", "mbse"},
    "algorithm": {"api", "tools", "mbse"},
    "api": {"tools", "mbse"},
}

# 全部层名（含不作为受检源、但作为禁入目标的层）
_ALL_LAYERS = frozenset({"data", "algorithm", "api", "tools", "mbse", "integrators"})

# 包根模块：自身不得 import 任何层（共享内核叶 + 数值层门面，ADR 0039）
_ROOT_NO_LAYER_MODULES = frozenset({"exceptions", "status", "spice_ext", "integrators"})

# 豁免的包根模块：组合根（装配全部层）与 maturin build.rs 生成的产物
_ROOT_EXEMPT = frozenset({"__init__", "_rust_abi"})


def layer_of(module: str) -> str | None:
    """从 ``e2m2e.<layer>...`` 提取层名。"""
    if not module.startswith("e2m2e."):
        return None
    parts = module.split(".")
    if len(parts) < 2:
        return None
    return parts[1]


def _package_anchor(rel: pathlib.Path) -> list[str]:
    """文件的 level=1 锚包（``__init__.py`` 是包本身，普通模块是所在包）。"""
    return ["e2m2e", *rel.parts[:-1]]


def _resolve(module: str | None, level: int, rel: pathlib.Path) -> list[str]:
    """把（可能相对的）import 目标解析为绝对模块路径。"""
    if level == 0:
        return [module] if module else []
    anchor = _package_anchor(rel)
    # level=n 表示上退 (n-1) 级；锚包本身已是 level=1 的落点。
    if len(anchor) < level:
        return []
    base = anchor[: len(anchor) - (level - 1)]
    return [*base, *(module.split(".") if module else [])]


def _targets(node: ast.AST, rel: pathlib.Path) -> list[str]:
    """枚举一条 import 语句指向的全部绝对模块路径（含别名展开）。"""
    resolved: list[str] = []
    if isinstance(node, ast.Import):
        resolved = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        module, level = node.module or None, node.level
        base = _resolve(module, level, rel)
        if not base:
            return []
        prefix = ".".join(base)
        if module or level == 0:
            # from X import a, b —— 本体与每个别名各算一个目标
            resolved = [prefix, *(f"{prefix}.{alias.name}" for alias in node.names)]
        else:
            # from . import a, b —— 别名即子模块
            resolved = [f"{prefix}.{alias.name}" for alias in node.names]
    return resolved


def _forbidden_for(rel: pathlib.Path) -> set[str] | None:
    """该文件适用的禁入层集合；None 表示不受检。"""
    if len(rel.parts) == 1:  # 包根模块
        stem = rel.parts[0][: -len(".py")] if rel.parts[0].endswith(".py") else rel.parts[0]
        if stem in _ROOT_EXEMPT:
            return None
        if stem in _ROOT_NO_LAYER_MODULES:
            return set(_ALL_LAYERS)
        return None
    return FORBIDDEN.get(rel.parts[0])


def check_file(path: pathlib.Path) -> list[str]:
    """检查单个文件的 import 方向，返回违规列表。"""
    rel = path.relative_to(ROOT)
    forbidden = _forbidden_for(rel)
    if not forbidden:
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except SyntaxError:
        return [f"{path}: 语法错误，无法检查"]
    seen: set[tuple[str, int, str]] = set()
    violations: list[str] = []
    for node in ast.walk(tree):
        for mod in _targets(node, rel):
            target = layer_of(mod)
            if target in forbidden:
                line = getattr(node, "lineno", 0)
                key = (str(rel), line, target)
                if key not in seen:
                    seen.add(key)
                    violations.append(f"{rel}:{line}: 越层 import {mod} (违反依赖方向)")
    return violations


def main() -> int:
    violations: list[str] = []
    for path in sorted(ROOT.rglob("*.py")):
        violations.extend(check_file(path))
    if violations:
        print("依赖方向违规：")
        for v in violations:
            print(f"  {v}")
        return 1
    print("依赖方向检查通过（相对 import 已解析；包根共享内核与 integrators 门面受检）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
