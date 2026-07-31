"""五层依赖方向检查（ADR 0012）。

硬规则：
- ``api/`` 不 import ``tools/``（工具层辅助，接口层不依赖）
- ``algorithm/`` 不 import ``api/``、``tools/``
- ``data/`` 不 import ``algorithm/``、``api/``、``tools/``（只依赖外部库）

扫描 ``e2m2e/`` 下各层包的 import 语句，违反方向即失败。
"""

from __future__ import annotations

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1] / "e2m2e"

# 层 → 禁止 import 的层
FORBIDDEN: dict[str, set[str]] = {
    "data": {"algorithm", "api", "tools"},
    "algorithm": {"api", "tools"},
    "api": {"tools"},
}


def layer_of(module: str) -> str | None:
    """从 ``e2m2e.<layer>...`` 提取层名。"""
    if not module.startswith("e2m2e."):
        return None
    parts = module.split(".")
    if len(parts) < 2:
        return None
    return parts[1]


def check_file(path: pathlib.Path) -> list[str]:
    """检查单个文件的 import 方向，返回违规列表。"""
    rel = path.relative_to(ROOT)
    src_layer = rel.parts[0]
    if src_layer not in FORBIDDEN:
        return []
    forbidden = FORBIDDEN[src_layer]
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except SyntaxError:
        return [f"{path}: 语法错误，无法检查"]
    violations: list[str] = []
    for node in ast.walk(tree):
        imports: list[str] = []
        if isinstance(node, ast.Import):
            imports = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports = [node.module]
        for mod in imports:
            target = layer_of(mod)
            if target in forbidden:
                line = getattr(node, "lineno", 0)
                violations.append(f"{rel}:{line}: {src_layer}/ import {mod} (违反依赖方向)")
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
    print("依赖方向检查通过（data/algorithm/api 无越层 import）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
