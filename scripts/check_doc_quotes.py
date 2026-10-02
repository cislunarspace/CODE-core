"""文档与代码注释的直角引号检查（issue #777）。

口径（AGENTS.md 的引号条款）：文档与代码内中文注释、docstring 不得使用直角引号
（U+300C/U+300D）与直角单引号（U+300E/U+300F）；中文引用词语统一用弯引号
（U+201C/U+201D），嵌套用弯单引号（U+2018/U+2019），同一文件内保持一致。本脚本
扫描仓库根 ``*.md``、``docs/`` 下的 md/rst/py（含 ADR）、``.out-of-scope/`` 下的
md、``e2m2e``/``tests``/``scripts`` 下的 py 与 ``crates`` 下的 rs，命中直角引号即
失败；`.github/`、`datasets/`、`kernels/` 与构建产物 ``_build/`` 不在扫描范围内。

豁免：仅 ``AGENTS.md`` 的规则正文需要引用被禁字符本身，放行四个直角引号字符；
其余文件（含历史 ADR）一律按口径归一。

误报处理：确需在正文里指称被禁字符时，改写为码点描述（如 U+300C）。

被禁字符在本脚本内一律用 ``\\u`` 转义构造（脚本自身也在扫描范围内），
``violations_for`` 为纯函数，供 ``tests/_meta/test_check_doc_quotes.py`` 直接
调用。
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# 被禁字符 -> 标签。
BANNED: dict[str, str] = {
    "\u300c": "直角引号",
    "\u300d": "直角引号",
    "\u300e": "直角单引号",
    "\u300f": "直角单引号",
}

# 相对路径 -> 该文件允许出现的被禁字符（不在表中 = 一个都不允许）。
EXEMPT: dict[str, frozenset[str]] = {
    # 规则正文引用被禁字符本身的元引用。
    "AGENTS.md": frozenset(BANNED),
}

# 扫描面（相对仓库根）。新增文档目录时在此登记。
SCAN_GLOBS: tuple[str, ...] = (
    "*.md",
    "docs/**/*.md",
    "docs/**/*.rst",
    "docs/**/*.py",
    ".out-of-scope/**/*.md",
    "e2m2e/**/*.py",
    "tests/**/*.py",
    "scripts/**/*.py",
    "crates/**/*.rs",
)

# 构建产物目录名，逐段比对。
SKIP_DIRS: frozenset[str] = frozenset({"_build"})


def violations_for(rel: str, text: str) -> list[str]:
    """返回 ``rel`` 内容的违规行列表（``相对路径:行号: 标签 内容``）。"""
    allowed = EXEMPT.get(rel, frozenset())
    violations: list[str] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        labels: list[str] = []
        for char, label in BANNED.items():
            if char in line and char not in allowed and label not in labels:
                labels.append(label)
        if labels:
            violations.append(f"{rel}:{lineno}: {'、'.join(labels)} {line.strip()}")
    return violations


def check_file(path: pathlib.Path) -> list[str]:
    """返回该文件违规行列表（``相对路径:行号: 标签 内容``）。"""
    rel = path.relative_to(ROOT).as_posix()
    return violations_for(rel, path.read_text(encoding="utf-8"))


def scan_paths() -> list[pathlib.Path]:
    """返回扫描面内待检文件（绝对路径，排序去重）。"""
    paths: set[pathlib.Path] = set()
    for pattern in SCAN_GLOBS:
        paths.update(ROOT.glob(pattern))
    return sorted(p for p in paths if p.is_file() and not SKIP_DIRS.intersection(p.parts))


def main() -> int:
    violations: list[str] = []
    for path in scan_paths():
        violations.extend(check_file(path))
    if violations:
        print("文档/注释引号口径违规（中文引用词语统一用弯引号）：")
        for v in violations:
            print(f"  {v}")
        print("整改口径见 AGENTS.md 引号条款；仅 AGENTS.md 规则正文的元引用豁免。")
        return 1
    print("文档引号检查通过（直角引号无残留）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
