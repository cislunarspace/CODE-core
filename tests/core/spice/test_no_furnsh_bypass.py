"""静态门控：禁止 e2m2e 生产代码绕过 ``SPICEManager`` 直接调 furnsh/unload。

背景（issue #334）：Python（spiceypy）与 Rust（cspice-sys）是两个独立 CSPICE
实例，内核池互不共享。``SPICEManager.load_kernel`` 是唯一在两侧同时 furnsh 的
公开入口；任何直接调 ``spiceypy.furnsh``/``spiceypy.unload`` 的生产路径都会让
Rust 实例内核池为空，下沉到 Rust 的力模型查询会报晦涩错误或杀进程。

本测试扫描 ``e2m2e/`` 下所有 ``.py``（排除测试、构建产物），断言白名单外无
``.furnsh(``/``.unload(`` 调用。匹配调用本身（``.furnsh(``）而非变量名，覆盖
``import spiceypy as X; X.furnsh(...)`` 等别名模式；排除注释行。

白名单（合法直接调用点）：
- ``e2m2e/data/kernels/manager.py``：``SPICEManager`` 实现本身
- ``e2m2e/data/kernels/_spice_loader.py``：spiceypy 惰性加载器（预留）
"""

from __future__ import annotations

import os
import re

import pytest

pytestmark = pytest.mark.l1

# 仓库根：tests/core/spice/ → 上溯四级
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_E2M2E_DIR = os.path.join(_REPO_ROOT, "e2m2e")

# 合法的直接 furnsh/unload 调用点（相对 e2m2e/ 的 POSIX 路径）。
_WHITELIST: set[str] = {
    "data/kernels/manager.py",
    "data/kernels/_spice_loader.py",
}

# 匹配 ``.furnsh(`` 或 ``.unload(`` 调用（前导点号排除定义/import）。
_CALL_RE = re.compile(r"\.(?:furnsh|unload)\s*\(")

# 行首注释（# ...）——整行跳过。不处理行内注释（生产代码里调用不会跟在代码后）。
_COMMENT_RE = re.compile(r"^\s*#")


def _iter_python_files(root: str):
    """遍历 ``root`` 下所有 ``.py``，yield 相对路径。"""
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".py"):
                abs_path = os.path.join(dirpath, name)
                yield os.path.relpath(abs_path, root)


def test_no_furnsh_unload_bypass_in_production_code():
    """e2m2e/ 下白名单外不得出现直接 ``.furnsh(``/``.unload(`` 调用。"""
    violations: list[str] = []
    for rel_path in _iter_python_files(_E2M2E_DIR):
        if rel_path in _WHITELIST:
            continue
        abs_path = os.path.join(_E2M2E_DIR, rel_path)
        with open(abs_path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if _COMMENT_RE.match(line):
                    continue
                if _CALL_RE.search(line):
                    violations.append(f"{rel_path}:{lineno}: {line.rstrip()}")

    assert not violations, (
        "发现绕过 SPICEManager 的直接 furnsh/unload 调用（应改走 "
        "SPICEManager.load_kernel / unload_kernel，见 issue #334）：\n" + "\n".join(violations)
    )
