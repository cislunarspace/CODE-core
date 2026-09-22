"""Sphinx 配置：e2m2e 在线文档站（issue #651）。

构建机不安装 e2m2e 本身（无 Rust 扩展、无 SPICE 内核），因此：

- 版本号不 import 包，用 tomllib 读仓库根 ``pyproject.toml`` 的 ``[project] version``
  （与 ``e2m2e/__init__.py`` 的 importlib.metadata 同源，Python >= 3.11 标准库）；
- API 参考由 sphinx-autoapi 静态解析源码生成，不执行任何模块代码。
"""

from __future__ import annotations

from pathlib import Path

import tomllib

_DOCS = Path(__file__).resolve().parent
_REPO = _DOCS.parent

with open(_REPO / "pyproject.toml", "rb") as _f:
    _pyproject = tomllib.load(_f)

project = "e2m2e"
author = _pyproject["project"]["authors"][0]["name"]
copyright = "2026, e2m2e authors"
release = _pyproject["project"]["version"]
version = release

extensions = [
    "autoapi.extension",
    "myst_parser",
]

language = "zh_CN"

# 主题：shibuya（明暗双模式、中文排版友好）。构建机只装文档工具链，
# 不 import 包本体。
html_theme = "shibuya"

# autoapi 的静态导入解析无法解析 Rust 扩展符号（e2m2e.integrators.*），
# 告警属预期而非文档缺陷；[docutils] 告警全部来自 autoapi 对 Google 风格
# docstring 的纯 rst 渲染（定义列表缩进等），本站自写页面已验证无此类问题。
suppress_warnings = ["autoapi.python_import_resolution", "docutils"]

exclude_patterns = [
    "_build",
    # myst_parser 注册 .md 后缀后，ADR 会被当作待构建文档产生大量 toctree 告警；
    # ADR 不是面向用户的文档，不收录进站点
    "adr",
    "adr/*",
]

# ---- sphinx-autoapi：静态解析源码生成 API 参考，构建机无需可运行的扩展 ----

autoapi_dirs = [str(_REPO / "e2m2e")]
autoapi_type = "python"
autoapi_add_toctree_entry = False
autoapi_output_dir = "autoapi"
# mbse 是独立顶层子系统（不在五层依赖链内），不在文档站收录范围。
# data/kernels 是隐式命名空间子包（无 __init__.py，sphinx-autoapi 的模块名
# 推导在其下截断出错）。
# algorithm/ 与 data/ 不入站：其 docstring 为 Google 风格且存在大量跨模块
# 重导出，autoapi 纯 rst 渲染会产生数百条结构告警与对象重复注册（issue #651
# 记录）；接口层 / integrators 门面 / 契约模块保持全量收录。
autoapi_ignore = [
    "*/mbse/*",
    "*/mbse/*/*",
    "*/data/kernels/*",
    "*/algorithm/*",
    "*/data/*",
    "*/_rust_abi.py",
]
