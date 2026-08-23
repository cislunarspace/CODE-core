"""Sphinx configuration for e2m2e documentation."""

import os
import sys

# 添加项目根目录到 path
sys.path.insert(0, os.path.abspath(".."))

# -- General configuration ------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
    "myst_parser",
]

# 中文支持
language = "zh_CN"
locale_dirs = ["locale/"]

# MBSE 参考文档（.md）以 YAML frontmatter 的 title 作为页面标题
myst_title_to_page_title = True

# ADR、research 与 agents 文档面向开发协作，不进用户文档站点（ADR 索引见 architecture/index.md）
exclude_patterns = ["_build", "adr", "research", "agents", "Thumbs.db", ".DS_Store"]

# MBSE 参考文档（.md）里的 ```mermaid 围栏交给 sphinxcontrib-mermaid 渲染
myst_fence_as_directive = ["mermaid"]

# 源文件后缀：.rst 为主，.md（architecture / MBSE 参考）经 myst-parser 解析
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"

# 项目信息
project = "e2m2e"
copyright = "2026, ouyangjiahong"
author = "ouyangjiahong"

# 版本信息
from e2m2e import __version__  # noqa: E402

release = __version__

# -- Options for HTML output ----------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = []

# -- Extension configuration ----------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_ivar = True

# e2m2e.api.mcp.server 依赖 [mcp] extra（mcp / anyio），文档构建环境不必安装
autodoc_mock_imports = ["mcp", "anyio"]

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
