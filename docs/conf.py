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
]

# 中文支持
language = "zh_CN"
locale_dirs = ["locale/"]

# 源文件后缀
source_suffix = ".rst"
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
html_static_path = ["_static"]

# -- Extension configuration ----------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_ivar = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
