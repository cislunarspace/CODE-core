"""
e2m2e - Earth to Moon, Moon to Earth Transfer Orbit Design Library

[English]

A Python library for designing and analyzing cislunar transfer orbits,
focused on orbit dynamics in the Circular Restricted Three-Body Problem
(CR3BP). Five-layer architecture (ADR 0011): data/ (data layer), crates/
(numerical layer), algorithm/ (algorithm layer), api/ (interface layer),
tools/ (tools layer).

Key capabilities: mission orbit design, station keeping, transfer design,
orbit prediction, spacetime coordinate conversion.

Repo-wide map and the journey of one orbit task: see the README section
"How to read this repository".

[简体中文]

一个用于设计和分析地月空间转移轨道的Python库，专注于圆型限制性三体问题
（CR3BP）中的轨道动力学。五层架构（ADR 0011）：data/（数据层）、crates/
（数值层）、algorithm/（算法层）、api/（接口层）、tools/（工具层）。

主要能力：任务轨道设计、轨道保持、转移轨道设计、轨道预报、时空坐标转换。

仓库全貌与一条任务链的走读见 README 的仓库怎么读一节。

作者: 天疆说
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("e2m2e")
except PackageNotFoundError:
    try:
        from pathlib import Path

        import tomllib

        _pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        if _pyproject.exists():
            with open(_pyproject, "rb") as _f:
                __version__ = tomllib.load(_f)["project"]["version"]
        else:
            __version__ = "0.0.0"
    except Exception:
        __version__ = "0.0.0"

__author__ = "天疆说"
__email__ = "ouyangjiahong22@nudt.edu.cn"

# 导入子包（五层 + mbse 独立顶层 + integrators 数值层门面）
from . import algorithm, api, data, integrators, mbse, tools

__all__ = [
    "data",
    "algorithm",
    "api",
    "tools",
    "mbse",
    "integrators",
]

# ---- Import-time Rust ABI 校验 / Import-time Rust ABI check ----
# 若 Rust 扩展已在进程内加载（如用户直引 e2m2e._integrators），立即校验版本，
# 避免过期二进制静默产生错误结果。扩展未加载时静默跳过（惰性，首次 Rust 使用
# 时由 spice_ext._check_rust_abi() 接管）。
# (If the Rust extension is already loaded in-process, e.g. via direct import of
# e2m2e._integrators, verify its version immediately so stale binaries cannot
# silently produce wrong results. If not loaded, skip quietly — lazy checking is
# then handled by spice_ext._check_rust_abi() at first Rust use.)
import sys as _sys

if "e2m2e._integrators" in _sys.modules:
    from .spice_ext import _check_rust_abi as _check_abi

    _check_abi()
