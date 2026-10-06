"""导出声明的架构契约测试（#797）。

公共模块维护显式 ``__all__`` 是 AGENTS.md 的公共面约定：星号导入与
``__init__`` re-export 都以它为准，漏列名字会让调用方拿到残缺的面。
本文件钉住两处曾经缺口的契约面：包根共享内核叶的导出声明，以及
``e2m2e.api.models`` 子包逐名 re-export 与 ``__all__`` 的一致性。
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

from e2m2e import exceptions, integrators, spice_ext, status
from e2m2e.api import models

pytestmark = pytest.mark.aux

_SHARED_KERNEL_LEAVES = (exceptions, status, spice_ext, integrators)


def test_shared_kernel_leaves_declare_exports():
    """四个共享内核叶都显式声明 ``__all__``，且声明的名字都在模块里。"""
    for module in _SHARED_KERNEL_LEAVES:
        declared = getattr(module, "__all__", None)
        assert declared is not None, f"{module.__name__} 缺少 __all__"
        assert declared, f"{module.__name__} 的 __all__ 为空"
        for name in declared:
            assert hasattr(module, name), f"{module.__name__}.__all__ 含未定义名字 {name}"


def test_exceptions_exports_exception_hierarchy():
    """exceptions 叶导出异常层次的三个公共类。"""
    assert set(exceptions.__all__) == {
        "E2M2EError",
        "RustExtensionUnavailableError",
        "PropagationFailure",
    }


def test_models_reexports_match_declared_exports():
    """models 子包的每个公开名字都在包级 ``__all__`` 里，且声明齐全。

    ``e2m2e.api.models`` 的契约是逐名 re-export 子模块公开面（模块
    docstring），子模块 ``__all__`` 与包级 ``__all__`` 任何一侧漏列都会
    让 ``from e2m2e.api.models import *`` 拿到残缺的面（#797）。
    """
    for info in pkgutil.iter_modules(models.__path__):
        submodule = importlib.import_module(f"{models.__name__}.{info.name}")
        for name in getattr(submodule, "__all__", []):
            assert name in models.__all__, f"{submodule.__name__} 的 {name} 未进包级 __all__"
    for name in models.__all__:
        assert name in vars(models), f"e2m2e.api.models.__all__ 含未定义名字 {name}"
