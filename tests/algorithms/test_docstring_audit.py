"""注释完备性审计测试。

通过 import 实际模块、读取运行时对象的 ``__doc__`` 来检查
e2m2e/algorithms/ 下各模块的 docstring 完备性。测的是行为
（运行时这个对象真的能拿到非空 docstring），而不是源码文本特征。

结构性源码检查（某行有注释、源码中不含某字符串等）已从此文件移除，
参见 issue #217：那类检查更适合交给 ruff 的 D 规则或 pre-commit AST lint。
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import pytest

# 行情：用 CJK 统一表意文字的 Unicode 范围判断是否含中文字符。
_CJK_FIRST = "一"
_CJK_LAST = "鿿"


def _has_chinese(text: str) -> bool:
    return any(_CJK_FIRST <= ch <= _CJK_LAST for ch in text)


def _import(dotted: str) -> Any:
    """按点号路径导入并返回模块对象，导入失败即让测试报错。"""
    import importlib

    return importlib.import_module(dotted)


def _doc(obj: Any) -> str:
    """取运行时对象的 docstring，剥离缩进，空时返回空串。"""
    return inspect.getdoc(obj) or ""


def _assert_doc(obj: Any, label: str) -> str:
    """断言对象有 docstring，返回 docstring 供后续断言使用。"""
    ds = _doc(obj)
    assert ds, f"{label} 缺少 docstring"
    return ds


def _public_functions(module: Any) -> list[tuple[str, Callable[..., Any]]]:
    """枚举模块中本模块定义的公开函数（不以下划线开头、不重导入）。"""
    out: list[tuple[str, Callable[..., Any]]] = []
    for name, obj in vars(module).items():
        if name.startswith("_"):
            continue
        if not inspect.isfunction(obj):
            continue
        if obj.__module__ != module.__name__:
            continue
        out.append((name, obj))
    return out


def _public_classes(module: Any) -> list[tuple[str, type]]:
    """枚举模块中本模块定义的公开类（不以下划线开头、不重导入）。"""
    out: list[tuple[str, type]] = []
    for name, obj in vars(module).items():
        if name.startswith("_"):
            continue
        if not inspect.isclass(obj):
            continue
        if obj.__module__ != module.__name__:
            continue
        out.append((name, obj))
    return out


# ---------------------------------------------------------------------------
# two_level_multiple_shooting.py
# ---------------------------------------------------------------------------


class TestTwoLevelMultipleShooting:
    MODULE = "e2m2e.algorithms.two_level_multiple_shooting"

    def test_module_docstring_exists(self):
        mod = _import(self.MODULE)
        ds = _assert_doc(mod, "two_level_multiple_shooting")
        assert _has_chinese(ds), "模块 docstring 应为中文"

    def test_dataclass_has_docstring(self):
        mod = _import(self.MODULE)
        ds = _assert_doc(mod.TwoLevelMultipleShootingResult, "TwoLevelMultipleShootingResult")
        assert _has_chinese(ds), "TwoLevelMultipleShootingResult docstring 应为中文"

    def test_class_has_docstring(self):
        mod = _import(self.MODULE)
        ds = _assert_doc(mod.TwoLevelMultipleShooting, "TwoLevelMultipleShooting")
        assert _has_chinese(ds), "TwoLevelMultipleShooting docstring 应为中文"

    @pytest.mark.parametrize(
        "name",
        [
            "correct",
            "_validate_inputs",
            "_run_level1",
            "_run_level2",
            "_compute_residuals",
            "_is_converged",
            "_result",
        ],
    )
    def test_method_has_docstring(self, name):
        mod = _import(self.MODULE)
        method = getattr(mod.TwoLevelMultipleShooting, name, None)
        assert method is not None, f"TwoLevelMultipleShooting.{name} 不存在"
        _assert_doc(method, f"TwoLevelMultipleShooting.{name}")

    @pytest.mark.parametrize(
        "name",
        [
            "_build_level1_constraint",
            "_build_level1_jacobian",
            "_build_level2_constraint",
            "_build_level2_patch_jacobian",
        ],
    )
    def test_helper_function_has_docstring(self, name):
        mod = _import(self.MODULE)
        func = getattr(mod, name, None)
        assert func is not None, f"{name} 不存在"
        _assert_doc(func, name)


# ---------------------------------------------------------------------------
# multiple_shooting.py
# ---------------------------------------------------------------------------


class TestMultipleShooting:
    MODULE = "e2m2e.algorithms.multiple_shooting"

    def test_module_docstring_exists(self):
        mod = _import(self.MODULE)
        ds = _assert_doc(mod, "multiple_shooting")
        assert _has_chinese(ds), "模块 docstring 应为中文"


# ---------------------------------------------------------------------------
# continuation.py
# ---------------------------------------------------------------------------


class TestContinuation:
    MODULE = "e2m2e.algorithms.continuation"

    def test_init_docstring_param_matches_signature(self):
        """__init__ docstring 中不应包含不存在的 param 参数"""
        mod = _import(self.MODULE)
        ds = _assert_doc(mod.Continuation.__init__, "Continuation.__init__")
        assert "- param:" not in ds, (
            "docstring 不应包含 param 参数（签名中实际参数名为 step）"
        )
        assert "step" in ds, "docstring 中应包含 step 参数说明"

    def test_infer_param_index_has_docstring(self):
        mod = _import(self.MODULE)
        _assert_doc(
            mod.Continuation._infer_param_index,
            "Continuation._infer_param_index",
        )

    def test_build_family_result_has_docstring(self):
        mod = _import(self.MODULE)
        _assert_doc(
            mod.Continuation._build_family_result,
            "Continuation._build_family_result",
        )


# ---------------------------------------------------------------------------
# stability.py
# ---------------------------------------------------------------------------


class TestStability:
    MODULE = "e2m2e.algorithms.stability"

    def test_pair_eigenvalues_has_args_returns(self):
        mod = _import(self.MODULE)
        ds = _assert_doc(
            mod.StabilityAnalysis._pair_eigenvalues,
            "StabilityAnalysis._pair_eigenvalues",
        )
        assert "Returns" in ds, "_pair_eigenvalues docstring 缺少 Returns 段"


# ---------------------------------------------------------------------------
# differential_correction.py
# ---------------------------------------------------------------------------


class TestDifferentialCorrection:
    MODULE = "e2m2e.algorithms.differential_correction"

    def test_iterate_correction_documents_callback(self):
        mod = _import(self.MODULE)
        ds = _assert_doc(
            mod.DifferentialCorrection.iterate_correction,
            "iterate_correction",
        )
        assert "callback" in ds, "iterate_correction docstring 缺少 callback 参数说明"

    def test_iterate_correction_documents_none_return(self):
        mod = _import(self.MODULE)
        ds = _assert_doc(
            mod.DifferentialCorrection.iterate_correction,
            "iterate_correction",
        )
        assert "None" in ds, "iterate_correction Returns 段应说明可能返回 None"


# ---------------------------------------------------------------------------
# strategies/ — 中文 docstring
# ---------------------------------------------------------------------------


class TestStrategiesChinese:
    STRATEGY_MODULES = [
        "e2m2e.algorithms.strategies.base",
        "e2m2e.algorithms.strategies.halo",
        "e2m2e.algorithms.strategies.symmetric_2d",
        "e2m2e.algorithms.strategies.symmetric_3d",
    ]

    @pytest.mark.parametrize("dotted", STRATEGY_MODULES, ids=lambda d: d.rsplit(".", 1)[1])
    def test_module_docstring_is_chinese(self, dotted):
        mod = _import(dotted)
        ds = _assert_doc(mod, dotted)
        assert _has_chinese(ds), f"{dotted} 模块 docstring 应为中文"

    @pytest.mark.parametrize("dotted", STRATEGY_MODULES, ids=lambda d: d.rsplit(".", 1)[1])
    def test_all_public_functions_have_chinese_docstring(self, dotted):
        mod = _import(dotted)
        for name, func in _public_functions(mod):
            ds = _assert_doc(func, f"{dotted}::{name}")
            assert _has_chinese(ds), f"{dotted}::{name} docstring 应为中文"

    @pytest.mark.parametrize("dotted", STRATEGY_MODULES, ids=lambda d: d.rsplit(".", 1)[1])
    def test_class_docstrings_are_chinese(self, dotted):
        mod = _import(dotted)
        for name, cls in _public_classes(mod):
            ds = _assert_doc(cls, f"{dotted}::{name}")
            assert _has_chinese(ds), f"{dotted}::{name} docstring 应为中文"
