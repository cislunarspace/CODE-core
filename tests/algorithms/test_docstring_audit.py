"""注释完备性审计测试

使用 AST 检查 e2m2e/algorithms/ 下各模块的 docstring 完备性。
每个测试对应 issue #37 中的一个审计项。
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

ALGORITHMS_DIR = Path(__file__).resolve().parents[2] / "e2m2e" / "algorithms"


def _parse_module(filepath: Path) -> ast.Module:
    return ast.parse(filepath.read_text(encoding="utf-8"), filename=str(filepath))


def _docstring(node: ast.AST) -> str:
    return ast.get_docstring(node) or ""


def _has_chinese(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in text)


# ---------------------------------------------------------------------------
# two_level_multiple_shooting.py
# ---------------------------------------------------------------------------


class TestTwoLevelMultipleShooting:
    FILE = ALGORITHMS_DIR / "two_level_multiple_shooting.py"

    def test_module_docstring_exists(self):
        tree = _parse_module(self.FILE)
        ds = _docstring(tree)
        assert ds, "two_level_multiple_shooting.py 缺少模块 docstring"
        assert _has_chinese(ds), "模块 docstring 应为中文"

    def test_dataclass_has_docstring(self):
        tree = _parse_module(self.FILE)
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef,)) and node.name == "TwoLevelMultipleShootingResult":
                ds = _docstring(node)
                assert ds, "TwoLevelMultipleShootingResult 缺少 docstring"
                assert _has_chinese(ds), "TwoLevelMultipleShootingResult docstring 应为中文"
                return
        pytest.fail("未找到 TwoLevelMultipleShootingResult 类")

    def test_class_has_docstring(self):
        tree = _parse_module(self.FILE)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "TwoLevelMultipleShooting":
                ds = _docstring(node)
                assert ds, "TwoLevelMultipleShooting 缺少 docstring"
                assert _has_chinese(ds), "TwoLevelMultipleShooting docstring 应为中文"
                return
        pytest.fail("未找到 TwoLevelMultipleShooting 类")

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
        tree = _parse_module(self.FILE)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                ds = _docstring(node)
                assert ds, f"TwoLevelMultipleShooting.{name} 缺少 docstring"
                return
        pytest.fail(f"未找到方法 {name}")

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
        tree = _parse_module(self.FILE)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                ds = _docstring(node)
                assert ds, f"{name} 缺少 docstring"
                return
        pytest.fail(f"未找到函数 {name}")


# ---------------------------------------------------------------------------
# multiple_shooting.py
# ---------------------------------------------------------------------------


class TestMultipleShooting:
    FILE = ALGORITHMS_DIR / "multiple_shooting.py"

    def test_module_docstring_exists(self):
        tree = _parse_module(self.FILE)
        ds = _docstring(tree)
        assert ds, "multiple_shooting.py 缺少模块 docstring"
        assert _has_chinese(ds), "模块 docstring 应为中文"


# ---------------------------------------------------------------------------
# continuation.py
# ---------------------------------------------------------------------------


class TestContinuation:
    FILE = ALGORITHMS_DIR / "continuation.py"

    def test_init_docstring_param_matches_signature(self):
        """__init__ docstring 中不应包含不存在的 param 参数"""
        tree = _parse_module(self.FILE)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Continuation":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        ds = _docstring(item)
                        assert ds, "Continuation.__init__ 缺少 docstring"
                        assert "- param:" not in ds, (
                            "docstring 不应包含 param 参数（签名中实际参数名为 step）"
                        )
                        assert "step" in ds, "docstring 中应包含 step 参数说明"
                        return
        pytest.fail("未找到 Continuation.__init__")

    def test_infer_param_index_has_docstring(self):
        tree = _parse_module(self.FILE)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef,)) and node.name == "_infer_param_index":
                ds = _docstring(node)
                assert ds, "_infer_param_index 缺少 docstring"
                return
        pytest.fail("未找到 _infer_param_index")

    def test_build_family_result_has_docstring(self):
        tree = _parse_module(self.FILE)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef,)) and node.name == "_build_family_result":
                ds = _docstring(node)
                assert ds, "_build_family_result 缺少 docstring"
                return
        pytest.fail("未找到 _build_family_result")

    def test_max_step_has_why_comment(self):
        """max_step 硬编码值 [0.04, 0.12, 0.12, 0.08] 应有 why-注释"""
        source = self.FILE.read_text(encoding="utf-8")
        assert "0.04" in source
        # 查找 max_step = np.array 附近是否有 why 注释
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if "max_step" in line and "0.04" in line:
                # 检查前后5行内是否有包含注释的行
                context = "\n".join(lines[max(0, i - 5) : i + 1])
                assert "#" in context, "max_step 硬编码值缺少 why-注释"
                return
        pytest.fail("未找到 max_step 定义")

    def test_step_factors_have_why_comment(self):
        """step_growth_factor 和 step_increase_factor 两套因子应有 why-注释解释混用"""
        source = self.FILE.read_text(encoding="utf-8")
        lines = source.split("\n")
        growth_line = None
        increase_line = None
        for i, line in enumerate(lines):
            if (
                "step_growth_factor" in line
                and "=" in line
                and "1.2" in line
                and growth_line is None
            ):
                growth_line = i
            if (
                "step_increase_factor" in line
                and "=" in line
                and "1.2" in line
                and increase_line is None
            ):
                increase_line = i
        if growth_line is not None and increase_line is not None:
            # 两套因子都存在，应至少有一处 why-注释解释为何需要两套
            span = lines[min(growth_line, increase_line) : max(growth_line, increase_line) + 2]
            context = "\n".join(span)
            assert "#" in context, (
                "step_growth_factor 和 step_increase_factor 两套因子混用，缺 why-注释"
            )


# ---------------------------------------------------------------------------
# stability.py
# ---------------------------------------------------------------------------


class TestStability:
    FILE = ALGORITHMS_DIR / "stability.py"

    def test_pair_eigenvalues_has_args_returns(self):
        tree = _parse_module(self.FILE)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef,)) and node.name == "_pair_eigenvalues":
                ds = _docstring(node)
                assert ds, "_pair_eigenvalues 缺少 docstring"
                assert "Returns" in ds, "_pair_eigenvalues docstring 缺少 Returns 段"
                return
        pytest.fail("未找到 _pair_eigenvalues")

    def test_pair_eigenvalues_tolerance_has_why_comment(self):
        """容差 0.01 应有 why-注释解释其选择"""
        source = self.FILE.read_text(encoding="utf-8")
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if "0.01" in line and "product" in line:
                context = "\n".join(lines[max(0, i - 2) : i + 1])
                assert "#" in context, "容差 0.01 缺少 why-注释"
                return
        pytest.fail("未找到 0.01 容差定义")

    def test_compute_stability_index_formula_correct(self):
        """docstring 和代码注释中应使用 1/λ 而非 λ_conj"""
        source = self.FILE.read_text(encoding="utf-8")
        lines = source.split("\n")
        in_method = False
        for line in lines:
            if "def compute_stability_index" in line:
                in_method = True
            elif in_method and line.startswith("    def "):
                break
            elif in_method and "λ_conj" in line:
                pytest.fail("compute_stability_index 中应使用 1/λ 而非 λ_conj")

    def test_detect_bifurcation_n_failed_logic(self):
        """n_failed 应追踪实际异常次数，而非 len(orbits) - len(bifurcation_points)"""
        import e2m2e.algorithms.stability as mod

        source = inspect.getsource(mod.StabilityAnalysis.detect_bifurcation_in_family)
        # 错误逻辑：n_failed = len(orbits) - len(bifurcation_points)
        # 正确逻辑：在 except 块中计数实际异常次数
        assert "len(orbits) - len(bifurcation_points)" not in source, (
            "n_failed 不应用 len(orbits) - len(bifurcation_points) 计算"
        )


# ---------------------------------------------------------------------------
# differential_correction.py
# ---------------------------------------------------------------------------


class TestDifferentialCorrection:
    FILE = ALGORITHMS_DIR / "differential_correction.py"

    def test_iterate_correction_documents_callback(self):
        tree = _parse_module(self.FILE)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef,)) and node.name == "iterate_correction":
                ds = _docstring(node)
                assert ds, "iterate_correction 缺少 docstring"
                assert "callback" in ds, "iterate_correction docstring 缺少 callback 参数说明"
                return
        pytest.fail("未找到 iterate_correction")

    def test_iterate_correction_documents_none_return(self):
        tree = _parse_module(self.FILE)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef,)) and node.name == "iterate_correction":
                ds = _docstring(node)
                assert ds, "iterate_correction 缺少 docstring"
                assert "None" in ds, "iterate_correction Returns 段应说明可能返回 None"
                return
        pytest.fail("未找到 iterate_correction")

    def test_no_dead_code_abs_gamma(self):
        """行 394 abs(gamma) 为 dead code，应已删除"""
        source = self.FILE.read_text(encoding="utf-8")
        lines = source.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            # 独立一行的 abs(gamma)（非赋值、非表达式的一部分）是 dead code
            if stripped == "abs(gamma)" or stripped == "abs(gamma);":
                pytest.fail(f"第 {i + 1} 行发现 dead code: abs(gamma)")


# ---------------------------------------------------------------------------
# strategies/ — 中文 docstring
# ---------------------------------------------------------------------------


class TestStrategiesChinese:
    STRATEGY_FILES = [
        ALGORITHMS_DIR / "strategies" / "base.py",
        ALGORITHMS_DIR / "strategies" / "halo.py",
        ALGORITHMS_DIR / "strategies" / "symmetric_2d.py",
        ALGORITHMS_DIR / "strategies" / "symmetric_3d.py",
    ]

    @pytest.mark.parametrize("filepath", STRATEGY_FILES, ids=lambda p: p.name)
    def test_module_docstring_is_chinese(self, filepath):
        tree = _parse_module(filepath)
        ds = _docstring(tree)
        assert ds, f"{filepath.name} 缺少模块 docstring"
        assert _has_chinese(ds), f"{filepath.name} 模块 docstring 应为中文"

    @pytest.mark.parametrize("filepath", STRATEGY_FILES, ids=lambda p: p.name)
    def test_all_public_functions_have_chinese_docstring(self, filepath):
        tree = _parse_module(filepath)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_"):
                    continue
                ds = _docstring(node)
                assert ds, f"{filepath.name}::{node.name} 缺少 docstring"
                assert _has_chinese(ds), f"{filepath.name}::{node.name} docstring 应为中文"

    @pytest.mark.parametrize("filepath", STRATEGY_FILES, ids=lambda p: p.name)
    def test_class_docstrings_are_chinese(self, filepath):
        tree = _parse_module(filepath)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                ds = _docstring(node)
                assert ds, f"{filepath.name}::{node.name} 缺少 docstring"
                assert _has_chinese(ds), f"{filepath.name}::{node.name} docstring 应为中文"
