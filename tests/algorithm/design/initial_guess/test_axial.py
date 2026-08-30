"""Axial 轨道初始猜测模块测试——导入契约。

compute_axial_initial_guess 的正确性测试（返回结构、x 轴对称初始条件、
面外速度非零、分岔振幅等）已按 ADR 0037 移出默认套件：其底层 Lyapunov
垂直临界分岔扫描固有 ~1 min（多条真实修正，见模块内
``_bifurcation_cache``），超出单测预算；同族修正/延拓测试因同一根因
同步移出（见 continuation/test_continuation_per_family.py docstring）。
"""

import pytest

from e2m2e.algorithm.family.axial_initial_guess import compute_axial_initial_guess

pytestmark = pytest.mark.orchestration


# =============================================================================
# Module identity
# =============================================================================


class TestModuleImport:
    """验证模块可直接导入，函数可直接调用。"""

    def test_module_importable(self):
        """axial_initial_guess 模块应可直接导入。"""
        assert compute_axial_initial_guess is not None

    def test_function_has_docstring(self):
        """公共函数应有文档字符串。"""
        assert compute_axial_initial_guess.__doc__ is not None
