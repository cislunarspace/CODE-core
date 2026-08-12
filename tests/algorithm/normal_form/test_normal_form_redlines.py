"""normal_form 静默退化改抛回归测试（#352）。

- ``_eval_coef`` 求不出（数值转换失败 / 含未提供符号）抛 ``ValueError``，
  不静默用 0 填（0 会让下游误以为该项不存在，污染哈密顿量）。
- ``_bdot2a`` 的 ``use_cr3bp`` 显式化：True 走纯 CR3BP 矩阵（不探 SPICE），
  False 走 SPICE 星历、失败改抛（不再静默退化为纯 CR3BP）。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.normal_form.dynamical_substitution import _bdot2a
from e2m2e.algorithm.normal_form.hamiltonian import _eval_coef

pytestmark = pytest.mark.orchestration


class FakeContext:
    force_cr3bp = True


class TestEvalCoefRaisesNotZero:
    def test_zero_coef_still_zero(self):
        """0 系数返回 0（稀疏多项式缺项，合法）。"""
        assert _eval_coef(0, {}) == 0.0

    def test_numeric_coef_ok(self):
        """数值系数正常求值。"""
        assert _eval_coef(1.5, {}) == 1.5

    def test_numeric_conversion_failure_raises(self):
        """数值转换失败抛 ValueError（修复前静默返回 0）。"""
        with pytest.raises(ValueError, match="无法求值"):
            _eval_coef(object(), {})

    def test_missing_symbol_raises(self):
        """sympy 系数含未提供参数时抛 ValueError（修复前静默返回 0）。"""
        sympy = pytest.importorskip("sympy")
        coef = sympy.Symbol("mystery_symbol")
        with pytest.raises(ValueError, match="未提供参数"):
            _eval_coef(coef, {})

    def test_known_symbols_evaluated(self):
        """参数齐备时正常求值（回归：不误伤 happy path）。"""
        sympy = pytest.importorskip("sympy")
        s = sympy.Symbol("a")
        assert _eval_coef(2 * s, {"a": 3.0}) == 6.0


class TestBdot2aExplicitCr3bp:
    def test_use_cr3bp_true_uses_rotation_matrix(self):
        """use_cr3bp=True：C_pq 恒为旋转矩阵，不探 SPICE。"""
        n = 3
        tlist = np.linspace(0.0, 1.0, n)
        B = np.tile([1.0, 0.0, 0.0], (n, 1))
        Bdot = np.tile([0.1, 0.2, 0.3], (n, 1))
        Bddot = np.tile([0.0, 0.0, 0.0], (n, 1))

        A, Adot = _bdot2a(FakeContext(), B, Bdot, Bddot, tlist, use_cr3bp=True)

        # A = -Bdot + C_pq @ B，C_pq 为 [[0,1,0],[-1,0,0],[0,0,0]]
        expected_A = -Bdot + np.tile([0.0, -1.0, 0.0], (n, 1))
        np.testing.assert_allclose(A, expected_A, atol=1e-12)
        # Adot = -Bddot + C_pq @ Bdot（dC_pq = 0）
        expected_Adot = -Bddot + np.tile([0.2, -0.1, 0.0], (n, 1))
        np.testing.assert_allclose(Adot, expected_Adot, atol=1e-12)
