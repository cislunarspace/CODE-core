"""``normal_form.polynomial`` 测试。

重点覆盖 ``poly_subs``（H→QF 映射的变量替换，对应 qiao Code09 的
``X = B·Y`` 符号代换）。该函数的「新变量命名约定」是易错边界，单独
用一组用例钉死。
"""

from __future__ import annotations

import pytest

pytest.importorskip("sympy")

import sympy as sp

from e2m2e.algorithm.normal_form.polynomial import (
    poly_subs,
)

# 标准正则坐标（替换后的「新变量」约定用这套命名）。
Q1, Q2, Q3, P1, P2, P3 = sp.symbols("q1 q2 q3 p1 p2 p3", real=True)
NEW_VARS = (Q1, Q2, Q3, P1, P2, P3)


# ---------------------------------------------------------------------------
# poly_subs：变量替换
# ---------------------------------------------------------------------------


class TestPolySubs:
    """``poly_subs`` 的变量替换语义。"""

    def test_numeric_identity_unit_matrix(self):
        """单位变换 ``x_i = y_i``（B=I）：替换后多项式不变。"""
        # 旧变量另起名字，避免与新变量 q..p 同名造成歧义。
        x1, x2 = sp.symbols("x1 x2", real=True)
        H = {(2, 1, 0, 0, 0, 0): 3}  # 3·x1^2·x2
        subs_map = {
            x1: Q1,
            x2: Q2,
            # 其余变量恒等映射（poly2expr 用 keys 当 variables，需补齐 6 个）
        }
        x3, x4, x5, x6 = sp.symbols("x3 x4 x5 x6", real=True)
        subs_map[x3] = Q3
        subs_map[x4] = P1
        subs_map[x5] = P2
        subs_map[x6] = P3

        result = poly_subs(H, subs_map)
        assert result == {(2, 1, 0, 0, 0, 0): 3}

    def test_numeric_linear_swap(self):
        """数值线性替换 ``x1 ↔ x2``：幂次随变量换位。"""
        x1, x2, x3, x4, x5, x6 = sp.symbols("x1 x2 x3 x4 x5 x6", real=True)
        H = {(2, 1, 0, 0, 0, 0): 1}  # x1^2·x2
        subs_map = {
            x1: Q2,  # 旧 x1 → 新 q2
            x2: Q1,  # 旧 x2 → 新 q1
            x3: Q3,
            x4: P1,
            x5: P2,
            x6: P3,
        }
        result = poly_subs(H, subs_map)
        # x1^2·x2 → q2^2·q1 = q1·q2^2 → 幂次 (1,2,0,0,0,0)
        assert result == {(1, 2, 0, 0, 0, 0): 1}

    def test_symbolic_matrix_B_substitution(self):
        """符号矩阵 B 的 ``X=B·Y`` 替换（Code09 核心场景）。

        ``H = x1^2·x2``，``x_i = Σ_j B[i,j]·y_j``，y 用 q..p 命名。
        替换后应是 3 次齐次多项式，系数为 B 元素的符号表达式。
        """
        x1, x2, x3, x4, x5, x6 = sp.symbols("x1 x2 x3 x4 x5 x6", real=True)
        b = sp.symbols("b1:7_1:7", real=True)  # b11..b66
        B = sp.Matrix([[b[i * 6 + j] for j in range(6)] for i in range(6)])

        subs_map = {}
        for i, xi in enumerate([x1, x2, x3, x4, x5, x6]):
            subs_map[xi] = sum(B[i, j] * NEW_VARS[j] for j in range(6))

        H = {(2, 1, 0, 0, 0, 0): 1}
        result = poly_subs(H, subs_map)

        # 3 次齐次：所有项的总阶数必须等于 3
        for pow_tuple in result:
            assert sum(pow_tuple) == 3, f"出现非 3 次项 {pow_tuple}"

        # 项数应为 C(3+6-1, 6-1) 的子集（实际由 B 结构决定，这里只验非平凡）
        assert len(result) > 1, "符号 B 替换后应展开成多项"

        # 抽查一项：q1^2·q2 的系数应含 b11^2·b21（x1^2 来自 b11·q1，x2 来自 b21·q1+b22·q2）
        # q1^2·q2 项 = (b11·q1)^2·(b22·q2) 的 b11^2·b22 部分 + 2·b11·b12·b21·q1^2·q2 ...
        # 更稳健的断言：q1^3 的系数恰为 b11^2·b21
        coef_q1_cubed = result.get((3, 0, 0, 0, 0, 0), 0)
        assert sp.expand(coef_q1_cubed - b[0] ** 2 * b[6]) == 0  # b11^2·b21

    def test_mixed_numeric_symbolic(self):
        """混合：旧变量既有数值替换又有符号替换。"""
        x1, x2, x3, x4, x5, x6 = sp.symbols("x1 x2 x3 x4 x5 x6", real=True)
        c = sp.symbols("c", real=True)
        H = {(1, 1, 0, 0, 0, 0): 5}  # 5·x1·x2
        subs_map = {
            x1: 2 * Q1,  # 数值缩放
            x2: c * Q2,  # 符号缩放
            x3: Q3,
            x4: P1,
            x5: P2,
            x6: P3,
        }
        result = poly_subs(H, subs_map)
        # 5·(2q1)·(c·q2) = 10c·q1·q2
        assert result == {(1, 1, 0, 0, 0, 0): 10 * c}

    def test_higher_order_term(self):
        """4 阶项 ``x1^4`` 经单位 B 替换不变。"""
        x1, x2, x3, x4, x5, x6 = sp.symbols("x1 x2 x3 x4 x5 x6", real=True)
        H = {(4, 0, 0, 0, 0, 0): 7}
        subs_map = {x: v for x, v in zip([x1, x2, x3, x4, x5, x6], NEW_VARS, strict=True)}
        result = poly_subs(H, subs_map)
        assert result == {(4, 0, 0, 0, 0, 0): 7}

    def test_new_variables_must_be_named_q_p(self):
        """约定：替换后的新变量必须命名为 q1..p3。

        若调用方用其他名字（如 y1..y6），expr2poly 会把它们误当常数。
        此测试钉死该约定——用错误命名的 subs_map 应让 poly_subs 抛出
        明确错误，而非静默返回错误结果。
        """
        x1 = sp.symbols("x1", real=True)
        y1 = sp.symbols("y1", real=True)  # 错误命名
        H = {(1, 0, 0, 0, 0, 0): 1}
        # subs_map 值含 y1（非 q..p 命名）
        subs_map = {x1: y1}
        # 补齐其余 5 个旧变量的恒等映射（仍用 q..p，凑成 6 元）
        x2, x3, x4, x5, x6 = sp.symbols("x2 x3 x4 x5 x6", real=True)
        subs_map.update({x2: Q2, x3: Q3, x4: P1, x5: P2, x6: P3})

        # 应抛 ValueError（新变量命名非法），而非静默把 y1 当常数
        with pytest.raises(ValueError, match="新变量"):
            poly_subs(H, subs_map)
