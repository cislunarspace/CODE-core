"""Legendre 展开 1/r，得到关于 ``(q1, q2, q3, p1, p2, p3)`` 的多项式字典。

对应 qiao ``Code02_Legendre_expr.py``。思路：

- 对单位矢量夹角的余弦 ``cos θ = (qv · r0v) / (q · r0)``，用 Legendre
  多项式递推 ``P_{n+1} = ((2n+1)/(n+1)) x P_n − (n/(n+1)) P_{n-1}``
  展开 ``1/|r - q| = (1/r0) Σ (q/r0)^n P_n(cos θ)``；
- 每一项代回 ``q = sqrt(q1² + q2² + q3²)``，使展开式成为 ``q1, q2, q3``
  的纯多项式（``P_n`` 是 ``cos θ`` 的多项式，因此 ``P_n((qv · r0v)/(q·r0))``
  乘 ``(q/r0)^n`` 后得到以 ``q`` 为分母的各阶齐次多项式，乘 ``r0`` 后
  的总幂次恰为 ``n``，但符号上每阶仍含分母，待 ``r0`` 替换回符号后分母消去）；
- 最后用 sympy 的 ``Poly`` 提取幂次 → 系数字典，再 ``trim_degree`` 截断
  到用户指定的阶数。

API：

- :class:`LegendreExpansionResult` —— 展开产物：幂次 → sympy 系数字典
  （``Le`` 标量场，其它天体调用方负责乘 ``-μ``）、``term_count``、``max_degree``。
- :func:`expand_legendre_1_over_r` —— 入口函数，构造标量场。
- :func:`expand_legendre_for_body` —— 给已构造好的 ``Le`` 标上 ``-μ``
  乘子，按幂次直接给到 ``hamiltonian.build_hamiltonian`` 使用。

本切片中 ``sympy`` 仅在函数内部惰性导入，``e2m2e.algorithm.normal_form``
顶层导入不强制依赖。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 共线平动点（库内几何稀疏）推荐阶数。
DEFAULT_COLLINEAR_ORDER: int = 10
# 三角平动点（L4/L5 距两天体都较远）通常阶数可略低；qiao 默认 8。
DEFAULT_TRIANGULAR_ORDER: int = 8


@dataclass(frozen=True)
class LegendreExpansionResult:
    """Legendre 展开 1/r 的结果。

    Attributes:
        polynomial: 幂次向量 (n1, n2, n3, n4, n5, n6) → sympy 系数
            的字典；``(0, 0, 0, 0, 0, 0)`` 对应常数项。
        term_count: ``polynomial`` 中非零项数。
        max_degree: 截断阶数（总阶数上限）。
        source_degree: sympy 推导时使用的最高阶数（在 ``max_degree``
            处可能已被 ``trim_degree`` 截掉部分项）。
    """

    polynomial: dict[tuple[int, ...], object]
    term_count: int = 0
    max_degree: int = 0
    source_degree: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# 1/r 的 Legendre 多项式展开（Code02 等价）
# ---------------------------------------------------------------------------


def expand_legendre_1_over_r(
    max_degree: int = DEFAULT_COLLINEAR_ORDER,
) -> LegendreExpansionResult:
    r"""对 1/\|(rx, ry, rz) − (q1, q2, q3)\| 做 Legendre 多项式展开。

    返回 sympy 系数多项式 dict：键为 ``(n1, n2, n3, n4, n5, n6)``，
    值为 sympy 系数（标量场不带 ``μ`` 乘子；乘子由调用方在
    :func:`expand_legendre_for_body` 或 ``hamiltonian`` 中补上）。

    Args:
        max_degree: 截断阶数。共线点通常 8–12；过大会令符号展开变慢、
            下游数值化耗时增加。``max_degree < 1`` 抛 :class:`ValueError`。

    Returns:
        :class:`LegendreExpansionResult`。
    """
    if max_degree < 1:
        raise ValueError(f"max_degree 必须 ≥ 1，得到 {max_degree}")

    # 局部导入：保持 ``import e2m2e.algorithm.normal_form`` 不强制依赖 sympy。
    from sympy import Integer, Matrix, expand, simplify, sqrt, symbols

    from .polynomial import expr2poly, trim_degree

    rx, ry, rz, q1, q2, q3, r0, q = symbols("rx ry rz q1 q2 q3 r0 q", real=True)
    p1, p2, p3 = symbols("p1 p2 p3", real=True)

    # 1/r 标量场与 p 无关：把 p 占位为 0，让 ``expr2poly`` 输出 6 元 dict。
    substitute_p_zero = {p1: Integer(0), p2: Integer(0), p3: Integer(0)}

    r0v = Matrix([rx, ry, rz])
    qv = Matrix([q1, q2, q3])

    # Legendre 多项式递推
    p_list = [Integer(1)]
    p_list.append(simplify(qv.dot(r0v) / (q * r0)))

    for n in range(1, max_degree):
        x = qv.dot(r0v) / (q * r0)
        next_p = simplify((2 * n + 1) / (n + 1) * x * p_list[n] - n / (n + 1) * p_list[n - 1])
        p_list.append(next_p)

    # 乘 (q/r0)^n 后求和再除以 r0
    weighted = [simplify(p * (q / r0) ** n) for n, p in enumerate(p_list)]
    total = Integer(0)
    for term in weighted:
        total += term
    total = simplify(total / r0)

    # q = sqrt(q1² + q2² + q3²)，分母的 q 与 r0 经乘除后抵消
    q_expr = sqrt(q1**2 + q2**2 + q3**2)
    total = simplify(total.subs(q, q_expr))

    # p 占位为 0，使 dict 维度为完整 (q1..p3) 6 元
    total = total.subs(substitute_p_zero)

    total = expand(total)
    raw = expr2poly(total, variables=(q1, q2, q3, p1, p2, p3))
    trimmed = trim_degree(raw, max_degree)

    return LegendreExpansionResult(
        polynomial=trimmed,
        term_count=sum(1 for v in trimmed.values() if v != 0),
        max_degree=max_degree,
        source_degree=max_degree,
        notes=(
            f"Legendre expansion of 1/r to order {max_degree}; "
            "Le scalar field carries no mu (call expand_legendre_for_body).",
        ),
    )


def expand_legendre_for_body(
    expansion: LegendreExpansionResult,
    mu: float,
) -> dict[tuple[int, ...], object]:
    """把 ``-μ · Le`` 形式应用到 Legendre 标量场，输出 ``hamiltonian`` 用 dict。

    Args:
        expansion: :func:`expand_legendre_1_over_r` 的结果。
        mu: 该天体的归一化引力常数（无量纲）。

    Returns:
        与 ``expansion.polynomial`` 同形状、键相同的 dict；每个系数
        变为原系数乘 ``-μ``。
    """
    result: dict[tuple[int, ...], object] = {}
    for pow_tuple, coef in expansion.polynomial.items():
        if coef == 0:
            continue
        # 标量场系数是 sympy 符号 + rx/ry/rz/r0；这里只乘浮点
        # ``-mu``，符号留待 ``hamiltonian`` 替换为 (rx*/ry*/rz*/r0*)。
        # coef 是 sympy 表达式（动态属性），用 Any 局部标注表达乘法语义。
        sympy_coef: Any = coef
        result[tuple(int(p) for p in pow_tuple)] = -float(mu) * sympy_coef
    return result


# ---------------------------------------------------------------------------
# 内部符号诊断辅助
# ---------------------------------------------------------------------------


def legendre_free_symbols(
    expansion: LegendreExpansionResult,
) -> set[object]:
    """返回 ``expansion.polynomial`` 中所有符号（用于 hamiltonian 替换检测）。"""
    symbols_set: set[object] = set()
    for coef in expansion.polynomial.values():
        try:
            # coef 是 sympy 表达式（动态属性），用 Any 表达 free_symbols 访问。
            sympy_coef: Any = coef
            symbols_set.update(sympy_coef.free_symbols)
        except AttributeError:
            # 非 sympy 系数（数值 0/1）
            continue
    return symbols_set


__all__ = [
    "DEFAULT_COLLINEAR_ORDER",
    "DEFAULT_TRIANGULAR_ORDER",
    "LegendreExpansionResult",
    "expand_legendre_1_over_r",
    "expand_legendre_for_body",
    "legendre_free_symbols",
]
