"""多项式 dict 工具：``{pow_tuple: coefficient}`` 的内部表示与运算。

对应 qiao 仓库的 ``poly_operator``（符号 / 混合系数）和
``list_operator``（数值时间系列系数）两组辅助函数，但只取本切片需要的
子集——``expr2poly``、``poly2expr``、``poly_poisson``、
``poly_simplify`` 与对应的 ``polylist_*`` 数值版本。

约定：

- 多项式 dict 的幂次向量固定为 ``(n1, n2, n3, n4, n5, n6)``，
  对应 ``(q1, q2, q3, p1, p2, p3)``；
- 系数可以是 sympy 符号、纯数值（``int``/``float``）、或长度为 ``M`` 的
  一维 ``numpy.ndarray`` （数值时间序列）；
- 零多项式统一表示为 ``{(0, 0, 0, 0, 0, 0): 0}``（或长度 ``M`` 的零数组）。

本模块属于 ``normal_form`` 包内部基础设施，只被同包的
``legendre`` / ``hamiltonian`` / 后续 ``reducer`` 调用，不对用户
直接暴露——上游接口请走 ``legendre`` 与 ``hamiltonian``。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    pass

# 对应 (q1, q2, q3, p1, p2, p3)，6 个正则坐标。
N_VARIABLES: int = 6


# ---------------------------------------------------------------------------
# 类型工具
# ---------------------------------------------------------------------------


def _zero_like(coef: object) -> object:
    """返回与 ``coef`` 同类型的“零”，支持 sympy、Python 数值与 ndarray。"""
    if isinstance(coef, np.ndarray):
        return np.zeros_like(coef)
    return 0


def _is_zero(coef: object, tol: float = 0.0) -> bool:
    """判断系数是否在数值上为零。

    对 ``ndarray`` 而言，只要存在分量绝对值大于 ``tol`` 即视为非零；
    对 sympy / Python 数值直接 ``== 0`` 比较。``tol`` 取 ``0`` 时即严格相等。
    """
    if isinstance(coef, np.ndarray):
        if tol <= 0:
            return not np.any(coef)
        return not np.any(np.abs(coef) > tol)
    try:
        if tol <= 0:
            return bool(coef == 0)
        # coef 可能是 sympy 表达式（动态属性），用 Any 表达 float() 调用。
        numeric: Any = coef
        return abs(float(numeric)) <= tol
    except (TypeError, ValueError):
        # sympy 符号无法直接判等：尝试数值化
        try:
            numeric = coef
            return abs(float(numeric)) <= tol
        except (TypeError, ValueError):
            return False


# ---------------------------------------------------------------------------
# 幂次向量工具
# ---------------------------------------------------------------------------


def order(pow_tuple: tuple[int, ...]) -> int:
    """返回幂次向量的总阶数（各分量之和）。"""
    return int(sum(pow_tuple))


def keys_by_order(
    poly: Mapping[tuple[int, ...], object],
) -> dict[int, list[tuple[int, ...]]]:
    """按总阶数分组返回幂次键。"""
    grouped: dict[int, list[tuple[int, ...]]] = {}
    for k in poly:
        grouped.setdefault(order(k), []).append(tuple(k))
    for deg in grouped:
        grouped[deg].sort()
    return grouped


def trim_degree(
    poly: Mapping[tuple[int, ...], object],
    max_degree: int,
) -> dict[tuple[int, ...], object]:
    """截断总阶数大于 ``max_degree`` 的项。"""
    if max_degree < 0:
        raise ValueError(f"max_degree 必须非负，得到 {max_degree}")
    result: dict[tuple[int, ...], object] = {}
    for pow_tuple, coef in poly.items():
        if order(pow_tuple) <= max_degree:
            result[tuple(pow_tuple)] = coef
    if not result:
        result[(0,) * N_VARIABLES] = 0
    return result


# ---------------------------------------------------------------------------
# sympy 符号 ↔ dict 转换
# ---------------------------------------------------------------------------


def expr2poly(
    expr: object,
    variables: tuple[object, ...] | None = None,
) -> dict[tuple[int, ...], object]:
    """把 sympy 表达式分解为幂次向量 → 系数 的 dict。

    与 qiao ``expr2poly`` 等价，使用 ``sp.Poly`` 提取单项式系数。
    零多项式返回 ``{(0,) * 6: 0}``。

    Args:
        expr: sympy 表达式。
        variables: 6 个 sympy 符号；默认 ``(q1, q2, q3, p1, p2, p3)``。
            ``None`` 时从 ``expr`` 中按名 ``q1, q2, q3, p1, p2, p3`` 解析；
            若 expr 中仍有同名以外的自由符号，多出的会被并入变量顺序中。

    Returns:
        幂次向量 → 系数 dict。
    """
    import sympy as sp

    if expr == 0:
        return {(0,) * N_VARIABLES: 0}

    if variables is None:
        # 从 expr.free_symbols 中按名 (q1, q2, q3, p1, p2, p3) 解析
        # 已有实例；缺失的位补齐（sympy.symbols 默认是 real=True）。
        name_to_sym = {s.name: s for s in getattr(expr, "free_symbols", set())}
        variables_list: list[object] = []
        for name in ("q1", "q2", "q3", "p1", "p2", "p3"):
            if name in name_to_sym:
                variables_list.append(name_to_sym[name])
            else:
                variables_list.append(sp.Symbol(name, real=True))
        variables = tuple(variables_list)

    expanded = sp.expand(expr)
    if expanded == 0:
        return {(0,) * len(variables): 0}

    poly = sp.Poly(expanded, *variables)
    raw = poly.as_dict()  # {(e1, ..., e6): coeff}
    result: dict[tuple[int, ...], object] = {}
    for monom, coef in raw.items():
        if coef != 0:
            result[tuple(int(x) for x in monom)] = coef
    if not result:
        result[tuple([0] * len(variables))] = 0
    return result


def poly2expr(
    poly: Mapping[tuple[int, ...], object],
    variables: tuple[object, ...] | None = None,
) -> object:
    """幂次向量 → 系数 dict 反向组装为 sympy 表达式。

    对应 qiao ``poly2expr``。零项跳过。
    """
    import sympy as sp

    if variables is None:
        variables = sp.symbols("q1 q2 q3 p1 p2 p3")

    # variables 是 sympy 符号元组（动态属性），用 Any 表达下标与幂运算。
    vars_any: Any = variables
    expr = sp.Integer(0)
    for pow_tuple, coef in poly.items():
        if _is_zero(coef):
            continue
        term = sp.sympify(coef)
        for i, p in enumerate(pow_tuple):
            if p:
                term *= vars_any[i] ** p
        expr += term
    return sp.expand(expr)


# ---------------------------------------------------------------------------
# 泊松括号
# ---------------------------------------------------------------------------


def poly_subs(
    poly: Mapping[tuple[int, ...], object],
    subs_map: Mapping[object, object],
) -> dict[tuple[int, ...], object]:
    """多项式变量替换：把每个旧变量替换为 ``subs_map`` 给出的表达式。

    用于 H→QF 映射（Code09）：``subs_map`` 把旧坐标 ``(x1..x6)`` 映到
    新坐标 ``(q1..p3)`` 的线性组合 ``X = B·Y``，即
    ``subs_map[x_i] = Σ_j B[i,j]·y_j``，其中 ``y_j`` **必须命名为**
    ``q1, q2, q3, p1, p2, p3``。替换后展开、按这套新变量的幂次重排系数。

    系数可以是 sympy 表达式（含 B 元素符号）或数值/ndarray（此时
    ``subs_map`` 的值也应是同形数值/符号，sympy 会广播）。

    Args:
        poly: 幂次向量 → 系数 dict（变量为 ``subs_map`` 的键，即旧变量）。
        subs_map: 旧变量符号 → 新变量表达式的映射。新变量必须命名为
            ``q1, q2, q3, p1, p2, p3``——否则会被 :func:`expr2poly`
            误当常数，本函数会对此做校验并抛 :class:`ValueError`。

    Returns:
        替换后的幂次向量 → 系数 dict（变量为新变量 ``q1..p3``）。

    Raises:
        ValueError: ``subs_map`` 值中出现了非 ``q1..p3`` 命名的自由符号。
    """
    import sympy as sp

    # 校验：替换后表达式的自由符号只能含 q1..p3（其余符号当作系数，
    # 如 B 矩阵元素 b_ij）。若出现其他「坐标名」（如 y1），说明调用方
    # 用错了命名——expr2poly 会把它误当常数，静默返回错误结果。
    allowed_coord_names = {"q1", "q2", "q3", "p1", "p2", "p3"}
    for replacement in subs_map.values():
        for sym in getattr(replacement, "free_symbols", set()):
            if (
                getattr(sym, "name", None)
                and sym.name in {"y1", "y2", "y3", "y4", "y5", "y6", "x1", "x2", "x3", "x4", "x5", "x6"}
            ):
                raise ValueError(
                    f"新变量 {sym.name!r} 命名非法：替换后的变量必须命名为 "
                    f"{sorted(allowed_coord_names)}，否则 expr2poly 会误当常数。"
                )

    expr = poly2expr(poly, variables=tuple(subs_map.keys()))
    substituted = expr.subs(subs_map)
    expanded = sp.expand(substituted)
    return expr2poly(expanded)


def poly_poisson(
    poly1: Mapping[tuple[int, ...], object],
    poly2: Mapping[tuple[int, ...], object],
) -> dict[tuple[int, ...], object]:
    """计算两个多项式 dict 的泊松括号 ``{poly1, poly2}``。

    6-DOF 辛流形 (q1, q2, q3, p1, p2, p3) 的泊松括号：

        {f, g} = Σ_{k=1}^{3} (∂f/∂q_k · ∂g/∂p_k − ∂f/∂p_k · ∂g/∂q_k)

    对单项式 ``f = c₁ ∏ x_i^{a_i}``, ``g = c₂ ∏ x_i^{b_i}``：

        {f, g} = c₁ c₂ Σ_k (a_k b_{k+3} - b_k a_{k+3}) x^{a+b-e_k-e_{k+3}}

    系数为 sympy 或 ndarray 时均适用。
    """
    result: dict[tuple[int, ...], object] = {}
    for pow1, coef1 in poly1.items():
        if _is_zero(coef1):
            continue
        for pow2, coef2 in poly2.items():
            if _is_zero(coef2):
                continue
            for k in range(3):
                a_k = pow1[k]
                a_pk = pow1[k + 3]
                b_k = pow2[k]
                b_pk = pow2[k + 3]
                if not ((a_k and b_pk) or (a_pk and b_k)):
                    continue
                new_pow: list[int] = [int(pow1[i]) + int(pow2[i]) for i in range(N_VARIABLES)]
                new_pow[k] -= 1
                new_pow[k + 3] -= 1
                if any(p < 0 for p in new_pow):
                    continue
                # coef1/coef2 为 sympy 表达式或 ndarray（动态属性），用 Any 表达乘法。
                c1: Any = coef1
                c2: Any = coef2
                coef = c1 * c2 * (a_k * b_pk - a_pk * b_k)
                if _is_zero(coef):
                    continue
                key = tuple(new_pow)
                if key in result:
                    result[key] = result[key] + coef
                else:
                    result[key] = coef
    if not result:
        result[(0,) * N_VARIABLES] = _zero_like(_sample_coef(poly1, poly2))
    return result


def poly_simplify(
    poly: Mapping[tuple[int, ...], object],
    eps: float = 1e-12,
) -> dict[tuple[int, ...], object]:
    """合并同幂次项并剔除小于 ``eps`` 的近零项。"""
    if not poly:
        return {(0,) * N_VARIABLES: 0}

    merged: dict[tuple[int, ...], object] = {}
    for pow_tuple, coef in poly.items():
        key = tuple(int(p) for p in pow_tuple)
        # coef 为 sympy 表达式或 ndarray（动态属性），用 Any 表达加法。
        existing: Any = merged.get(key, 0)
        merged[key] = existing + coef

    result: dict[tuple[int, ...], object] = {}
    for pow_tuple, coef in merged.items():
        if not _is_zero(coef, tol=eps):
            result[pow_tuple] = coef
    if not result:
        result[(0,) * N_VARIABLES] = _zero_like(_sample_coef(poly))
    return result


def polylist_simplify(
    poly: Mapping[tuple[int, ...], npt.NDArray[np.floating]],
    eps: float = 1e-15,
) -> dict[tuple[int, ...], npt.NDArray[np.floating]]:
    """数值版 ``poly_simplify``：合并同幂次项，剔除均幅值过小的时间序列。

    与 qiao ``polylist_simplify`` 等价，使用 ``mean abs`` 作为阈值。
    """
    if not poly:
        return {(0,) * N_VARIABLES: np.zeros(1)}

    result: dict[tuple[int, ...], npt.NDArray[np.floating]] = {}
    for pow_tuple, coef in poly.items():
        key = tuple(int(p) for p in pow_tuple)
        mean_abs = float(np.mean(np.abs(coef)))
        if mean_abs <= eps:
            continue
        if key in result:
            result[key] = result[key] + coef
        else:
            result[key] = coef
    if not result:
        sample = next(iter(poly.values()))
        return {(0,) * N_VARIABLES: np.zeros_like(sample)}
    return result


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


def _sample_coef(
    poly1: Mapping[tuple[int, ...], object],
    poly2: Mapping[tuple[int, ...], object] | None = None,
) -> object:
    """从多项式 dict 中取一个样本系数用于生成零值。

    ``poly2`` 省略或为空时退化为只看 ``poly1``。
    """
    for poly in (poly1, poly2):
        if poly:
            return next(iter(poly.values()))
    return 0


__all__ = [
    "N_VARIABLES",
    "order",
    "keys_by_order",
    "trim_degree",
    "expr2poly",
    "poly2expr",
    "poly_subs",
    "poly_poisson",
    "poly_simplify",
    "polylist_simplify",
]
