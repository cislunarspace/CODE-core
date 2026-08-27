"""多项式 dict 工具：``{pow_tuple: coefficient}`` 的内部表示与运算。

对应 qiao 仓库的 ``poly_operator`` （符号 / 混合系数）和
``list_operator`` （数值时间系列系数）两组辅助函数，但只取需要的
子集——``expr2poly``、``poly2expr``、``poly_poisson``、
``poly_simplify`` 与对应的 ``polylist_*`` 数值版本。

约定：

- 多项式 dict 的幂次向量固定为 ``(n1, n2, n3, n4, n5, n6)``，
  对应 ``(q1, q2, q3, p1, p2, p3)``；
- 系数可以是 sympy 符号、纯数值（``int``/``float``）、或长度为 ``M`` 的
  一维 ``numpy.ndarray`` （数值时间序列）；
- 零多项式统一表示为 ``{(0, 0, 0, 0, 0, 0): 0}`` （或长度 ``M`` 的零数组）。

本模块属于 ``normal_form`` 包内部基础设施，只被同包的
``legendre`` / ``hamiltonian`` / ``reducer`` 调用，不对用户
直接暴露——上游接口请走 ``legendre`` 与 ``hamiltonian``。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    pass

# 对应 (q1, q2, q3, p1, p2, p3)，6 个正则坐标。
N_VARIABLES: int = 6

Backend = Literal["rust", "python"]


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
    对 Python ``complex`` 按模长；对 sympy / 实数值直接比较。
    ``tol`` 取 ``0`` 时即严格相等。
    """
    if isinstance(coef, np.ndarray):
        if tol <= 0:
            return not np.any(coef)
        return not np.any(np.abs(coef) > tol)
    # 复标量：float(complex) 会抛 TypeError，须按模长，与 Rust 核一致
    if isinstance(coef, (complex, np.complexfloating)):
        return abs(complex(coef)) <= tol
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
    *,
    backend: Backend = "rust",
) -> dict[int, list[tuple[int, ...]]]:
    """按总阶数分组返回幂次键。

    默认 ``backend='rust'``；``backend='python'`` 仅作显式等价性对照。
    """
    if backend not in ("rust", "python"):
        raise ValueError("backend 须为 'rust' 或 'python'")
    if backend == "python":
        return _keys_by_order_python(poly)

    from e2m2e.integrators import keys_by_order_py, require_rust_extension

    require_rust_extension("keys_by_order_py")
    pows = [[int(p) for p in k] for k in poly]
    grouped_list = keys_by_order_py(pows)
    return {
        int(deg): [tuple(int(x) for x in pow_t) for pow_t in keys] for deg, keys in grouped_list
    }


def _keys_by_order_python(
    poly: Mapping[tuple[int, ...], object],
) -> dict[int, list[tuple[int, ...]]]:
    """Python 参照：按总阶数分组返回幂次键。"""
    grouped: dict[int, list[tuple[int, ...]]] = {}
    for k in poly:
        grouped.setdefault(order(k), []).append(tuple(k))
    for deg in grouped:
        grouped[deg].sort()
    return grouped


def trim_degree(
    poly: Mapping[tuple[int, ...], object],
    max_degree: int,
    *,
    backend: Backend = "rust",
) -> dict[tuple[int, ...], object]:
    """截断总阶数大于 ``max_degree`` 的项。

    默认 ``backend='rust'``（数值系数）；含 sympy 符号系数时自动走 Python
    参照路径。``backend='python'`` 仅作显式等价性对照。
    """
    if backend not in ("rust", "python"):
        raise ValueError("backend 须为 'rust' 或 'python'")
    if max_degree < 0:
        raise ValueError(f"max_degree 必须非负，得到 {max_degree}")
    if backend == "python" or not _is_numeric_poly(poly):
        return _trim_degree_python(poly, max_degree)

    from e2m2e.integrators import require_rust_extension, trim_degree_py

    require_rust_extension("trim_degree_py")
    pows, flat, series_len, meta = _pack_numeric_poly(poly)
    if not pows:
        return {(0,) * N_VARIABLES: 0}
    out_pows, out_flat, out_len = trim_degree_py(pows, flat, series_len, int(max_degree))
    return _unpack_numeric_poly(out_pows, out_flat, out_len, meta)


def _trim_degree_python(
    poly: Mapping[tuple[int, ...], object],
    max_degree: int,
) -> dict[tuple[int, ...], object]:
    """Python 参照：截断总阶数大于 ``max_degree`` 的项。"""
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
    # 如 B 矩阵元素 b_ij）。若出现其他坐标名（如 y1），说明调用方
    # 用错了命名——expr2poly 会把它误当常数，静默返回错误结果。
    allowed_coord_names = {"q1", "q2", "q3", "p1", "p2", "p3"}
    disallowed_coord_names = {
        "y1",
        "y2",
        "y3",
        "y4",
        "y5",
        "y6",
        "x1",
        "x2",
        "x3",
        "x4",
        "x5",
        "x6",
    }
    for replacement in subs_map.values():
        for sym in getattr(replacement, "free_symbols", set()):
            if getattr(sym, "name", None) and sym.name in disallowed_coord_names:
                raise ValueError(
                    f"新变量 {sym.name!r} 命名非法：替换后的变量必须命名为 "
                    f"{sorted(allowed_coord_names)}，否则 expr2poly 会误当常数。"
                )

    expr: Any = poly2expr(poly, variables=tuple(subs_map.keys()))
    substituted = expr.subs(subs_map)
    expanded = sp.expand(substituted)
    return expr2poly(expanded)


def poly_poisson(
    poly1: Mapping[tuple[int, ...], object],
    poly2: Mapping[tuple[int, ...], object],
    *,
    backend: Backend = "rust",
) -> dict[tuple[int, ...], object]:
    """计算两个多项式 dict 的泊松括号 ``{poly1, poly2}``。

    6-DOF 辛流形 (q1, q2, q3, p1, p2, p3) 的泊松括号：

        {f, g} = Σ_{k=1}^{3} (∂f/∂q_k · ∂g/∂p_k − ∂f/∂p_k · ∂g/∂q_k)

    对单项式 ``f = c₁ ∏ x_i^{a_i}``, ``g = c₂ ∏ x_i^{b_i}``：

        {f, g} = c₁ c₂ Σ_k (a_k b_{k+3} - b_k a_{k+3}) x^{a+b-e_k-e_{k+3}}

    默认 ``backend='rust'`` 走数值核（标量/时间序列、实/复）；含 sympy
    符号系数时自动走 Python。``backend='python'`` 仅作显式等价性对照，
    绝不作为运行时静默回退。
    """
    if backend not in ("rust", "python"):
        raise ValueError("backend 须为 'rust' 或 'python'")
    if backend == "python" or not (_is_numeric_poly(poly1) and _is_numeric_poly(poly2)):
        return _poly_poisson_python(poly1, poly2)

    from e2m2e.integrators import poly_poisson_py, require_rust_extension

    require_rust_extension("poly_poisson_py")
    pows1, flat1, len1, meta1 = _pack_numeric_poly(poly1)
    pows2, flat2, len2, meta2 = _pack_numeric_poly(poly2)
    series_len = _align_series_len(len1, len2)
    # 一侧标量一侧序列时广播重打包
    if series_len != len1:
        pows1, flat1, _, meta1 = _pack_numeric_poly(poly1, series_len=series_len)
    if series_len != len2:
        pows2, flat2, _, meta2 = _pack_numeric_poly(poly2, series_len=series_len)
    out_meta = _merge_meta(meta1, meta2)
    out_pows, out_flat, out_len = poly_poisson_py(pows1, flat1, pows2, flat2, series_len)
    return _unpack_numeric_poly(out_pows, out_flat, out_len, out_meta)


def _poly_poisson_python(
    poly1: Mapping[tuple[int, ...], object],
    poly2: Mapping[tuple[int, ...], object],
) -> dict[tuple[int, ...], object]:
    """Python 参照：6-DOF 辛 Poisson 括号。"""
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
    *,
    backend: Backend = "rust",
) -> dict[tuple[int, ...], object]:
    """合并同幂次项并剔除小于 ``eps`` 的近零项。

    默认 ``backend='rust'``；含 sympy 符号系数时自动走 Python。
    ``backend='python'`` 仅作显式等价性对照。
    """
    if backend not in ("rust", "python"):
        raise ValueError("backend 须为 'rust' 或 'python'")
    if backend == "python" or not _is_numeric_poly(poly):
        return _poly_simplify_python(poly, eps=eps)

    from e2m2e.integrators import poly_simplify_py, require_rust_extension

    require_rust_extension("poly_simplify_py")
    if not poly:
        return {(0,) * N_VARIABLES: 0}
    pows, flat, series_len, meta = _pack_numeric_poly(poly)
    out_pows, out_flat, out_len = poly_simplify_py(pows, flat, series_len, float(eps))
    return _unpack_numeric_poly(out_pows, out_flat, out_len, meta)


def _poly_simplify_python(
    poly: Mapping[tuple[int, ...], object],
    eps: float = 1e-12,
) -> dict[tuple[int, ...], object]:
    """Python 参照：合并同幂次并剔除近零项。"""
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
    poly: Mapping[tuple[int, ...], Any],
    eps: float = 1e-15,
    *,
    backend: Backend = "rust",
) -> dict[tuple[int, ...], Any]:
    """数值版 ``poly_simplify``：合并同幂次项，剔除均幅值过小的时间序列。

    与 qiao ``polylist_simplify`` 等价，使用 ``mean abs`` 作为阈值。
    系数为时间序列 ``ndarray`` （实或复值皆可，仅做幅值阈值与逐项累加）。

    默认 ``backend='rust'``；``backend='python'`` 仅作显式等价性对照。
    """
    if backend not in ("rust", "python"):
        raise ValueError("backend 须为 'rust' 或 'python'")
    if backend == "python":
        return _polylist_simplify_python(poly, eps=eps)

    from e2m2e.integrators import polylist_simplify_py, require_rust_extension

    require_rust_extension("polylist_simplify_py")
    if not poly:
        return {(0,) * N_VARIABLES: np.zeros(1)}
    if not _is_numeric_poly(poly):
        raise TypeError("polylist_simplify 的 Rust 路径只接受数值 ndarray 系数")
    pows, flat, series_len, meta = _pack_numeric_poly(poly)
    out_pows, out_flat, out_len = polylist_simplify_py(pows, flat, series_len, float(eps))
    return _unpack_numeric_poly(out_pows, out_flat, out_len, meta)


def _polylist_simplify_python(
    poly: Mapping[tuple[int, ...], Any],
    eps: float = 1e-15,
) -> dict[tuple[int, ...], Any]:
    """Python 参照：时间序列 mean-abs 阈值化简。"""
    if not poly:
        return {(0,) * N_VARIABLES: np.zeros(1)}

    result: dict[tuple[int, ...], Any] = {}
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


def _is_numeric_coef(coef: object) -> bool:
    """判断系数是否为 Rust 核可处理的数值（Python 数 / 复数 / ndarray）。"""
    if isinstance(coef, np.ndarray):
        return np.issubdtype(coef.dtype, np.number)
    return isinstance(coef, (int, float, complex, np.floating, np.integer, np.complexfloating))


def _is_numeric_poly(poly: Mapping[tuple[int, ...], object]) -> bool:
    """多项式全部系数为数值时返回 True；空多项式视为数值。"""
    if not poly:
        return True
    return all(_is_numeric_coef(c) for c in poly.values())


def _pack_numeric_poly(
    poly: Mapping[tuple[int, ...], object],
    *,
    series_len: int | None = None,
) -> tuple[list[list[int]], list[float], int, dict[str, Any]]:
    """把数值多项式打包为 Rust 核输入。

    返回 ``(pows, flat_re_im, series_len, meta)``。``meta`` 记录输出形态：
    ``kind`` ∈ {``scalar_real``, ``scalar_complex``, ``series_real``, ``series_complex``}。
    """
    if not poly:
        return [], [], series_len or 1, {"kind": "scalar_real", "series_len": series_len or 1}

    items = list(poly.items())
    samples = [np.asarray(c) for _, c in items]
    has_complex = any(np.iscomplexobj(s) for s in samples)
    lengths = []
    for s in samples:
        sample_flat = s.ravel()
        lengths.append(int(sample_flat.size) if sample_flat.size > 0 else 1)
    inferred = max(lengths) if lengths else 1
    # 标量（size=1）可与序列广播
    if series_len is None:
        series_len = inferred
    if series_len < 1:
        raise ValueError(f"series_len 必须 ≥ 1，得到 {series_len}")

    is_series = series_len > 1 or any(length > 1 for length in lengths)
    if has_complex:
        kind = "series_complex" if is_series else "scalar_complex"
    else:
        kind = "series_real" if is_series else "scalar_real"

    pows: list[list[int]] = []
    flat: list[float] = []
    for pow_tuple, coef in items:
        pows.append([int(p) for p in pow_tuple])
        arr = np.asarray(coef)
        if arr.ndim == 0 or arr.size == 1:
            val = complex(arr.ravel()[0]) if arr.size else 0j
            re = float(val.real)
            im = float(val.imag)
            for _ in range(series_len):
                flat.extend([re, im])
        else:
            vec = arr.ravel()
            if vec.size != series_len:
                raise ValueError(f"系数序列长度不一致：期望 {series_len}，得到 {vec.size}")
            if np.iscomplexobj(vec):
                for z in vec:
                    flat.extend([float(np.real(z)), float(np.imag(z))])
            else:
                for x in vec:
                    flat.extend([float(x), 0.0])
    meta: dict[str, Any] = {"kind": kind, "series_len": series_len}
    return pows, flat, series_len, meta


def _unpack_numeric_poly(
    pows: list[list[int]],
    flat: list[float],
    series_len: int,
    meta: Mapping[str, Any],
) -> dict[tuple[int, ...], object]:
    """把 Rust 核输出还原为 Python 多项式 dict。"""
    kind = str(meta.get("kind", "scalar_real"))
    result: dict[tuple[int, ...], object] = {}
    stride = series_len * 2
    for i, p in enumerate(pows):
        key = tuple(int(x) for x in p)
        chunk = flat[i * stride : (i + 1) * stride]
        if kind == "scalar_real":
            result[key] = float(chunk[0])
        elif kind == "scalar_complex":
            result[key] = complex(chunk[0], chunk[1])
        elif kind == "series_real":
            result[key] = np.asarray([chunk[2 * j] for j in range(series_len)], dtype=float)
        else:  # series_complex
            result[key] = np.asarray(
                [complex(chunk[2 * j], chunk[2 * j + 1]) for j in range(series_len)],
                dtype=complex,
            )
    if not result:
        if kind.startswith("series"):
            dtype = complex if "complex" in kind else float
            result[(0,) * N_VARIABLES] = np.zeros(series_len, dtype=dtype)
        elif kind == "scalar_complex":
            result[(0,) * N_VARIABLES] = 0j
        else:
            result[(0,) * N_VARIABLES] = 0.0
    return result


def _align_series_len(len1: int, len2: int) -> int:
    """两侧 series_len 对齐：允许 1 与 N 广播，禁止 N≠M。"""
    if len1 == len2:
        return len1
    if len1 == 1:
        return len2
    if len2 == 1:
        return len1
    raise ValueError(f"两侧时间序列长度不一致：{len1} vs {len2}")


def _merge_meta(meta1: Mapping[str, Any], meta2: Mapping[str, Any]) -> dict[str, Any]:
    """合并两侧输出形态：任一侧复/序列则输出复/序列。"""
    k1 = str(meta1.get("kind", "scalar_real"))
    k2 = str(meta2.get("kind", "scalar_real"))
    series = "series" in k1 or "series" in k2
    complex_ = "complex" in k1 or "complex" in k2
    if series and complex_:
        kind = "series_complex"
    elif series:
        kind = "series_real"
    elif complex_:
        kind = "scalar_complex"
    else:
        kind = "scalar_real"
    series_len = max(int(meta1.get("series_len", 1)), int(meta2.get("series_len", 1)))
    return {"kind": kind, "series_len": series_len}


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
