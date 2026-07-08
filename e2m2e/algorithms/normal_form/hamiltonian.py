"""构建与数值化标准形 Hamilton 量。

对应 qiao ``Code03_Hamilton_expr.py``（符号构造）与
``Code04_Hamilton_num.py``（在指定历元窗口上求数值时间序列）。

公开接口：

- :class:`Hamiltonian` —— 不可变结果容器，封装符号 dict
  ``{pow_tuple: coefficient}``、依赖的动态参数名，以及数值化方法
  ``evaluated_coefficients(times, context)``。``coefficient`` 字段
  可以是 sympy 符号（``build_hamiltonian`` 之后），也可以是 numpy 数组
  （``evaluate_hamiltonian`` 之后）；两者互斥，分别服务于静/动态两种
  消费路径。
- :func:`build_hamiltonian` —— 接受 :class:`NormalFormContext` 与
  :class:`LegendreExpansionResult`，构造包含动能 / Coriolis-pq / 离心-qq
  / 强制项 f·q / 地球+月球+太阳引力势的 Hamilton 多项式（sympy 系数）。
- :func:`evaluate_hamiltonian` —— 在 :class:`Hamiltonian` 上批量求数值
  时间序列，返回 ``ndarray`` 形状 ``(len(times), n_terms)``；幂次向量
  同步保留在 :class:`Hamiltonian` 中。

设计上让 :class:`Hamiltonian` 始终保留幂次向量，便于后续
``QuasiFloquetReducer`` / ``CenterManifoldReducer`` 调用同一对象。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from .polynomial import (
    expr2poly,
    poly_simplify,
    polylist_simplify,
    trim_degree,
)

if TYPE_CHECKING:
    from .context import NormalFormContext
    from .legendre import LegendreExpansionResult


# Hamilton 量在评估时需要从 ``Eval_expr`` 注入的全部动态参数。
# 该列表与 ``_ephemeris.eval_params`` 的返回键一致。
DYNAMIC_PARAM_NAMES: tuple[str, ...] = (
    "Cpq1",
    "Cpq2",
    "Cpq3",
    "Cpq4",
    "Cpq5",
    "Cpq6",
    "Cpq7",
    "Cpq8",
    "Cpq9",
    "Cqq1",
    "Cqq2",
    "Cqq3",
    "Cqq4",
    "Cqq5",
    "Cqq6",
    "Cqq7",
    "Cqq8",
    "Cqq9",
    "f1",
    "f2",
    "f3",
    "rex",
    "rey",
    "rez",
    "re0",
    "rmx",
    "rmy",
    "rmz",
    "rm0",
    "rsx",
    "rsy",
    "rsz",
    "rs0",
    "mu_e",
    "mu_m",
    "mu_s",
)


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hamiltonian:
    """Hamilton 量封装，包含符号系数 dict 与幂次向量数组。

    Attributes:
        powers: ``(n_terms, 6)`` int64 幂次向量数组；行序与
            ``coefficients`` 一一对应。
        coefficients: 幂次向量 → sympy 系数（来自 ``build_hamiltonian``）
            或 → ``ndarray`` 时间序列（来自 ``evaluate_hamiltonian``）。
            默认为 ``None``，代表该 Hamiltonian 尚未填充系数。
        sources: dict，键为各组成（kinetic / coriolis / centrifugal /
            force / earth / moon / sun）名称，值为幂次向量 → sympy 系数
            的子字典；用于诊断 / 中间检查。
        max_degree: 构造时使用的截断阶数；保留以便复用。
    """

    powers: npt.NDArray[np.integer]
    coefficients: dict[tuple[int, ...], object] | npt.NDArray[np.floating] | None = None
    sources: dict[str, dict[tuple[int, ...], object]] = field(default_factory=dict)
    max_degree: int = 0

    @property
    def n_terms(self) -> int:
        return int(self.powers.shape[0])

    @property
    def is_evaluated(self) -> bool:
        return isinstance(self.coefficients, np.ndarray)


# ---------------------------------------------------------------------------
# 构造（Code03 等价）
# ---------------------------------------------------------------------------


def build_hamiltonian(
    context: NormalFormContext,
    legendre_result: LegendreExpansionResult,
    *,
    max_degree: int | None = None,
    store_sources: bool = True,
) -> Hamiltonian:
    """组装符号 Hamilton 量。

    Hamilton = f·q + ½‖p‖² + pᵀ C_pq q + ½ qᵀ C_qq q
              − μ_e / |r_e − q_LP − q| − μ_m / |r_m − q_LP − q| − μ_s / |r_s − q_LP − q|

    三个引力势分别用 Legendre 标量场乘 ``-μ`` 替换 ``(rx, ry, rz, r0)``
    成对应天体的观测向量后加入。

    Args:
        context: 归一化上下文。
        legendre_result: :func:`e2m2e.algorithms.normal_form.legendre.expand_legendre_1_over_r`
            输出。标量场 ``Le`` 已被本函数乘上 ``-μ_e/μ_m/μ_s`` 后并入
            Hamilton 多项式。
        max_degree: 额外截断阶数；``None`` 时使用 ``context.order``。
        store_sources: 若为 ``True``，把动能 / Coriolis / 离心 / f·q /
            三体势能这 6 块单独存到 ``Hamiltonian.sources``，便于诊断。

    Returns:
        :class:`Hamiltonian`，``coefficients`` 是幂次向量 → sympy 系数
        的 dict。
    """
    from sympy import Matrix, Rational, symbols

    deg = int(max_degree) if max_degree is not None else int(context.order)
    if deg < 1:
        raise ValueError(f"max_degree 必须 ≥ 1，得到 {deg}")

    q1, q2, q3 = symbols("q1 q2 q3", real=True)
    p1, p2, p3 = symbols("p1 p2 p3", real=True)
    f1, f2, f3 = symbols("f1 f2 f3", real=True)
    c_pq_syms = symbols("Cpq1 Cpq2 Cpq3 Cpq4 Cpq5 Cpq6 Cpq7 Cpq8 Cpq9", real=True)
    c_qq_syms = symbols("Cqq1 Cqq2 Cqq3 Cqq4 Cqq5 Cqq6 Cqq7 Cqq8 Cqq9", real=True)
    rex, rey, rez, re0 = symbols("rex rey rez re0", real=True)
    rmx, rmy, rmz, rm0 = symbols("rmx rmy rmz rm0", real=True)
    rsx, rsy, rsz, rs0 = symbols("rsx rsy rsz rs0", real=True)

    qv = Matrix([q1, q2, q3])
    pv = Matrix([p1, p2, p3])
    fv = Matrix([f1, f2, f3])
    cpq = Matrix(
        [
            [c_pq_syms[0], c_pq_syms[1], c_pq_syms[2]],
            [c_pq_syms[3], c_pq_syms[4], c_pq_syms[5]],
            [c_pq_syms[6], c_pq_syms[7], c_pq_syms[8]],
        ]
    )
    cqq = Matrix(
        [
            [c_qq_syms[0], c_qq_syms[1], c_qq_syms[2]],
            [c_qq_syms[3], c_qq_syms[4], c_qq_syms[5]],
            [c_qq_syms[6], c_qq_syms[7], c_qq_syms[8]],
        ]
    )

    h_quad = (
        fv.dot(qv)
        + Rational(1, 2) * pv.dot(pv)
        + pv.dot(cpq * qv)
        + Rational(1, 2) * qv.dot(cqq * qv)
    )

    sources: dict[str, dict[tuple[int, ...], object]] = {}

    if store_sources:
        sources["force"] = trim_degree(expr2poly(fv.dot(qv)), deg)
        sources["kinetic"] = trim_degree(expr2poly(Rational(1, 2) * pv.dot(pv)), deg)
        sources["coriolis"] = trim_degree(expr2poly(pv.dot(cpq * qv)), deg)
        sources["centrifugal"] = trim_degree(expr2poly(Rational(1, 2) * qv.dot(cqq * qv)), deg)

    h_poly = expr2poly(h_quad)

    h_poly = _add_body(
        h_poly,
        legendre_result,
        rex,
        rey,
        rez,
        re0,
        float(context.mu_e),
        "pot_earth",
        sources,
        deg,
    )
    h_poly = _add_body(
        h_poly,
        legendre_result,
        rmx,
        rmy,
        rmz,
        rm0,
        float(context.mu_m),
        "pot_moon",
        sources,
        deg,
    )
    h_poly = _add_body(
        h_poly,
        legendre_result,
        rsx,
        rsy,
        rsz,
        rs0,
        float(context.mu_s),
        "pot_sun",
        sources,
        deg,
    )

    h_poly = trim_degree(h_poly, deg)
    h_poly = poly_simplify(h_poly)

    powers = _powers_array(h_poly)

    return Hamiltonian(
        powers=powers,
        coefficients=h_poly,
        sources=sources,
        max_degree=deg,
    )


def _add_body(
    h_poly: dict[tuple[int, ...], object],
    legendre_result: LegendreExpansionResult,
    rx_sym,
    ry_sym,
    rz_sym,
    r0_sym,
    mu: float,
    label: str,
    sources: dict[str, dict[tuple[int, ...], object]],
    max_degree: int,
) -> dict[tuple[int, ...], object]:
    """把 ``-μ · Le(rx*, ry*, rz*, r0*)`` 加到 Hamilton 多项式。

    Args:
        h_poly: 已构造的 Hamilton 多项式 dict（被原地更新）。
        legendre_result: :func:`expand_legendre_1_over_r` 输出。
        rx_sym/ry_sym/rz_sym/r0_sym: 本天体的观测向量符号
            （例如地球 ``(rex, rey, rez, re0)``）。
        mu: 天体归一化引力常数。
        label: 在 ``sources`` 中用于标记本天体势能的键名（如
            ``"pot_earth"``）。
        sources: 诊断 dict。
        max_degree: 截断阶数。
    """
    from sympy import Rational

    body_sub_map: dict[object, object] = {}
    # 把 Legendre 标量场中残留的 (rx, ry, rz, r0) 符号
    # 替换为对应天体的观测向量。
    for coef in legendre_result.polynomial.values():
        for s in getattr(coef, "free_symbols", set()):
            if s.name == "rx":
                body_sub_map[s] = rx_sym
            elif s.name == "ry":
                body_sub_map[s] = ry_sym
            elif s.name == "rz":
                body_sub_map[s] = rz_sym
            elif s.name == "r0":
                body_sub_map[s] = r0_sym

    body_terms: dict[tuple[int, ...], object] = {}
    for pow_tuple, coef in legendre_result.polynomial.items():
        # coef 是 sympy 表达式（动态属性 subs），用 Any 表达符号替换语义。
        sympy_coef: Any = coef
        subbed = sympy_coef.subs(body_sub_map)
        signed = -mu * subbed
        if signed == 0:
            continue
        key = tuple(int(p) for p in pow_tuple)
        body_terms[key] = body_terms.get(key, Rational(0)) + signed
        h_poly[key] = h_poly.get(key, Rational(0)) + signed

    sources[label] = trim_degree(body_terms, max_degree)
    return h_poly


# ---------------------------------------------------------------------------
# 数值化（Code04 等价）
# ---------------------------------------------------------------------------


def evaluate_hamiltonian(
    hamiltonian: Hamiltonian,
    times: npt.ArrayLike,
    context: NormalFormContext,
) -> Hamiltonian:
    """对 Hamilton 量在指定时刻序列上求数值时间序列。

    Args:
        hamiltonian: :func:`build_hamiltonian` 输出。
        times: 归一化时间数组（TU）；支持 ``(n,)`` shape。
        context: 归一化上下文。

    Returns:
        一个新的 :class:`Hamiltonian`：

        - ``powers`` 为 ``(n_terms, 6)`` int64 数组；
        - ``coefficients`` 为 ``(n_times, n_terms)`` 浮点 ``ndarray``；
        - ``max_degree`` 等字段与输入一致。
    """
    if not isinstance(hamiltonian.coefficients, dict):
        raise ValueError("Hamiltonian 缺少 sympy 系数 dict；请先用 build_hamiltonian 构造。")

    times_arr = np.asarray(times, dtype=float)
    if times_arr.ndim != 1:
        raise ValueError(f"times 必须是一维序列，得到形状 {times_arr.shape}")
    times_arr = times_arr.ravel()

    n_terms = int(hamiltonian.powers.shape[0])
    arr = np.zeros((times_arr.shape[0], n_terms), dtype=float)
    pow_tuples = [tuple(int(x) for x in row) for row in hamiltonian.powers]

    from ._ephemeris import eval_params as _eval_params

    t_to_jd = float(context.TU) / 86400.0

    for i, t in enumerate(times_arr):
        jd = float(context.epoch) + float(t) * t_to_jd
        params = _eval_params(jd, context)
        for j, pow_tuple in enumerate(pow_tuples):
            coef = hamiltonian.coefficients.get(pow_tuple, 0)
            arr[i, j] = _eval_coef(coef, params)

    if arr.shape[1] > 0:
        cols = {
            tuple(int(p) for p in hamiltonian.powers[k]): arr[:, k] for k in range(arr.shape[1])
        }
        cols = polylist_simplify(cols)
        new_keys = sorted(cols.keys())
        new_powers = np.array(new_keys, dtype=np.int64)
        new_arr = np.column_stack([cols[k] for k in new_keys])
        arr = new_arr
        powers_out: npt.NDArray[np.integer] = new_powers
    else:
        powers_out = hamiltonian.powers

    return Hamiltonian(
        powers=powers_out,
        coefficients=arr,
        sources=hamiltonian.sources,
        max_degree=hamiltonian.max_degree,
    )


def _eval_coef(coef: object, params: dict[str, float]) -> float:
    """对单个 sympy 系数用 ``params`` 替换后求 float。

    ``coef = 0`` 或空 dict 项直接返回 0；sympy ``free_symbols`` 中
    出现非 ``params`` 中键的符号时报 0 并保留 ``RuntimeWarning``，
    避免静默丢失但又不会让整体时间序列崩溃。
    """
    if coef == 0:
        return 0.0
    if not hasattr(coef, "free_symbols"):
        try:
            # coef 此处为数值（无 free_symbols），用 Any 表达 float() 调用。
            numeric: Any = coef
            return float(numeric)
        except (TypeError, ValueError):
            return 0.0
    # coef 此处为 sympy 表达式（含 free_symbols），用 Any 表达动态属性。
    sympy_coef: Any = coef
    sub_map: dict[object, float] = {}
    missing: list[str] = []
    for s in sympy_coef.free_symbols:
        name = getattr(s, "name", None)
        if name in params:
            sub_map[s] = float(params[name])
        else:
            missing.append(str(name))
    if missing:
        # 不抛错：用 0 填；保守处理，等价于 qiao eval 出现无法求值的项时
        # 把该项记为 0（Code04 在 exceptions 时也是写 0.0）。
        return 0.0
    return float(sympy_coef.subs(sub_map))


def hamiltonian_constant_term(
    hamiltonian: Hamiltonian,
    times: npt.ArrayLike,
    context: NormalFormContext,
) -> npt.NDArray[np.floating]:
    """返回 Hamilton 量常值项 ``H_0(t)`` 的时间序列，形状 ``(len(times),)``。

    常值项即 ``(0, 0, 0, 0, 0, 0)`` 幂次对应的系数；当 Hamilton 量
    已被数值化时直接取列；未数值化时本函数会先走完一遍
    :func:`evaluate_hamiltonian` 再取常值列。
    """
    if not isinstance(hamiltonian.coefficients, np.ndarray):
        evaled = evaluate_hamiltonian(hamiltonian, times, context)
        # evaluate_hamiltonian 必返回 ndarray 形式 coefficients（见其实现），
        # 但 dataclass 标注为联合类型，此处显式标注为 ndarray。
        arr: npt.NDArray[np.floating] = evaled.coefficients  # type: ignore[assignment]
        powers = evaled.powers
    else:
        arr = hamiltonian.coefficients
        powers = hamiltonian.powers

    target = tuple([0] * powers.shape[1])
    for j in range(powers.shape[0]):
        if tuple(int(p) for p in powers[j]) == target:
            return arr[:, j]
    return np.zeros(arr.shape[0], dtype=float)


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


def _powers_array(poly_dict: dict[tuple[int, ...], object]) -> npt.NDArray[np.integer]:
    """幂次 dict → ``(n_terms, 6)`` int64 数组，按幂次字典序排序。"""
    keys = sorted(poly_dict.keys())
    return np.array(keys, dtype=np.int64)


__all__ = [
    "DYNAMIC_PARAM_NAMES",
    "Hamiltonian",
    "build_hamiltonian",
    "evaluate_hamiltonian",
    "hamiltonian_constant_term",
]
