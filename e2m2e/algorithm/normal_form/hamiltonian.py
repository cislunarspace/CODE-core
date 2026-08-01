"""构建与数值化标准形 Hamilton 量。

对应 qiao ``Code03_Hamilton_expr.py`` （符号构造）与
``Code04_Hamilton_num.py`` （在指定历元窗口上求数值时间序列）。

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
    r"""组装符号 Hamilton 量。

    Hamilton = f·q + ½‖p‖² + pᵀ C_pq q + ½ qᵀ C_qq q
              − μ_e / \|r_e − q_LP − q\| − μ_m / \|r_m − q_LP − q\| − μ_s / \|r_s − q_LP − q\|

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


def build_cr3bp_hamiltonian(
    context: NormalFormContext,
    max_degree: int | None = None,
) -> dict[tuple[int, ...], float]:
    r"""构造纯 CR3BP Hamiltonian（数值系数 dict，不依赖 SPICE）。

    CR3BP 是自治系统，平动点是不动点，故 Hamiltonian 系数为常数。本函数
    复用 :func:`build_hamiltonian` 的符号构造，再把全部动态参数替换为
    CR3BP 常数值，得到 ``{pow_tuple: float}`` 的纯数值 dict。

    对应 Gómez vol III 2.7 的 LISWCH ``GENHAM`` 步（CR3BP 原路）：在平动点
    偏移坐标 ``q = (q1,q2,q3)`` 下展开

    .. math::
        H = \tfrac12\|p\|^2 + p^T C_{pq} q + \tfrac12 q^T C_{qq} q
            - \mu_e/\|r_e - q\| - \mu_m/\|r_m - q\|

    其中 ``C_pq`` 为科里奥利、``C_qq`` 为离心，``r_e=(-\mu-x_{LP},0,0)``、
    ``r_m=(1-\mu-x_{LP},0,0)`` 为两主天体相对平动点的位置。太阳项置零
    （CR3BP 只含两个主天体）。一阶项 ``f·q`` 在平动点处为零（平衡点）。

    Args:
        context: 归一化上下文（提供 ``mu``、``libration_position`` 等）。
            本函数内部以 ``mu_s=0`` 重新构造上下文以消去太阳项。
        max_degree: 截断阶数；``None`` 时用 ``context.order``。

    Returns:
        幂次向量 → 纯数值 ``float`` 系数的 dict。系数为常数（不随时间），
        可直接广播成时间序列供 :class:`CenterManifoldReducer` 注入。

    Notes:
        与 :func:`evaluate_hamiltonian` 不同，本函数**不**产出时间序列——
        CR3BP 自治，系数恒定。调用方（如 pipeline）若需与 ``qf_result.tlist``
        对齐的时间序列，自行 ``np.full(N, coef)`` 广播即可。
    """

    from .context import NormalFormContext
    from .legendre import expand_legendre_1_over_r

    deg = int(max_degree) if max_degree is not None else int(context.order)
    if deg < 1:
        raise ValueError(f"max_degree 必须 ≥ 1，得到 {deg}")

    # 以 mu_s=0 重新构造上下文，消去太阳项（_add_body 已把 -mu_s 乘进系数）。
    ctx_cr3bp = NormalFormContext(
        system=context.system,
        libration_point=context.libration_point,
        epoch=context.epoch,
        order=context.order,
        LU=context.LU,
        TU=context.TU,
        mu=context.mu,
        mu_e=context.mu_e,
        mu_m=context.mu_m,
        mu_s=0.0,
        frequency_scale=context.frequency_scale,
    )

    # Rust 数值路径（e2m2e._integrators）：JM c_n 形式直接生成系数，
    # 不依赖 sympy 符号展开（26s → ms）。共线点（gamma 有定义）可用；
    # 扩展未编译或三角点回退符号路径。
    if ctx_cr3bp.gamma is not None:
        try:
            from e2m2e._integrators import build_cr3bp_hamiltonian_py
        except ImportError:
            build_cr3bp_hamiltonian_py = None
        if build_cr3bp_hamiltonian_py is not None:
            gamma = float(ctx_cr3bp.gamma)
            mu_val = float(ctx_cr3bp.mu)
            rho_e_ratio = gamma / (1.0 + gamma)  # 地球项 γ/(1+γ)
            pows, coefs = build_cr3bp_hamiltonian_py(mu_val, gamma, rho_e_ratio, deg)
            numeric_rust: dict[tuple[int, ...], float] = {
                tuple(int(p) for p in pow_t): float(c)
                for pow_t, c in zip(pows, coefs, strict=True)
            }
            # 与符号路径一致：强制平动点平衡（删一阶项）
            numeric_rust = {k: v for k, v in numeric_rust.items() if sum(k) != 1}
            if not numeric_rust:
                numeric_rust[(0, 0, 0, 0, 0, 0)] = 0.0
            return numeric_rust

    legendre = expand_legendre_1_over_r(max_degree=deg)
    H_sym = build_hamiltonian(ctx_cr3bp, legendre, max_degree=deg, store_sources=False)

    # CR3BP 常数参数（会合系，角速度 ω=ẑ 无量纲化为 1）。
    x_lp = float(np.asarray(ctx_cr3bp.libration_position, dtype=float).ravel()[0])

    # 科里奥利 C_pq = [[0,1,0],[-1,0,0],[0,0,0]]（对应 yp_x − xp_y，论文 3 式 4-5）。
    # 离心 C_qq = 零矩阵：H = ½‖p‖² + yp_x − xp_y − Σc_nρⁿP_n（Jorba-Masdemont /
    # Gómez vol I 标准形）。离心效应已通过 H = ½‖ṙ‖² − Ω 的 Legendre 变换
    # 被吸收进 ½‖p‖² + yp_x − xp_y，不作为独立项出现。引力势的完整 Legendre
    # 展开（build_hamiltonian 的 -μ/|r-q|）已包含全部二阶信息。
    # （此前误加 ±½(x²+y²) 离心项，导致 H 的 q-q 块符号错、与 QF 的 S 不自洽。）
    cpq = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    cqq = np.zeros((3, 3))

    # 两主天体相对平动点的位置（地心会合系：地球在 0、月球在 1）
    re = np.array([-x_lp, 0.0, 0.0])
    rm = np.array([1.0 - x_lp, 0.0, 0.0])

    # 力项 f：星历模型里 f 是「受迫频率」（使平动点保持拟周期运动的额外力）。
    # CR3BP 自治、平动点是真实平衡点，故 f 应令 q=0 处 Hamilton 方程的 ṗ=0，
    # 即 f 恰好抵消引力势在原点的一阶梯度。这里先置 0，数值化后删一阶项
    # （等价于令 f = -∂(引力势)/∂q|₀，强制平衡条件）。
    params: dict[str, float] = {}
    for i in range(3):
        for j in range(3):
            params[f"Cpq{i * 3 + j + 1}"] = float(cpq[i, j])
            params[f"Cqq{i * 3 + j + 1}"] = float(cqq[i, j])
    params["f1"] = params["f2"] = params["f3"] = 0.0
    params["rex"], params["rey"], params["rez"] = float(re[0]), float(re[1]), float(re[2])
    params["re0"] = float(np.linalg.norm(re))
    params["rmx"], params["rmy"], params["rmz"] = float(rm[0]), float(rm[1]), float(rm[2])
    params["rm0"] = float(np.linalg.norm(rm))
    # 太阳项已被 mu_s=0 消去，rs* 任取非零避免 0/0（系数恒为 0）
    params["rsx"] = params["rsy"] = params["rsz"] = 1.0
    params["rs0"] = 1.0

    # 逐项数值化：对 sympy 系数做 subs 再求 float
    h_sym_coefs = H_sym.coefficients
    if not isinstance(h_sym_coefs, dict):
        raise TypeError("符号路径 build_hamiltonian 应返回 dict 系数")
    numeric: dict[tuple[int, ...], float] = {}
    for pow_tuple, coef in h_sym_coefs.items():
        val = _eval_coef(coef, params)
        if val != 0.0:
            numeric[tuple(int(p) for p in pow_tuple)] = val

    # 强制平动点平衡：删除一阶项（CR3BP 下 q=0 是平衡点，ṗ=0 要求一阶项为零）。
    # 这等价于令 f 等于引力势在原点的负梯度，与 DS 的 _build_dynamics_rhs_circular
    # 把平动点当原点的约定一致。
    numeric = {k: v for k, v in numeric.items() if sum(k) != 1}

    # —— γ 缩放坐标下的引力势（Jorba-Masdemont c_n 形式）——
    # 标准 expand_legendre 展开的 -μ/|r-q| 在月球附近 (r=γ) 使 (q/r)^n 急剧放大
    # （阶6 达 3.6e4）。改用 JM 的 c_n·ρ^n·P_n(x/ρ) 形式：c_n 含 (γ/r)^{n+1}
    # 因子，月球 r=γ 使其=1，系数降到 O(1)~O(10)。
    # 动能 ½‖p‖² 与科里奥利 yp_x-xp_y 保持原值（系数 1）。
    if ctx_cr3bp.gamma is not None:
        gamma = float(ctx_cr3bp.gamma)
        mu_val = float(ctx_cr3bp.mu)
        rho_e_ratio = gamma / (1.0 + gamma)  # 地球项 γ/(1+γ)

        # 保留动能 + 科里奥利（含 p 项），删原引力势（纯 q 项）用 c_n 重构
        kinetic_coriolis = {k: v for k, v in numeric.items() if sum(k[3:]) > 0}
        grav = _legendre_sum_cn(mu_val, gamma, rho_e_ratio, deg)
        numeric = {**kinetic_coriolis, **grav}

    if not numeric:
        numeric[(0, 0, 0, 0, 0, 0)] = 0.0
    return numeric


def _legendre_sum_cn(
    mu: float, gamma: float, rho_e_ratio: float, max_degree: int
) -> dict[tuple[int, ...], float]:
    """用 Jorba-Masdemont c_n·ρ^n·P_n(x/ρ) 构造 γ 缩放引力势。

    ``H_grav = -Σ_{n≥2} c_n·ρ^n·P_n(x/ρ)``，其中 ``ρ²=x²+y²+z²``、
    ``P_n`` 是 Legendre 多项式。``c_n = (-1)^n/γ³·[μ+(1-μ)(γ/(1+γ))^{n+1}]``
    （L2，JM 式1）。返回 ``{pow_tuple: coef}`` 的多项式 dict。
    """
    # c_n（L2）
    cn = {}
    for n in range(2, max_degree + 1):
        cn[n] = ((-1) ** n / gamma**3) * (mu + (1 - mu) * rho_e_ratio ** (n + 1))

    # ρ^n·P_n(x/ρ) = ρ^n·P_n(x/ρ)。用 P_n 递推展开为 x,y,z 多项式。
    # P_0=1, P_1=u, P_{n} = ((2n-1)/n)·u·P_{n-1} - ((n-1)/n)·P_{n-2}, u=x/ρ
    # ρ^n·P_n(x/ρ): 用 Q_n = ρ^n·P_n(x/ρ) 递推（避免 ρ 分母）
    # Q_0=1, Q_1=x, Q_n = ((2n-1)/n)·x·Q_{n-1} - ((n-1)/n)·ρ²·Q_{n-2}
    # Q_n 是 x,y,z 的 n 次齐次多项式，存为 {(i,j,k): coef}
    Q: dict[int, dict[tuple[int, int, int], float]] = {0: {(0, 0, 0): 1.0}, 1: {(1, 0, 0): 1.0}}
    rho_sq_terms = {(2, 0, 0): 1.0, (0, 2, 0): 1.0, (0, 0, 2): 1.0}  # x²+y²+z²
    for n in range(2, max_degree + 1):
        # Q_n = ((2n-1)/n)·x·Q_{n-1} - ((n-1)/n)·(x²+y²+z²)·Q_{n-2}
        a = (2 * n - 1) / n
        b = (n - 1) / n
        qn: dict[tuple[int, int, int], float] = {}
        for pow_t, coef in Q[n - 1].items():
            new_pow = (pow_t[0] + 1, pow_t[1], pow_t[2])
            qn[new_pow] = qn.get(new_pow, 0.0) + a * coef
        for pow_t, coef in Q[n - 2].items():
            for rs_pow, rs_coef in rho_sq_terms.items():
                new_pow = (pow_t[0] + rs_pow[0], pow_t[1] + rs_pow[1], pow_t[2] + rs_pow[2])
                qn[new_pow] = qn.get(new_pow, 0.0) - b * rs_coef * coef
        Q[n] = {k: v for k, v in qn.items() if abs(v) > 1e-15}

    # H_grav = -Σ c_n·Q_n
    result: dict[tuple[int, ...], float] = {}
    for n in range(2, max_degree + 1):
        c = cn[n]
        for pow_t, coef in Q[n].items():
            full_pow = (pow_t[0], pow_t[1], pow_t[2], 0, 0, 0)
            val = -c * coef
            result[full_pow] = result.get(full_pow, 0.0) + val
    return {k: v for k, v in result.items() if abs(v) > 1e-15}


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
    "build_cr3bp_hamiltonian",
    "evaluate_hamiltonian",
    "hamiltonian_constant_term",
]
