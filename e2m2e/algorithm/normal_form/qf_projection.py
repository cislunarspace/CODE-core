"""H→QF 映射：把 Hamiltonian 投影到 quasi-Floquet 坐标。

对应 qiao ``Code09_Kamilton_QF.m``。把平动点偏移坐标 ``(q1..p3)`` 下的
Hamiltonian 高阶项（阶 ≥ 3），经 quasi-Floquet 变换矩阵 ``B(t)`` 做线性
替换 ``X = B·Y``，重排为 QF 坐标 ``Y`` 的多项式，逐时刻代入 ``B(t)`` 数值
得到时间序列系数。

二阶项不投影——:class:`CenterManifoldReducer` 的 ``_assemble_hamiltonian``
自加实标准形二阶项（``λ q₁p₁ + ω/2(q²+p²)``）；本函数只提供 ≥3 阶非线性
项，供 Lie 变换消去双曲-中心耦合。

两条实现路径：

1. **Rust 数值展开**（默认）：``e2m2e._integrators.project_hamiltonian_qf_py``
   对每个采样时刻做 ``X = B·Y`` 的数值多项式展开（multinomial 组合枚举），
   不依赖符号运算。CR3BP 标量系数路径下毫秒级完成。
2. **sympy 符号展开**（回退）：扩展未编译、或系数为时间序列（星历路径）
   时回退到 :func:`_project_hamiltonian_to_qf_sympy`（原实现，符号替换
   ``x_i → Σ_j B[i,j]·y_j`` + 逐时刻求值，较慢）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .polynomial import poly_subs

if TYPE_CHECKING:
    from .quasi_floquet import QuasiFloquetResult

__all__ = ["project_hamiltonian_to_qf"]


def project_hamiltonian_to_qf(
    hamiltonian_terms: dict[tuple[int, ...], float],
    qf_result: QuasiFloquetResult,
) -> dict[tuple[int, ...], npt.NDArray[np.floating]]:
    """把 Hamiltonian 高阶项投影到 QF 坐标。

    Args:
        hamiltonian_terms: 平动点偏移坐标 ``(q1..p3)`` 下的 Hamiltonian
            系数 dict ``{pow_tuple: float}``（如
            :func:`build_cr3bp_hamiltonian` 输出）。阶 < 3 的项被丢弃。
            系数为标量（CR3BP 自治）或 ``ndarray`` 时间序列。
        qf_result: quasi-Floquet 结果，提供 ``B(t)`` 与 ``tlist``。

    Returns:
        ``{pow_tuple: coef_array}``，``coef_array`` 长度与
        ``qf_result.tlist`` 一致。变量为 QF 坐标（命名为 q1..p3）。
    """
    tlist = np.asarray(qf_result.tlist, dtype=float).ravel()
    N = tlist.size

    # 收集标量系数的 ≥3 阶项；含时间序列系数时走符号回退
    pows: list[list[int]] = []
    coefs: list[float] = []
    scalar_only = True
    for pow_tuple, coef in hamiltonian_terms.items():
        if sum(pow_tuple) < 3:
            continue
        arr = np.asarray(coef, dtype=float).ravel()
        if arr.size != 1:
            scalar_only = False
            break
        pows.append([int(p) for p in pow_tuple])
        coefs.append(float(arr[0]))

    try:
        from e2m2e._integrators import project_hamiltonian_qf_py
    except ImportError:
        project_hamiltonian_qf_py = None

    if scalar_only and pows and project_hamiltonian_qf_py is not None:
        b_seq = [np.asarray(qf_result.B(t), dtype=float).ravel().tolist() for t in tlist]
        out_pows, out_coefs = project_hamiltonian_qf_py(pows, coefs, b_seq)
        coef_matrix = np.asarray(out_coefs, dtype=float)  # (M, K)
        result: dict[tuple[int, ...], npt.NDArray[np.floating]] = {
            tuple(p): coef_matrix[:, k] for k, p in enumerate(out_pows)
        }
        cleaned = {k: v for k, v in result.items() if np.any(np.abs(v) > 1e-14)}
        if not cleaned:
            cleaned = {(0, 0, 0, 0, 0, 0): np.zeros(N)}
        return cleaned

    return _project_hamiltonian_to_qf_sympy(hamiltonian_terms, qf_result)


def _project_hamiltonian_to_qf_sympy(
    hamiltonian_terms: dict[tuple[int, ...], float],
    qf_result: QuasiFloquetResult,
) -> dict[tuple[int, ...], npt.NDArray[np.floating]]:
    """符号路径（原实现）：sympy 替换展开 + lambdify 逐时刻求值。"""
    import sympy as sp

    tlist = np.asarray(qf_result.tlist, dtype=float).ravel()
    N = tlist.size

    # 旧变量（平动点偏移坐标）命名为 x1..x6，避免与新变量 q1..p3 同名
    x_syms = sp.symbols("x1:7", real=True)
    # 新变量（QF 坐标）命名为 q1..p3（poly_subs 的约定）
    y_syms = sp.symbols("q1 q2 q3 p1 p2 p3", real=True)
    # B 矩阵元素符号 b11..b66
    b_syms = sp.symbols("b1:7_1:7", real=True)
    B_sym = sp.Matrix([[b_syms[i * 6 + j] for j in range(6)] for i in range(6)])
    Y = sp.Matrix(y_syms)

    # subs_map: x_i → Σ_j B[i,j]·y_j
    subs_map = {x_syms[i]: sum(B_sym[i, j] * Y[j] for j in range(6)) for i in range(6)}

    # 逐项符号替换，收集 {new_pow: sym_expr(含 b_ij)}
    symbolic: dict[tuple[int, ...], object] = {}
    for pow_tuple, coef in hamiltonian_terms.items():
        if sum(pow_tuple) < 3:
            continue
        # 单项式 C·x^n（C 为数值系数，可能标量或时间序列）
        single = {tuple(pow_tuple): coef}
        substituted = poly_subs(single, subs_map)
        for new_pow, sym_coef in substituted.items():
            existing = symbolic.get(new_pow, sp.Integer(0))
            symbolic[new_pow] = sp.expand(existing + sp.sympify(sym_coef))

    if not symbolic:
        return {(0, 0, 0, 0, 0, 0): np.zeros(N)}

    # 把每个符号系数编译成 lambda(b11,b12,...,b66)，一次性向量化求值
    # 整条时间序列（B(t) 的 36 个元素逐时刻作为数组参数，numpy 广播）。
    b_list = list(b_syms)
    lambdas = {pow_tuple: sp.lambdify(b_list, expr, "numpy") for pow_tuple, expr in symbolic.items()}
    B_all = np.stack([np.asarray(qf_result.B(t), dtype=float).ravel() for t in tlist])  # (N, 36)

    result: dict[tuple[int, ...], npt.NDArray[np.floating]] = {}
    for pow_tuple, fn in lambdas.items():
        val = fn(*[B_all[:, j] for j in range(36)])
        result[pow_tuple] = np.asarray(val, dtype=float).ravel()

    # 剔除全零序列（数值噪声或抵消）
    cleaned = {k: v for k, v in result.items() if np.any(np.abs(v) > 1e-14)}
    if not cleaned:
        cleaned = {(0, 0, 0, 0, 0, 0): np.zeros(N)}
    return cleaned
