"""高阶中心流形化简（qiao Code10–Code11）。

在 quasi-Floquet 坐标（切片 #172）基础上，用逐阶 Lie 变换消去
Hamiltonian 的双曲-中心耦合项，把非线性 Hamiltonian 化简为仅依赖
作用量的函数：

- **Step 1 ``"invariant"``**（qiao ``Code10_DepartInvarManifold``）：
  消去 ``q_1^k·p_1^l``（``k ≠ l``）的双曲交叉项，判别条件
  ``pow(1)==pow(4)`` 保留双曲"循环坐标" ``I_1 = q_1·p_1``。经此步后
  双曲方向与中心方向解耦，中心部分可独立分析；
- **Step 2 ``"center"``**（qiao ``Code11_DepartCenterManifold``）：
  在 Step 1 结果上，消去两个中心方向 ``(q_2,p_2)``/``(q_3,p_3)`` 之间
  的非共振耦合，更严判别条件
  ``pow(1)==pow(4) && pow(2)==pow(5) && pow(3)==pow(6)``，结果只剩
  作用量 ``I_1 = q_1·p_1``、``I_2 = (q_2²+p_2²)/2``、
  ``I_3 = (q_3²+p_3²)/2``。

算法（两步结构相同）：

1. 逐阶（``3..max_order``）解同调方程 ``{H_2, W_i} = -H_i^{待消}}``。
   ``W_i`` 由频域 ODE 求解器 :func:`_solve_wfunc_fft` /
   :func:`_solve_wfunc_fft_imag` 给出特解 ``ẏ = k·y + f(t)``；
   特征频率 ``k = (j_1−i_1)·λ + (j_2−i_2)·i·ω_p + (j_3−i_3)·i·ω_v``；
2. ``W_i`` 对各高阶 Hamiltonian 的贡献由 Poisson 括号链
   ``ad_{W_i}^n / n!``（Lie 级数）累加，再加上 ``Ẇ_i`` 项更新各阶；
3. 第 ``i`` 阶按判别条件删去被 ``W_i`` 消去的项。

数学上每步都应在末尾用实变换 ``D⁻¹`` 映回实坐标——但由于虚变换 ``D``
是常值辛矩阵且同调方程已显式消去耦合，本模块直接在实 ``QF`` 坐标上
操作（``D`` 只用于推导 ``k`` 的形式），与 qiao 数值实现等价。

单位约定：内部 Hamiltonian 系数全部在 qiao 归一化单位（TU）下运算；
与 SI 之间换算只能经 :class:`NormalFormContext` 与 :mod:`.units`。
``sympy`` 仅在 Legendre/Hamiltonian 构造时惰性导入；本模块不依赖
sympy。``joblib`` 为可选优化（qiao 用其并行 Poisson 括号），本模块
串行实现并显式注明。

Public API：

- :class:`CenterManifoldResult` —— 化简结果句柄；
- :class:`CenterManifoldReducer` —— 上下文绑定的 reducer。

输入 :class:`QuasiFloquetResult` 只提供二阶实标准形 ``D``（频率
``λ``、``ω_p``、``ω_v``）；高阶非线性 Hamiltonian 项需经
``hamiltonian_terms`` 参数注入（对应 qiao ``Code09`` 的
``L?_QF_Hamilton.npz``：``{pow_tuple: coef_array}``，系数为时间序列
``ndarray``）。若不注入，reducer 退化为只含二阶项的平凡情形
（仍能跑通，但无高阶项可消，仅供 smoke）。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import factorial
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from .polynomial import poly_poisson, polylist_simplify
from .quasi_floquet import QuasiFloquetResult

if TYPE_CHECKING:
    from .context import NormalFormContext

# ---------------------------------------------------------------------------
# 默认参数
# ---------------------------------------------------------------------------

#: 默认展开阶数（与 qiao ``Code10``/``Code11`` 一致）。
DEFAULT_MAX_ORDER: int = 10
#: 频域 ODE 求解器边界延拓比例（qiao ``Solve_Wfunc_fft`` 默认值）。
_DEFAULT_EXTENSION_RATIO: float = 0.2
#: MAD 离群抑制阈值乘子（qiao ``Solve_Wfunc_fft_imag`` 默认值）。
_DEFAULT_MAD_KVAL: float = 1e6
#: 数值微分中心差分阶数（qiao ``list_deriv`` 默认值）。
_DEFAULT_DERIV_N: int = 14


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CenterManifoldResult:
    """中心流形化简结果句柄。

    Attributes:
        context: 关联 :class:`NormalFormContext`。
        order: 化简截断阶数（``max_order``）。
        W_series: 各步、各阶生成函数 ``W`` 的系数表。结构
            ``{step_name: {order: {pow_tuple: coef_array}}}``，封装访问；
            ``step_name ∈ {"invariant", "center"}``。系数为**复值**时间
            序列 ``ndarray``（``invariant`` 步 W 天然实值、虚部≈0；
            ``center`` 步 W 为纯虚、实部≈0）——与 qiao Code10/Code11
            输出的复值 ``.npz`` 一致，供 ``coord_trans`` 的 QF↔CM Lie
            流在复域消费。不直接暴露 qiao 的 ``powers``/``coefficients``
            扁平数组命名。
        hamiltonian_terms: 化简后 Hamiltonian 的
            ``{pow_tuple: coef_array}`` 系数表（实 QF 坐标）。
        steps_performed: 实际执行的化简步骤名元组（按顺序）。
        metadata: 自由扩展字段（频率、项数等诊断信息）。
    """

    context: NormalFormContext
    order: int
    W_series: dict[str, dict[int, dict[tuple[int, ...], npt.NDArray[np.complex128]]]]
    hamiltonian_terms: dict[tuple[int, ...], npt.NDArray[np.floating]]
    steps_performed: tuple[str, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def max_hyperbolic_coupling(self) -> float:
        """化简后剩余双曲-中心交叉项系数的最大绝对值。

        "双曲-中心交叉项"指同时涉及双曲方向 ``q_1``/``p_1``（指数位
        0/3）与中心方向（指数位 1/4 或 2/5）的 Hamiltonian 项；理想
        化简后这类项应被消去（系数 ≈ 0）。本属性是步骤 ``"invariant"``
        化简效果的核心诊断量：值越小，化简越彻底。

        Returns:
            ``max_{k∈terms} |coef_k|``，其中 ``k`` 遍历
            ``(pow[0]+pow[3] > 0) and (pow[1]+pow[4]+pow[2]+pow[5] > 0)``
            的项；无此类项时返回 ``0.0``。
        """
        mx = 0.0
        for pow_tuple, coef in self.hamiltonian_terms.items():
            hyper = pow_tuple[0] + pow_tuple[3]
            center = pow_tuple[1] + pow_tuple[4] + pow_tuple[2] + pow_tuple[5]
            if hyper > 0 and center > 0:
                m = float(np.max(np.abs(coef))) if np.ndim(coef) else abs(float(coef))
                if m > mx:
                    mx = m
        return mx

    def W_for(self, step: str, order: int) -> dict[tuple[int, ...], npt.NDArray[np.complex128]]:
        """封装访问器：取出某步某阶的 ``W`` 系数表。

        Args:
            step: 步骤名（``"invariant"`` / ``"center"``）。
            order: 阶数。

        Returns:
            ``{pow_tuple: coef_array}``；该步/阶未执行或无项时返回 ``{}``。

        Raises:
            KeyError: ``step`` 未执行。
        """
        if step not in self.W_series:
            raise KeyError(f"步骤 {step!r} 未执行；已执行：{self.steps_performed}")
        return dict(self.W_series[step].get(order, {}))


# ---------------------------------------------------------------------------
# 频域 ODE 求解器（迁移自 qiao Solve_Wfunc_fft / Solve_Wfunc_fft_imag）
# ---------------------------------------------------------------------------


def _characteristic_freq(pow_tuple: tuple[int, ...], lam: float, wp: float, wv: float) -> complex:
    """同调方程的特征频率 ``k``（复值）。

    ``k = (j_1 − i_1)·λ + (j_2 − i_2)·i·ω_p + (j_3 − i_3)·i·ω_v``，
    其中 ``pow_tuple = (i_1, i_2, i_3, j_1, j_2, j_3)`` 对应
    ``(q_1, q_2, q_3, p_1, p_2, p_3)``。``λ`` 是双曲特征指数（实），
    ``ω_p``/``ω_v`` 是平面/垂直中心频率。对应 qiao
    ``Solve_Wfunc_fft`` 中 ``k`` 的定义。

    ``k == 0``（共振项）时同调方程不可解，调用方负责跳过此类项。
    """
    i1, i2, i3, j1, j2, j3 = (int(p) for p in pow_tuple)
    return complex(
        (j1 - i1) * lam,
        (j2 - i2) * wp + (j3 - i3) * wv,
    )


def _solve_wfunc_fft(
    tlist: npt.NDArray[np.floating],
    forcing: npt.NDArray[np.floating],
    k: complex,
    *,
    extension_ratio: float = _DEFAULT_EXTENSION_RATIO,
) -> npt.NDArray[np.complex128]:
    """频域求解 ``ẏ = k·y + f(t)`` 的特解（实值 ``f``）。

    迁移自 qiao ``Solve_Wfunc_fft``：对 ``f`` 做镜像边界延拓后 FFT，
    乘传递函数 ``H(ω) = 1/(iω − k)`` 再 IFFT，截取原区间。频域 ODE
    求解的离散泄漏由边界延拓抑制（qiao 用 mirror，本实现一致）。

    Args:
        tlist: ``(N,)`` 等距时间序列。
        forcing: ``(N,)`` 强迫项 ``f(t)``。
        k: 特征频率（复值）。
        extension_ratio: 边界延拓比例。

    Returns:
        ``(N,)`` 复值特解 ``y(t)``。
    """
    t_arr = np.asarray(tlist, dtype=float).ravel()
    f_vals = np.asarray(forcing, dtype=float).ravel()
    N = t_arr.size
    if N < 2:
        return np.zeros(N, dtype=complex)
    dt = float(t_arr[1] - t_arr[0])
    M = int(np.ceil(extension_ratio * N))
    # mirror 边界延拓
    f_ext = np.concatenate([f_vals[:M][::-1], f_vals, f_vals[-M:][::-1]])
    N_ext = f_ext.size

    F = np.fft.fft(f_ext)
    if N_ext % 2 == 0:
        freq = np.arange(-N_ext // 2, N_ext // 2, dtype=float)
    else:
        freq = np.arange(-(N_ext - 1) // 2, (N_ext - 1) // 2 + 1, dtype=float)
    freq = freq * (1.0 / (N_ext * dt))
    omega = 2.0 * np.pi * freq
    omega[np.abs(omega) < 1e-12] = 1e-12  # 避免零频奇异

    H = 1.0 / (1j * omega - k)
    H = np.fft.fftshift(H)
    y_ext = np.fft.ifft(F * H)
    return y_ext[M : M + N]


def _limit_fft_outliers_mad(
    fft_result: npt.NDArray[np.complex128], k_val: float
) -> tuple[npt.NDArray[np.complex128], bool]:
    """MAD（中位绝对偏差）离群抑制。

    迁移自 qiao ``Solve_Wfunc_fft_imag._limit_fft_outliers_mad``：对 FFT
    幅值超过 ``med_A + k_val·MAD/0.6745`` 的频点按比例缩回阈值，抑制
    共振峰在频域产生的数值尖刺。

    Returns:
        ``(corrected_fft, corrected_flag)``。
    """
    A = np.abs(fft_result)
    med_A = float(np.median(A))
    mad = float(np.median(np.abs(A - med_A)))
    # MAD=0 时频谱无统计离散度（如常系数输入，FFT 仅零频非零），
    # ``threshold = med_A + k·0 = med_A`` 会把所有高于中位数的频点
    # （含唯一非零的零频）当成离群点缩到 med_A，使整个 W 归零。此时
    # "离群点"概念不成立——抑制对象是共振尖刺，而此类信号无尖刺，
    # 跳过即可。对 qiao 真实时变输入（MAD>0）零影响。
    if mad == 0.0:
        return fft_result.copy(), False
    threshold = med_A + k_val * (mad / 0.6745)
    outliers = threshold < A
    corrected = bool(np.any(outliers))
    if not corrected:
        return fft_result.copy(), False
    fft_limited = fft_result.copy()
    idx = np.where(outliers)[0]
    scale = threshold / A[idx]
    fft_limited[idx] = fft_result[idx] * scale
    return fft_limited, corrected


def _solve_wfunc_fft_imag(
    tlist: npt.NDArray[np.floating],
    forcing: npt.NDArray[np.floating],
    k: complex,
    *,
    extension_ratio: float = _DEFAULT_EXTENSION_RATIO,
    mad_kval: float = _DEFAULT_MAD_KVAL,
) -> tuple[npt.NDArray[np.complex128], bool]:
    """频域求解 ``ẏ = k·y + f(t)``（复值 ``f``，含 MAD 离群抑制）。

    迁移自 qiao ``Solve_Wfunc_fft_imag``：结构与 :func:`_solve_wfunc_fft`
    一致，但额外对 FFT 结果做 MAD 离群抑制，返回 ``(y, corrected)``。
    ``corrected=True`` 表示发生了离群抑制（调用方据此切换 ``Ẇ`` 的
    数值微分路径）。
    """
    t_arr = np.asarray(tlist, dtype=float).ravel()
    f_vals = np.asarray(forcing, dtype=complex).ravel()
    N = t_arr.size
    if N < 2:
        return np.zeros(N, dtype=complex), False
    dt = float(t_arr[1] - t_arr[0])
    M = int(np.ceil(extension_ratio * N))
    f_ext = np.concatenate([f_vals[:M][::-1], f_vals, f_vals[-M:][::-1]])
    N_ext = f_ext.size

    F = np.fft.fft(f_ext)
    if N_ext % 2 == 0:
        freq = np.arange(-N_ext // 2, N_ext // 2, dtype=float)
    else:
        freq = np.arange(-(N_ext - 1) // 2, (N_ext - 1) // 2 + 1, dtype=float)
    freq = freq * (1.0 / (N_ext * dt))
    omega = 2.0 * np.pi * freq
    omega[np.abs(omega) < 1e-12] = 1e-12

    H = 1.0 / (1j * omega - k)
    H = np.fft.fftshift(H)
    Y_freq = F * H
    y_ext = np.fft.ifft(Y_freq)
    y = y_ext[M : M + N]

    Y_corrected, corrected = _limit_fft_outliers_mad(Y_freq, mad_kval)
    if corrected:
        y_corr = np.fft.ifft(Y_corrected)[M : M + N]
        return y_corr, True
    return y, False


# ---------------------------------------------------------------------------
# 高阶数值微分（迁移自 qiao list_deriv）
# ---------------------------------------------------------------------------


def _vandermonde_deriv_coeffs(n: int, mode: int) -> npt.NDArray[np.floating]:
    """Vandermonde 系统求数值微分系数。

    ``mode=0`` 中心、``1`` 前向、``-1`` 后向，节点数 ``n+1``。
    对应 qiao ``list_deriv._vandermonde_coeffs``。
    """
    if mode == 0:
        k = np.arange(-n // 2, n // 2 + 1)
    elif mode == 1:
        k = np.arange(0, n + 1)
    elif mode == -1:
        k = np.arange(-n, 1)
    else:
        raise ValueError(f"未知 mode: {mode}")
    if k.size <= 1:
        return np.array([0.0])
    V = np.vander(k, increasing=True).T
    b = np.zeros(n + 1)
    b[1] = 1.0
    return np.linalg.solve(V, b)


def list_deriv(
    y: npt.ArrayLike,
    h: float,
    n: int = _DEFAULT_DERIV_N,
    swi: int = 4,
    ord_boundary: int = 10,
) -> npt.NDArray[np.floating]:
    """等距采样高阶数值微分。

    迁移自 qiao ``list_deriv``：内部 ``n`` 阶中心差分，两端用
    ``swi`` 点前/后向差分（阶 ``ord_boundary``），过渡区用递降阶中心
    差分。复值 ``y`` 自动转实/虚部分别处理。

    Args:
        y: ``(N,)`` 等距采样数据（实或复）。
        h: 采样间距。
        n: 中心差分阶（偶数）。
        swi: 端点前/后向差分点数。
        ord_boundary: 端点差分阶。

    Returns:
        ``(N,)`` 导数 ``dy/dt``。
    """
    y_arr = np.asarray(y)
    if np.iscomplexobj(y_arr):
        return list_deriv(y_arr.real, h, n, swi, ord_boundary) + 1j * list_deriv(
            y_arr.imag, h, n, swi, ord_boundary
        )
    y_arr = y_arr.ravel()
    N = y_arr.size
    dy = np.zeros(N, dtype=float)
    half = n // 2

    # 内部：n 阶中心差分
    if n < N:
        c = _vandermonde_deriv_coeffs(n, mode=0)
        for idx_k, kk in enumerate(range(-half, half + 1)):
            dy[half : N - half] += c[idx_k] * y_arr[half + kk : N - half + kk]
        dy[half : N - half] /= h

    # 前向/后向端点
    for i in range(min(swi, N)):
        ord_eff = min(ord_boundary, N - i - 1)
        if ord_eff < 1:
            dy[i] = (y_arr[i + 1] - y_arr[i]) / h if i + 1 < N else 0.0
        else:
            c_fwd = _vandermonde_deriv_coeffs(ord_eff, mode=1)
            dy[i] = float(np.dot(c_fwd, y_arr[i : i + ord_eff + 1])) / h

    for i in range(min(swi, N)):
        ri = N - 1 - i
        ord_eff = min(ord_boundary, ri)
        if ord_eff < 1:
            dy[ri] = (y_arr[ri] - y_arr[ri - 1]) / h if ri > 0 else 0.0
        else:
            c_bwd = _vandermonde_deriv_coeffs(ord_eff, mode=-1)
            dy[ri] = float(np.dot(c_bwd, y_arr[ri - ord_eff : ri + 1])) / h

    # 过渡区（swi..half-1）：递降阶中心差分
    for i in range(swi, min(half, N - swi)):
        span = 2 * i
        if span < 2:
            continue
        c_center = _vandermonde_deriv_coeffs(span, mode=0)
        offset = span // 2
        acc = 0.0
        for idx_k, kk in enumerate(range(-offset, offset + 1)):
            acc += c_center[idx_k] * y_arr[i + kk]
        dy[i] = acc / h
        ri = N - 1 - i
        acc = 0.0
        for idx_k, kk in enumerate(range(-offset, offset + 1)):
            acc += c_center[idx_k] * y_arr[ri + kk]
        dy[ri] = acc / h

    return dy


# ---------------------------------------------------------------------------
# 同调方程判别条件
# ---------------------------------------------------------------------------


def _is_invariant_term(pow_tuple: tuple[int, ...]) -> bool:
    """Step 1（invariant）保留条件：``pow(1)==pow(4)``。

    满足此条件的 ``q_1^k·p_1^k`` 项是双曲"循环坐标"作用量
    ``I_1 = q_1·p_1`` 的分量，**保留**；其余 ``q_1^k·p_1^l (k≠l)`` 双曲
    交叉项被 ``W`` 消去。对应 qiao ``Code10`` 的 ``if n1 == n4`` 分支。
    """
    return int(pow_tuple[0]) == int(pow_tuple[3])


def _is_center_term(pow_tuple: tuple[int, ...]) -> bool:
    """Step 2（center）保留条件：三对共轭全部平衡。

    ``pow(1)==pow(4) && pow(2)==pow(5) && pow(3)==pow(6)``。满足时该项
    仅依赖作用量（``I_1``、``I_2``、``I_3`` 的幂次），**保留**；其余
    含中心方向非共振耦合的项被消去。对应 qiao ``Code11`` 的
    ``if n1 == n4 and n2 == n5 and n3 == n6`` 分支。
    """
    return (
        int(pow_tuple[0]) == int(pow_tuple[3])
        and int(pow_tuple[1]) == int(pow_tuple[4])
        and int(pow_tuple[2]) == int(pow_tuple[5])
    )


# ---------------------------------------------------------------------------
# 单步 Lie 变换（Code10 / Code11 通用骨架）
# ---------------------------------------------------------------------------


def _polylist_to_complex(
    poly: Mapping[tuple[int, ...], npt.ArrayLike],
) -> dict[tuple[int, ...], npt.NDArray[np.complex128]]:
    """把实/复值系数表统一转为复值（W 封装用，保留 invariant/center 两步的复值 W）。"""
    return {k: np.asarray(v, dtype=complex) for k, v in poly.items()}


def _lie_transform_step(
    H_by_order: dict[int, dict[tuple[int, ...], npt.NDArray[np.floating]]],
    *,
    max_order: int,
    tlist: npt.NDArray[np.floating],
    lam: float,
    wp: float,
    wv: float,
    keep_criterion,
    delete_criterion,
    use_imag_solver: bool,
) -> tuple[
    dict[int, dict[tuple[int, ...], npt.NDArray[np.floating]]],
    dict[int, dict[tuple[int, ...], npt.NDArray[np.complex128]]],
]:
    """执行一步中心流形化简的 Lie 变换（Code10/Code11 通用骨架）。

    Args:
        H_by_order: ``{order: {pow: coef}}``，被原地更新。
        max_order: 截断阶数。
        tlist: 时间序列。
        lam, wp, wv: 频率（双曲/平面/垂直）。
        keep_criterion: 保留判据（决定**是否对该项求 W**）；满足者跳过
            （W 记零）。Code10 用 ``_is_invariant_term``（``pow1==pow4``），
            Code11 用 ``_is_center_term``（三对全平衡）。
        delete_criterion: 该阶处理完后**删除**不满足此判据的项。
            Code10 与 Code11 不同：Code10 删所有非作用量项（三对全平衡），
            Code11 只删真正求过 W 的项（``list_iseliminate``）。
        use_imag_solver: ``True`` 用复值 + MAD 求解器（Step 2），
            ``False`` 用实值求解器（Step 1）。

    Returns:
        ``(H_by_order, W_series)``：``W_series`` 为
        ``{order: {pow: W_coef}}``（复值，``W_coef`` 时间序列）。

    Notes:
        忠实迁移 qiao ``Code10``/``Code11`` 的逐阶循环：

        1. 对每个 ``order ∈ [3, max_order]``，遍历 ``H_order`` 中不满足
           ``keep_criterion`` 的项，求同调方程特解 ``W`` 与 ``Ẇ``；
        2. Poisson 括号链累加 ``ad_W^n / n!`` 到更高阶 Hamiltonian
           （``j`` 从 2 起，``cur_order = j + order − 2`` 递增）；
        3. 加 ``Ẇ`` 项，按 ``delete_criterion`` 删除该阶残余/新生项。
    """
    W_series: dict[int, dict[tuple[int, ...], npt.NDArray[np.complex128]]] = {}
    N = tlist.size

    for order in range(3, max_order + 1):
        if order not in H_by_order:
            H_by_order[order] = {}
        H_order = H_by_order[order]

        W_temp: dict[tuple[int, ...], npt.NDArray[np.complex128]] = {}
        Wd_temp: dict[tuple[int, ...], npt.NDArray[np.complex128]] = {}
        eliminated: list[tuple[int, ...]] = []

        for pow_tuple, coef_arr in list(H_order.items()):
            if keep_criterion(pow_tuple):
                # 保留项（共振/作用量项）：不生成 W，W 记零
                W_temp[pow_tuple] = np.zeros(N, dtype=complex)
                continue

            coef_c = np.asarray(coef_arr, dtype=complex)
            k = _characteristic_freq(pow_tuple, lam, wp, wv)
            # Step 1：Ẇ = -k·W - coef（qiao Code10 同调方程闭合形式）。
            # Step 2：先用复值+MAD 求 W；若 MAD 抑制触发则 Ẇ 走数值微分，
            #         否则 Ẇ = k·W + coef（再取负，见下方 if-else 实现）。
            if not use_imag_solver:
                W_func = _solve_wfunc_fft(tlist, coef_c.real, k)
                Wd_func = -k * W_func - coef_c
            else:
                W_func, corrected = _solve_wfunc_fft_imag(tlist, coef_c, k)
                if not corrected:
                    Wd_func = k * W_func + coef_c
                else:
                    dt = float(np.mean(np.diff(tlist)))
                    Wd_func = list_deriv(W_func, dt)
                Wd_func = -Wd_func

            W_temp[pow_tuple] = W_func
            Wd_temp[pow_tuple] = Wd_func
            eliminated.append(pow_tuple)

        W_series[order] = W_temp

        # Poisson 括号链：ad_W^n / n! 累加到高阶 Hamiltonian
        for j in range(2, max_order):
            if j not in H_by_order or not H_by_order[j]:
                continue
            H_j = H_by_order[j]
            cur_order = j + order - 2
            if cur_order > max_order:
                continue
            num = 1
            # poly_poisson 在本数值路径下输入/输出系数均为 ndarray，但其签名
            # 为 object（兼顾 sympy 路径）；用 Any 局部标注表达 cascade 运算。
            p_prev: Any = H_j
            while cur_order <= max_order:
                p: dict[tuple[int, ...], Any] = poly_poisson(p_prev, W_temp)
                for kk in p:
                    p[kk] = p[kk] / factorial(num)
                num += 1
                if cur_order not in H_by_order:
                    H_by_order[cur_order] = {}
                target = H_by_order[cur_order]
                for kk, v in p.items():
                    existing: Any = target.get(kk, 0.0)
                    target[kk] = existing + v
                cur_order += order - 2
                p_prev = p

        # 加 Ẇ 项
        if order in H_by_order:
            target = H_by_order[order]
            for kk, v in Wd_temp.items():
                # Wd 项为复值；与既有（可能复值）累加项相加。
                cur: Any = target.get(kk, 0.0)
                target[kk] = cur + v

        # 按 delete_criterion 删除该阶残余/新生项（Code10 与 Code11 不同）
        if order in H_by_order:
            H_by_order[order] = {
                k: v for k, v in H_by_order[order].items() if delete_criterion(k, eliminated)
            }
            if not H_by_order[order]:
                H_by_order[order] = {(0, 0, 0, 0, 0, 0): np.zeros(N)}

    # 化简各阶
    H_by_order = {o: polylist_simplify(v) for o, v in H_by_order.items() if v}
    return H_by_order, W_series


def _delete_invariant(pow_tuple, eliminated: list[tuple[int, ...]]) -> bool:
    """Code10 删除判据：保留三对全平衡项（即作用量项）。

    qiao ``Code10`` 第 131–134 行：``k[0]==k[3] and k[1]==k[4] and
    k[2]==k[5]``。注意 Code10 用 ``keep_criterion=_is_invariant_term``
    决定**求 W**（只对 ``pow1≠pow4`` 求解），但删除时更严，连中心非共振
    项一并清掉。这是 qiao 的实际行为：Step 1 把 Hamiltonian 直接化到
    作用量形式（Step 2 在此基础上再做精修，结构相同但 ``keep`` 更严、
    无项可删时退化为恒等）。
    """
    return (
        int(pow_tuple[0]) == int(pow_tuple[3])
        and int(pow_tuple[1]) == int(pow_tuple[4])
        and int(pow_tuple[2]) == int(pow_tuple[5])
    )


def _delete_center(pow_tuple, eliminated: list[tuple[int, ...]]) -> bool:
    """Code11 删除判据：保留非 eliminated 的项。

    qiao ``Code11`` 第 146–148 行：只删 ``list_iseliminate``（即本阶求
    过 W 的项），其他项（包括 Poisson cascade 新生成的）保留。Code11
    的 ``keep_criterion`` 已是三对全平衡，故非 eliminated 项天然是作用量项。
    """
    return pow_tuple not in eliminated


# ---------------------------------------------------------------------------
# Reducer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CenterManifoldReducer:
    """中心流形化简 reducer（上下文绑定）。

    通过 :meth:`reduce` 在 quasi-Floquet 化简后的 Hamiltonian 上执行
    一步或两步 Lie 变换，消去双曲-中心耦合与中心方向间非共振耦合，
    输出 :class:`CenterManifoldResult`。

    Args:
        context: 归一化上下文（提供频率 ``ω_p``/``ω_v``、特征指数 λ）。
        max_order: Lie 变换截断阶数，默认 ``10``（与 qiao 一致）。
    """

    context: NormalFormContext
    max_order: int = DEFAULT_MAX_ORDER

    def reduce(
        self,
        qf_result: QuasiFloquetResult,
        hamiltonian_terms: Mapping[tuple[int, ...], npt.ArrayLike] | None = None,
        steps: tuple[str, ...] | None = None,
    ) -> CenterManifoldResult:
        """对 ``qf_result`` 执行中心流形化简。

        Args:
            qf_result: quasi-Floquet 变换结果（切片 #172），提供实标准形
                ``D``（频率 ``λ``/``ω_p``/``ω_v``）与采样时间 ``tlist``。
            hamiltonian_terms: 高阶 Hamiltonian 系数表
                ``{pow_tuple: coef_array}``（对应 qiao ``Code09`` 的
                ``L?_QF_Hamilton.npz``）。``None`` 时 reducer 只用二阶实
                标准形项（平凡情形，仅供 smoke）。系数为时间序列
                ``ndarray``，长度应与 ``qf_result.tlist`` 一致。
            steps: 要执行的步骤元组，按顺序；默认
                ``("invariant", "center")``。可选单步 ``("invariant",)``
                或 ``("center",)``。

        Returns:
            :class:`CenterManifoldResult`。

        Raises:
            ValueError: ``steps`` 含非法名；``max_order`` 非正；
                ``hamiltonian_terms`` 系数长度与 ``tlist`` 不一致。
        """
        valid_steps = {"invariant", "center"}
        steps_resolved = tuple(steps) if steps is not None else ("invariant", "center")
        bad = [s for s in steps_resolved if s not in valid_steps]
        if bad:
            raise ValueError(f"steps 只能含 {sorted(valid_steps)}，得到非法值：{bad}")
        if self.max_order < 1:
            raise ValueError(f"max_order 必须为正，得到 {self.max_order}")

        # 频率（实标准形 D 同源）
        lam = float(self.context.characteristic_exponent)
        nu1, nu2 = self.context.central_frequencies
        wp, wv = float(nu1), float(nu2)

        tlist = np.asarray(qf_result.tlist, dtype=float).ravel()
        N = tlist.size

        # 组装初始 Hamiltonian 多项式表 {order: {pow: coef}}
        H_by_order = self._assemble_hamiltonian(qf_result, hamiltonian_terms, tlist, N, lam, wp, wv)

        W_all: dict[str, dict[int, dict[tuple[int, ...], npt.NDArray[np.complex128]]]] = {}
        steps_done: list[str] = []
        for step in steps_resolved:
            if step == "invariant":
                H_by_order, W_step = _lie_transform_step(
                    H_by_order,
                    max_order=self.max_order,
                    tlist=tlist,
                    lam=lam,
                    wp=wp,
                    wv=wv,
                    keep_criterion=_is_invariant_term,
                    delete_criterion=_delete_invariant,
                    use_imag_solver=False,
                )
            else:  # center
                H_by_order, W_step = _lie_transform_step(
                    H_by_order,
                    max_order=self.max_order,
                    tlist=tlist,
                    lam=lam,
                    wp=wp,
                    wv=wv,
                    keep_criterion=_is_center_term,
                    delete_criterion=_delete_center,
                    use_imag_solver=True,
                )
            # W 保留复值：Step 1（invariant）的双曲特征频率 λ 为实数，W
            # 天然实值；Step 2（center）的中心频率为纯虚 iω，同调方程特解
            # W 为纯虚（实部≈0）。coord_trans 的 QF↔CM Lie 流在复域操作，
            # 必须拿到完整复值 W，故此处不取实部。对应 qiao Code10/Code11
            # 输出复值 ``L?_InvarManifold.npz`` / ``L?_CenterManifold.npz``。
            W_all[step] = {o: _polylist_to_complex(w) for o, w in W_step.items()}
            steps_done.append(step)

        # 汇总化简后 Hamiltonian（所有阶合并成一个 pow→coef 表）
        final_terms: dict[tuple[int, ...], npt.NDArray[np.floating]] = {}
        for o in sorted(H_by_order.keys()):
            for k, v in H_by_order[o].items():
                v_arr = np.real(np.asarray(v, dtype=complex))
                if k in final_terms:
                    final_terms[k] = final_terms[k] + v_arr
                else:
                    final_terms[k] = v_arr
        final_terms = polylist_simplify(final_terms)

        # 化简前最大双曲-中心耦合（诊断：用原始 hamiltonian_terms）
        pre_coupling = _max_hyperbolic_center_coupling(self._flat_terms(hamiltonian_terms, N))

        return CenterManifoldResult(
            context=self.context,
            order=int(self.max_order),
            W_series=W_all,
            hamiltonian_terms=final_terms,
            steps_performed=tuple(steps_done),
            metadata={
                "lambda": lam,
                "wp": wp,
                "wv": wv,
                "n_samples": N,
                "pre_hyperbolic_center_coupling": pre_coupling,
                "max_order": int(self.max_order),
            },
        )

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _assemble_hamiltonian(
        self,
        qf_result: QuasiFloquetResult,
        hamiltonian_terms: Mapping[tuple[int, ...], npt.ArrayLike] | None,
        tlist: npt.NDArray[np.floating],
        N: int,
        lam: float,
        wp: float,
        wv: float,
    ) -> dict[int, dict[tuple[int, ...], npt.NDArray[np.floating]]]:
        """组装 Lie 变换输入的 Hamiltonian 多项式表。

        二阶项来自实标准形（``λ·q_1·p_1 + (ω_p/2)(q_2²+p_2²)
        + (ω_v/2)(q_3²+p_3²)``，对应 qiao ``Code09`` 的 ``NormalForm_poly``）；
        高阶项来自 ``hamiltonian_terms``（若提供）。按总阶数分组。
        """
        H_by_order: dict[int, dict[tuple[int, ...], npt.NDArray[np.floating]]] = {}

        def _put(deg: int, key: tuple[int, ...], val: np.ndarray) -> None:
            H_by_order.setdefault(deg, {})
            if key in H_by_order[deg]:
                H_by_order[deg][key] = H_by_order[deg][key] + val
            else:
                H_by_order[deg][key] = val

        # 二阶实标准形（qiao Code09 NormalForm_poly）
        _put(2, (1, 0, 0, 1, 0, 0), lam * np.ones(N))
        _put(2, (0, 2, 0, 0, 0, 0), (wp / 2) * np.ones(N))
        _put(2, (0, 0, 0, 0, 2, 0), (wp / 2) * np.ones(N))
        _put(2, (0, 0, 2, 0, 0, 0), (wv / 2) * np.ones(N))
        _put(2, (0, 0, 0, 0, 0, 2), (wv / 2) * np.ones(N))

        # 高阶项（Code09 输出）
        if hamiltonian_terms:
            for pow_tuple, coef in hamiltonian_terms.items():
                key = tuple(int(p) for p in pow_tuple)
                deg = sum(key)
                if deg < 1 or deg > self.max_order:
                    continue
                arr = np.asarray(coef, dtype=float).ravel()
                if arr.size == 1:
                    arr = np.full(N, float(arr[0]))
                if arr.size != N:
                    raise ValueError(
                        f"hamiltonian_terms 系数长度 {arr.size} 与 tlist "
                        f"长度 {N} 不一致（pow={key}）"
                    )
                _put(deg, key, arr)
        return H_by_order

    @staticmethod
    def _flat_terms(
        hamiltonian_terms: Mapping[tuple[int, ...], npt.ArrayLike] | None,
        N: int,
    ) -> dict[tuple[int, ...], npt.NDArray[np.floating]]:
        """把注入的高阶项规整为扁平 ``{pow: coef}`` 表（诊断用）。"""
        out: dict[tuple[int, ...], npt.NDArray[np.floating]] = {}
        if hamiltonian_terms:
            for pow_tuple, coef in hamiltonian_terms.items():
                arr = np.asarray(coef, dtype=float).ravel()
                if arr.size == 1:
                    arr = np.full(N, float(arr[0]))
                out[tuple(int(p) for p in pow_tuple)] = arr
        return out


def _max_hyperbolic_center_coupling(
    terms: Mapping[tuple[int, ...], npt.ArrayLike],
) -> float:
    """计算 Hamiltonian 表中双曲-中心交叉项系数的最大绝对值。

    双曲方向指数位 0/3（``q_1``/``p_1``），中心方向指数位 1/4/2/5
    （``q_2``/``p_2``/``q_3``/``p_3``）。同时涉及时即为交叉项。
    """
    mx = 0.0
    for pow_tuple, coef in terms.items():
        hyper = int(pow_tuple[0]) + int(pow_tuple[3])
        center = int(pow_tuple[1]) + int(pow_tuple[4]) + int(pow_tuple[2]) + int(pow_tuple[5])
        if hyper > 0 and center > 0:
            arr = np.asarray(coef, dtype=float).ravel()
            m = float(np.max(np.abs(arr))) if arr.size else 0.0
            if m > mx:
                mx = m
    return mx


__all__ = [
    "DEFAULT_MAX_ORDER",
    "CenterManifoldReducer",
    "CenterManifoldResult",
    "list_deriv",
]
