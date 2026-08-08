"""quasi-Floquet 变换矩阵 ``B(t)`` （Code7–Code9）。

对应 qiao ``Code08_QuasiFloquet.py`` （矩阵法）与
``Code08_QuasiFloquet_LA.py`` （李代数法）：从动力学替代轨道出发，求解

.. math::

    \\dot{B}(t) = M(t)\\,B(t) - B(t)\\,D,\\qquad B^{T} J B = J,

其中 ``M(t)`` 是替代轨道邻域的时变线性化矩阵，``D`` 是实标准形矩阵
（一双曲方向 ``±λ``，两中心方向 ``±i ω_p``、``±i ω_v``），``J`` 是 6 维
辛矩阵。``B(t)`` 把受迫线性化系统化为常系数的 ``Ẏ = D·Y``。

两种实现：

- **矩阵法** （``method="matrix"``）：把 ``B`` 展平为 36 维状态直接积分
  ``Ḃ = M·B − B·D``，事后用 Newton 迭代把每个采样点的 ``B`` 投影到
  最近的辛矩阵（qiao ``Code08_QuasiFloquet``）；
- **李代数法** （``method="lie_algebra"``）：参数化 ``B = B₀·exp(ξ)``，
  ``ξ ∈ sp(6, R)``（21 维），在李代数里做修正，``B`` 自动保辛（qiao
  ``Code08_QuasiFloquet_LA``）；
- **常数法** （``method="constant"``）：CR3BP 下 ``M`` 是常数矩阵，
  方程 ``Ḃ = M·B − B·D`` 有常数解 ``B = V``（把 ``M`` 化到实标准形
  ``D`` 的变换矩阵）。``V`` 的元素 O(1)、不随 ``e^{λt}`` 增长，投影到
  QF 坐标的 Hamiltonian 系数保持常数——中心流形化简的同调方程退化为
  代数除法（Gómez vol III §2.7.1），无需 FFT 频域求解。矩阵法/多点
  打靶在 CR3BP 下解出的时变解（``B(0)=I`` 前向或 ``B(T)=I`` 末端）使
  系数随窗口变化，短窗口下 FFT 求解器因频率分辨率不足产生系统偏差。

辛保持性的数值保证：当 ``M(t)`` 与 ``D`` 都是 Hamilton 矩阵
（``MᵀJ + JM = 0``）时，``BᵀJB`` 是 ``Ḃ`` 方程的精确首次积分，因此
本模块刻意构造 ``M(t) = J·S(t)`` （``S`` 对称）以保证辛约束在
``<1e-12`` 量级成立；矩阵法仍在末尾做一次辛投影兜底。

Public API：

- :class:`QuasiFloquetResult` —— 结果句柄；
- :class:`QuasiFloquetReducer` —— 上下文绑定的 reducer，通过
  :meth:`reduce` 给出 :class:`QuasiFloquetResult`。

单位约定：``M(t)``、``D``、``B(t)`` 全部在 qiao 归一化单位（TU）下
运算；与 SI 之间的换算只能经由 :class:`NormalFormContext` 与
:mod:`.units` 模块。本模块不暴露 qiao 的 ``QFtrans_mat`` 原始结构。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .dynamical_substitution import DynamicalSubstituteResult

if TYPE_CHECKING:
    from .context import NormalFormContext

# ---------------------------------------------------------------------------
# 辛结构与实标准形
# ---------------------------------------------------------------------------

#: 6 维辛矩阵 ``J = [[0, I₃], [-I₃, 0]]`` （位置在前、动量在后）。
J6: npt.NDArray[np.floating] = np.block(
    [[np.zeros((3, 3)), np.eye(3)], [-np.eye(3), np.zeros((3, 3))]]
)


def real_normal_form_matrix(lam: float, wp: float, wv: float) -> npt.NDArray[np.floating]:
    """构造 6×6 实标准形矩阵 ``D``。

    对应 qiao ``Global_File`` 中固化的 ``Mat_D``：一双曲方向 ``±λ``、
    平面内中心方向 ``±i ω_p``、垂直中心方向 ``±i ω_v``。基底顺序与
    qiao 一致：``[p₁, q₁, q₂, p₂, ...]``——指数 0 双曲、指数 1/4 平面
    中心对、指数 2/5 垂直中心对。

    Args:
        lam: 双曲特征指数 λ（>0）。
        wp: 平面内中心频率 ω_p（>0）。
        wv: 垂直中心频率 ω_v（>0）。

    Returns:
        ``(6, 6)`` 实标准形矩阵 ``D``。
    """
    D = np.zeros((6, 6), dtype=float)
    D[0, 0] = lam
    D[3, 3] = -lam
    D[1, 4] = wp
    D[4, 1] = -wp
    D[2, 5] = wv
    D[5, 2] = -wv
    return D


def real_normal_form_transform(
    M: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """CR3BP 常数 QF 变换矩阵 ``V``：``X = V·Y`` 把常数 ``M`` 化为实标准形。

    ``M`` 必须是**常数 Hamilton 矩阵**（``MᵀJ + JM = 0``，如
    :func:`_cr3bp_hamiltonian_linearization` 的输出）——Hamilton 矩阵的
    特征向量满足 J 正交性，实标准形基 ``V`` 才能同时辛归一化与对角化
    （速度框架 ``[[0,I],[S,−2Ω×]]`` 不是 Hamilton 矩阵，其基无法辛
    归一化，``symplectic_project`` 会破坏对角化）。特征值为一对实
    ``±λ`` 与两对纯虚 ``±i·ω_p``、``±i·ω_v``。``V`` 的列按
    :func:`real_normal_form_matrix` 的基底顺序取实标准形基：

    .. math::

        V = [v_λ,\\ \\mathrm{Re}(v_{ω_p}),\\ \\mathrm{Re}(v_{ω_v}),
             v_{-λ},\\ \\mathrm{Im}(v_{ω_p}),\\ \\mathrm{Im}(v_{ω_v})]

    并做辛归一化（``v_iᵀJv_{i+3} = 1`` 的列缩放 + 符号）与
    :func:`symplectic_project` 精修，使 ``VᵀJV = J``、``V⁻¹MV = D``。
    注：``ω_p`` 与 ``ω_v`` 按频率升序无法区分面内/面外（L1/L2 面内
    频率更大），本函数按**特征向量的 z 分量占优**区分：z 占优为
    ``ω_v``（垂直）、x/y 占优为 ``ω_p``（平面）。

    Args:
        M: ``(6, 6)`` 常数 Hamilton 线性化矩阵。

    Returns:
        ``(V, D)``：实标准形变换矩阵与实标准形（后者即
        :func:`real_normal_form_matrix` 输出，从 ``V`` 还原）。
    """
    M = np.asarray(M, dtype=float)
    eigvals, eigvecs = np.linalg.eig(M)
    tol = 1e-8 * max(1.0, float(np.max(np.abs(eigvals))))

    v_pos: npt.NDArray[np.complexfloating] | None = None  # +λ
    v_neg: npt.NDArray[np.complexfloating] | None = None  # -λ
    v_imag: list[tuple[float, npt.NDArray[np.complexfloating]]] = []  # +iω 族
    for ev, vec in zip(eigvals, eigvecs.T, strict=True):
        if abs(ev.imag) <= tol:  # 实特征值 ±λ
            if ev.real > 0 and v_pos is None:
                v_pos = vec
            elif ev.real < 0 and v_neg is None:
                v_neg = vec
        elif abs(ev.real) <= tol and ev.imag > 0:  # 正虚特征值 +iω
            v_imag.append((float(ev.imag), vec))
    if v_pos is None or v_neg is None or len(v_imag) < 2:
        raise ValueError(
            f"M 的特征值结构不符合共线平动点（需要 ±λ、±iω_p、±iω_v），"
            f"实际谱：{sorted(round(abs(ev), 6) for ev in eigvals)}"
        )

    # 面内/面外按特征向量方向区分：z 占优 = 垂直（ω_v），x/y 占优 = 平面（ω_p）
    def _z_dominance(v: npt.NDArray[np.complexfloating]) -> float:
        return float(np.abs(v[2]) / (np.abs(v).max() + 1e-300))

    if _z_dominance(v_imag[0][1]) > _z_dominance(v_imag[1][1]):
        v_wp, v_wv = v_imag[1][1], v_imag[0][1]
    else:
        v_wp, v_wv = v_imag[0][1], v_imag[1][1]

    # 列顺序：[q1, q2, q3, p1, p2, p3] = [v_λ, Re(v_ωp), Re(v_ωv), v_-λ, Im(v_ωp), Im(v_ωv)]
    V = np.zeros((6, 6), dtype=float)
    V[:, 0] = np.real(v_pos)
    V[:, 3] = np.real(v_neg)
    V[:, 1] = np.real(v_wp)
    V[:, 4] = np.imag(v_wp)
    V[:, 2] = np.real(v_wv)
    V[:, 5] = np.imag(v_wv)

    # 辛归一化：每对 (q_i, p_i) 的 J 内积 s_i = v_qiᵀ J v_pi 应非零；
    # 缩放 v_qi·a、v_pi·a（a = 1/√|s_i|），符号取正。
    for i in range(3):
        s = float(V[:, i] @ J6 @ V[:, i + 3])
        if abs(s) < 1e-14:
            raise ValueError(f"V 的第 {i} 对列 J 内积为零，无法辛归一化")
        a = 1.0 / np.sqrt(abs(s))
        V[:, i] *= a
        V[:, i + 3] *= a
        if s < 0:
            V[:, i + 3] *= -1.0

    V = symplectic_project(V)
    D = np.linalg.solve(V, M @ V)  # V⁻¹MV
    return V, D


def _cr3bp_hamiltonian_linearization(
    r_lp: npt.NDArray[np.floating],
    mu: float,
) -> npt.NDArray[np.floating]:
    """CR3BP 共线平动点处的 Hamiltonian 框架线性化矩阵（Hamilton 矩阵）。

    动量坐标 ``(q, p)`` 下 ``H₂ = ½‖p‖² + pᵀ·C·q − ½qᵀ·S_grav·q``
    （``C = Ω×`` 为旋转耦合、``S_grav`` 为纯引力势 Hessian，无离心），
    正则方程线性化：

    .. math::

        M_H = \\begin{pmatrix} C & I \\\\ S_{grav} & C \\end{pmatrix}

    满足 ``M_HᵀJ + J·M_H = 0``（Hamilton 矩阵），其特征向量具有
    J 正交性，可构造同时辛归一化与对角化的实标准形变换
    （:func:`real_normal_form_transform`）。速度框架
    ``[[0, I], [S_eff, −2Ω×]]`` 不是 Hamilton 矩阵，不可用于此。

    Args:
        r_lp: 平动点位置（地心会合系，如 ``(1+γ, 0, 0)``）。
        mu: 系统质量比。

    Returns:
        ``(6, 6)`` Hamilton 线性化矩阵 ``M_H``。
    """
    C = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # Ω×
    S = _cr3bp_hessian_symmetric(np.asarray(r_lp, dtype=float).ravel(), mu)
    S_grav = S - np.diag([1.0, 1.0, 0.0])  # 去掉有效势离心（动量框架不含离心）
    M = np.zeros((6, 6), dtype=float)
    M[:3, :3] = C
    M[:3, 3:] = np.eye(3)
    M[3:, :3] = S_grav
    M[3:, 3:] = C
    return M


# ---------------------------------------------------------------------------
# sp(6) 李代数工具（迁移自 qiao Calc_Phi_Lie_algebra.py）
# ---------------------------------------------------------------------------


def build_sp6_basis() -> list[npt.NDArray[np.floating]]:
    """构造 sp(6, R) 的 21 维正交基 ``{E_k}`` （Frobenius 归一）。

    三块结构：``A`` 块 ``n²=9``（``E_ij = e_i e_jᵀ − e_{n+j} e_{n+i}ᵀ``）、
    ``B`` 块 ``n(n+1)/2=6``（右上对称）、``C`` 块 ``n(n+1)/2=6``
    （左下对称）。对应 qiao ``_build_sp6_basis``。
    """
    n = 3
    E_list: list[npt.NDArray[np.floating]] = []
    for i in range(n):
        for j in range(n):
            tmp = np.zeros((2 * n, 2 * n))
            tmp[i, j] = 1.0
            tmp[n + j, n + i] = -1.0
            E_list.append(tmp / np.linalg.norm(tmp, "fro"))
    for i in range(n):
        for j in range(i, n):
            tmp = np.zeros((2 * n, 2 * n))
            tmp[i, n + j] = 1.0
            if i != j:
                tmp[j, n + i] = 1.0
            E_list.append(tmp / np.linalg.norm(tmp, "fro"))
    for i in range(n):
        for j in range(i, n):
            tmp = np.zeros((2 * n, 2 * n))
            tmp[n + i, j] = 1.0
            if i != j:
                tmp[n + j, i] = 1.0
            E_list.append(tmp / np.linalg.norm(tmp, "fro"))
    return E_list


def sp6_to_vector(
    mat: npt.NDArray[np.floating], basis: list[npt.NDArray[np.floating]]
) -> npt.NDArray[np.floating]:
    """sp(6) 矩阵 → 21 维系数向量（Frobenius 内积投影）。"""
    return np.array([float(np.sum(mat * E_k)) for E_k in basis], dtype=float)


def vector_to_sp6(
    xi: npt.ArrayLike, basis: list[npt.NDArray[np.floating]]
) -> npt.NDArray[np.floating]:
    """21 维系数向量 → sp(6) 矩阵 ``M = Σ ξ_k E_k``。"""
    xi_arr = np.asarray(xi, dtype=float).ravel()
    out = np.zeros((6, 6), dtype=float)
    for k, E_k in enumerate(basis):
        out += xi_arr[k] * E_k
    return out


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuasiFloquetResult:
    """quasi-Floquet 变换结果句柄。

    Attributes:
        context: 关联 :class:`NormalFormContext`。
        order: 展开阶数（与 ``context.order`` 一致）。
        tlist: 采样时间数组 ``(n,)``，归一化 TU。
        B_samples: ``B(t)`` 在采样点的堆叠，形状 ``(n, 6, 6)``。
        D: 6×6 实标准形矩阵。
        method: 实际使用的求解方法（``"matrix"`` 或 ``"lie_algebra"``）。
        M_samples: 时变线性化矩阵 ``M(t)`` 在采样点的堆叠，形状
            ``(n, 6, 6)``；诊断用，``None`` 表示未保留。
        metadata: 自由扩展字段。
    """

    context: NormalFormContext
    order: int
    tlist: npt.NDArray[np.floating]
    B_samples: npt.NDArray[np.floating]
    D: npt.NDArray[np.floating]
    method: str
    M_samples: npt.NDArray[np.floating] | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def max_symplectic_error(self) -> float:
        """所有采样点上 ``‖BᵀJB − J‖∞`` 的最大值。"""
        if self.B_samples.shape[0] == 0:
            return float("nan")
        err = 0.0
        for B in self.B_samples:
            e = float(np.max(np.abs(B.T @ J6 @ B - J6)))
            if e > err:
                err = e
        return err

    def B(self, t: float) -> npt.NDArray[np.floating]:
        """在时刻 ``t`` 处线性插值 ``B(t)``。

        Args:
            t: 归一化时间 TU。允许 ``tlist`` 之外的端点外推。

        Returns:
            ``(6, 6)`` 插值矩阵。
        """
        t_arr = np.asarray(self.tlist, dtype=float).ravel()
        if t_arr.size == 0:
            raise ValueError("tlist 为空，无法插值")
        if t_arr.size == 1:
            return np.array(self.B_samples[0], dtype=float)
        B_flat = np.array(self.B_samples, dtype=float).reshape(t_arr.size, 36)
        out = np.empty(36, dtype=float)
        for k in range(36):
            out[k] = float(np.interp(t, t_arr, B_flat[:, k]))
        return out.reshape(6, 6)


# ---------------------------------------------------------------------------
# 时变线性化矩阵 M(t)
# ---------------------------------------------------------------------------


def _cr3bp_hessian_symmetric(r: npt.NDArray[np.floating], mu: float) -> npt.NDArray[np.floating]:
    """地心会合系下有效势 ``Ω`` 的对称 Hessian（无量纲）。

    坐标系与 qiao ``Dynfunc_rho`` 一致：地心会合系（地球在原点、月球在
    ``(1,0,0)``）。``r`` 为该系下的位置（如 ``ρ + 平动点位置``）。

    有效势 ``Ω = ½(x²+y²) + (1−μ)/r₁ + μ/r₂``（离心绕地心，仅 x-y 平面），
    其 Hessian 对角为 ``(1+2c₂, 1−c₂, −c₂)``，其中
    ``c₂ = (1−μ)/r₁³ + μ/r₂³``，``r₁=|r|``（到地球）、``r₂=|r−(1,0,0)|``（到月球）。
    """
    mu1 = 1.0 - mu
    r1 = r  # 到地球（地心会合系地球在原点）
    r2 = r - np.array([1.0, 0.0, 0.0])  # 到月球（月球在 (1,0,0)）
    d1 = float(np.linalg.norm(r1))
    d2 = float(np.linalg.norm(r2))
    d1i3 = 1.0 / d1**3
    d2i3 = 1.0 / d2**3
    d1i5 = 1.0 / d1**5
    d2i5 = 1.0 / d2**5

    # 离心仅 x-y 平面（z 方向无离心）+ 引力 Hessian。
    # 此前用质心系（地球 −μ）并把离心误加到 z，导致与 constants 频率不自洽。
    c2 = mu1 * d1i3 + mu * d2i3
    S = np.diag([1.0 - c2, 1.0 - c2, -c2]).astype(float)
    S += mu1 * 3.0 * d1i5 * np.outer(r1, r1)
    S += mu * 3.0 * d2i5 * np.outer(r2, r2)
    return S


def _build_M_at(
    ds_result: DynamicalSubstituteResult,
) -> tuple[
    Callable[[float], npt.NDArray[np.floating]],
    npt.NDArray[np.floating],
]:
    """从替代轨道构造 ``M(t)`` 的**连续**求值器与采样点栈。

    ``M(t)`` 是**动量框架**（Hamiltonian 正则坐标 ``(q, p)``）的线性化
    （:func:`_cr3bp_hamiltonian_linearization` 同款，Hamilton 矩阵，
    ``MᵀJ + JM = 0``）：

    ``M = [[C, I₃], [S_grav, C]]``，``C = Ω×``（旋转耦合）、
    ``S_grav`` 为纯引力势 Hessian（无离心）。

    选动量框架而非速度框架 ``[[0, I], [S_eff, −2Ω×]]`` 的原因：
    (1) QF 变换 ``B`` 的消费方是 Hamiltonian（:func:`build_cr3bp_hamiltonian`
    的 ``(q,p)`` 坐标），``B`` 必须与 H 同框架；(2) 动量框架是 Hamilton
    矩阵，特征向量具有 J 正交性，实标准形变换 ``V`` 可同时辛归一化与
    对角化，且 ``BᵀJB`` 是 ``Ḃ`` 方程的首次积分（速度框架两者皆不成立）。

    关键：返回的求值器 ``M_at(t)`` 在任意 ``t`` 上**重新解析地**计算
    ``S_grav``（先对轨道位置线性插值，再算对称 Hessian），因此在 ODE 的
    每个自适应步上 ``M(t)`` 都是精确的 Hamilton 矩阵。这避免了「预计算
    ``M`` 再线性插值」会让 ``MᵀJ+JM=0`` 在节点之间失效、从而破坏辛
    守恒的问题。

    返回 ``(M_at, M_stack)``：``M_at`` 为连续求值器，``M_stack`` 为
    在 ``ds_result.tlist`` 采样点上的 ``(n,6,6)`` 数组（诊断用）。
    """
    Xlist = np.asarray(ds_result.Xlist, dtype=float)
    if Xlist.ndim != 2 or Xlist.shape[1] != 6:
        raise ValueError(f"Xlist 必须是 (n,6)，得到 {Xlist.shape}")
    n = Xlist.shape[0]
    mu = float(ds_result.context.mu)
    r_lp = np.asarray(ds_result.context.libration_position, dtype=float).ravel()
    t_arr = np.asarray(ds_result.tlist, dtype=float).ravel()

    # 旋转耦合张量 C = Ω×（[[0,-1,0],[1,0,0],[0,0,0]]）
    omega_x = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    def assemble(rho: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        S_eff = _cr3bp_hessian_symmetric(rho + r_lp, mu)
        S_grav = S_eff - np.diag([1.0, 1.0, 0.0])  # 去掉有效势离心（动量框架不含）
        M = np.zeros((6, 6), dtype=float)
        M[:3, :3] = omega_x  # C
        M[:3, 3:] = np.eye(3)
        M[3:, :3] = S_grav
        M[3:, 3:] = omega_x  # C
        return M

    def M_at(t: float) -> npt.NDArray[np.floating]:
        rho = np.array(
            [float(np.interp(t, t_arr, Xlist[:, k])) for k in range(3)],
            dtype=float,
        )
        return assemble(rho)

    M_stack = np.zeros((n, 6, 6), dtype=float)
    for i in range(n):
        M_stack[i] = assemble(Xlist[i, :3])

    return M_at, M_stack


# ---------------------------------------------------------------------------
# 辛投影
# ---------------------------------------------------------------------------


def symplectic_project(
    B: npt.NDArray[np.floating],
    *,
    tol: float = 1e-14,
    max_iter: int = 50,
) -> npt.NDArray[np.floating]:
    """把 ``B`` 投影到最近的辛矩阵（Newton 迭代，qiao 风格）。

    迭代 ``B ← B − ½ B (Jᵀ S)``，其中 ``S = BᵀJB − J``。
    """
    B = np.array(B, dtype=float)
    for _ in range(max_iter):
        S = B.T @ J6 @ B - J6
        if float(np.linalg.norm(S, "fro")) < tol:
            break
        B = B - 0.5 * B @ (J6.T @ S)
    return B


# ---------------------------------------------------------------------------
# 矩阵法：直接积分 Ḃ = M·B − B·D
# ---------------------------------------------------------------------------


def _qf_rhs_factory(
    M_at: Callable[[float], npt.NDArray[np.floating]],
    D: npt.NDArray[np.floating],
) -> Callable[[float, npt.ArrayLike], npt.NDArray[np.floating]]:
    """构造 36 维 QF 右端项 ``Ḃ = M(t)·B − B·D``。

    ``M_at`` 在 ODE 自适应步的任意时刻重新计算 Hamilton 矩阵，从而
    ``MᵀJ+JM=0`` 在整个积分区间精确成立。
    """

    def rhs(t: float, X: npt.ArrayLike) -> npt.NDArray[np.floating]:
        B = np.asarray(X, dtype=float).reshape(6, 6)
        M = M_at(t)
        dB = M @ B - B @ D
        return dB.ravel()

    return rhs


def _solve_qf_matrix(
    M_at: Callable[[float], npt.NDArray[np.floating]],
    D: npt.NDArray[np.floating],
    tlist: npt.NDArray[np.floating],
    *,
    rtol: float = 1e-11,
    atol: float = 1e-13,
    segment: float | None = None,
) -> npt.NDArray[np.floating]:
    """矩阵法：从 ``B(t_0)=I`` 出发 DOP853 积分 ``Ḃ = M·B − B·D``。

    qiao ``Code08`` 在 36 维空间做多点打靶 + Newton；对 smoke 级小窗口，
    单次初值积分 + 末尾辛投影即可：``M``、``D`` 都是 Hamilton 矩阵，
    ``BᵀJB=J`` 是精确首次积分，前向积分的辛误差仅由 ODE 容差累积，
    ``symplectic_project`` 把残余误差拉回 ``<1e-13``。

    Args:
        segment: 分段辛重投影的段长（归一化 TU）。``None`` 时单次积分到底
            （仅适合 ``λT < 10`` 的小窗口）。非 ``None`` 时把积分区间按
            ``segment`` 分短段，每段末用 :func:`symplectic_project` 把 ``B``
            拉回辛群——抑制双曲方向 ``e^(λt)`` 增长导致的辛误差累积。
            段长取 ``0.4``–``0.8``（与 qiao ``node_step`` 一致，使单段
            ``e^(λ·segment)`` 有限）。这避免长窗口的 overflow，但 ``B`` 的
            双曲分量仍按 ``e^(λt)`` 物理增长（中心流形约化本就如此）。

            .. note::
               分段 + 投影是 ``qiao`` 完整多点打靶（块三对角 Newton）的最小
               替代。对 ``λT ≳ 40``（如 L2 的 30 天窗口）仍可能精度不足，
               需完整多点打靶（见 issue #328）。
    """
    from ._solve_ivp_rust import solve_ivp_rust

    rhs = _qf_rhs_factory(M_at, D)
    B0 = np.eye(6, dtype=float).ravel()

    t_arr = np.asarray(tlist, dtype=float).ravel()

    if segment is None or segment <= 0:
        sol = solve_ivp_rust(
            fun=rhs,
            t_span=(float(t_arr[0]), float(t_arr[-1])),
            y0=B0,
            t_eval=t_arr,
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"QF 矩阵法积分失败：{sol.message}")
        return sol.y.T.reshape(-1, 6, 6)

    # 分段积分 + 每段末辛投影。段边界取 segment 的整数倍，落在 [t0, tf] 内。
    return _solve_qf_segmented(rhs, B0, t_arr, float(segment), rtol, atol)


def _solve_qf_segmented(
    rhs: Callable[[float, npt.ArrayLike], npt.NDArray[np.floating]],
    B0: npt.NDArray[np.floating],
    t_arr: npt.NDArray[np.floating],
    segment: float,
    rtol: float,
    atol: float,
) -> npt.NDArray[np.floating]:
    """分段积分 ``Ḃ = M·B − B·D``，每段末辛投影。

    段边界为 ``segment`` 的整数倍（夹在 ``[t_arr[0], t_arr[-1]]`` 之间）。
    每段独立 DOP853 积分，段末用 :func:`symplectic_project` 把 ``B`` 拉回
    辛群，抑制 ``e^(λt)`` 增长导致的辛误差累积。段内对 ``t_arr`` 的采样点
    用段积分的 ``t_eval`` 直接输出（Rust ``solve_ivp_py`` 无稠密输出插值器，
    故把采样点并入段 ``t_eval`` 网格，而非事后插值）。
    """
    from ._solve_ivp_rust import solve_ivp_rust

    t0 = float(t_arr[0])
    tf = float(t_arr[-1])
    seg_ends = np.arange(t0 + segment, tf + 0.5 * segment, segment)
    seg_ends = seg_ends[seg_ends < tf - 1e-12]  # 不含末点（末点单独处理）

    N = t_arr.size
    B_out = np.zeros((N, 6, 6), dtype=float)
    B_out[0] = B0.reshape(6, 6)

    cur_B = B0.copy()
    cur_t = t0

    next_i = 1  # t_arr[0] 已填
    for t_end in seg_ends:
        t_hi = float(t_end)
        # 段内采样点：包含段首 cur_t + t_arr 中落在 (cur_t, t_hi] 的点
        seg_mask = (t_arr > cur_t - 1e-12) & (t_arr <= t_hi + 1e-12)
        seg_t_eval = np.concatenate([[cur_t], t_arr[seg_mask]])
        sol = solve_ivp_rust(
            fun=rhs,
            t_span=(cur_t, t_hi),
            y0=cur_B,
            t_eval=seg_t_eval,
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"QF 分段积分失败（段 [{cur_t}, {t_end}]）：{sol.message}")
        seg_y = np.asarray(sol.y)
        # seg_t_eval[0] = cur_t（段首，已由上段末填，跳过）；
        # seg_t_eval[1:] 对应 t_arr 中 seg_mask 为 True 的点，按顺序写入 B_out
        sample_indices = np.where(seg_mask)[0]
        for m, idx in enumerate(sample_indices):
            B_out[idx] = seg_y[:, m + 1].reshape(6, 6)
        next_i = int(sample_indices[-1] + 1) if sample_indices.size > 0 else next_i
        # 段末投影，作为下段初值
        cur_B = symplectic_project(sol.y[:, -1].reshape(6, 6)).ravel()
        cur_t = t_hi

    # 末段：积分到 tf
    if cur_t < tf - 1e-12:
        sol = solve_ivp_rust(
            fun=rhs,
            t_span=(cur_t, tf),
            y0=cur_B,
            t_eval=t_arr[next_i:],
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"QF 末段积分失败：{sol.message}")
        for k, _t in enumerate(sol.t):
            B_out[next_i + k] = sol.y[:, k].reshape(6, 6)

    return B_out


# ---------------------------------------------------------------------------
# 多点打靶法：短弧 STM + 块三对角求解（qiao Code08）
# ---------------------------------------------------------------------------


def _solve_qf_multipoint(
    M_at: Callable[[float], npt.NDArray[np.floating]],
    D: npt.NDArray[np.floating],
    tlist: npt.NDArray[np.floating],
    *,
    node_step: float = 0.8,
    rtol: float = 1e-11,
    atol: float = 1e-13,
) -> npt.NDArray[np.floating]:
    """多点打靶法求 quasi-Floquet 变换 ``B(t)``（qiao Code08 路径）。

    把区间 ``[t0, tf]`` 按 ``node_step`` 分短段，每段用 36 维向量化 STM
    ``Φ_i = expm((I⊗M − D^T⊗I)·node_step)``（短弧 ``e^(λ·node_step)`` 有限，
    不 overflow），再解节点连续性方程 ``Φ_i·B_i = B_{i+1}``（边界 ``B_N=I``）
    得各节点 ``B_i``，节点间用段 STM 单步稠密化。

    与单次积分 (:func:`_solve_qf_matrix`) 的根本区别：长窗口下 ``B(t)`` 的
    双曲分量 ``e^(λt)`` 物理增长不可避免，但单次积分在 ``e^(λt)`` 大时浮点
    精度累积丢失（辛性破坏）；多点打靶把长积分换成短弧 STM 的代数连锁，
    每段 STM 精度保持，连锁求解是纯线性代数——辛误差不随窗口增长。

    对 CR3BP（``M`` 常数），所有段 STM 相同，连续性方程有显式解
    ``B_i = Φ^{-(N-i)}·B_N``（反向递推）；对时变 ``M``，逐段算 STM 后用
    块三对角 Thomas 算法求解。

    Args:
        node_step: 节点间距（TU），默认 ``0.8``（与 qiao ``Code08`` 一致，
            使 ``e^(λ·node_step)`` 有限）。L2 ``λ≈2.16`` 时 ``0.8`` 给
            ``e^1.73≈5.6``。
        rtol/atol: 段 STM 数值积分容差（时变 ``M`` 用；常数 ``M`` 用
            ``expm`` 解析）。

    Returns:
        ``(N, 6, 6)`` 采样点 ``B(t)``。``B(tlist[-1]) = I``（末节点边界）。
    """
    from scipy.linalg import expm

    t_arr = np.asarray(tlist, dtype=float).ravel()
    t0, tf = float(t_arr[0]), float(t_arr[-1])

    # 节点时刻（含端点）
    t_nodes = np.arange(t0, tf + 0.5 * node_step, node_step)
    if t_nodes[-1] > tf - 1e-12:
        t_nodes[-1] = tf
    elif abs(t_nodes[-1] - tf) > 1e-12:
        t_nodes = np.append(t_nodes, tf)
    n_nodes = t_nodes.size

    # 检测 M 是否常数（解析 CR3BP 线性化）：多点采样比较。
    # 注意 DS 替代轨道让 M(t)=J·S(ρ(t)) 随轨道位置变化，即使底层是 CR3BP，
    # 此时 M 非常数，须走时变分支（块三对角 Thomas）。
    M0 = np.asarray(M_at(t0), dtype=float)
    M_const = all(
        np.allclose(M0, np.asarray(M_at(tt), dtype=float), atol=1e-12)
        for tt in [t0 + 0.1 * (tf - t0), 0.5 * (t0 + tf), t0 + 0.9 * (tf - t0)]
    )

    I6 = np.eye(6)

    if M_const:
        # CR3BP：段 STM 解析，反向递推求节点 B。
        # 用 expm(-A36·dt) 作段逆（比 inv(expm(A36·dt)) 精确，避免大条件数求逆）。
        # A36 是 vec(Ḃ)=vec(MB−BD) 的矩阵形式；numpy ravel 行优先，故
        # vec(MB)=(M⊗I)vec(B)、vec(BD)=(I⊗D^T)vec(B)，A36 = M⊗I − I⊗D^T。
        A36 = np.kron(M0, I6) - np.kron(I6, D.T)
        Phi_inv_equal = expm(-A36 * node_step)

        B_nodes: list[npt.NDArray[np.floating]] = [np.zeros((6, 6)) for _ in range(n_nodes)]
        B_nodes[-1] = I6.copy()  # B_N = I
        for i in range(n_nodes - 2, -1, -1):
            dt = t_nodes[i + 1] - t_nodes[i]
            phi_inv_i = Phi_inv_equal if abs(dt - node_step) < 1e-12 else expm(-A36 * dt)
            B_nodes[i] = (phi_inv_i @ B_nodes[i + 1].ravel()).reshape(6, 6)
    else:
        # 时变 M：逐段积分 STM，块三对角 Thomas 求解
        B_nodes = _multipoint_thomas(M_at, D, t_nodes, rtol, atol)

    # 节点间稠密化：对 t_arr 每个点，找到所在段 [t_nodes[k], t_nodes[k+1]]，
    # 用段 STM 单步传播 B_nodes[k] 到该点。
    return _densify_b_multipoint(M_at, D, M_const, M0, t_arr, t_nodes, B_nodes, rtol, atol)


def _multipoint_thomas(
    M_at: Callable[[float], npt.NDArray[np.floating]],
    D: npt.NDArray[np.floating],
    t_nodes: npt.NDArray[np.floating],
    rtol: float,
    atol: float,
) -> list[npt.NDArray[np.floating]]:
    """时变 M 的块三对角多点打靶（qiao Code08:251-329 路径）。

    高斯-牛顿迭代解连续性 ``Φ_i·B_i − B_{i+1} = 0``（边界 ``B_N=I``），
    正则化 ``D_i = I + Φ_i·Φ_i^T`` 保证可逆（Levenberg-Marquardt 式）。
    """
    from ._solve_ivp_rust import solve_ivp_rust

    I6 = np.eye(6)
    n_nodes = t_nodes.size
    I36 = np.eye(36)

    # 逐段算 36×36 STM Φ_i：积分 vec(Φ) 的 1296 维流，Φ̇ = A36(t)·Φ，Φ(t0)=I36。
    # 对应 qiao Calc_Phi_QF 的 Dynfunc_Phi_QF（reshape X 为 36×36 再乘 A）。
    Phi_list: list[npt.NDArray[np.floating]] = []
    for k in range(n_nodes - 1):

        def rhs_phi(t: float, X1296: npt.ArrayLike) -> npt.NDArray[np.floating]:
            Phi = np.asarray(X1296, dtype=float).reshape(36, 36)
            # A36(t) = M(t)⊗I − I⊗D^T（行优先 vec(B)）；Ḃ 各列 = A36·Φ 各列
            Mt = M_at(t)
            A36_t = np.kron(Mt, I6) - np.kron(I6, D.T)
            return (A36_t @ Phi).ravel()

        sol = solve_ivp_rust(
            fun=rhs_phi,
            t_span=(float(t_nodes[k]), float(t_nodes[k + 1])),
            y0=I36.ravel(),
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"多点打靶段 STM 积分失败（段 {k}）：{sol.message}")
        Phi_list.append(sol.y[:, -1].reshape(36, 36))

    # 节点 B 初猜（随机小矩阵，末节点 I）
    rng = np.random.default_rng(0)
    B: list[npt.NDArray[np.floating]] = [rng.standard_normal((6, 6)) * 0.1 for _ in range(n_nodes)]
    B[-1] = np.eye(6)

    # 高斯-牛顿迭代
    for _ in range(30):
        # 前向消元
        D_blk: list = [None] * (n_nodes - 1)
        L_blk: list = [None] * (n_nodes - 1)
        X_res: list = [None] * (n_nodes - 1)
        max_err = 0.0
        for i in range(n_nodes - 1):
            Phi = Phi_list[i]
            Ai = Phi
            Xf = Phi @ B[i].ravel()
            X_res[i] = Xf - B[i + 1].ravel()
            max_err = max(max_err, np.max(np.abs(X_res[i])))
            if i == 0:
                D_blk[i] = I36 + Ai @ Ai.T
            else:
                L_blk[i] = -Ai @ np.linalg.inv(D_blk[i - 1])
                D_blk[i] = I36 + Ai @ Ai.T - L_blk[i] @ D_blk[i - 1] @ L_blk[i].T
                X_res[i] = X_res[i] - L_blk[i] @ X_res[i - 1]
        if max_err < 1e-13:
            break
        # 后向回代
        Y: list = [None] * (n_nodes - 1)
        for i in range(n_nodes - 2, -1, -1):
            sol_i = np.linalg.solve(D_blk[i], X_res[i])
            if i < n_nodes - 2:
                sol_i = sol_i - L_blk[i + 1].T @ Y[i + 1]
            Y[i] = sol_i
        # 修正
        for i in range(n_nodes):
            if i == 0:
                dB = -Phi_list[0].T @ Y[0]
            elif i == n_nodes - 1:
                dB = Y[i - 1]
            else:
                dB = Y[i - 1] - Phi_list[i].T @ Y[i]
            B[i] = B[i] + dB.reshape(6, 6)
        B[-1] = np.eye(6)  # 固定末节点
    return B


def _densify_b_multipoint(
    M_at: Callable[[float], npt.NDArray[np.floating]],
    D: npt.NDArray[np.floating],
    M_const: bool,
    M0: npt.NDArray[np.floating],
    t_arr: npt.NDArray[np.floating],
    t_nodes: npt.NDArray[np.floating],
    B_nodes: list[npt.NDArray[np.floating]],
    rtol: float,
    atol: float,
) -> npt.NDArray[np.floating]:
    """节点间稠密化：对每个采样点，用段 STM 单步传播节点 ``B`` 到该点。"""
    from scipy.linalg import expm

    from ._solve_ivp_rust import solve_ivp_rust

    I6 = np.eye(6)
    n = t_arr.size
    n_nodes = t_nodes.size
    B_out = np.zeros((n, 6, 6), dtype=float)
    rhs36 = _qf_rhs_factory(M_at, D)

    # 预计算常数 M 的 36×36 矩阵（CR3BP 加速）。行优先：A36 = M⊗I − I⊗D^T。
    A36_const = np.kron(M0, I6) - np.kron(I6, D.T) if M_const else None

    for i, t in enumerate(t_arr):
        # 找 t 所在段 [t_nodes[k], t_nodes[k+1]]
        if t <= t_nodes[0]:
            B_out[i] = B_nodes[0]
            continue
        if t >= t_nodes[-1]:
            B_out[i] = B_nodes[-1]
            continue
        k = int(np.searchsorted(t_nodes, t)) - 1
        k = max(0, min(k, n_nodes - 2))
        t_lo = float(t_nodes[k])
        dt = float(t) - t_lo
        if dt <= 0:
            B_out[i] = B_nodes[k]
            continue

        if M_const:
            # 解析段 STM
            assert A36_const is not None  # M_const 分支必然已初始化
            phi_t = expm(A36_const * dt)
            B_out[i] = (phi_t @ B_nodes[k].ravel()).reshape(6, 6)
        else:
            # 时变 M：从节点 B_nodes[k] 积分 36 维 vec(B) 到 t。
            # rhs36(t, B_6x6) 返回 Ḃ 的 6×6，这里对 vec(B)（36 维）操作。
            def rhs_vec(tt: float, X36: npt.ArrayLike) -> npt.NDArray[np.floating]:
                return rhs36(tt, np.asarray(X36).ravel().reshape(6, 6)).ravel()

            sol = solve_ivp_rust(
                fun=rhs_vec,
                t_span=(t_lo, float(t)),
                y0=B_nodes[k].ravel(),
                rtol=rtol,
                atol=atol,
            )
            B_out[i] = sol.y[:, -1].reshape(6, 6)
    return B_out


# ---------------------------------------------------------------------------
# 李代数法：B = B0·exp(ξ)
# ---------------------------------------------------------------------------


def _solve_qf_lie(
    M_at: Callable[[float], npt.NDArray[np.floating]],
    D: npt.NDArray[np.floating],
    tlist: npt.NDArray[np.floating],
    *,
    rtol: float = 1e-11,
    atol: float = 1e-13,
    n_substeps: int = 40,
) -> npt.NDArray[np.floating]:
    """李代数法：在辛群 ``Sp(6)`` 上推进 ``B(t)``，自动保辛。

    参数化 ``B = exp(ξ)``、``ξ ∈ sp(6, R)``。要解 ``Ḃ = M B − B D``，
    不能简单令 ``ξ̇ = B⁻¹ Ḃ``：``d/dt exp(ξ) = exp(ξ)·ξ̇`` 仅当
    ``[ξ, ξ̇] = 0`` 时成立，一般情形需带 ``dexp`` 修正项
    ``ξ̇ = ad_ξ/(1−e^{−ad_ξ})(B⁻¹Ḃ)``。该项的 Bernoulli 级数在
    ``‖ξ‖ ≳ 2π``（双曲方向 ``e^{λT}`` 增长后）发散，矩阵函数法在
    ``1−e^{−ad_ξ}`` 奇异处失效，二者均不可靠。

    改用 **commutator-free 4 阶 Lie group 积分器**（每步
    ``B ← B·exp(h·ξ_k)``，``ξ_k`` 由 RK4 加权的体速度
    ``B⁻¹Ḃ`` 的 sp(6) 投影给出）：``B·exp(sp(6))`` 恒辛，且 RK4
    以 ``h⁴`` 收敛到 ``Ḃ = M B − B D`` 的真解。这是数值稳健、
    自动保辛（无需末尾投影）、忠实于 qiao 李代数法意图的实现。

    Args:
        n_substeps: 相邻采样点之间的均匀子步数；``h⁴`` 误差随其增大
            而下降，``40`` 对 smoke 级窗口给出与矩阵法 ``<1e-5`` 的
            一致性。非公开调参，保留在签名内仅供测试覆盖。
    """
    from scipy.linalg import expm

    basis = build_sp6_basis()
    t_arr = np.asarray(tlist, dtype=float).ravel()
    n = t_arr.size

    def body_velocity(t: float, B: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """体速度 ``B⁻¹ Ḃ`` 投影到 sp(6)（``M``、``D`` Hamilton 时无损）。"""
        Bdot = M_at(t) @ B - B @ D
        return vector_to_sp6(sp6_to_vector(np.linalg.solve(B, Bdot), basis), basis)

    def step(t: float, h: float, B: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """一步 4 阶 commutator-free Lie group RK4。"""
        k1 = body_velocity(t, B)
        k2 = body_velocity(t + h / 2, B @ expm(h / 2 * k1))
        k3 = body_velocity(t + h / 2, B @ expm(h / 2 * k2))
        k4 = body_velocity(t + h, B @ expm(h * k3))
        return B @ expm(h * (k1 + 2 * k2 + 2 * k3 + k4) / 6)

    B_samples = np.zeros((n, 6, 6), dtype=float)
    B = np.eye(6, dtype=float)
    B_samples[0] = B
    for i in range(1, n):
        h = (float(t_arr[i]) - float(t_arr[i - 1])) / n_substeps
        t = float(t_arr[i - 1])
        for _ in range(n_substeps):
            B = step(t, h, B)
            t += h
        B_samples[i] = B
    return B_samples


# ---------------------------------------------------------------------------
# Reducer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuasiFloquetReducer:
    """quasi-Floquet 变换 reducer（上下文绑定）。

    通过 :meth:`reduce` 从一条 :class:`DynamicalSubstituteResult` 提取
    时变线性化 ``M(t)``、求解 ``Ḃ = M·B − B·D``，并施加辛约束，
    输出可插值的 :class:`QuasiFloquetResult`。

    Args:
        context: 归一化上下文（提供频率 ``ω_p``/``ω_v``、特征指数 λ）。
        method: ``"matrix"`` （默认，36 维直接积分 + 辛投影）或
            ``"lie_algebra"`` （21 维 sp(6) 参数化，自动保辛）。
        project: 矩阵法是否在末尾做辛投影兜底（默认 ``True``）。
        rtol: ODE 相对容差。
        atol: ODE 绝对容差。
        segment: 矩阵法分段辛重投影的段长（TU）。``None``（默认）单次积分，
            适合 ``λT < 10`` 的小窗口；非 ``None`` 时分短段 + 每段辛投影，
            抑制双曲方向 ``e^(λt)`` 增长导致的 overflow（详见
            :func:`_solve_qf_matrix`）。
    """

    context: NormalFormContext
    method: str = "matrix"
    project: bool = True
    rtol: float = 1e-11
    atol: float = 1e-13
    segment: float | None = None

    def reduce(self, ds_result: DynamicalSubstituteResult) -> QuasiFloquetResult:
        """对 ``ds_result`` 执行 quasi-Floquet 变换。

        Args:
            ds_result: 切片 #171 的动力学替代结果，至少提供 ``tlist``、
                ``Xlist`` 与 ``context``。

        Returns:
            :class:`QuasiFloquetResult`。

        Raises:
            ValueError: ``method`` 非法，或 ``ds_result`` 数据不足。
        """
        if self.method not in ("matrix", "lie_algebra", "multipoint", "constant"):
            raise ValueError(
                f"method 必须是 'matrix' / 'lie_algebra' / 'multipoint' / 'constant'，"
                f"得到 {self.method!r}"
            )
        if ds_result.Xlist.shape[0] < 2:
            raise ValueError(f"ds_result 至少需要 2 个采样点，得到 {ds_result.Xlist.shape[0]}")

        # —— 实标准形 D ——
        # 非 constant 方法用 qiao Global_File 固化频率；constant 方法（CR3BP）
        # 在下方用 V 还原的 V⁻¹MV（M 特征值），保证 D 与 B 严格自洽。
        nu1, nu2 = self.context.central_frequencies
        lam = float(self.context.characteristic_exponent)
        D = real_normal_form_matrix(lam, float(nu1), float(nu2))

        # —— 时变线性化 M(t) ——
        M_at, M_stack = _build_M_at(ds_result)

        # —— 求解 B(t) ——
        if self.method == "constant":
            # CR3BP：M(t) 常数，QF 变换退化为常数实标准形变换 V
            # （X = V·Y 使 Ẏ = D·Y）。B 不随 e^{λt} 增长，QF 坐标下的
            # Hamiltonian 系数保持常数，同调方程退化为代数除法。
            # D 用 V 还原的 V⁻¹MV（M 特征值），与 context 固化频率的
            # 0.3% 级失谐不再进入同调方程。V 必须从 Hamiltonian 框架
            # 的线性化构造（Hamilton 矩阵，基可同时辛归一化与对角化）。
            r_lp = np.asarray(ds_result.context.libration_position, dtype=float).ravel()
            M_H = _cr3bp_hamiltonian_linearization(r_lp, float(ds_result.context.mu))
            V, D = real_normal_form_transform(M_H)
            B_samples = np.stack([V for _ in range(len(ds_result.tlist))])
        elif self.method == "matrix":
            B_samples = _solve_qf_matrix(
                M_at, D, ds_result.tlist, rtol=self.rtol, atol=self.atol, segment=self.segment
            )
            if self.project:
                B_samples = np.array([symplectic_project(B) for B in B_samples], dtype=float)
        elif self.method == "multipoint":
            # 多点打靶（qiao Code08）：segment 作 node_step，默认 0.8。
            node_step = 0.8 if self.segment is None else float(self.segment)
            B_samples = _solve_qf_multipoint(
                M_at, D, ds_result.tlist, node_step=node_step, rtol=self.rtol, atol=self.atol
            )
            if self.project:
                B_samples = np.array([symplectic_project(B) for B in B_samples], dtype=float)
        else:
            B_samples = _solve_qf_lie(M_at, D, ds_result.tlist, rtol=self.rtol, atol=self.atol)

        return QuasiFloquetResult(
            context=self.context,
            order=int(self.context.order),
            tlist=np.asarray(ds_result.tlist, dtype=float).copy(),
            B_samples=B_samples,
            D=D,
            method=self.method,
            M_samples=M_stack,
            metadata={
                "lambda": lam,
                "wp": float(nu1),
                "wv": float(nu2),
                "n_samples": int(B_samples.shape[0]),
                "project": bool(self.project) if self.method == "matrix" else False,
                "source_orbit_residual": float(ds_result.residual_norm),
            },
        )


__all__ = [
    "J6",
    "QuasiFloquetReducer",
    "QuasiFloquetResult",
    "build_sp6_basis",
    "real_normal_form_matrix",
    "sp6_to_vector",
    "symplectic_project",
    "vector_to_sp6",
]
