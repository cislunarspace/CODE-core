"""quasi-Floquet 变换矩阵 ``B(t)``（Code7–Code9）。

对应 qiao ``Code08_QuasiFloquet.py``（矩阵法）与
``Code08_QuasiFloquet_LA.py``（李代数法）：从动力学替代轨道出发，求解

.. math::

    \\dot{B}(t) = M(t)\\,B(t) - B(t)\\,D,\\qquad B^{T} J B = J,

其中 ``M(t)`` 是替代轨道邻域的时变线性化矩阵，``D`` 是实标准形矩阵
（一双曲方向 ``±λ``，两中心方向 ``±i ω_p``、``±i ω_v``），``J`` 是 6 维
辛矩阵。``B(t)`` 把受迫线性化系统化为常系数的 ``Ẏ = D·Y``。

两种实现：

- **矩阵法**（``method="matrix"``）：把 ``B`` 展平为 36 维状态直接积分
  ``Ḃ = M·B − B·D``，事后用 Newton 迭代把每个采样点的 ``B`` 投影到
  最近的辛矩阵（qiao ``Code08_QuasiFloquet``）；
- **李代数法**（``method="lie_algebra"``）：参数化 ``B = B₀·exp(ξ)``，
  ``ξ ∈ sp(6, R)``（21 维），在李代数里做修正，``B`` 自动保辛（qiao
  ``Code08_QuasiFloquet_LA``）。

辛保持性的数值保证：当 ``M(t)`` 与 ``D`` 都是 Hamilton 矩阵
（``MᵀJ + JM = 0``）时，``BᵀJB`` 是 ``Ḃ`` 方程的精确首次积分，因此
本模块刻意构造 ``M(t) = J·S(t)``（``S`` 对称）以保证辛约束在
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

#: 6 维辛矩阵 ``J = [[0, I₃], [-I₃, 0]]``（位置在前、动量在后）。
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


# ---------------------------------------------------------------------------
# sp(6) 李代数工具（迁移自 qiao Calc_Phi_Lie_algebra.py）
# ---------------------------------------------------------------------------


def build_sp6_basis() -> list[npt.NDArray[np.floating]]:
    """构造 sp(6, R) 的 21 维正交基 ``{E_k}``（Frobenius 归一）。

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
    """纯 CR3BP 会合系下二阶导 ``Ω`` 的对称 Hessian（无量纲）。

    返回对称矩阵 ``S = ∂²Ω/∂x_i ∂x_j``，对应
    ``ρ̈ = −∂Ω/∂ρ`` 在会合系下的二阶导。后续用 ``M = J·S_block`` 拼
    Hamilton 矩阵（``J`` 的位置/动量顺序与 qiao 一致）。这里只关心围绕
    替代轨道的线性化，因此位置取 ``r = ρ + 平动点位置``。
    """
    mu1 = 1.0 - mu
    r1 = r - np.array([-mu, 0.0, 0.0])  # 到地球
    r2 = r - np.array([1.0 - mu, 0.0, 0.0])  # 到月球
    d1 = float(np.linalg.norm(r1))
    d2 = float(np.linalg.norm(r2))
    d1i3 = 1.0 / d1**3
    d2i3 = 1.0 / d2**3
    d1i5 = 1.0 / d1**5
    d2i5 = 1.0 / d2**5

    # 二阶导矩阵：(3/r^5) r⊗r − (1/r^3) I，对每个主天体加权求和
    S = np.eye(3) * (1.0 - mu1 * d1i3 - mu * d2i3)
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

    会合系 CR3BP 线性化方程为
    ``δρ̈ = S(ρ)·δρ − 2 Ω×δρ̇``（``S`` 对称、``Ω=ẑ``），写成 Hamilton
    形 ``[δρ̇; δρ̈] = M·[δρ; δρ̇]`` 即
    ``M = [[−Ω×, I₃], [S, −Ω×]]``，等价于 ``M = J·[[S, Ω×], [−Ω×, 0]]``
    仍为 Hamilton 矩阵。

    关键：返回的求值器 ``M_at(t)`` 在任意 ``t`` 上**重新解析地**计算
    ``S``（先对轨道位置线性插值，再算对称 Hessian），因此在 ODE 的每个
    自适应步上 ``M(t)`` 都是精确的 Hamilton 矩阵。这避免了「预计算
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

    # 旋转系角速度张量 [[0,-1,0],[1,0,0],[0,0,0]]（对应 Ω×v = Omega_x @ v）
    omega_x = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    def assemble(rho: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        S = _cr3bp_hessian_symmetric(rho + r_lp, mu)
        M = np.zeros((6, 6), dtype=float)
        M[:3, :3] = -omega_x  # −Ω×（Coriolis 位置块）
        M[:3, 3:] = np.eye(3)
        M[3:, :3] = S
        M[3:, 3:] = -omega_x
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
) -> npt.NDArray[np.floating]:
    """矩阵法：从 ``B(t_0)=I`` 出发 DOP853 积分 ``Ḃ = M·B − B·D``。

    qiao ``Code08`` 在 36 维空间做多点打靶 + Newton；对 smoke 级小窗口，
    单次初值积分 + 末尾辛投影即可：``M``、``D`` 都是 Hamilton 矩阵，
    ``BᵀJB=J`` 是精确首次积分，前向积分的辛误差仅由 ODE 容差累积，
    ``symplectic_project`` 把残余误差拉回 ``<1e-13``。
    """
    from scipy.integrate import solve_ivp

    rhs = _qf_rhs_factory(M_at, D)
    B0 = np.eye(6, dtype=float).ravel()

    t_arr = np.asarray(tlist, dtype=float).ravel()
    sol = solve_ivp(
        rhs,
        (float(t_arr[0]), float(t_arr[-1])),
        B0,
        t_eval=t_arr,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"QF 矩阵法积分失败：{sol.message}")
    return sol.y.T.reshape(-1, 6, 6)


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
        method: ``"matrix"``（默认，36 维直接积分 + 辛投影）或
            ``"lie_algebra"``（21 维 sp(6) 参数化，自动保辛）。
        project: 矩阵法是否在末尾做辛投影兜底（默认 ``True``）。
        rtol: ODE 相对容差。
        atol: ODE 绝对容差。
    """

    context: NormalFormContext
    method: str = "matrix"
    project: bool = True
    rtol: float = 1e-11
    atol: float = 1e-13

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
        if self.method not in ("matrix", "lie_algebra"):
            raise ValueError(f"method 必须是 'matrix' 或 'lie_algebra'，得到 {self.method!r}")
        if ds_result.Xlist.shape[0] < 2:
            raise ValueError(f"ds_result 至少需要 2 个采样点，得到 {ds_result.Xlist.shape[0]}")

        # —— 实标准形 D（qiao Global_File 频率）——
        nu1, nu2 = self.context.central_frequencies
        lam = float(self.context.characteristic_exponent)
        D = real_normal_form_matrix(lam, float(nu1), float(nu2))

        # —— 时变线性化 M(t) ——
        M_at, M_stack = _build_M_at(ds_result)

        # —— 求解 B(t) ——
        if self.method == "matrix":
            B_samples = _solve_qf_matrix(M_at, D, ds_result.tlist, rtol=self.rtol, atol=self.atol)
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
