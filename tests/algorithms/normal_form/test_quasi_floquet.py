"""``normal_form.quasi_floquet`` 测试。

覆盖（issue #172）：

- :class:`QuasiFloquetReducer` 可用，``reduce`` 返回
  :class:`QuasiFloquetResult`；
- ``B(t)`` 在采样点上 ``Bᵀ J B = J`` 误差 ``< 1e-12``——矩阵法与
  李代数法各一个硬性数值断言；
- 实标准形 ``D`` 与 qiao ``Global_File`` 频率一致；
- sp(6) 基构造、往返 ``sp6_to_vector``/``vector_to_sp6`` 数值正确；
- qiao ``.npz`` 回归 fixture 用 ``pytest.skip`` 守卫（本仓库没有）。

输入 :class:`DynamicalSubstituteResult` 的构造思路与
``test_dynamical_substitution.py`` 一致（复用其 ``NormalFormContext`` /
``earth_moon_system`` fixture、纯 CR3BP 退路），但 quasi-Floquet 对输入
轨道的「良态性」敏感：发散轨道会让 ``B(t)`` 沿双曲方向 ``exp(λT)`` 爆
炸，辛投影退化。因此这里用 **L1 线性化复特征向量** 构造一条物理意义
明确的小 Lyapunov 轨道作为输入——这正是 quasi-Floquet 变换的设计场景。
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from e2m2e.algorithm.dynamics import LibrationPoint
from e2m2e.algorithm.normal_form.constants import JD0_J2000
from e2m2e.algorithm.normal_form.dynamical_substitution import (
    DynamicalSubstituteResult,
)
from e2m2e.algorithm.normal_form.quasi_floquet import (
    J6,
    QuasiFloquetReducer,
    QuasiFloquetResult,
    build_sp6_basis,
    real_normal_form_matrix,
    sp6_to_vector,
    symplectic_project,
    vector_to_sp6,
)

# ---------------------------------------------------------------------------
# 公共 fixture（复用 test_dynamical_substitution 的范式）
# ---------------------------------------------------------------------------


@pytest.fixture
def l1_context(earth_moon_system):
    """L1 共线点上下文（与 slice 2 测试一致）。"""
    from e2m2e.algorithm.normal_form import NormalFormContext

    return NormalFormContext(
        system=earth_moon_system,
        libration_point=LibrationPoint.L1,
        epoch=JD0_J2000,
        order=4,
    )


def _cr3bp_hessian(r, mu):
    """会合系 Ω 的对称 Hessian（与 quasi_floquet._cr3bp_hessian_symmetric 同式）。"""
    mu1 = 1.0 - mu
    r1 = r - np.array([-mu, 0.0, 0.0])
    r2 = r - np.array([1.0 - mu, 0.0, 0.0])
    d1 = float(np.linalg.norm(r1))
    d2 = float(np.linalg.norm(r2))
    S = np.eye(3) * (1.0 - mu1 / d1**3 - mu / d2**3)
    S += mu1 * 3.0 / d1**5 * np.outer(r1, r1)
    S += mu * 3.0 / d2**5 * np.outer(r2, r2)
    return S


def _small_lyapunov_orbit(l1_context, *, amp: float = 1e-3, T: float = 0.6, n: int = 31):
    """构造一条围绕 L1 的小 Lyapunov 轨道作为 quasi-Floquet 输入。

    从 L1 线性化矩阵 ``M₀`` 的**平面中心方向**（最大纯虚特征值）取实
    特征向量作为初始扰动，在纯 CR3BP 下积分 ``T``，得到一条振幅 ``O(amp)``
    的有界小轨道。这是 quasi-Floquet 变换的设计场景（围绕平动点的小
    振幅中心运动），避免了发散轨道导致 ``B(t)`` 沿双曲方向 ``exp(λT)``
    爆炸、辛投影退化的数值病态。

    返回 ``(tlist, Xlist)``，``Xlist`` 形状 ``(n, 6)``。
    """
    mu = float(l1_context.mu)
    r_lp = np.asarray(l1_context.libration_position, dtype=float).ravel()
    S0 = _cr3bp_hessian(r_lp, mu)
    omega_x = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    M0 = np.zeros((6, 6))
    M0[:3, :3] = -omega_x
    M0[:3, 3:] = np.eye(3)
    M0[3:, :3] = S0
    M0[3:, 3:] = -omega_x
    # 最大纯虚特征值（平面中心对）
    w, V = np.linalg.eig(M0)
    imag_idx = [
        i for i in range(6) if abs(w[i].imag) > 1e-6 and abs(w[i].real) < 1e-8 and w[i].imag > 0
    ]
    assert imag_idx, "L1 线性化应含纯虚特征值（中心方向）"
    i_planar = max(imag_idx, key=lambda i: w[i].imag)
    v = V[:, i_planar].real
    x0 = amp * v / np.linalg.norm(v)

    def rhs(t, X):
        rho = X[:3]
        rd = X[3:]
        e = np.array([-mu, 0.0, 0.0])
        m = np.array([1.0 - mu, 0.0, 0.0])
        d1 = rho + r_lp - e
        d2 = rho + r_lp - m
        dd = (
            -(1.0 - mu) * d1 / np.linalg.norm(d1) ** 3
            - mu * d2 / np.linalg.norm(d2) ** 3
            - 2.0 * np.cross([0.0, 0.0, 1.0], rd)
            - np.cross([0.0, 0.0, 1.0], np.cross([0.0, 0.0, 1.0], rho))
        )
        return np.concatenate([rd, dd])

    tlist = np.linspace(0.0, T, n)
    sol = solve_ivp(rhs, (0.0, T), x0, t_eval=tlist, rtol=1e-12, atol=1e-14)
    assert sol.success
    return tlist, sol.y.T


@pytest.fixture
def small_orbit_ds_result(l1_context) -> DynamicalSubstituteResult:
    """围绕 L1 的小 Lyapunov 轨道，包装成 :class:`DynamicalSubstituteResult`。

    quasi-Floquet reducer 只读 ``tlist``/``Xlist``/``context``；``W_poly``
    等字段在此测试中无关紧要（留空 dict）。
    """
    tlist, Xlist = _small_lyapunov_orbit(l1_context)
    return DynamicalSubstituteResult(
        context=l1_context,
        order=int(l1_context.order),
        substitute_orbit=Xlist,
        tlist=tlist,
        Xlist=Xlist,
        W_poly={},
        Wdot_poly={},
    )


@pytest.fixture
def matrix_reducer(l1_context) -> QuasiFloquetReducer:
    """矩阵法 reducer。"""
    return QuasiFloquetReducer(context=l1_context, method="matrix")


@pytest.fixture
def lie_reducer(l1_context) -> QuasiFloquetReducer:
    """李代数法 reducer。"""
    return QuasiFloquetReducer(context=l1_context, method="lie_algebra")


# ---------------------------------------------------------------------------
# 构造与接口
# ---------------------------------------------------------------------------


def test_reducer_is_constructible(l1_context):
    """``QuasiFloquetReducer`` 可用上下文构造，默认走矩阵法。"""
    reducer = QuasiFloquetReducer(context=l1_context)
    assert reducer.context is l1_context
    assert reducer.method == "matrix"
    assert reducer.project is True


def test_reducer_rejects_unknown_method(l1_context):
    """``method`` 非法时抛 :class:`ValueError`。"""
    reducer = QuasiFloquetReducer(context=l1_context, method="bogus")
    with pytest.raises(ValueError, match="method"):
        reducer.reduce(_trivial_ds_result(l1_context))


def test_reducer_rejects_too_few_samples(l1_context):
    """``ds_result`` 采样点不足 2 个时抛 :class:`ValueError`。"""
    reducer = QuasiFloquetReducer(context=l1_context)
    tlist = np.array([0.0])
    Xlist = np.zeros((1, 6))
    ds = DynamicalSubstituteResult(
        context=l1_context,
        order=l1_context.order,
        substitute_orbit=Xlist,
        tlist=tlist,
        Xlist=Xlist,
        W_poly={},
        Wdot_poly={},
    )
    with pytest.raises(ValueError, match="采样点"):
        reducer.reduce(ds)


def test_reduce_returns_result_with_required_fields(matrix_reducer, small_orbit_ds_result):
    """``reduce`` 返回 :class:`QuasiFloquetResult`，字段齐备。"""
    result = matrix_reducer.reduce(small_orbit_ds_result)
    assert isinstance(result, QuasiFloquetResult)
    assert result.context is matrix_reducer.context
    assert result.order == int(matrix_reducer.context.order)
    assert result.method == "matrix"
    # B_samples 形状 (n, 6, 6)
    assert result.B_samples.ndim == 3
    assert result.B_samples.shape[1:] == (6, 6)
    assert result.B_samples.shape[0] == result.tlist.shape[0]
    # 实标准形 D 是 6×6
    assert result.D.shape == (6, 6)
    # metadata 记录频率
    assert {"lambda", "wp", "wv", "n_samples"} <= set(result.metadata)


# ---------------------------------------------------------------------------
# 核心数值约束：B^T J B = J 误差 < 1e-12（硬性，必须真实测试）
# ---------------------------------------------------------------------------


def test_matrix_method_preserves_symplecticity(matrix_reducer, small_orbit_ds_result):
    """矩阵法：``B(t)`` 在所有采样点上 ``‖Bᵀ J B − J‖∞ < 1e-12``。

    数学保证：``M(t)`` 与 ``D`` 都是 Hamilton 矩阵，``Bᵀ J B`` 是
    ``Ḃ = M·B − B·D`` 的精确首次积分；末尾再做一次辛投影兜底，把
    ODE 容差累积的残余误差（~1e-11）拉回机器精度。
    """
    result = matrix_reducer.reduce(small_orbit_ds_result)
    assert result.max_symplectic_error < 1e-12, (
        f"矩阵法辛误差过大：{result.max_symplectic_error:.2e}"
    )


def test_lie_algebra_method_preserves_symplecticity(lie_reducer, small_orbit_ds_result):
    """李代数法：``B(t)`` 在所有采样点上 ``‖Bᵀ J B − J‖∞ < 1e-12``。

    ``B=exp(ξ)``、``ξ ∈ sp(6)`` 自动保辛，无需投影。
    """
    result = lie_reducer.reduce(small_orbit_ds_result)
    assert result.max_symplectic_error < 1e-12, (
        f"李代数法辛误差过大：{result.max_symplectic_error:.2e}"
    )


def test_both_methods_share_same_normal_form(l1_context, small_orbit_ds_result):
    """矩阵法与李代数法的实标准形 ``D`` 应完全一致（同源频率）。"""
    D_ref = real_normal_form_matrix(
        float(l1_context.characteristic_exponent),
        float(l1_context.central_frequencies[0]),
        float(l1_context.central_frequencies[1]),
    )
    r_mat = QuasiFloquetReducer(context=l1_context, method="matrix").reduce(small_orbit_ds_result)
    r_lie = QuasiFloquetReducer(context=l1_context, method="lie_algebra").reduce(
        small_orbit_ds_result
    )
    np.testing.assert_allclose(r_mat.D, D_ref)
    np.testing.assert_allclose(r_lie.D, D_ref)


def test_both_methods_agree_on_B(l1_context, small_orbit_ds_result):
    """矩阵法与李代数法应解同一个方程 ``Ḃ = M·B − B·D``，给出一致的 ``B(t)``。

    这是 issue #172「可共用一套测试」的实质要求：两条入口都是对
    ``Ḃ = M·B − B·D`` 的等价数值实现（矩阵法 36 维直接积分 + 辛投影，
    李代数法 commutator-free Lie group RK4、自动保辛），故 ``B(t)`` 在
    采样点上须数值一致。容差放宽到 ``1e-4``：李代数法用固定子步 4 阶
    格式（~5e-6 误差），矩阵法用 DOP853（~1e-11），二者之差由李代数法
    主导。
    """
    r_mat = QuasiFloquetReducer(context=l1_context, method="matrix").reduce(small_orbit_ds_result)
    r_lie = QuasiFloquetReducer(context=l1_context, method="lie_algebra").reduce(
        small_orbit_ds_result
    )
    # B(t_0)=I 是共同初值，应精确成立
    np.testing.assert_allclose(r_mat.B_samples[0], np.eye(6), atol=1e-9)
    np.testing.assert_allclose(r_lie.B_samples[0], np.eye(6), atol=1e-9)
    # 整条 B(t) 数值一致
    np.testing.assert_allclose(r_lie.B_samples, r_mat.B_samples, atol=1e-4)


# ---------------------------------------------------------------------------
# B(t) 在 t_0 处 = I（初值条件）
# ---------------------------------------------------------------------------


def test_B_at_t0_is_identity(matrix_reducer, small_orbit_ds_result):
    """``B(t_0)`` 应为单位矩阵（初值 ``B(0)=I``）。"""
    result = matrix_reducer.reduce(small_orbit_ds_result)
    np.testing.assert_allclose(result.B_samples[0], np.eye(6), atol=1e-9)


def test_B_interpolation_round_trips(matrix_reducer, small_orbit_ds_result):
    """``result.B(t)`` 在采样点上等于 ``B_samples``。"""
    result = matrix_reducer.reduce(small_orbit_ds_result)
    t_mid = float(result.tlist[len(result.tlist) // 2])
    np.testing.assert_allclose(result.B(t_mid), result.B_samples[len(result.tlist) // 2])


# ---------------------------------------------------------------------------
# M(t) 与 D 的 Hamilton 性（数值自检）
# ---------------------------------------------------------------------------


def test_D_is_hamiltonian(l1_context):
    """实标准形 ``D`` 应满足 ``Dᵀ J + J D = 0``（Hamilton 矩阵）。"""
    D = real_normal_form_matrix(
        float(l1_context.characteristic_exponent),
        float(l1_context.central_frequencies[0]),
        float(l1_context.central_frequencies[1]),
    )
    assert np.linalg.norm(D.T @ J6 + J6 @ D) < 1e-12


def test_M_samples_are_hamiltonian(matrix_reducer, small_orbit_ds_result):
    """``M(t)`` 在各采样点应是 Hamilton 矩阵（保证辛守恒）。"""
    result = matrix_reducer.reduce(small_orbit_ds_result)
    assert result.M_samples is not None
    for M in result.M_samples:
        assert np.linalg.norm(M.T @ J6 + J6 @ M) < 1e-12


# ---------------------------------------------------------------------------
# 常数 QF 变换（CR3BP：B = 实标准形变换 V）
# ---------------------------------------------------------------------------


def test_cr3bp_hamiltonian_linearization_is_hamiltonian(l1_context):
    """动量框架线性化 ``M_H`` 应为 Hamilton 矩阵（``MᵀJ + JM = 0``）。

    速度框架 ``[[0,I],[S,−2Ω×]]`` 不满足此性质（特征向量无 J 正交性，
    无法构造同时辛归一化与对角化的实标准形变换）；动量框架
    ``[[C,I],[S_grav,C]]`` 满足且特征值等于物理频率。
    """
    from e2m2e.algorithm.normal_form.quasi_floquet import (
        _cr3bp_hamiltonian_linearization,
    )

    M_H = _cr3bp_hamiltonian_linearization(l1_context.libration_position, l1_context.mu)
    assert np.linalg.norm(M_H.T @ J6 + J6 @ M_H) < 1e-10
    # 特征值与 qiao 权威频率一致（1% 内；γ/μ 数值舍入引起的小偏差）
    ev = np.linalg.eigvals(M_H)
    lam_num = max(abs(e.real) for e in ev)
    om_num = sorted(abs(e.imag) for e in ev if abs(e.real) < 1e-8)
    assert abs(lam_num - l1_context.characteristic_exponent) < 0.01 * l1_context.characteristic_exponent
    assert abs(om_num[0] - l1_context.central_frequencies[1]) < 0.01 * l1_context.central_frequencies[1]
    assert abs(om_num[-1] - l1_context.central_frequencies[0]) < 0.01 * l1_context.central_frequencies[0]


def test_real_normal_form_transform_is_symplectic_diagonalizing(l1_context):
    """``V`` 辛归一化且 ``V⁻¹MV = D``（实标准形结构、频率正确）。"""
    from e2m2e.algorithm.normal_form.quasi_floquet import (
        _cr3bp_hamiltonian_linearization,
        real_normal_form_transform,
    )

    M_H = _cr3bp_hamiltonian_linearization(l1_context.libration_position, l1_context.mu)
    V, D = real_normal_form_transform(M_H)
    # 辛性
    assert np.linalg.norm(V.T @ J6 @ V - J6) < 1e-10
    # 对角化：V⁻¹MV = D（实标准形结构，非对角块仅 ±ω 位置）
    D_check = np.linalg.solve(V, M_H @ V)
    mask = np.zeros((6, 6), dtype=bool)
    mask[0, 0] = mask[3, 3] = True
    mask[1, 4] = mask[4, 1] = True
    mask[2, 5] = mask[5, 2] = True
    assert np.abs(D_check[~mask]).max() < 1e-10
    # 频率：λ > 0、ωp > ωv > 0（L1/L2 面内频率更大）
    assert D_check[0, 0] > 0
    assert D_check[1, 4] > D_check[2, 5] > 0
    # 还原的 D 与 qiao 权威频率一致（1% 内）
    assert abs(D_check[0, 0] - l1_context.characteristic_exponent) < 0.01 * l1_context.characteristic_exponent
    assert abs(D_check[1, 4] - l1_context.central_frequencies[0]) < 0.01 * l1_context.central_frequencies[0]
    assert abs(D_check[2, 5] - l1_context.central_frequencies[1]) < 0.01 * l1_context.central_frequencies[1]


# ---------------------------------------------------------------------------
# sp(6) 基与代数工具
# ---------------------------------------------------------------------------


def test_sp6_basis_has_21_elements():
    """sp(6, R) 维度 ``2n² + n = 21``。"""
    basis = build_sp6_basis()
    assert len(basis) == 21


def test_sp6_basis_is_hamiltonian_and_orthonormal():
    """每个基元素是 Hamilton 矩阵，且彼此 Frobenius 正交。"""
    basis = build_sp6_basis()
    for E in basis:
        assert np.linalg.norm(E.T @ J6 + J6 @ E) < 1e-12
    for i in range(len(basis)):
        for j in range(i + 1, len(basis)):
            ip = float(np.sum(basis[i] * basis[j]))
            assert abs(ip) < 1e-12


def test_sp6_vector_roundtrip():
    """``sp6_to_vector``/``vector_to_sp6`` 互为精确逆。"""
    basis = build_sp6_basis()
    rng = np.random.default_rng(0)
    xi = rng.normal(size=21)
    M = vector_to_sp6(xi, basis)
    xi2 = sp6_to_vector(M, basis)
    np.testing.assert_allclose(xi, xi2, atol=1e-12)


# ---------------------------------------------------------------------------
# symplectic_project 单元测试
# ---------------------------------------------------------------------------


def test_symplectic_project_recovers_exact_matrix():
    """精确辛矩阵的投影应不变；近辛矩阵应被拉回。"""
    basis = build_sp6_basis()
    rng = np.random.default_rng(1)
    xi = rng.normal(scale=0.1, size=21)
    B_exact = expm(vector_to_sp6(xi, basis))
    B_projected = symplectic_project(B_exact)
    np.testing.assert_allclose(B_projected, B_exact, atol=1e-12)
    # 加扰动后投影应让辛误差降到 < 1e-13
    B_noisy = B_exact + 1e-3 * rng.normal(size=(6, 6))
    B_back = symplectic_project(B_noisy)
    assert float(np.max(np.abs(B_back.T @ J6 @ B_back - J6))) < 1e-13


def test_symplectic_project_handles_large_norm():
    """投影对范数较大的辛矩阵（双曲方向增长）也应收敛。"""
    basis = build_sp6_basis()
    xi = np.zeros(21)
    xi[0] = 8.0  # 第一个基对应双曲方向，B 范数 ~ exp(8)
    B_exact = expm(vector_to_sp6(xi, basis))
    rng = np.random.default_rng(3)
    B = B_exact + 1e-9 * np.linalg.norm(B_exact) * rng.normal(size=(6, 6))
    B_back = symplectic_project(B)
    assert float(np.max(np.abs(B_back.T @ J6 @ B_back - J6))) < 1e-12


# ---------------------------------------------------------------------------
# qiao .npz 回归 fixture（本仓库没有，skip 守卫）
# ---------------------------------------------------------------------------


def test_qiao_npz_regression_is_skipped_without_fixture():
    """qiao ``L1_QFtrans.npz`` fixture 在本仓库不存在，必须 skip。

    本测试仅验证 skip 守卫生效；一旦外部 fixture 引入，可在此替换为真实
    数值比对。
    """
    from pathlib import Path

    fixture = Path(__file__).parent / "data" / "L1_QFtrans.npz"
    if not fixture.exists():
        pytest.skip(f"qiao .npz fixture 不存在：{fixture}")
    data = np.load(fixture, allow_pickle=True)
    assert "tlist" in data and "QFtrans_mat" in data


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _trivial_ds_result(l1_context) -> DynamicalSubstituteResult:
    """构造一个仅用于触发 reducer 校验的最小占位结果。"""
    tlist = np.array([0.0, 1.0], dtype=float)
    Xlist = np.zeros((2, 6), dtype=float)
    return DynamicalSubstituteResult(
        context=l1_context,
        order=l1_context.order,
        substitute_orbit=Xlist,
        tlist=tlist,
        Xlist=Xlist,
        W_poly={},
        Wdot_poly={},
    )
