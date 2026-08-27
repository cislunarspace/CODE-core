"""``normal_form.coord_trans`` + ``catalog`` 测试。

覆盖：

- :mod:`coord_trans` 5 段叶子函数各自的往返（``rho↔em``、``em↔ds``、
  ``ds↔qf``、``qf↔cm``、``cm↔param``）；
- 端到端链 ``rho → param → rho`` 误差在容差内（**硬性数值约束**）：
  线性段（rho↔em、em↔ds、ds↔qf、cm↔param）接近机器精度（~1e-12），
  高阶 Lie 级数段（qf↔cm）允许稍宽（~1e-8），分开断言；
- :class:`LibrationCatalogTransformer` 可绑定预计算系数表做 OO 入口；

输入三个子结果（DS / QF / CM）的构造沿用 ``test_center_manifold.py`` /
``test_quasi_floquet.py`` 的范式：本切片的变换链只读 ``W_poly``、
``B_at(t)``、``W_series``，故用 ``B=I`` 的最小 QF 结果 + 注入高阶项的
CM 结果 + 构造的小 DS 结果。SPICE 不可用时 rho↔em 段退化到纯 CR3BP
（``p = ρ̇``），往返仍为机器精度。
"""

from __future__ import annotations

import numpy as np
import pytest

# sympy 是 normal-form optional dep；未安装时整个文件 skip（不 error）。
pytest.importorskip("sympy")

from e2m2e.algorithm.normal_form.catalog import (
    LibrationCatalogData,
    LibrationCatalogTransformer,
)
from e2m2e.algorithm.normal_form.center_manifold import (
    CenterManifoldReducer,
)
from e2m2e.algorithm.normal_form.coord_trans import (
    cm_to_param,
    cm_to_qf,
    ds_to_em,
    ds_to_qf,
    em_to_ds,
    em_to_rho,
    param_to_cm,
    param_to_rho,
    qf_to_cm,
    qf_to_ds,
    rho_to_em,
    rho_to_param,
)
from e2m2e.algorithm.normal_form.dynamical_substitution import (
    DynamicalSubstituteResult,
)
from e2m2e.algorithm.normal_form.quasi_floquet import (
    QuasiFloquetResult,
    real_normal_form_matrix,
)
from e2m2e.data.templates.enums import LibrationPoint

pytestmark = pytest.mark.theory


# ---------------------------------------------------------------------------
# 公共 fixture（沿用 test_center_manifold / test_quasi_floquet 的范式）
# ---------------------------------------------------------------------------


@pytest.fixture
def l1_context(earth_moon_system):
    """L1 共线点上下文。"""
    from e2m2e.algorithm.normal_form import NormalFormContext
    from e2m2e.algorithm.normal_form.constants import JD0_J2000

    return NormalFormContext(
        system=earth_moon_system,
        libration_point=LibrationPoint.L1,
        epoch=JD0_J2000,
        order=4,
    )


def _make_ds_result(l1_context, *, n: int = 96, T: float = 3.0):
    """构造最小 :class:`DynamicalSubstituteResult`。

    用一条围绕原点的小正弦轨道作为 ``Xlist``，由其数值微分得 ``W_poly``
    （6 个线性项系数时间序列）。本切片变换链只读 ``W_poly`` 与 ``tlist``，
    故 ``fft_components`` 等字段留默认/空。
    """
    tlist = np.linspace(0.0, T, n)
    amp = 1e-3
    # 简单正弦轨道（rho, rhodot）
    A = amp * np.cos(tlist)[:, None] * np.array([[1.0, 0.5, -0.3]])
    Adot = -amp * np.sin(tlist)[:, None] * np.array([[1.0, 0.5, -0.3]])
    B = amp * 0.8 * np.sin(tlist)[:, None] * np.array([[0.2, -0.6, 0.4]])
    Bdot = amp * 0.8 * np.cos(tlist)[:, None] * np.array([[0.2, -0.6, 0.4]])
    Xlist = np.concatenate([A, B], axis=1)  # (n,6): [A,B]（位置在 0:3、动量在 3:6）

    # W_poly: 前 3 线性项 = A（位置层），后 3 = B（动量层），与
    # DynamicalSubstituteCorrector._build_W 的 pow_units 约定一致。
    pow_units = [
        (1, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0),
        (0, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 1, 0),
        (0, 0, 0, 0, 0, 1),
    ]
    W_poly = {}
    Wdot_poly = {}
    for k, p in enumerate(pow_units):
        W_poly[p] = Xlist[:, k]
        Wdot_poly[p] = (Adot if k < 3 else Bdot)[:, k - (3 if k >= 3 else 0)]

    return DynamicalSubstituteResult(
        context=l1_context,
        order=int(l1_context.order),
        substitute_orbit=Xlist,
        tlist=tlist,
        Xlist=Xlist,
        W_poly=W_poly,
        Wdot_poly=Wdot_poly,
    )


def _make_qf_result(l1_context, *, n: int = 96, T: float = 3.0, B_identity: bool = True):
    """构造最小 :class:`QuasiFloquetResult`。

    ``B_identity=True`` 时 ``B(t)=I``（DS↔QF 退化为恒等，便于把往返误差
    归因到其他段）；``False`` 时用一组随机辛接近的矩阵。本切片变换链只
    读 ``B_at(t)``。
    """
    lam = float(l1_context.characteristic_exponent)
    nu1, nu2 = l1_context.central_frequencies
    D = real_normal_form_matrix(lam, float(nu1), float(nu2))
    tlist = np.linspace(0.0, T, n)
    if B_identity:
        B_samples = np.stack([np.eye(6, dtype=float) for _ in range(n)])
    else:
        rng = np.random.default_rng(0)
        B_samples = np.stack([np.eye(6) + 1e-2 * rng.standard_normal((6, 6)) for _ in range(n)])
    return QuasiFloquetResult(
        context=l1_context,
        order=int(l1_context.order),
        tlist=tlist,
        B_samples=B_samples,
        D=D,
        method="matrix",
    )


def _make_cm_result(l1_context, qf_result, *, max_order: int = 5, with_terms: bool = True):
    """构造 :class:`CenterManifoldResult`（注入高阶 Hamiltonian 项后化简）。

    注入双曲-中心交叉项使 ``W_series`` 非平凡——这是 QF↔CM 高阶 Lie 级数
    的数值内容。``with_terms=False`` 时退化到平凡（无高阶项，W_series 全空，
    qf↔cm 段退化为恒等）。
    """
    reducer = CenterManifoldReducer(context=l1_context, max_order=max_order)
    n = qf_result.tlist.size
    if not with_terms:
        return reducer.reduce(qf_result)
    terms = {
        (2, 1, 0, 1, 0, 0): 0.1 * np.ones(n),
        (1, 2, 0, 2, 0, 0): 0.05 * np.ones(n),
        (0, 3, 0, 0, 1, 0): 0.08 * np.ones(n),
    }
    return reducer.reduce(qf_result, hamiltonian_terms=terms)


def _interp_W_series_at_t(cm_result, qf_result, t):
    """测试用镜像：在 t 处插值 CM 的 W_series（与 coord_trans.__init__ 一致）。

    时间网格用 ``qf_result.tlist``（与 ``CenterManifoldReducer.reduce`` 生成
    W_series 的时间轴一致），而非任意兜底网格：这是正向数值
    正确性关键（往返测试因正逆相消掩盖不了正向错误）。
    """
    t_arr = np.asarray(qf_result.tlist, dtype=float).ravel()
    out: dict[int, dict[tuple[int, ...], complex]] = {}
    for step_data in cm_result.W_series.values():
        for order, poly in step_data.items():
            if not poly:
                continue
            out.setdefault(order, {})
            for pow_tuple, coef_arr in poly.items():
                arr = np.asarray(coef_arr, dtype=complex).ravel()
                if arr.size == 0:
                    continue
                if arr.size == 1:
                    val = complex(arr[0])
                else:
                    re = float(np.interp(t, t_arr, arr.real))
                    im = float(np.interp(t, t_arr, arr.imag))
                    val = complex(re, im)
                out[order][pow_tuple] = out[order].get(pow_tuple, 0j) + val
    return out


# ---------------------------------------------------------------------------
# rho ↔ EM
# ---------------------------------------------------------------------------


def test_rho_em_roundtrip_machine_precision(l1_context):
    """rho → EM → rho 在纯 CR3BP 退路下为机器精度（p = ρ̇）。"""
    rng = np.random.default_rng(42)
    X_rho = 1e-3 * rng.standard_normal(6)
    t = 1.23
    X_em = rho_to_em(X_rho, t, l1_context)
    X_back = em_to_rho(X_em, t, l1_context)
    np.testing.assert_allclose(X_back, X_rho, atol=1e-14)


def test_rho_em_rejects_bad_shape(l1_context):
    """非 (6,) 输入报 ValueError。"""
    with pytest.raises(ValueError, match="X_rho"):
        rho_to_em(np.zeros(7), 0.0, l1_context)
    with pytest.raises(ValueError, match="X_em"):
        em_to_rho(np.zeros(3), 0.0, l1_context)


# ---------------------------------------------------------------------------
# EM ↔ DS
# ---------------------------------------------------------------------------


def test_em_ds_roundtrip_machine_precision(l1_context):
    """EM → DS → EM 平移往返为机器精度。"""
    ds = _make_ds_result(l1_context)
    rng = np.random.default_rng(7)
    X_em = 1e-3 * rng.standard_normal(6)
    t = 1.5
    # 内联插值 W(t) 以便单元隔离
    from e2m2e.algorithm.normal_form.coord_trans.em_ds import _interp_W_at

    W_at_t = _interp_W_at(ds.W_poly, np.asarray(ds.tlist).ravel(), t)
    X_ds = em_to_ds(X_em, W_at_t)
    X_back = ds_to_em(X_ds, W_at_t)
    np.testing.assert_allclose(X_back, X_em, atol=1e-14)


def test_em_ds_translation_correctness():
    """平移关系正确：Q = q − B、P = p + A，[A,B]=W。"""
    W = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # A=[.1,.2,.3], B=[.4,.5,.6]
    X_em = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    X_ds = em_to_ds(X_em, W)
    # Q = q - B
    np.testing.assert_allclose(X_ds[:3], [0.6, 1.5, 2.4])
    # P = p + A
    np.testing.assert_allclose(X_ds[3:], [4.1, 5.2, 6.3])


# ---------------------------------------------------------------------------
# DS ↔ QF
# ---------------------------------------------------------------------------


def test_ds_qf_roundtrip_identity(l1_context):
    """B=I 时 DS↔QF 恒等；往返机器精度。"""
    B = np.eye(6)
    rng = np.random.default_rng(11)
    X_ds = 1e-3 * rng.standard_normal(6)
    X_qf = ds_to_qf(X_ds, B)
    np.testing.assert_allclose(X_qf, X_ds, atol=1e-14)
    X_back = qf_to_ds(X_qf, B)
    np.testing.assert_allclose(X_back, X_ds, atol=1e-14)


def test_ds_qf_roundtrip_general_matrix():
    """一般可逆 B 的往返机器精度。"""
    rng = np.random.default_rng(23)
    B = np.eye(6) + 1e-2 * rng.standard_normal((6, 6))
    assert abs(np.linalg.det(B)) > 0.1  # 可逆
    X_ds = 1e-3 * rng.standard_normal(6)
    X_qf = ds_to_qf(X_ds, B)
    X_back = qf_to_ds(X_qf, B)
    np.testing.assert_allclose(X_back, X_ds, atol=1e-12)


def test_ds_qf_relation_correctness():
    """X_QF = B⁻¹·X_DS 数学关系正确。"""
    B = 2.0 * np.eye(6)
    X_ds = np.ones(6)
    X_qf = ds_to_qf(X_ds, B)
    np.testing.assert_allclose(X_qf, 0.5 * np.ones(6))


# ---------------------------------------------------------------------------
# QF ↔ CM（高阶 Lie 级数）
# ---------------------------------------------------------------------------


def test_qf_cm_roundtrip_trivial_no_W(l1_context):
    """无高阶项（W_series 全空）时 QF↔CM 恒等；往返机器精度。"""
    qf = _make_qf_result(l1_context)
    cm = _make_cm_result(l1_context, qf, with_terms=False)
    W_at_t = _interp_W_series_at_t(cm, qf, 1.0)
    rng = np.random.default_rng(5)
    X_qf = 1e-3 * rng.standard_normal(6)
    X_cm = qf_to_cm(X_qf, W_at_t)
    X_back = cm_to_qf(X_cm, W_at_t)
    np.testing.assert_allclose(X_back, X_qf, atol=1e-12)


def test_qf_cm_roundtrip_with_high_order(l1_context):
    """有非平凡 W_series 时 QF↔CM 往返在 1e-8 内（高阶 Lie 级数容差）。"""
    qf = _make_qf_result(l1_context)
    cm = _make_cm_result(l1_context, qf, with_terms=True, max_order=5)
    # 确认 W_series 非平凡
    nonempty = sum(
        1
        for step in cm.W_series.values()
        for poly in step.values()
        for v in poly.values()
        if np.any(np.asarray(v) != 0)
    )
    assert nonempty > 0, "注入高阶项后 W_series 应非平凡"

    W_at_t = _interp_W_series_at_t(cm, qf, 1.0)
    rng = np.random.default_rng(13)
    X_qf = 1e-3 * rng.standard_normal(6)
    X_cm = qf_to_cm(X_qf, W_at_t)
    X_back = cm_to_qf(X_cm, W_at_t)
    # 高阶 Lie 级数往返：允许稍宽
    np.testing.assert_allclose(X_back, X_qf, atol=1e-8)


def test_qf_cm_re_basis_change_is_involution():
    """实/复基底变换 D 是对合：D⁻¹·D = I（逐元素验证）。"""
    from e2m2e.algorithm.normal_form.coord_trans.qf_cm import _D, _D_INV

    np.testing.assert_allclose(_D @ _D_INV, np.eye(6), atol=1e-14)
    np.testing.assert_allclose(_D_INV @ _D, np.eye(6), atol=1e-14)


def test_qf_cm_rust_matches_python_multi_order(l1_context):
    """多阶 W 下 Rust 与 Python 参照正反向分量一致。"""
    qf = _make_qf_result(l1_context)
    cm = _make_cm_result(l1_context, qf, with_terms=True, max_order=5)
    W_at_t = _interp_W_series_at_t(cm, qf, 1.0)
    rng = np.random.default_rng(17)
    X_qf = 1e-3 * rng.standard_normal(6)

    X_cm_rust = qf_to_cm(X_qf, W_at_t, backend="rust")
    X_cm_py = qf_to_cm(X_qf, W_at_t, backend="python")
    np.testing.assert_allclose(X_cm_rust, X_cm_py, atol=1e-9, rtol=1e-9)

    X_back_rust = cm_to_qf(X_cm_rust, W_at_t, backend="rust")
    X_back_py = cm_to_qf(X_cm_py, W_at_t, backend="python")
    np.testing.assert_allclose(X_back_rust, X_back_py, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(X_back_rust, X_qf, atol=1e-8)


def test_qf_cm_near_zero_monomial_and_empty_W():
    """靠近零坐标的单项式与空 W 恒等：Rust/Python 一致。"""
    # 空 W：恒等
    X = np.array([1e-4, -2e-4, 3e-4, -4e-4, 5e-4, -6e-4])
    np.testing.assert_allclose(qf_to_cm(X, {}, backend="rust"), X, atol=1e-14)
    np.testing.assert_allclose(qf_to_cm(X, {}, backend="python"), X, atol=1e-14)

    # 含在零坐标处 n_j=1 的项（0^0 边界）：不得产生 NaN
    W = {
        3: {
            (0, 0, 0, 1, 0, 0): 1e-3 + 2e-4j,
            (2, 0, 0, 0, 0, 0): -5e-4 + 1e-4j,
        },
        4: {
            (1, 0, 0, 1, 0, 0): 2e-4 - 1e-4j,
        },
    }
    X0 = np.zeros(6)
    X0[0] = 1e-3  # 仅 q1 非零
    rust = qf_to_cm(X0, W, backend="rust")
    py = qf_to_cm(X0, W, backend="python")
    assert np.all(np.isfinite(rust)) and np.all(np.isfinite(py))
    np.testing.assert_allclose(rust, py, atol=1e-10)
    # 小振幅 + 弱 W 时往返应可逆；系数相对状态不小，放宽到 1e-7
    back = cm_to_qf(rust, W, backend="rust")
    np.testing.assert_allclose(back, X0, atol=1e-7)


def test_qf_cm_rejects_bad_backend():
    """非法 backend 显式报错，禁止 auto。"""
    with pytest.raises(ValueError, match="backend"):
        qf_to_cm(np.zeros(6), {}, backend="auto")  # type: ignore[arg-type]


def test_hamilton_flow_rhs_matches_termwise():
    """向量化 Hamilton 流右端与逐项 oracle 逐位一致（含 0^0=1 边界）。

    对应 qiao ``qpQF2qpCM.__main__`` 的自检逻辑。
    """
    from e2m2e.algorithm.normal_form.coord_trans.qf_cm import (
        _hamilton_flow_rhs,
        _pack_wpoly,
    )

    W = {
        (0, 0, 0, 0, 0, 0): 0.5 + 0.0j,
        (2, 0, 0, 0, 0, 0): 1.3 + 0.2j,
        (0, 0, 0, 1, 0, 0): 0.7 - 0.4j,
        (1, 1, 0, 1, 1, 0): -0.6 + 1.1j,
        (0, 0, 2, 0, 0, 3): 2.0 + 0.0j,
        (1, 2, 3, 4, 5, 6): 0.3 - 0.7j,
    }
    rng = np.random.default_rng(0)
    test_X = [
        np.zeros(6, dtype=complex),
        np.array([1.0, 0, 0, 0, 0.5, 0], dtype=complex),
        np.array([0.3, -0.2, 0.1, -0.4, 0.6, 0.9], dtype=complex),
        rng.standard_normal(6) + 1j * rng.standard_normal(6),
        rng.standard_normal(6) + 1j * rng.standard_normal(6),
    ]
    exps, coefs = _pack_wpoly(W)

    def termwise(X, W_poly):
        q1, q2, q3, p1, p2, p3 = X
        dX = np.zeros(6, dtype=complex)
        for (n1, n2, n3, n4, n5, n6), val in W_poly.items():
            if n4 > 0:
                dX[0] += val * n4 * q1**n1 * q2**n2 * q3**n3 * p1 ** (n4 - 1) * p2**n5 * p3**n6
            if n5 > 0:
                dX[1] += val * n5 * q1**n1 * q2**n2 * q3**n3 * p1**n4 * p2 ** (n5 - 1) * p3**n6
            if n6 > 0:
                dX[2] += val * n6 * q1**n1 * q2**n2 * q3**n3 * p1**n4 * p2**n5 * p3 ** (n6 - 1)
            if n1 > 0:
                dX[3] -= val * n1 * q1 ** (n1 - 1) * q2**n2 * q3**n3 * p1**n4 * p2**n5 * p3**n6
            if n2 > 0:
                dX[4] -= val * n2 * q1**n1 * q2 ** (n2 - 1) * q3**n3 * p1**n4 * p2**n5 * p3**n6
            if n3 > 0:
                dX[5] -= val * n3 * q1**n1 * q2**n2 * q3 ** (n3 - 1) * p1**n4 * p2**n5 * p3**n6
        return dX

    worst = 0.0
    for X in test_X:
        v = _hamilton_flow_rhs(X.astype(complex), exps, coefs)
        ref = termwise(X.astype(complex), W)
        worst = max(worst, float(np.max(np.abs(v - ref))))
    assert worst < 1e-12, f"向量化与逐项偏差 {worst:.3e}"


def test_w_series_interp_uses_real_tlist(l1_context):
    """W_series 插值必须用 ``qf_result.tlist``，而非任意兜底网格。

    回归守卫：``CenterManifoldResult`` 不存 tlist，``_interp_W_series_at_t``
    若退回 ``dt=0.1`` 兜底网格，会在非采样点上给出错误系数（差异可达 ~1），
    导致正向 ``rho_to_param`` 数值错误，且往返测试因正逆相消掩盖该错误。
    本测试注入一项随时间线性变化的 Hamiltonian 项，使化简后的 W_series
    含随时间变化的系数，在非采样点上断言插值值等于用真实 ``qf.tlist``
    的解析值，且与 ``dt=0.1`` 兜底值明显不同。
    """
    from e2m2e.algorithm.normal_form.center_manifold import CenterManifoldReducer
    from e2m2e.algorithm.normal_form.coord_trans import _interp_W_series_at_t

    qf = _make_qf_result(l1_context, n=96, T=3.0)
    # 注入一项随时间线性变化的 Hamiltonian 项，使 W_series 非常值
    n = qf.tlist.size
    terms = {
        (2, 1, 0, 1, 0, 0): np.linspace(0.05, 0.15, n),
        (1, 2, 0, 2, 0, 0): 0.05 * np.ones(n),
        (0, 3, 0, 0, 1, 0): 0.08 * np.ones(n),
    }
    reducer = CenterManifoldReducer(context=l1_context, max_order=5)
    cm = reducer.reduce(qf, hamiltonian_terms=terms)

    # 取一个随时间显著变化的 (step, order, pow) 项
    target = None
    for step_data in cm.W_series.values():
        for order, poly in step_data.items():
            for pow_tuple, arr in poly.items():
                a = np.asarray(arr, dtype=complex).ravel()
                variation = max(float(np.ptp(a.real)), float(np.ptp(a.imag)))
                if a.size > 2 and variation > 1e-6:
                    target = (order, pow_tuple, a)
                    break
            if target:
                break
        if target:
            break
    assert target is not None, "W_series 应有随时间变化的项"

    order, pow_tuple, arr = target
    # 非采样点：真实网格 dt≈0.0316，兜底 dt=0.1，t=1.0 上两者插值明显不同
    t = 1.0
    tlist = np.asarray(qf.tlist, dtype=float).ravel()
    expected = complex(
        np.interp(t, tlist, arr.real),
        np.interp(t, tlist, arr.imag),
    )
    fallback_tlist = np.arange(arr.size) * 0.1
    wrong = complex(
        np.interp(t, fallback_tlist, arr.real),
        np.interp(t, fallback_tlist, arr.imag),
    )
    assert abs(expected - wrong) > 1e-3, "测试前提不成立：真实网格与兜底网格在该点应明显不同"

    coeffs = _interp_W_series_at_t(cm, qf, t)
    got = coeffs[order][pow_tuple]
    np.testing.assert_allclose(got, expected, atol=1e-12)
    assert abs(got - wrong) > 1e-3, "插值不应使用 dt=0.1 兜底网格"


# ---------------------------------------------------------------------------
# CM ↔ param
# ---------------------------------------------------------------------------


def test_cm_param_roundtrip_machine_precision():
    """CM → param → CM 往返机器精度（作用量-角变量）。"""
    rng = np.random.default_rng(29)
    X_cm = 1e-3 * rng.standard_normal(6)
    # 保证中心对幅值非零（避免 atan2(0,0) 退化）
    X_cm[1] += 0.05
    X_cm[4] += 0.03
    X_cm[2] += 0.04
    X_cm[5] += 0.02
    X_param = cm_to_param(X_cm)
    X_back = param_to_cm(X_param)
    np.testing.assert_allclose(X_back, X_cm, atol=1e-14)


def test_cm_param_action_angle_correctness():
    """作用量-角变量公式正确：I2=(q2²+p2²)/2, θ2=atan2(p2,q2)。"""
    X_cm = np.array([1.0, 0.3, -0.4, 2.0, 0.5, 0.6])
    q1, q2, q3, p1, p2, p3 = X_cm
    X_param = cm_to_param(X_cm)
    # 双曲方向原样
    np.testing.assert_allclose(X_param[0], q1)
    np.testing.assert_allclose(X_param[1], p1)
    # 作用量
    np.testing.assert_allclose(X_param[2], 0.5 * (q2**2 + p2**2))
    np.testing.assert_allclose(X_param[4], 0.5 * (q3**2 + p3**2))
    # 角变量
    np.testing.assert_allclose(X_param[3], np.arctan2(p2, q2))
    np.testing.assert_allclose(X_param[5], np.arctan2(p3, q3))


def test_cm_param_rejects_bad_shape():
    """非 (6,) 输入报 ValueError。"""
    with pytest.raises(ValueError, match="X_cm"):
        cm_to_param(np.zeros(4))
    with pytest.raises(ValueError, match="X_param"):
        param_to_cm(np.zeros(8))


def test_param_to_cm_zero_action():
    """I2=I3=0 时中心对还原为 0（√(2I)·cos/sin = 0）。"""
    X_param = np.array([1.0, 2.0, 0.0, 0.5, 0.0, 0.7])
    X_cm = param_to_cm(X_param)
    np.testing.assert_allclose(X_cm[1], 0.0)
    np.testing.assert_allclose(X_cm[2], 0.0)
    np.testing.assert_allclose(X_cm[4], 0.0)
    np.testing.assert_allclose(X_cm[5], 0.0)
    # 双曲方向原样
    np.testing.assert_allclose(X_cm[0], 1.0)
    np.testing.assert_allclose(X_cm[3], 2.0)


# ---------------------------------------------------------------------------
# 端到端链：rho → param → rho
# ---------------------------------------------------------------------------


@pytest.fixture
def catalog_data(l1_context):
    """聚合三个子结果的 :class:`LibrationCatalogData`（B=I + 非平凡 W_series）。"""
    ds = _make_ds_result(l1_context)
    qf = _make_qf_result(l1_context)
    cm = _make_cm_result(l1_context, qf, with_terms=True, max_order=5)
    return LibrationCatalogData(
        context=l1_context,
        ds_result=ds,
        qf_result=qf,
        cm_result=cm,
    )


def test_end_to_end_roundtrip_with_high_order(catalog_data, l1_context):
    """端到端 rho → param → rho：含高阶 Lie 级数时在 1e-7 内。

    硬性数值约束：高阶 Lie 级数段误差主导，整体放宽到 ~1e-7。
    """
    rng = np.random.default_rng(202)
    X_rho = 1e-3 * rng.standard_normal(6)
    t = 1.35
    X_param = rho_to_param(
        X_rho,
        t,
        l1_context,
        catalog_data.ds_result,
        catalog_data.qf_result,
        catalog_data.cm_result,
    )
    X_back = param_to_rho(
        X_param,
        t,
        l1_context,
        catalog_data.ds_result,
        catalog_data.qf_result,
        catalog_data.cm_result,
    )
    np.testing.assert_allclose(X_back, X_rho, atol=1e-7)


def test_end_to_end_at_multiple_times(catalog_data, l1_context):
    """端到端往返在多个时刻都成立（插值路径覆盖）。"""
    rng = np.random.default_rng(303)
    X_rho = 1e-3 * rng.standard_normal(6)
    for t in [0.0, 0.5, 1.0, 2.0, 2.9]:
        X_param = rho_to_param(
            X_rho,
            t,
            l1_context,
            catalog_data.ds_result,
            catalog_data.qf_result,
            catalog_data.cm_result,
        )
        X_back = param_to_rho(
            X_param,
            t,
            l1_context,
            catalog_data.ds_result,
            catalog_data.qf_result,
            catalog_data.cm_result,
        )
        np.testing.assert_allclose(X_back, X_rho, atol=1e-7, err_msg=f"t={t}")


# ---------------------------------------------------------------------------
# LibrationCatalogTransformer（OO 入口）
# ---------------------------------------------------------------------------


def test_transformer_roundtrip(catalog_data):
    """``LibrationCatalogTransformer`` 端到端 OO 往返。"""
    tr = LibrationCatalogTransformer(data=catalog_data)
    rng = np.random.default_rng(505)
    X_rho = 1e-3 * rng.standard_normal(6)
    t = 1.5
    X_param = tr.rho_to_param(X_rho, t)
    X_back = tr.param_to_rho(X_param, t)
    np.testing.assert_allclose(X_back, X_rho, atol=1e-7)
