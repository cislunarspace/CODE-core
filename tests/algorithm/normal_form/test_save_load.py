"""``NormalFormResult`` 序列化（save / load）往返测试。

覆盖：

- :meth:`NormalFormResult.save` 存成 ``.npz``；
- :meth:`NormalFormResult.load` 从 ``.npz`` 重建等价对象；
- W_series 三层嵌套 dict（含 tuple 键 + 复值）正确序列化/反序列化；
- context 重建后标量参数与原一致；
- ``catalog_transformer.rho_to_param`` / ``param_to_rho`` 往返输出一致
  （atol=1e-12）。
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sympy")

from e2m2e.algorithm.dynamics import LibrationPoint
from e2m2e.algorithm.normal_form.catalog import (
    LibrationCatalogData,
    LibrationCatalogTransformer,
)
from e2m2e.algorithm.normal_form.center_manifold import CenterManifoldReducer
from e2m2e.algorithm.normal_form.dynamical_substitution import DynamicalSubstituteResult
from e2m2e.algorithm.normal_form.quasi_floquet import QuasiFloquetResult, real_normal_form_matrix
from e2m2e.algorithm.normal_form.types import NormalFormResult
from e2m2e.data.templates import ConvergenceState, FailureCause

pytestmark = pytest.mark.theory


# ---------------------------------------------------------------------------
# 公共 fixture（沿用 test_catalog.py 的范式）
# ---------------------------------------------------------------------------


@pytest.fixture
def l1_context(earth_moon_system):
    from e2m2e.algorithm.normal_form import NormalFormContext
    from e2m2e.algorithm.normal_form.constants import JD0_J2000

    return NormalFormContext(
        system=earth_moon_system,
        libration_point=LibrationPoint.L1,
        epoch=JD0_J2000,
        order=4,
    )


def _make_ds_result(l1_context, *, n=96, T=3.0):
    tlist = np.linspace(0.0, T, n)
    amp = 1e-3
    A = amp * np.cos(tlist)[:, None] * np.array([[1.0, 0.5, -0.3]])
    Adot = -amp * np.sin(tlist)[:, None] * np.array([[1.0, 0.5, -0.3]])
    B = amp * 0.8 * np.sin(tlist)[:, None] * np.array([[0.2, -0.6, 0.4]])
    Bdot = amp * 0.8 * np.cos(tlist)[:, None] * np.array([[0.2, -0.6, 0.4]])
    Xlist = np.concatenate([A, B], axis=1)

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


def _make_qf_result(l1_context, *, n=96, T=3.0):
    lam = float(l1_context.characteristic_exponent)
    nu1, nu2 = l1_context.central_frequencies
    D = real_normal_form_matrix(lam, float(nu1), float(nu2))
    tlist = np.linspace(0.0, T, n)
    B_samples = np.stack([np.eye(6, dtype=float) for _ in range(n)])
    return QuasiFloquetResult(
        context=l1_context,
        order=int(l1_context.order),
        tlist=tlist,
        B_samples=B_samples,
        D=D,
        method="matrix",
    )


def _make_cm_result(l1_context, qf_result, *, max_order=5, with_terms=True):
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


def _make_normal_form_result(l1_context, *, n=96, T=3.0, max_order=5, with_terms=True):
    ds = _make_ds_result(l1_context, n=n, T=T)
    qf = _make_qf_result(l1_context, n=n, T=T)
    cm = _make_cm_result(l1_context, qf, max_order=max_order, with_terms=with_terms)
    catalog = LibrationCatalogData(
        context=l1_context,
        ds_result=ds,
        qf_result=qf,
        cm_result=cm,
    )
    transformer = LibrationCatalogTransformer(data=catalog)
    return NormalFormResult(
        context=l1_context,
        order=l1_context.order,
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="测试化简完成",
        ds_result=ds,
        qf_result=qf,
        cm_result=cm,
        catalog_transformer=transformer,
    )


# ---------------------------------------------------------------------------
# save / load 基本往返
# ---------------------------------------------------------------------------


def test_save_creates_file(l1_context, tmp_path):
    """save 应创建 .npz 文件。"""
    result = _make_normal_form_result(l1_context)
    path = tmp_path / "test.npz"
    result.save(path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_load_returns_normal_form_result(l1_context, tmp_path):
    """load 应返回 NormalFormResult 实例。"""
    result = _make_normal_form_result(l1_context)
    path = tmp_path / "test.npz"
    result.save(path)
    loaded = NormalFormResult.load(path)
    assert loaded.status is ConvergenceState.CONVERGED
    assert loaded.cause is FailureCause.NONE
    assert loaded.message == "测试化简完成"
    assert isinstance(loaded, NormalFormResult)


# ---------------------------------------------------------------------------
# context 重建
# ---------------------------------------------------------------------------


def test_context_scalar_params_match(l1_context, tmp_path):
    """重建后 context 标量参数与原一致。"""
    result = _make_normal_form_result(l1_context)
    path = tmp_path / "test.npz"
    result.save(path)
    loaded = NormalFormResult.load(path)

    assert loaded.context.mu == pytest.approx(l1_context.mu, abs=1e-15)
    assert pytest.approx(l1_context.LU, abs=1e-10) == loaded.context.LU
    assert pytest.approx(l1_context.TU, abs=1e-10) == loaded.context.TU
    assert loaded.context.epoch == pytest.approx(l1_context.epoch, abs=1e-10)
    assert loaded.context.order == l1_context.order
    assert loaded.context.libration_point == l1_context.libration_point
    np.testing.assert_allclose(
        loaded.context.libration_position, l1_context.libration_position, atol=1e-14
    )


# ---------------------------------------------------------------------------
# W_series 三层嵌套 dict
# ---------------------------------------------------------------------------


def test_w_series_roundtrip(l1_context, tmp_path):
    """W_series 三层嵌套 dict（含 tuple 键 + 复值）正确往返。"""
    result = _make_normal_form_result(l1_context, with_terms=True)
    path = tmp_path / "test.npz"
    result.save(path)
    loaded = NormalFormResult.load(path)

    orig_cm = result.cm_result
    load_cm = loaded.cm_result
    assert orig_cm is not None and load_cm is not None

    # 验证 W_series 结构一致
    assert set(orig_cm.W_series.keys()) == set(load_cm.W_series.keys())
    for step in orig_cm.W_series:
        assert set(orig_cm.W_series[step].keys()) == set(load_cm.W_series[step].keys())
        for order in orig_cm.W_series[step]:
            orig_poly = orig_cm.W_series[step][order]
            load_poly = load_cm.W_series[step][order]
            assert set(orig_poly.keys()) == set(load_poly.keys())
            for pow_tuple in orig_poly:
                np.testing.assert_allclose(
                    np.asarray(load_poly[pow_tuple], dtype=complex),
                    np.asarray(orig_poly[pow_tuple], dtype=complex),
                    atol=1e-14,
                    err_msg=f"W_series[{step!r}][{order}][{pow_tuple}] 不一致",
                )


# ---------------------------------------------------------------------------
# 子结果数组往返
# ---------------------------------------------------------------------------


def test_ds_result_arrays_roundtrip(l1_context, tmp_path):
    """DS 结果数组（tlist、Xlist、W_poly、Wdot_poly）正确往返。"""
    result = _make_normal_form_result(l1_context)
    path = tmp_path / "test.npz"
    result.save(path)
    loaded = NormalFormResult.load(path)

    orig = result.ds_result
    load = loaded.ds_result
    assert orig is not None and load is not None

    np.testing.assert_allclose(load.tlist, orig.tlist, atol=1e-14)
    np.testing.assert_allclose(load.Xlist, orig.Xlist, atol=1e-14)

    for pow_tuple in orig.W_poly:
        np.testing.assert_allclose(load.W_poly[pow_tuple], orig.W_poly[pow_tuple], atol=1e-14)
    for pow_tuple in orig.Wdot_poly:
        np.testing.assert_allclose(load.Wdot_poly[pow_tuple], orig.Wdot_poly[pow_tuple], atol=1e-14)


def test_qf_result_arrays_roundtrip(l1_context, tmp_path):
    """QF 结果数组（tlist、B_samples、D、M_samples）正确往返。"""
    result = _make_normal_form_result(l1_context)
    path = tmp_path / "test.npz"
    result.save(path)
    loaded = NormalFormResult.load(path)

    orig = result.qf_result
    load = loaded.qf_result
    assert orig is not None and load is not None

    np.testing.assert_allclose(load.tlist, orig.tlist, atol=1e-14)
    np.testing.assert_allclose(load.B_samples, orig.B_samples, atol=1e-14)
    np.testing.assert_allclose(load.D, orig.D, atol=1e-14)
    assert load.method == orig.method


def test_cm_result_hamiltonian_terms_roundtrip(l1_context, tmp_path):
    """CM hamiltonian_terms 正确往返。"""
    result = _make_normal_form_result(l1_context, with_terms=True)
    path = tmp_path / "test.npz"
    result.save(path)
    loaded = NormalFormResult.load(path)

    orig = result.cm_result
    load = loaded.cm_result
    assert orig is not None and load is not None

    assert set(orig.hamiltonian_terms.keys()) == set(load.hamiltonian_terms.keys())
    for pow_tuple in orig.hamiltonian_terms:
        np.testing.assert_allclose(
            load.hamiltonian_terms[pow_tuple],
            orig.hamiltonian_terms[pow_tuple],
            atol=1e-14,
        )


# ---------------------------------------------------------------------------
# catalog_transformer 往返（核心验收标准）
# ---------------------------------------------------------------------------


def test_transformer_rho_to_param_matches_original(l1_context, tmp_path):
    """save → load 后 catalog_transformer.rho_to_param 输出与原一致（atol=1e-12）。"""
    result = _make_normal_form_result(l1_context, with_terms=True)
    path = tmp_path / "test.npz"
    result.save(path)
    loaded = NormalFormResult.load(path)

    assert result.catalog_transformer is not None
    assert loaded.catalog_transformer is not None

    rng = np.random.default_rng(42)
    X_rho = 1e-3 * rng.standard_normal(6)
    t = 1.23

    orig_param = result.catalog_transformer.rho_to_param(X_rho, t)
    load_param = loaded.catalog_transformer.rho_to_param(X_rho, t)
    np.testing.assert_allclose(load_param, orig_param, atol=1e-12)


def test_transformer_param_to_rho_matches_original(l1_context, tmp_path):
    """save → load 后 catalog_transformer.param_to_rho 输出与原一致（atol=1e-12）。"""
    result = _make_normal_form_result(l1_context, with_terms=True)
    path = tmp_path / "test.npz"
    result.save(path)
    loaded = NormalFormResult.load(path)

    assert result.catalog_transformer is not None
    assert loaded.catalog_transformer is not None

    rng = np.random.default_rng(77)
    X_param = 1e-3 * rng.standard_normal(6)
    X_param[2] += 0.05  # I2 非零
    X_param[4] += 0.03  # I3 非零
    t = 0.8

    orig_rho = result.catalog_transformer.param_to_rho(X_param, t)
    load_rho = loaded.catalog_transformer.param_to_rho(X_param, t)
    np.testing.assert_allclose(load_rho, orig_rho, atol=1e-12)


# ---------------------------------------------------------------------------
# 端到端：save → load → rho_to_param → param_to_rho 往返
# ---------------------------------------------------------------------------


def test_end_to_end_roundtrip_after_save_load(l1_context, tmp_path):
    """save → load 后端到端 rho ↔ param 往返在 1e-7 内。"""
    result = _make_normal_form_result(l1_context, with_terms=True)
    path = tmp_path / "test.npz"
    result.save(path)
    loaded = NormalFormResult.load(path)

    assert loaded.catalog_transformer is not None

    rng = np.random.default_rng(999)
    X_rho = 1e-3 * rng.standard_normal(6)
    t = 1.5
    X_param = loaded.catalog_transformer.rho_to_param(X_rho, t)
    X_back = loaded.catalog_transformer.param_to_rho(X_param, t)
    np.testing.assert_allclose(X_back, X_rho, atol=1e-7)


# ---------------------------------------------------------------------------
# 无子结果时 save/load 不报错
# ---------------------------------------------------------------------------


def test_save_load_without_sub_results(l1_context, tmp_path):
    """无子结果的 NormalFormResult 也能正常 save/load。"""
    result = NormalFormResult(context=l1_context, order=l1_context.order)
    path = tmp_path / "test.npz"
    result.save(path)
    loaded = NormalFormResult.load(path)
    assert loaded.ds_result is None
    assert loaded.qf_result is None
    assert loaded.cm_result is None
    assert loaded.catalog_transformer is None


# ---------------------------------------------------------------------------
# ds_result 的 fft_components 序列化
# ---------------------------------------------------------------------------


def test_fft_components_roundtrip(l1_context, tmp_path):
    """fft_components 正确往返（含 tuple coef 字段）。"""
    from e2m2e.algorithm.normal_form.fft import FFTComponent

    ds = _make_ds_result(l1_context)
    # 注入 fft_components
    fft_comps = {
        "x": [
            FFTComponent(freq=0.1, amp_s=0.01, amp_c=0.02, amp=0.022),
            FFTComponent(freq=0.5, amp_s=0.0, amp_c=0.03, amp=0.03),
        ],
        "y": [FFTComponent(freq=0.2, amp_s=0.0, amp_c=0.01, amp=0.01)],
        "z": [],
    }
    ds_with_fft = DynamicalSubstituteResult(
        context=l1_context,
        order=int(l1_context.order),
        substitute_orbit=ds.substitute_orbit,
        tlist=ds.tlist,
        Xlist=ds.Xlist,
        W_poly=ds.W_poly,
        Wdot_poly=ds.Wdot_poly,
        fft_components=fft_comps,
    )
    qf = _make_qf_result(l1_context)
    cm = _make_cm_result(l1_context, qf, with_terms=False)
    catalog = LibrationCatalogData(
        context=l1_context,
        ds_result=ds_with_fft,
        qf_result=qf,
        cm_result=cm,
    )
    result = NormalFormResult(
        context=l1_context,
        order=l1_context.order,
        ds_result=ds_with_fft,
        qf_result=qf,
        cm_result=cm,
        catalog_transformer=LibrationCatalogTransformer(data=catalog),
    )
    path = tmp_path / "test.npz"
    result.save(path)
    loaded = NormalFormResult.load(path)

    assert set(loaded.ds_result.fft_components.keys()) == {"x", "y", "z"}
    assert len(loaded.ds_result.fft_components["x"]) == 2
    assert len(loaded.ds_result.fft_components["y"]) == 1
    assert len(loaded.ds_result.fft_components["z"]) == 0
    np.testing.assert_allclose(loaded.ds_result.fft_components["x"][0].freq, 0.1, atol=1e-14)


# ---------------------------------------------------------------------------
# 持久化格式迁移
# ---------------------------------------------------------------------------


def test_load_rejects_old_format_without_status_fields(l1_context, tmp_path):
    """旧 NPZ 缺少状态三元组时必须要求重新计算。"""
    path = tmp_path / "legacy.npz"
    np.savez(
        path,
        _fmt_version=np.array(1),
        _ctx_system=np.frombuffer(b"CR3BP_System", dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="旧格式"):
        NormalFormResult.load(path)


def test_load_rejects_missing_status_fields(l1_context, tmp_path):
    """标称新格式缺状态字段时必须给出明确迁移错误。"""
    path = tmp_path / "incomplete.npz"
    np.savez(path, _fmt_version=np.array(2))

    with pytest.raises(ValueError, match="缺少必需字段"):
        NormalFormResult.load(path)
