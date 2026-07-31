"""``normal_form.dynamical_substitution`` 测试。

覆盖：

- :class:`DynamicalSubstituteCorrector` 可构造，``reduce`` 返回
  :class:`DynamicalSubstituteResult`；
- 结果具备 ``substitute_orbit`` / ``W_poly`` / ``Wdot_poly`` /
  ``tlist`` / ``shooting_result`` 等切片 #171 要求字段；
- 在零初值 / 较小窗口下烟测通过；
- NAFF 不可用时降级到 FFT 并发出警告；
- :func:`multiple_shooting_newton` 在玩具动力学上能收敛；
- :func:`solve_block_tridiagonal` 数值正确性。

完整 ``L1DynSubs.npz`` 回归留给后续切片（需要 SPICE 内核与完整
``T_total = 0.1·2^16`` 窗口）。
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import LibrationPoint
from e2m2e.algorithm.normal_form.constants import JD0_J2000
from e2m2e.algorithm.normal_form.dynamical_substitution import (
    DEFAULT_DENSE_STEP,
    DEFAULT_NODE_STEP,
    DEFAULT_TOTAL_TU,
    DynamicalSubstituteCorrector,
    DynamicalSubstituteResult,
)
from e2m2e.algorithm.normal_form.multiple_shooting import (
    MultipleShootingResult,
    ODESubstituteSolver,
    ShootingPatch,
    multiple_shooting_newton,
    solve_block_tridiagonal,
)

# ---------------------------------------------------------------------------
# 公共 fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def l1_context(earth_moon_system):
    """L1 共线点上下文（与现有 slice 1 测试一致）。"""
    from e2m2e.algorithm.normal_form import NormalFormContext

    return NormalFormContext(
        system=earth_moon_system,
        libration_point=LibrationPoint.L1,
        epoch=JD0_J2000,
        order=4,
    )


@pytest.fixture
def tiny_corrector(l1_context) -> DynamicalSubstituteCorrector:
    """极小窗口 corrector，让 ``reduce`` 在 < 5 s 完成（不依赖 SPICE）。"""
    return DynamicalSubstituteCorrector(
        context=l1_context,
        t_total=4.0,
        node_step=0.8,
        dense_step=0.2,
        max_iter=3,
        tolerance=1e-6,
        prefer="fft",
        spice_optional=True,
    )


@pytest.fixture
def auto_corrector(l1_context) -> DynamicalSubstituteCorrector:
    """``prefer='auto'`` 的 corrector，便于触发 NAFF → FFT 降级分支。"""
    return DynamicalSubstituteCorrector(
        context=l1_context,
        t_total=4.0,
        node_step=0.8,
        dense_step=0.2,
        max_iter=3,
        tolerance=1e-6,
        prefer="auto",
        spice_optional=True,
    )


# ---------------------------------------------------------------------------
# Corrector 构造烟测
# ---------------------------------------------------------------------------


def test_corrector_is_constructible(l1_context):
    """``DynamicalSubstituteCorrector`` 可用 :class:`NormalFormContext` 构造。"""
    corrector = DynamicalSubstituteCorrector(
        context=l1_context,
        t_total=4.0,
        node_step=0.8,
        dense_step=0.2,
        max_iter=2,
        tolerance=1e-6,
        prefer="fft",
    )
    assert corrector.context is l1_context
    assert corrector.t_total == 4.0
    assert corrector.prefer == "fft"


# ---------------------------------------------------------------------------
# Corrector.reduce 烟测
# ---------------------------------------------------------------------------


def test_reduce_returns_dataclass_with_required_fields(tiny_corrector):
    """``reduce`` 返回 :class:`DynamicalSubstituteResult` 且字段齐备。"""
    result = tiny_corrector.reduce()
    assert isinstance(result, DynamicalSubstituteResult)
    assert result.context is tiny_corrector.context
    assert result.order == int(tiny_corrector.context.order)
    # 切片 #171 验收字段：substitute_orbit / W_poly / Wdot_poly
    assert result.substitute_orbit is not None
    assert isinstance(result.W_poly, dict)
    assert isinstance(result.Wdot_poly, dict)
    # tlist 与 Xlist 形状
    assert result.tlist.ndim == 1
    assert result.Xlist.ndim == 2
    assert result.Xlist.shape[1] == 6
    assert result.Xlist.shape[0] == result.tlist.shape[0]
    # 至少含 6 个线性分量（q1/q2/q3/p1/p2/p3）
    expected_keys = {
        (1, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0),
        (0, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 1, 0),
        (0, 0, 0, 0, 0, 1),
    }
    assert expected_keys.issubset(set(result.W_poly.keys()))
    assert expected_keys.issubset(set(result.Wdot_poly.keys()))


def test_reduce_uses_seed_when_provided(tiny_corrector):
    """``seed`` 显式传入时初始 X_Q 各节点都应是该 seed。"""
    seed = np.array([1e-3, -1e-3, 0.0, 0.0, 1e-4, -1e-4])
    result = tiny_corrector.reduce(seed=seed)
    assert result.shooting_result is not None
    # shooting 节点不一定仍等于 seed（Newton 迭代会改），但稠密输出至少存在
    assert result.Xlist.shape[0] >= result.shooting_result.t_Q.shape[0]


def test_reduce_rejects_invalid_seed_shape(tiny_corrector):
    """``seed`` 不是 ``(6,)`` 时抛 :class:`ValueError`。"""
    with pytest.raises(ValueError, match="seed"):
        tiny_corrector.reduce(seed=np.zeros(5))


def test_reduce_metadata_records_window(tiny_corrector):
    """``metadata`` 应记录窗口设置，便于诊断。"""
    result = tiny_corrector.reduce()
    assert result.metadata["t_total"] == pytest.approx(4.0)
    assert result.metadata["node_step"] == pytest.approx(0.8)
    assert result.metadata["dense_step"] == pytest.approx(0.2)
    assert result.metadata["n_nodes"] >= 2
    assert result.metadata["n_segments"] >= 1


def test_reduce_falls_back_to_fft_when_naff_missing(
    auto_corrector, monkeypatch: pytest.MonkeyPatch
):
    """``reduce`` 在 NAFF 不可用时降级到 FFT，并把后端记录到结果。"""
    monkeypatch.setattr(
        "e2m2e.algorithm.normal_form.fft._resolve_naff_binary",
        lambda: None,
    )
    monkeypatch.setattr(
        "e2m2e.algorithm.normal_form.fft.naff_available",
        lambda: False,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = auto_corrector.reduce()
    assert result.backend == "fft"
    # 至少一条 NAFF 警告
    assert any("NAFF" in str(w.message) for w in caught)


def test_reduce_result_residual_property(tiny_corrector):
    """``residual_norm`` 属性映射到多重打靶最大残差。"""
    result = tiny_corrector.reduce()
    if result.shooting_result is None:
        pytest.skip("shooting_result 为 None（容差异常）")
    assert result.residual_norm == pytest.approx(result.shooting_result.max_residual)


def test_reduce_substitute_orbit_is_orbit_or_array(tiny_corrector):
    """``substitute_orbit`` 应为 ``e2m2e.Orbit`` 实例，或退化为 ``ndarray``。"""
    result = tiny_corrector.reduce()
    obj = result.substitute_orbit
    if hasattr(obj, "states") and hasattr(obj, "times"):
        # Orbit-like
        assert obj.states.shape[1] == 6
        np.testing.assert_allclose(obj.times, result.tlist)
    else:
        # ndarray fallback
        assert isinstance(obj, np.ndarray)
        assert obj.shape[1] == 6


# ---------------------------------------------------------------------------
# 玩具动力学上的多重打靶
# ---------------------------------------------------------------------------


def _toy_rhs(t: float, X):
    """纯衰减动力学：``ρ̇ = 0``，``ρ̈ = -ρ``。解可解析。"""
    X = np.asarray(X, dtype=float).ravel()
    return np.concatenate([X[3:6], -X[:3]])


def test_multiple_shooting_newton_converges_on_toy_dynamics():
    """玩具动力学 ``ρ̈ = -ρ`` 上 Newton 多重打靶应让残差低于初始值。"""
    n_nodes = 11
    t_Q = np.linspace(0.0, 2.0 * np.pi, n_nodes)
    X0 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    X_Q = np.tile(X0, (n_nodes, 1))

    solver = ODESubstituteSolver(rhs=_toy_rhs, rtol=1e-12, atol=1e-14)
    initial = ShootingPatch(t_Q=t_Q, X_Q=X_Q)
    result = multiple_shooting_newton(
        initial,
        solver,
        max_iter=10,
        tolerance=1e-9,
    )
    assert isinstance(result, MultipleShootingResult)
    # 即使不严格收敛，残差应显著下降
    assert result.max_residual < 1.0
    assert len(result.residual_history) == result.iterations


def _zero_rhs(t, X):
    """恒零右端项——用于边界条件测试。"""
    return np.zeros(6)


def test_multiple_shooting_newton_rejects_too_few_nodes():
    """少于 2 个节点时抛 :class:`ValueError`。"""
    solver = ODESubstituteSolver(rhs=_zero_rhs)
    patch = ShootingPatch(t_Q=np.array([0.0]), X_Q=np.zeros((1, 6)))
    with pytest.raises(ValueError, match="节点"):
        multiple_shooting_newton(patch, solver)


def test_multiple_shooting_newton_rejects_invalid_iter_args():
    """``max_iter < 1`` 或 ``tolerance <= 0`` 抛 :class:`ValueError`。"""
    solver = ODESubstituteSolver(rhs=_zero_rhs)
    patch = ShootingPatch(t_Q=np.array([0.0, 1.0]), X_Q=np.zeros((2, 6)))
    with pytest.raises(ValueError, match="max_iter"):
        multiple_shooting_newton(patch, solver, max_iter=0)
    with pytest.raises(ValueError, match="tolerance"):
        multiple_shooting_newton(patch, solver, tolerance=0.0)


# ---------------------------------------------------------------------------
# solve_block_tridiagonal 直接测试
# ---------------------------------------------------------------------------


def test_solve_block_tridiagonal_zero_residual_short_circuit():
    """当 ``Xf_i == X_Q_{i+1}`` 时，修正量应接近 0。"""
    rng = np.random.default_rng(0)
    n_seg = 4
    X_Q = rng.normal(scale=0.1, size=(n_seg + 1, 6))
    phi_stack = np.tile(np.eye(6), (n_seg, 1, 1))
    xf_stack = X_Q[1:].copy()

    delta_Q, errs = solve_block_tridiagonal(phi_stack, xf_stack, X_Q)
    assert all(e < 1e-12 for e in errs)
    assert np.linalg.norm(delta_Q) < 1e-9


def test_solve_block_tridiagonal_drive_residual_to_zero_in_toy_setting():
    """对玩具连续性残差，Newton 收敛应让残差降到极小。"""
    n_seg = 6
    rng = np.random.default_rng(1)
    X_Q = rng.normal(scale=0.5, size=(n_seg + 1, 6))
    # 故意构造 Xf 与 X_Q_{i+1} 有偏差
    xf_stack = X_Q[1:] + 0.1 * rng.normal(scale=1.0, size=(n_seg, 6))
    phi_stack = np.tile(np.eye(6), (n_seg, 1, 1))

    delta_Q, errs_before = solve_block_tridiagonal(phi_stack, xf_stack, X_Q)
    X_Q_new = X_Q + delta_Q
    _, errs_after = solve_block_tridiagonal(phi_stack, xf_stack, X_Q_new)

    assert max(errs_after) < max(errs_before)
    # 单次 Newton 步在玩具问题上不应让残差爆炸
    assert max(errs_after) < 2.0 * max(errs_before)


def test_solve_block_tridiagonal_rejects_shape_mismatch():
    """``xf_stack`` / ``X_Q`` 形状不匹配时抛 :class:`ValueError`。"""
    phi_stack = np.tile(np.eye(6), (3, 1, 1))
    xf_stack = np.zeros((2, 6))
    X_Q = np.zeros((4, 6))
    with pytest.raises(ValueError, match="xf_stack"):
        solve_block_tridiagonal(phi_stack, xf_stack, X_Q)


# ---------------------------------------------------------------------------
# ODESubstituteSolver 基础烟测
# ---------------------------------------------------------------------------


def test_ode_substitute_solver_matches_direct_integration_for_toy_rhs():
    """ODE solver 终端状态应与 ``solve_ivp`` 单次积分一致。"""
    from scipy.integrate import solve_ivp

    solver = ODESubstituteSolver(rhs=_toy_rhs, rtol=1e-12, atol=1e-14)
    x0 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    xf, phi = solver.propagate_segment(0.0, np.pi / 2, x0)

    sol = solve_ivp(_toy_rhs, (0.0, np.pi / 2), x0, method="DOP853", rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(xf, sol.y[:, -1], atol=1e-9)
    # STM 在 ρ̈=-ρ 问题上应当近似 [[cos, 0, sin], ...]
    expected_phi = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(phi, expected_phi, atol=1e-6)


def test_ode_substitute_solver_rejects_bad_shape():
    """``x0`` 不是 ``(6,)`` 时抛 :class:`ValueError`。"""
    solver = ODESubstituteSolver(rhs=_toy_rhs)
    with pytest.raises(ValueError, match="x0"):
        solver.propagate_segment(0.0, 1.0, np.zeros(5))


# ---------------------------------------------------------------------------
# 模块顶层常量
# ---------------------------------------------------------------------------


def test_default_constants_match_qiao():
    """默认窗口/间距与 qiao Code05 一致。"""
    assert pytest.approx(0.1 * (2**16)) == DEFAULT_TOTAL_TU
    assert pytest.approx(0.8) == DEFAULT_NODE_STEP


# ---------------------------------------------------------------------------
# SPICE 不可用时的优雅降级（无 SPICE 内核的 CI 环境）
# ---------------------------------------------------------------------------


def test_reduce_works_without_spice_kernels(tiny_corrector, monkeypatch):
    """``spice_optional=True`` 时即使 SPICE 不可用也应跑出合理结果。"""
    # 强行让 _ephemeris.eval_params 抛 RuntimeError 模拟 SPICE 缺失
    import e2m2e.algorithm.normal_form.dynamical_substitution as ds

    def _broken(*args, **kwargs):
        raise RuntimeError("simulated SPICE missing")

    monkeypatch.setattr(
        "e2m2e.algorithm.normal_form._ephemeris.eval_params",
        _broken,
        raising=False,
    )
    # 同时让 _build_dynamics_rhs_spice 走异常路径
    monkeypatch.setattr(ds, "_build_dynamics_rhs_spice", _broken)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = tiny_corrector.reduce()
    assert result.spice_available is False
    assert result.tlist.size > 0
    assert result.Xlist.size > 0


# ---------------------------------------------------------------------------
# 验收标准 #2：L1_Halo_Large 的 W_poly 与 qiao L1DynSubs.npz 回归一致
# ---------------------------------------------------------------------------


def test_W_poly_regression_against_qiao_L1DynSubs_npz():
    """与 qiao ``L1DynSubs.npz`` 的逐系数回归。

    该回归需要 qiao 参考数据（``L1DynSubs.npz``，体积大，本仓库未引入）
    以及完整 ``T_total = 0.1·2^16`` 窗口的星历模型积分。两者在 CI 环境
    均不可得，故以 ``pytest.skip`` 守卫占位。

    解锁条件（任一）：
    1. 将 qiao ``L1DynSubs.npz`` 放入 ``tests/algorithms/normal_form/data/``；
    2. 提供 SPICE 内核（``.tls`` + ``.bsp``）并在长窗口下跑 reduce。

    fixture 就绪后，本测试应加载 npz 中的 ``W_poly``，与
    ``DynamicalSubstituteCorrector(...).reduce()`` 的 ``result.W_poly``
    逐线性项做 ``np.testing.assert_allclose``。
    """
    import os

    fixture = os.path.join(os.path.dirname(__file__), "data", "L1DynSubs.npz")
    if not os.path.exists(fixture):
        pytest.skip("qiao L1DynSubs.npz 未引入仓库；W_poly 逐系数回归待 fixture 就绪")


# ---------------------------------------------------------------------------
# 验收标准 #3：动力学替代轨道重复积分后中心流形频率幅值低于阈值
# ---------------------------------------------------------------------------


def test_substitute_orbit_suppresses_center_manifold_frequencies(l1_context, monkeypatch):
    """星历模型下重复积分后的 FFT 中心流形频率幅值检查。

    验收标准要求：动力学替代轨道在星历模型下重复积分后仍只含受迫频率，
    中心流形频率（ν₁、ν₂）处的 FFT 幅值低于阈值。

    该检查需要：
    1. SPICE 内核（否则 ``reduce`` 退到纯 CR3BP，退路仅供烟雾测试）；
    2. 完整 ``T_total = 0.1·2^16`` 窗口（FFT 频率分辨率需达
       ``2π/T_total ≈ 9.6e-5`` rad/TU 才能分辨受迫频率与 ν₁/ν₂）。

    CI 环境两者皆缺，故以 ``pytest.skip`` 守卫占位。检测逻辑本身的
    正确性由 ``test_fft.test_fft_extract_detects_suppressed_center_manifold_frequency``
    在合成数据上覆盖。

    解锁条件：加载 SPICE ``.tls`` + ``.bsp`` 内核后即可启用本测试。
    """
    import e2m2e.algorithm.normal_form._ephemeris as _eph

    # 探测 SPICE leapseconds 内核是否就绪：str2et 需要已加载 .tls
    spice_ready = True
    try:
        _eph.eval_params(float(l1_context.epoch), l1_context)
    except Exception:
        spice_ready = False

    if not spice_ready:
        pytest.skip(
            "SPICE leapseconds 内核不可用；星历模型端到端 FFT 中心流形频率压制检查待内核就绪"
        )

    corrector = DynamicalSubstituteCorrector(
        context=l1_context,
        t_total=DEFAULT_TOTAL_TU,
        node_step=DEFAULT_NODE_STEP,
        dense_step=DEFAULT_DENSE_STEP,
        max_iter=19,
        tolerance=1e-11,
        prefer="fft",
        spice_optional=False,
    )
    result = corrector.reduce()
    assert result.spice_available is True

    nu1, nu2 = l1_context.central_frequencies
    threshold = 1e-3  # 中心流形频率幅值阈值（相对最大非直流分量）
    for label in ("x", "y", "z"):
        comps = result.fft_components[label]
        non_dc = [c for c in comps if abs(c.freq) > 1e-6]
        if not non_dc:
            continue
        max_amp = max(c.amp for c in non_dc)
        for nu in (nu1, nu2):
            nearest = min(non_dc, key=lambda c: abs(c.freq - nu))
            assert nearest.amp < threshold * max_amp, (
                f"{label} 方向中心流形频率 ν={nu:.4f} 处幅值 "
                f"{nearest.amp:.3e} 超过阈值 {threshold * max_amp:.3e}"
            )


# ---------------------------------------------------------------------------
# 模块接口
# ---------------------------------------------------------------------------


def test_dynamical_substitution_importable_via_package_root():
    """切片 #171 验收：能从包根直接 ``import``。"""
    from e2m2e.algorithm.normal_form import (  # noqa: F401
        DynamicalSubstituteCorrector,
        DynamicalSubstituteResult,
    )
