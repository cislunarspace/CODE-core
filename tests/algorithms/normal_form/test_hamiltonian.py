"""``normal_form.hamiltonian`` 测试。"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

# sympy 是 normal-form optional dep；未安装时整个文件 skip（不 error）。
pytest.importorskip("sympy")

from e2m2e.algorithms.normal_form import NormalFormContext
from e2m2e.algorithms.normal_form.constants import JD0_J2000
from e2m2e.algorithms.normal_form.hamiltonian import (
    DYNAMIC_PARAM_NAMES,
    Hamiltonian,
    build_hamiltonian,
    evaluate_hamiltonian,
    hamiltonian_constant_term,
)
from e2m2e.algorithms.normal_form.legendre import expand_legendre_1_over_r
from e2m2e.core import LibrationPoint

# ---------------------------------------------------------------------------
# 公共 fixture：可复用的上下文与 Legendre 展开
# ---------------------------------------------------------------------------

_QIAO_FIXTURE_L1 = os.path.join(
    "/home/ouyangjiahong/codes/qiao",
    "Results/Result_HamiltonFunc/1_Ephemeris_Model/L1_EM_Hamilton.mat",
)


@pytest.fixture
def l1_context(earth_moon_system) -> NormalFormContext:
    """L1 共线点 :class:`NormalFormContext`（不含 SPICE 依赖）。"""
    return NormalFormContext(
        system=earth_moon_system,
        libration_point=LibrationPoint.L1,
        epoch=JD0_J2000,
        order=4,
    )


@pytest.fixture
def l1_legendre():
    """L1 共线点 4 阶 Legendre 展开——评估时间 < 1 s。"""
    return expand_legendre_1_over_r(4)


@pytest.fixture
def l1_hamiltonian(l1_context, l1_legendre) -> Hamiltonian:
    """符号 Hamilton 量（仅 sympy 系数，未数值化）。"""
    return build_hamiltonian(l1_context, l1_legendre, max_degree=4)


# ---------------------------------------------------------------------------
# SPICE 与 qiao fixture 可用性检测
# ---------------------------------------------------------------------------

_SPICE_KERNEL_DIR = os.environ.get(
    "SPICE_KERNEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "kernels"),
)


def _has_spice_kernels() -> bool:
    """检查 SPICE ``.tls`` + ``.bsp`` 都可用。"""
    if not os.path.isdir(_SPICE_KERNEL_DIR):
        return False
    has_tls = any(f.endswith(".tls") for f in os.listdir(_SPICE_KERNEL_DIR))
    has_bsp = any(f.endswith(".bsp") for f in os.listdir(_SPICE_KERNEL_DIR))
    return has_tls and has_bsp


_has_spice = _has_spice_kernels()
_has_qiao_fixture = os.path.exists(_QIAO_FIXTURE_L1)

requires_spice = pytest.mark.skipif(
    not _has_spice,
    reason="SPICE kernels (.tls + .bsp) not available",
)

requires_qiao = pytest.mark.skipif(
    not (_has_spice and _has_qiao_fixture),
    reason="SPICE kernels or qiao L1_EM_Hamilton.mat not available",
)


@pytest.fixture
def spice_loaded():
    """加载 leapseconds + 任一可用 SPK；不可用时跳过。"""
    if not _has_spice:
        pytest.skip("SPICE kernels not available")
    import spiceypy as spice

    for fname in ("naif0012.tls", "naif0011.tls"):
        path = os.path.join(_SPICE_KERNEL_DIR, fname)
        if os.path.exists(path):
            spice.furnsh(path)
    for fname in ("de440.bsp", "de440s.bsp", "de438.bsp", "de435.bsp", "de430.bsp"):
        path = os.path.join(_SPICE_KERNEL_DIR, fname)
        if os.path.exists(path):
            spice.furnsh(path)
            return path
    pytest.skip("No SPICE ephemeris kernel (.bsp) found")


# ---------------------------------------------------------------------------
# 构造烟测（纯符号，不需 SPICE）
# ---------------------------------------------------------------------------


def test_build_hamiltonian_returns_dataclass(l1_context, l1_legendre):
    """``build_hamiltonian`` 必须返回 :class:`Hamiltonian`。"""
    h = build_hamiltonian(l1_context, l1_legendre, max_degree=4)
    assert isinstance(h, Hamiltonian)
    assert isinstance(h.coefficients, dict)
    assert h.n_terms > 0
    assert h.max_degree == 4
    assert not h.is_evaluated


def test_build_hamiltonian_powers_match_coefficients(l1_hamiltonian):
    """幂次向量数组与系数 dict 一一对应。"""
    assert l1_hamiltonian.n_terms == len(l1_hamiltonian.coefficients)
    keys_in_dict = set(l1_hamiltonian.coefficients.keys())
    keys_in_array = {tuple(int(p) for p in row) for row in l1_hamiltonian.powers}
    assert keys_in_dict == keys_in_array


def test_build_hamiltonian_max_degree_filters_terms(l1_context, l1_legendre):
    """阶数截断必须生效。"""
    h = build_hamiltonian(l1_context, l1_legendre, max_degree=2)
    for pow_tuple in h.coefficients:
        assert sum(int(p) for p in pow_tuple) <= 2


def test_build_hamiltonian_rejects_invalid_max_degree(l1_context, l1_legendre):
    """``max_degree < 1`` 报 :class:`ValueError`。"""
    with pytest.raises(ValueError, match="max_degree"):
        build_hamiltonian(l1_context, l1_legendre, max_degree=0)


def test_build_hamiltonian_sources_are_recorded(l1_hamiltonian):
    """``store_sources=True`` 时输出含 6 块组成的诊断 dict。"""
    sources = l1_hamiltonian.sources
    expected_keys = {
        "force",
        "kinetic",
        "coriolis",
        "centrifugal",
        "pot_earth",
        "pot_moon",
        "pot_sun",
    }
    assert expected_keys.issubset(set(sources.keys()))


def test_hamiltonian_contains_kinetic_diagonal_terms(l1_hamiltonian):
    """动能 ½‖p‖² 对角项必为 ½。"""
    diag = (
        l1_hamiltonian.coefficients[(0, 0, 0, 2, 0, 0)],
        l1_hamiltonian.coefficients[(0, 0, 0, 0, 2, 0)],
        l1_hamiltonian.coefficients[(0, 0, 0, 0, 0, 2)],
    )
    for c in diag:
        assert c == pytest.approx(0.5, abs=1e-12)


def test_dynamic_param_names_cover_eval_params_keys():
    """动态参数名集合至少包含 Eval_expr 主要项。"""
    needed = {
        "Cpq1",
        "Cpq5",
        "Cqq5",
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
    }
    assert needed.issubset(set(DYNAMIC_PARAM_NAMES))


# ---------------------------------------------------------------------------
# 数值化与已知值（需要 SPICE 内核）
# ---------------------------------------------------------------------------


@requires_spice
def test_evaluate_hamiltonian_runs(spice_loaded, l1_hamiltonian, l1_context):
    """对一组时刻求值 Hamilton 量，运行无异常。"""
    times = np.linspace(0.0, 5.0, 6)
    evaled = evaluate_hamiltonian(l1_hamiltonian, times, l1_context)
    assert evaled.is_evaluated
    assert isinstance(evaled.coefficients, np.ndarray)
    assert evaled.coefficients.shape == (len(times), evaled.n_terms)
    # 常数项应在 J2000 附近 ≈ -862.5
    target = (0, 0, 0, 0, 0, 0)
    for j in range(evaled.n_terms):
        if tuple(int(p) for p in evaled.powers[j]) == target:
            assert evaled.coefficients[0, j] == pytest.approx(-862.5, abs=0.5)
            break
    else:
        pytest.fail("常数项 (0,0,0,0,0,0) 不在 evaled.powers 中")


@requires_spice
def test_hamiltonian_constant_term_matches_qiao_value(spice_loaded, l1_hamiltonian, l1_context):
    """L1 Hamilton 常数项与 qiao ``L1_EM_Hamilton.mat`` 在 t=0 一致。"""
    times = np.array([0.0])
    h0 = hamiltonian_constant_term(l1_hamiltonian, times, l1_context)
    assert h0[0] == pytest.approx(-862.50648692, abs=1e-6)


@requires_qiao
def test_evaluate_hamiltonian_against_qiao_fixture(spice_loaded, l1_context, l1_legendre):
    """与 qiao ``L1_EM_Hamilton.mat`` 在 t=0 处逐项比对。

    qiao fixture 在 N=15 上有 687 项；本测试只用 order=4，因此 qiao 中
    高阶项不在我们这边。匹配率衡量我们 order=4 内能重现的那部分。
    """
    import scipy.io as sio

    qiao_data = sio.loadmat(_QIAO_FIXTURE_L1, squeeze_me=False)
    qiao_h_poly = qiao_data["H_poly"]

    qiao_lookup: dict[tuple[int, ...], float] = {}
    for i in range(qiao_h_poly.shape[0]):
        pv = qiao_h_poly[i, 0].ravel().astype(int)
        col = qiao_h_poly[i, 1].ravel()
        qiao_lookup[tuple(int(x) for x in pv)] = float(col[0])

    h = build_hamiltonian(l1_context, l1_legendre, max_degree=4)
    times = np.array([0.0])
    evaled = evaluate_hamiltonian(h, times, l1_context)
    arr = evaled.coefficients
    our_lookup = {
        tuple(int(x) for x in evaled.powers[j]): float(arr[0, j]) for j in range(evaled.n_terms)
    }

    n_match = 0
    n_total = 0
    for k, v in qiao_lookup.items():
        ours = our_lookup.get(k)
        if ours is None:
            continue
        n_total += 1
        if abs(v) < 1e-12 and abs(ours) < 1e-12:
            n_match += 1
            continue
        tol = max(1e-7, abs(v) * 1e-6)
        if abs(ours - v) < tol:
            n_match += 1

    # order=4 至少匹配 30 项；qiao N=15 多出来的 600+ 项不在我们这边
    assert n_total >= 30, f"我们的项数太少：{n_total}（qiao={len(qiao_lookup)}）"
    assert n_match >= n_total - 5, f"匹配数 {n_match}/{n_total}"


@requires_spice
def test_evaluate_hamiltonian_time_series_bounded(spice_loaded, l1_hamiltonian, l1_context):
    """时间序列的常数项 H_0(t) 在 [0, 10] TU 内单调有界（不应发散）。"""
    times = np.linspace(0.0, 10.0, 11)
    h0 = hamiltonian_constant_term(l1_hamiltonian, times, l1_context)
    assert h0.min() > -1100.0
    assert h0.max() < -700.0
    assert h0.std() < 50.0


@requires_spice
def test_build_evaluate_l1_order4_within_timeout(spice_loaded, l1_context):
    """构造 + 一次 evaluate 不应超过 30 s（SPICE 内核已加载）。"""
    leg = expand_legendre_1_over_r(4)
    t0 = time.time()
    h = build_hamiltonian(l1_context, leg, max_degree=4)
    _ = evaluate_hamiltonian(h, np.linspace(0.0, 5.0, 5), l1_context)
    elapsed = time.time() - t0
    assert elapsed < 30.0, f"耗时 {elapsed:.1f} s 过长"


# ---------------------------------------------------------------------------
# 不需要 SPICE 的解析/接口测试
# ---------------------------------------------------------------------------


def test_evaluate_hamiltonian_over_invalid_times_fails(l1_hamiltonian):
    """非一维 ``times`` 必须报 :class:`ValueError`。"""
    bad_times = np.zeros((3, 3))
    with pytest.raises(ValueError, match="times"):
        evaluate_hamiltonian(l1_hamiltonian, bad_times, None)  # type: ignore[arg-type]


def test_polylist_simplify_zero_epsilon_drops_only_zero_columns():
    """``polylist_simplify`` 在 ``eps=0`` 仍保留所有非零列。"""
    from e2m2e.algorithms.normal_form.polynomial import polylist_simplify

    powers = {
        (0, 0, 0, 0, 0, 0): np.array([1.0]),
        (0, 1, 0, 0, 0, 0): np.array([0.0]),
        (1, 0, 0, 0, 0, 0): np.array([2.0]),
    }
    simplified = polylist_simplify(powers, eps=0.0)
    assert set(simplified.keys()) == {
        (0, 0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0, 0),
    }


def test_evaluate_hamiltonian_requires_sympy_coefficients():
    """无 sympy 系数的 :class:`Hamiltonian` 报 :class:`ValueError`。"""
    from e2m2e.algorithms.normal_form.hamiltonian import Hamiltonian

    powers = np.zeros((1, 6), dtype=np.int64)
    h = Hamiltonian(
        powers=powers,
        coefficients=None,
        sources={},
        max_degree=1,
    )
    with pytest.raises(ValueError, match="sympy"):
        evaluate_hamiltonian(h, np.array([0.0]), None)  # type: ignore[arg-type]
