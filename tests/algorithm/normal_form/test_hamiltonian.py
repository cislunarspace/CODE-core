"""``normal_form.hamiltonian`` 测试。"""

from __future__ import annotations

import time

import numpy as np
import pytest
from kernel_helpers import requires_spice

# sympy 是 normal-form optional dep；未安装时整个文件 skip（不 error）。
pytest.importorskip("sympy")

from e2m2e.algorithm.normal_form import NormalFormContext
from e2m2e.algorithm.normal_form.constants import JD0_J2000
from e2m2e.algorithm.normal_form.hamiltonian import (
    DYNAMIC_PARAM_NAMES,
    Hamiltonian,
    build_hamiltonian,
    evaluate_hamiltonian,
    hamiltonian_constant_term,
)
from e2m2e.algorithm.normal_form.legendre import expand_legendre_1_over_r
from e2m2e.data.templates.enums import LibrationPoint

pytestmark = pytest.mark.theory


# ---------------------------------------------------------------------------
# 公共 fixture：可复用的上下文与 Legendre 展开
# ---------------------------------------------------------------------------


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
# 构造烟测（纯符号，不需 SPICE）；SPICE 通用可用性探测见 kernel_helpers
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


@pytest.mark.spice
@requires_spice
def test_evaluate_hamiltonian_runs(spice_manager, l1_hamiltonian, l1_context):
    """对一组时刻求值 Hamilton 量，运行无异常。"""
    times = np.linspace(0.0, 5.0, 6)
    evaled = evaluate_hamiltonian(l1_hamiltonian, times, l1_context)
    assert evaled.is_evaluated
    assert isinstance(evaled.coefficients, np.ndarray)
    assert evaled.coefficients.shape == (len(times), evaled.n_terms)
    # 常数项按定义为负（引力势主导）且有限
    target = (0, 0, 0, 0, 0, 0)
    for j in range(evaled.n_terms):
        if tuple(int(p) for p in evaled.powers[j]) == target:
            col = evaled.coefficients[:, j]
            assert np.all(np.isfinite(col))
            assert np.all(col < 0)
            break
    else:
        pytest.fail("常数项 (0,0,0,0,0,0) 不在 evaled.powers 中")


@pytest.mark.spice
@requires_spice
def test_hamiltonian_constant_term_matches_point_mass_definition(
    spice_manager, l1_hamiltonian, l1_context
):
    """常值项 H_0 按定义等于三天体点质量势零阶项之和 ``-(μe/re0 + μm/rm0 + μs/rs0)``。

    Hamilton 定义中动能/科里奥利/离心项在 ``q=p=0`` 处为零，引力势
    Legendre 展开的零阶项即 ``-μ/|r|``。参照值由 ``_ephemeris.eval_params``
    的星历输入现场计算——输入共享、公式独立：构造装配错误（错号、错幂次
    映射）会使两边分离。这是 ADR 0013 的定义级断言，替代原对外部软件
    输出值的比对。
    """
    from e2m2e.algorithm.normal_form._ephemeris import eval_params

    times = np.linspace(0.0, 5.0, 6)
    h0 = hamiltonian_constant_term(l1_hamiltonian, times, l1_context)

    t_to_jd = float(l1_context.TU) / 86400.0
    for i, t in enumerate(times):
        params = eval_params(float(l1_context.epoch) + float(t) * t_to_jd, l1_context)
        expected = -(
            params["mu_e"] / params["re0"]
            + params["mu_m"] / params["rm0"]
            + params["mu_s"] / params["rs0"]
        )
        assert h0[i] == pytest.approx(expected, rel=1e-9)


@pytest.mark.spice
@requires_spice
def test_build_evaluate_l1_order4_within_timeout(spice_manager, l1_context):
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
    from e2m2e.algorithm.normal_form.polynomial import polylist_simplify

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
    from e2m2e.algorithm.normal_form.hamiltonian import Hamiltonian

    powers = np.zeros((1, 6), dtype=np.int64)
    h = Hamiltonian(
        powers=powers,
        coefficients=None,
        sources={},
        max_degree=1,
    )
    with pytest.raises(ValueError, match="sympy"):
        evaluate_hamiltonian(h, np.array([0.0]), None)  # type: ignore[arg-type]
