"""区间化分区判定（classify_state_interval，issue #785）的理论测试。

DA 数值底座来自 #784 原语：判据链按截断 Taylor 多项式保守包围。包围口径
是多项式包围而非解析验证性包围，Monte Carlo 包含性断言带 1e-9 容差吸收
O(|h|^(k+1)) 截断余项；DA 上下文是进程级全局状态，实现内部每次调用自建，
测试不跨调用持 Da 对象。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from kernel_helpers import requires_native_symbols

from e2m2e.algorithm.spatiography import (
    IntervalStateDiagnostics,
    RegionId,
    classify_by_semi_major_axis_interval,
    classify_state,
    classify_state_interval,
    jacobi_critical_values,
    jacobi_topology_case_interval,
    primer_cr3bp_system,
)

pytestmark = [
    pytest.mark.theory,
    requires_native_symbols("da_init_py", "da_initialized_py", "da_truncation_order_py", "Da"),
]

#: 共享 Primer CR3BP 系统（平动点求根结果在系统对象上缓存，module 级复用）。
_SYS = primer_cr3bp_system()

_T = int(RegionId.TERRESTRIAL)
_CI = int(RegionId.CISLUNAR_INNER_SECULAR)
_CO = int(RegionId.CISLUNAR_OUTER_RESONANT)
_CU = int(RegionId.CIRCUMLUNAR)
_TL = int(RegionId.TRANSLUNAR)
_H = int(RegionId.HELIOCENTRIC)


def test_zero_width_anchor_bitwise_matches_point_classification():
    """零宽锚：半宽全零退化为点判定，各标量复制为退化区间（精确相等）。"""
    state = [0.5, 0.1, 0.0, 0.0, 1.0, 0.0]
    diag = classify_state(state, frame="synodic_barycentric_nd", system=_SYS)
    result = classify_state_interval(state, [0.0] * 6, frame="synodic_barycentric_nd", system=_SYS)
    assert isinstance(result, IntervalStateDiagnostics)
    for interval, scalar in (
        (result.r_geocentric_km, diag.r_geocentric_km),
        (result.rho_selenocentric_km, diag.rho_selenocentric_km),
        (result.a_geocentric_km, diag.a_geocentric_km),
        (result.a_over_a_moon, diag.a_over_a_moon),
        (result.jacobi_constant, diag.jacobi_constant),
    ):
        assert interval == (scalar, scalar)
    assert result.zone_ids_possible == diag.zone_ids
    assert result.zone_ids_certain == diag.zone_ids
    assert result.topology_case_min == diag.topology_case
    assert result.topology_case_max == diag.topology_case
    assert result.ambiguous_critical_values == ()
    assert result.open_necks == diag.open_necks
    assert result.status == diag.status
    assert result.message == diag.message


def test_table1_interval_band_membership():
    """table1 区间规则：退化区间回归点版口径，跨界区间只进 possible。"""
    possible, certain = classify_by_semi_major_axis_interval((0.05, 0.06), system=_SYS)
    assert possible == (_T,)
    assert certain == (_T,)

    possible, certain = classify_by_semi_major_axis_interval((0.33, 0.35), system=_SYS)
    assert possible == (_CI, _CO)
    assert certain == ()

    possible, certain = classify_by_semi_major_axis_interval((0.5, 0.6), system=_SYS)
    assert possible == (_CO,)
    assert certain == (_CO,)

    possible, certain = classify_by_semi_major_axis_interval((0.86, 1.2), system=_SYS)
    assert _CO in possible
    assert _CU in possible
    assert _TL in possible
    assert certain == ()

    possible, certain = classify_by_semi_major_axis_interval((4.0, float("inf")), system=_SYS)
    assert possible == (_TL, _H)


def test_table4_interval_overlapping_bands():
    """table4 区间规则：跨界区间同时命中 SC/CR 两带，certain 为空。"""
    possible, certain = classify_by_semi_major_axis_interval((0.30, 0.40), reference="table4")
    assert possible == (_CI, _CO)
    assert certain == ()


def test_interval_classifier_rejects_bad_reference():
    with pytest.raises(ValueError, match="reference"):
        classify_by_semi_major_axis_interval((0.1, 0.2), reference="bogus")


def test_topology_case_interval_boundaries():
    """Case 区间：C 上界给最小 Case、下界给最大 Case；严格内部穿越才列歧义。"""
    crits = jacobi_critical_values(_SYS)
    c1, c2, c4 = crits["C1"], crits["C2"], crits["C4"]
    assert jacobi_topology_case_interval((c1 - 0.01, c1 + 0.01), crits) == (1, 2, ("C1",))
    assert jacobi_topology_case_interval((c1 + 0.01, c1 + 0.02), crits) == (1, 1, ())
    assert jacobi_topology_case_interval((c4 + 0.01, c4 + 0.02), crits) == (4, 4, ())
    case_min, case_max, ambiguous = jacobi_topology_case_interval(
        (crits["C4"] - 0.5, c2 + 0.5), crits
    )
    assert case_min < case_max
    assert "C2" in ambiguous
    assert "C3" in ambiguous


def test_end_to_end_band_crossing_ambiguity():
    """标称 a/a☾ 落在 5:1 共振界、速度半宽跨界的盒：INNER/OUTER 均可能、均不确证。"""
    gamma = 1.0 - _SYS.mu
    r = math.hypot(0.5 + _SYS.mu, 0.1)
    v = math.sqrt(gamma * (2.0 / r - 1.0 / 0.342))
    result = classify_state_interval(
        [0.5, 0.1, 0.0, 0.0, v, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.01, 0.0],
        frame="synodic_barycentric_nd",
        system=_SYS,
    )
    assert result.a_over_a_moon[0] <= result.a_over_a_moon[1]
    assert _CI in result.zone_ids_possible
    assert _CO in result.zone_ids_possible
    assert _CI not in result.zone_ids_certain
    assert _CO not in result.zone_ids_certain
    assert 1 <= result.topology_case_min <= result.topology_case_max <= 5


def test_monte_carlo_inclusion():
    """盒内采样点判定须落入区间包围（容差吸收截断余项）与 possible 标签集。"""
    nominal = np.array([0.5, 0.1, 0.0, 0.0, 1.0, 0.0])
    h = np.full(6, 1e-3)
    result = classify_state_interval(nominal, h, frame="synodic_barycentric_nd", system=_SYS)
    rng = np.random.default_rng(20261003)
    for sample in rng.uniform(-1.0, 1.0, size=(200, 6)):
        diag = classify_state(nominal + sample * h, frame="synodic_barycentric_nd", system=_SYS)
        assert set(diag.zone_ids) <= set(result.zone_ids_possible)
        assert result.topology_case_min <= diag.topology_case <= result.topology_case_max
        assert (
            result.r_geocentric_km[0] - 1e-9
            <= diag.r_geocentric_km
            <= result.r_geocentric_km[1] + 1e-9
        )
        assert (
            result.rho_selenocentric_km[0] - 1e-9
            <= diag.rho_selenocentric_km
            <= result.rho_selenocentric_km[1] + 1e-9
        )
        assert (
            result.jacobi_constant[0] - 1e-9
            <= diag.jacobi_constant
            <= result.jacobi_constant[1] + 1e-9
        )
        if math.isfinite(diag.a_over_a_moon):
            assert (
                result.a_over_a_moon[0] - 1e-9
                <= diag.a_over_a_moon
                <= result.a_over_a_moon[1] + 1e-9
            )


def test_box_splitting_nested_and_monotone():
    """逐维对分：子盒区间嵌套于全盒区间（相对容差），possible 标签集单调收缩。

    ``Da.bound()`` 是区间算术保守包围，松紧随展开点不同而变化（依赖效应），
    子盒界可小幅越出全盒界：实测量级 ~5e-7 相对（z 维标称 0、r 对 z 偶对称
    时最明显），远大于机器精度但远小于带宽；故取相对 1e-6 容差而非逐位嵌套。
    """
    nominal = np.array([0.5, 0.1, 0.0, 0.0, 1.0, 0.0])
    h = np.full(6, 1e-3)
    full = classify_state_interval(nominal, h, frame="synodic_barycentric_nd", system=_SYS)
    for i in range(6):
        sub_h = h.copy()
        sub_h[i] = h[i] / 2.0
        for sign in (-1.0, 1.0):
            sub_nominal = nominal.copy()
            sub_nominal[i] += sign * h[i] / 2.0
            sub = classify_state_interval(
                sub_nominal, sub_h, frame="synodic_barycentric_nd", system=_SYS
            )
            for sub_iv, full_iv in (
                (sub.r_geocentric_km, full.r_geocentric_km),
                (sub.rho_selenocentric_km, full.rho_selenocentric_km),
                (sub.a_over_a_moon, full.a_over_a_moon),
                (sub.jacobi_constant, full.jacobi_constant),
            ):
                slack_lo = 1e-6 * max(1.0, abs(full_iv[0]))
                slack_hi = 1e-6 * max(1.0, abs(full_iv[1]))
                assert sub_iv[0] >= full_iv[0] - slack_lo
                assert sub_iv[1] <= full_iv[1] + slack_hi
            assert set(sub.zone_ids_possible) <= set(full.zone_ids_possible)


def test_escape_and_mixed_boxes():
    """整盒逃逸 a=(inf,inf) 不分带（镜像点版 a=inf 跳过）；跨 0 盒上界无界。"""
    gamma = 1.0 - _SYS.mu
    r = math.hypot(0.5 + _SYS.mu, 0.1)
    v_esc = math.sqrt(2.0 * gamma / r)
    escaped = classify_state_interval(
        [0.5, 0.1, 0.0, 0.0, 1.2 * v_esc, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.01, 0.0],
        frame="synodic_barycentric_nd",
        system=_SYS,
    )
    assert escaped.a_geocentric_km == (float("inf"), float("inf"))
    assert escaped.zone_ids_possible == ()
    assert escaped.zone_ids_certain == ()

    mixed = classify_state_interval(
        [0.5, 0.1, 0.0, 0.0, v_esc, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.01, 0.0],
        frame="synodic_barycentric_nd",
        system=_SYS,
    )
    assert mixed.a_over_a_moon[1] == float("inf")
    assert _TL in mixed.zone_ids_possible


def test_interval_input_validation():
    state = [0.5, 0.1, 0.0, 0.0, 1.0, 0.0]
    widths = [1e-3] * 6
    with pytest.raises(ValueError, match="frame"):
        classify_state_interval(state, widths, frame="gcrs_km")
    with pytest.raises(ValueError, match="6 维"):
        classify_state_interval(state[:5], widths, frame="synodic_barycentric_nd")
    with pytest.raises(ValueError, match="half_widths"):
        classify_state_interval(state, widths[:5], frame="synodic_barycentric_nd")
    with pytest.raises(ValueError, match="≥ 0"):
        classify_state_interval(state, [-1e-3] + widths[1:], frame="synodic_barycentric_nd")
    with pytest.raises(ValueError, match="truncation_order"):
        classify_state_interval(state, widths, frame="synodic_barycentric_nd", truncation_order=0)


def test_interval_rejects_primary_singularity():
    """标称置于地心奇点（地球位于 (-mu,0,0)）：判据不可微，拒绝而非返 NaN。"""
    with pytest.raises(ValueError, match="奇点"):
        classify_state_interval(
            [-_SYS.mu, 0.0, 0.0, 0.0, 0.5, 0.0],
            [1e-3] * 6,
            frame="synodic_barycentric_nd",
            system=_SYS,
        )
