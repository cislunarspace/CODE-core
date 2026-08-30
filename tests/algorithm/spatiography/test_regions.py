"""区域分类器的理论与边界情形测试。

设计依据（论文校验结论，ADR 0041）：分区边界 deliberate overlap——互斥区间
设计已被 Table 1/4 的实际结构否定（5:4ζ=0.86 在 circumlunar 包络内、L2 与
4:5ζ 同值、a_TP/SOI 落在 translunar 带内部）。
"""

from __future__ import annotations

import pytest

from e2m2e.algorithm.spatiography import (
    REGION_LEGEND,
    RegionId,
    classify_by_semi_major_axis,
    classify_state,
    jacobi_critical_values,
    jacobi_topology_case,
    primer_cr3bp_system,
    table4_bands,
)
from e2m2e.algorithm.spatiography.constants import PRIMER_DEFAULTS
from e2m2e.algorithm.spatiography.scales import hill_radius_earth

pytestmark = pytest.mark.theory

_T = RegionId.TERRESTRIAL
_CI = RegionId.CISLUNAR_INNER_SECULAR
_CO = RegionId.CISLUNAR_OUTER_RESONANT
_CU = RegionId.CIRCUMLUNAR
_TL = RegionId.TRANSLUNAR
_H = RegionId.HELIOCENTRIC


def test_legend_has_no_umbrella_cislunar():
    """命名铁律：cislunar 只出现在两个带级名中，绝无伞式 cislunar 值。"""
    names = list(REGION_LEGEND.values())
    assert "cislunar" not in names
    assert "cislunar_inner_secular" in names
    assert "cislunar_outer_resonant" in names
    assert "geolunar" not in " ".join(names)  # 总称留给文档层，不作区域名


@pytest.mark.parametrize(
    "x, expected",
    [
        (0.05, [_T]),
        (0.20, [_CI]),
        (0.50, [_CO]),
        (0.86, [_CO, _CU]),  # 5:4ζ 与 circumlunar 包络交叠
        (0.90, [_CU]),
        (1.00, [_CU]),
        (1.30, [_TL]),
        (2.50, [_TL]),
        (4.50, [_TL, _H]),  # 越出地球 Hill 球
    ],
)
def test_table1_classification_multi_label(x, expected):
    assert classify_by_semi_major_axis(x) == [int(e) for e in expected]


def test_primary_label_precedence_on_overlap():
    """include_overlaps=False：0.86 处 circumlunar（enclave）优先于外带。"""
    assert classify_by_semi_major_axis(0.86, include_overlaps=False) == [int(_CU)]


@pytest.mark.parametrize(
    "x, expected",
    [
        (0.20, [_CI]),
        (0.34, [_CI, _CO]),  # SC 与 CR 有意重叠带 0.33–0.35
        (0.86, [_CO, _CU]),  # CR 与 CG 重叠带 0.84–0.89
        (1.10, [_CU, _TL]),  # CG 与 IT 重叠带 1.08–1.16
        (1.95, [_TL]),  # IT 与 OT 重叠、同属 translunar，去重后单标签
        (3.10, [_TL]),  # OT 与 TF 重叠带
        (4.50, [_H]),  # TF 止于地球 Hill，之外仅 heliocentric
    ],
)
def test_table4_classification_overlap_bands(x, expected):
    assert classify_by_semi_major_axis(x, reference="table4") == [int(e) for e in expected]


def test_table4_band_edges_derive_from_primer_constants():
    bands = table4_bands()
    # SC 下界用解析 r_L/a☾（0.1273）替换表值 0.13：0.128 已在内带。
    assert 0.127 < bands.lower[0] < 0.128
    assert classify_by_semi_major_axis(bands.lower[0] + 1e-4) == [int(_CI)]
    # TF 上界用解析地球 Hill（3.9034）替换表值 3.90。
    assert bands.upper[-1] == pytest.approx(
        hill_radius_earth() / PRIMER_DEFAULTS.moon_a_km, rel=1e-6
    )


def test_invalid_reference_rejected():
    with pytest.raises(ValueError, match="reference"):
        classify_by_semi_major_axis(0.5, reference="table3")


def test_jacobi_topology_case_progression():
    system = primer_cr3bp_system()
    crits = jacobi_critical_values(system)
    assert crits["C1"] > crits["C2"] > crits["C3"] > crits["C4"]
    assert crits["C4"] == pytest.approx(crits["C5"])
    c1, c2, c3, c4 = crits["C1"], crits["C2"], crits["C3"], crits["C4"]
    assert jacobi_topology_case(c1 + 0.01, crits) == (1, ())
    assert jacobi_topology_case((c1 + c2) / 2.0, crits) == (2, ("L1",))
    assert jacobi_topology_case((c2 + c3) / 2.0, crits) == (3, ("L1", "L2"))
    assert jacobi_topology_case((c3 + c4) / 2.0, crits)[0] == 4
    assert jacobi_topology_case(c4 - 0.01, crits)[0] == 5


def test_classify_state_frame_consistency_km_vs_nd():
    """同一状态按 km 与无量纲两种 frame 声明应给出一致诊断。"""

    from e2m2e.algorithm.spatiography.constants import PRIMER_DEFAULTS as K

    state_nd = [0.5, 0.1, 0.0, 0.0, 1.0, 0.0]
    n = K.cr3bp_mean_motion_rad_s
    state_km = [
        state_nd[0] * K.moon_a_km,
        state_nd[1] * K.moon_a_km,
        0.0,
        state_nd[3] * K.moon_a_km * n,
        state_nd[4] * K.moon_a_km * n,
        0.0,
    ]
    diag_nd = classify_state(state_nd, frame="synodic_barycentric_nd")
    diag_km = classify_state(state_km, frame="synodic_barycentric_km")
    assert diag_km.a_over_a_moon == pytest.approx(diag_nd.a_over_a_moon, rel=1e-9)
    assert diag_km.jacobi_constant == pytest.approx(diag_nd.jacobi_constant, rel=1e-9)
    assert diag_km.topology_case == diag_nd.topology_case
    assert diag_km.zone_ids == diag_nd.zone_ids
    assert diag_nd.topology_case == 3  # C2..C3 之间：L1/L2 双颈开


def test_classify_state_diagnostics_fields():
    diag = classify_state([0.5, 0.1, 0.0, 0.0, 1.0, 0.0], frame="synodic_barycentric_nd")
    assert diag.r_geocentric_km > 0.0
    assert diag.rho_selenocentric_km > 0.0
    assert diag.a_geocentric_km == pytest.approx(135936.2, rel=1e-4)
    assert diag.zone_ids == (int(_CO),)
    assert diag.message == "ok"


def test_classify_state_rejects_bad_frame_and_shape():
    with pytest.raises(ValueError, match="frame"):
        classify_state([0.5] * 6, frame="gcrs_km")
    with pytest.raises(ValueError, match="6"):
        classify_state([0.5] * 5, frame="synodic_barycentric_nd")
