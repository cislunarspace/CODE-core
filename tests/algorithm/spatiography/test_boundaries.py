"""边界几何生成器的结构与数值测试。"""

from __future__ import annotations

import pytest

from e2m2e.algorithm.spatiography import (
    ae_curves,
    hill_radius_moon,
    laplace_radius_geolunar,
    soi_laplace_moon,
    synodic_planar_elements,
)

pytestmark = pytest.mark.theory


def _labels(result) -> list[str]:
    return [element.label for element in result.elements]


def test_synodic_planar_default_set_contains_all_families():
    result = synodic_planar_elements(resolution=64)
    labels = _labels(result)
    assert len(result) == 13  # 7 圆 + Battin 闭合曲线 + L1–L5
    for name in (
        "Laplace radius r_L (geolunar)",
        "Moon Hill sphere rho_H",
        "Moon Battin SOI rho_B(psi)",
        "L1",
        "L5",
    ):
        assert name in labels
    assert {element.kind for element in result.elements} == {"circle", "polyline", "point"}


def test_circles_carry_radius_and_center_in_barycentric_km():
    from e2m2e.algorithm.spatiography.constants import PRIMER_DEFAULTS

    result = synodic_planar_elements(resolution=32, boundary_set=["laplace_radius"])
    element = result.elements[0]
    assert element.kind == "circle"
    assert element.radius_km == pytest.approx(laplace_radius_geolunar(), rel=1e-9)
    earth_x = -PRIMER_DEFAULTS.moon_mass_parameter * PRIMER_DEFAULTS.moon_a_km
    assert element.center_km[0] == pytest.approx(earth_x, rel=1e-9)
    assert element.points_km.shape == (32, 3)
    # 闭合：首尾相接（sin(2π) 有 ~1e-11 三角尾差，用绝对容差）。
    assert element.points_km[0] == pytest.approx(element.points_km[-1], abs=1e-6)


def test_battin_curve_axial_extrema_match_paper_values():
    """闭合曲线在 psi=0（背地）/ psi=pi（朝地）处取论文轴向值 64201/52009 km。"""
    result = synodic_planar_elements(resolution=360, boundary_set=["moon_battin"])
    element = result.elements[0]
    assert element.kind == "polyline"
    offsets = element.points_km[:, :2] - element.center_km[:2]
    radii = np_hypot(offsets)
    assert radii[0] == pytest.approx(64201.0, rel=1e-4)  # psi=0：+x 背地
    assert radii[180] == pytest.approx(52009.0, rel=1e-4)  # psi=pi：朝地
    # 全局最大不在轴向（约 66.4 Mm @ psi~78°）——论文仅引用轴向值。
    assert radii.max() == pytest.approx(66389.0, rel=5e-3)
    assert radii.max() > radii[0]


def np_hypot(offsets):
    return (offsets[:, 0] ** 2 + offsets[:, 1] ** 2) ** 0.5


def test_libration_points_are_exact_and_planar():
    result = synodic_planar_elements(resolution=32, boundary_set=["libration_points"])
    by_label = {element.label: element for element in result.elements}
    l1 = by_label["L1"].center_km
    l2 = by_label["L2"].center_km
    moon_x = (1.0 - 0.012150587) * 383397.7725  # mu_bar 精度内
    assert l1[1] == 0.0 and l1[2] == 0.0
    assert l1[0] < moon_x < l2[0]  # L1 在地月之间、L2 在月外
    # 论文表值 57868/64347 km 为级数口径（月心距）；精确解差 1.2–2.3%（ADR 0041）。
    l1_mooncentric = moon_x - l1[0]
    l2_mooncentric = l2[0] - moon_x
    assert l1_mooncentric == pytest.approx(57868.0, rel=0.03)
    assert l2_mooncentric == pytest.approx(64347.0, rel=0.03)


def test_unknown_boundary_element_rejected():
    with pytest.raises(ValueError, match="边界元素"):
        synodic_planar_elements(boundary_set=["bogus"])
    with pytest.raises(ValueError, match="曲线族"):
        ae_curves(boundary_set=["bogus"])


def test_ae_curves_family_formulas():
    from e2m2e.algorithm.spatiography.constants import PRIMER_DEFAULTS as K

    result = ae_curves(n_points=200, boundary_set=["graze", "hill_apocenter"])
    by_label = {element.label: element for element in result.elements}
    graze = by_label["Earth grazing a(1-e)=R+"].points_ae
    mask = graze[:, 0] > 3.0 * K.earth_ref_radius_km
    assert mask.any()
    e_expected = 1.0 - K.earth_ref_radius_km / graze[mask, 0]
    assert graze[mask, 1] == pytest.approx(e_expected, rel=1e-9)
    hill = by_label["Earth Hill apocenter a(1+e)=r_H"].points_ae
    assert (hill[:, 1] <= 1.0 + 1e-12).all()
    assert (hill[:, 0] <= 1.02 * 1.4966e6 * 1.001 + 1.0).all()


def test_ae_resonance_verticals_cover_all_geocentric_ladder():
    from e2m2e.algorithm.spatiography import resonance_centers

    result = ae_curves(n_points=32, boundary_set=["resonance_verticals"])
    verticals = {
        element.label: element for element in result.elements if element.kind == "vertical_ae"
    }
    geocentric = [
        center
        for center in resonance_centers("all").centers
        if center.kind != "exterior_terrestrial_selenocentric"
    ]
    assert len(geocentric) == 9 + 9 + 6
    for center in geocentric:
        assert verticals[center.label].a_km == pytest.approx(center.a_km, rel=1e-9)
    assert verticals["r_L"].a_km == pytest.approx(laplace_radius_geolunar(), rel=1e-9)


def test_ae_tisserand_contour_passes_near_moon_circular_point():
    from e2m2e.algorithm.spatiography.constants import PRIMER_DEFAULTS as K

    result = ae_curves(n_points=400, boundary_set=["tisserand_contours"])
    t3 = [element for element in result.elements if element.label == "Tisserand T_M=3"][0]
    idx = int((t3.points_ae[:, 0] - K.moon_a_km).__abs__().argmin())
    assert t3.points_ae[idx, 1] < 0.01  # T=3 等值线过 (a☾, e≈0) 参考点


def test_resolution_validation():
    with pytest.raises(ValueError, match="resolution"):
        synodic_planar_elements(resolution=4)
    with pytest.raises(ValueError, match="n_points"):
        ae_curves(n_points=8)


def test_moon_circle_radii_are_consistent_with_scales():
    result = synodic_planar_elements(resolution=32, boundary_set=["moon_hill", "moon_soi"])
    radii = {element.label: element.radius_km for element in result.elements}
    assert radii["Moon Hill sphere rho_H"] == pytest.approx(hill_radius_moon(), rel=1e-9)
    assert radii["Moon SOI (Laplace-Tisserand)"] == pytest.approx(soi_laplace_moon(), rel=1e-9)
