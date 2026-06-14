"""ConicalShadowModel 圆锥阴影模型测试。

纯几何路径（``_body_flux_factor`` / ``_combine_body_fluxes``）免 SPICE，与
``test_drag.py`` 范式一致。系统感知路径（``flux_factor(t, state, system)``）
由 ``test_srp_transform.py`` 同级覆盖。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.core.forces.shadow import ConicalShadowModel, ShadowModel

_AU_KM = 149597870.691
_R_EARTH = 6378.1363
_R_SUN = 695700.0


def test_conical_shadow_model_is_shadow_model() -> None:
    """ConicalShadowModel 是 ShadowModel 的具体子类。"""
    assert isinstance(ConicalShadowModel(), ShadowModel)


def test_body_flux_factor_full_sun_returns_one() -> None:
    """SC 远离遮挡体、日体角距远大于视角径之和 → 全光照 1.0。"""
    model = ConicalShadowModel()
    sun_pos = np.array([0.0, 0.0, 0.0])
    body_pos = np.array([_AU_KM, 0.0, 0.0])
    # 体正上方 10000 km，明显在阴影锥外
    sc_pos = np.array([_AU_KM, 1.0e7, 0.0])

    flux = model._body_flux_factor(sc_pos, body_pos, sun_pos, _R_EARTH, _R_SUN)
    assert flux == pytest.approx(1.0)


def test_body_flux_factor_deep_umbra_returns_zero() -> None:
    """SC 在遮挡体背日轴上、深本影锥内 → 0.0。"""
    model = ConicalShadowModel()
    sun_pos = np.array([0.0, 0.0, 0.0])
    body_pos = np.array([_AU_KM, 0.0, 0.0])
    # 地球背日侧 ~地月距离处，c=0 < b-a → 本影
    sc_pos = np.array([_AU_KM + 384000.0, 0.0, 0.0])

    flux = model._body_flux_factor(sc_pos, body_pos, sun_pos, _R_EARTH, _R_SUN)
    assert flux == pytest.approx(0.0, abs=1e-15)


def _lens_overlap_flux(a: float, b: float, c: float) -> float:
    """独立参考：两圆（角径 a、b、圆心距 c）重叠面积的可见份额。

    标准透镜面积公式（与 GMAT 的 x-y 表述代数等价但形式不同，作交叉验证）。
    """
    area = (
        a * a * np.arccos((c * c + a * a - b * b) / (2.0 * c * a))
        + b * b * np.arccos((c * c + b * b - a * a) / (2.0 * c * b))
        - 0.5 * np.sqrt(max(0.0, (a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)))
    )
    return 1.0 - area / (np.pi * a * a)


def test_body_flux_factor_penumbra_matches_lens_formula() -> None:
    """半影区内点 → 0 < flux < 1，与独立透镜面积公式一致（驱动 M&G 面积分支）。"""
    model = ConicalShadowModel()
    sun_pos = np.array([0.0, 0.0, 0.0])
    body_pos = np.array([_AU_KM, 0.0, 0.0])
    # 背日侧 1e6 km、偏轴 5000 km → 半影
    sc_pos = np.array([_AU_KM + 1.0e6, 5000.0, 0.0])

    flux = model._body_flux_factor(sc_pos, body_pos, sun_pos, _R_EARTH, _R_SUN)
    assert 0.0 < flux < 1.0

    # 独立参考：从同一几何重算 a, b, c，套透镜公式
    sc_to_sun = sun_pos - sc_pos
    sc_to_body = body_pos - sc_pos
    d_sun = np.linalg.norm(sc_to_sun)
    d_body = np.linalg.norm(sc_to_body)
    a = np.arcsin(_R_SUN / d_sun)
    b = np.arcsin(_R_EARTH / d_body)
    c = np.arccos(np.dot(sc_to_sun, sc_to_body) / (d_sun * d_body))
    expected = _lens_overlap_flux(a, b, c)
    np.testing.assert_allclose(flux, expected, rtol=1e-9)


def test_body_flux_factor_umbra_boundary_cone_tip_is_zero() -> None:
    """本影锥尖（遮挡体与太阳表现等大、c=0）→ 0.0（验收：本影边界光压=0）。"""
    model = ConicalShadowModel()
    sun_pos = np.array([0.0, 0.0, 0.0])
    body_pos = np.array([_AU_KM, 0.0, 0.0])
    # 本影锥尖距离 d = Rb·AU/(Rs-Rb)，此处 b=a、c=0
    d_tip = _R_EARTH * _AU_KM / (_R_SUN - _R_EARTH)
    sc_pos = np.array([_AU_KM + d_tip, 0.0, 0.0])

    flux = model._body_flux_factor(sc_pos, body_pos, sun_pos, _R_EARTH, _R_SUN)
    assert flux == pytest.approx(0.0, abs=1e-6)


def test_body_flux_factor_anteumbra_matches_one_minus_b2_over_a2() -> None:
    """环形食（遮挡体小于太阳、c≤a−b）→ 1−(b/a)²。"""
    model = ConicalShadowModel()
    # 小天体（R=1000 km）置于 SC 与太阳之间、在日盘内 → annular
    sun_pos = np.array([0.0, 0.0, 0.0])
    sc_pos = np.array([1.0e7, 0.0, 0.0])
    body_pos = np.array([1.0e7 - 1.0e5, 0.0, 0.0])
    r_body = 1000.0

    flux = model._body_flux_factor(sc_pos, body_pos, sun_pos, r_body, _R_SUN)

    d_sun = np.linalg.norm(sun_pos - sc_pos)
    d_body = np.linalg.norm(body_pos - sc_pos)
    a = np.arcsin(_R_SUN / d_sun)
    b = np.arcsin(r_body / d_body)
    expected = 1.0 - (b / a) ** 2
    np.testing.assert_allclose(flux, expected, rtol=1e-12)


def test_body_flux_factor_sunny_side_is_full_sun() -> None:
    """SC 在遮挡体向日侧（体在 SC 背后）→ 全光照 1.0（c=π ≫ a+b）。"""
    model = ConicalShadowModel()
    sun_pos = np.array([0.0, 0.0, 0.0])
    body_pos = np.array([_AU_KM, 0.0, 0.0])
    # SC 在体与太阳之间（向日侧 1e5 km）
    sc_pos = np.array([_AU_KM - 1.0e5, 0.0, 0.0])

    flux = model._body_flux_factor(sc_pos, body_pos, sun_pos, _R_EARTH, _R_SUN)
    assert flux == pytest.approx(1.0)


def test_body_flux_factor_guard_sun_radius_ge_dsun() -> None:
    """守卫：sun_radius ≥ satToSunDist（SC 在日冕内）→ 1.0，避免 arcsin 定义域错。"""
    model = ConicalShadowModel()
    sun_pos = np.array([0.0, 0.0, 0.0])
    body_pos = np.array([_AU_KM, 0.0, 0.0])
    # SC 距太阳 1e5 km ≪ sun_radius(695700) → 触发守卫
    sc_pos = np.array([1.0e5, 0.0, 0.0])

    flux = model._body_flux_factor(sc_pos, body_pos, sun_pos, _R_EARTH, _R_SUN)
    assert flux == pytest.approx(1.0)


def test_body_flux_factor_guard_body_radius_ge_dbody() -> None:
    """守卫：body_radius ≥ satToBodyDist（SC 在遮挡体内）→ 0.0。"""
    model = ConicalShadowModel()
    sun_pos = np.array([0.0, 0.0, 0.0])
    body_pos = np.array([_AU_KM, 0.0, 0.0])
    # SC 距地心 1000 km ≪ R_EARTH → 触发守卫
    sc_pos = np.array([_AU_KM + 1000.0, 0.0, 0.0])

    flux = model._body_flux_factor(sc_pos, body_pos, sun_pos, _R_EARTH, _R_SUN)
    assert flux == pytest.approx(0.0)


# --------------------------- 多遮挡体合成（GMT-6543）---------------------------


def _combine(model, factors, angular_radii, sep_angles):
    """构造体方向单位向量（均在 xy 平面，依次偏移 sep_angles）后调用合成。"""
    dirs = [np.array([np.cos(s), np.sin(s), 0.0]) for s in sep_angles]
    return model._combine_body_fluxes(list(factors), list(angular_radii), dirs)


def test_combine_single_partial_body_passthrough() -> None:
    """单体部分阴影 → 直接返回该体份额。"""
    model = ConicalShadowModel()
    flux = _combine(model, [0.6], [0.01], [0.0])
    assert flux == pytest.approx(0.6)


def test_combine_any_umbra_short_circuits_to_zero() -> None:
    """任一遮挡体本影（factor=0）→ 总份额 0。"""
    model = ConicalShadowModel()
    flux = _combine(model, [0.0, 0.5], [0.01, 0.01], [0.0, 0.05])
    assert flux == pytest.approx(0.0)


def test_combine_two_partials_non_overlapping_inclusion_exclusion() -> None:
    """两体部分阴影、日盘上不重叠（a1+a2 < c12）→ f1+f2−1（包容排斥）。"""
    model = ConicalShadowModel()
    # a1+a2 = 0.02 < c12 = 0.05 → 不重叠
    flux = _combine(model, [0.7, 0.8], [0.01, 0.01], [0.0, 0.05])
    assert flux == pytest.approx(0.7 + 0.8 - 1.0)


def test_combine_two_partials_overlapping_takes_min() -> None:
    """两体部分阴影、日盘上重叠（a1+a2 ≥ c12）→ min(f1, f2)。"""
    model = ConicalShadowModel()
    # a1+a2 = 0.02 >= c12 = 0.01 → 重叠
    flux = _combine(model, [0.7, 0.8], [0.01, 0.01], [0.0, 0.01])
    assert flux == pytest.approx(0.7)


def test_combine_one_full_one_partial_takes_min() -> None:
    """一全光照(f=1) + 一部分阴影 → min = 部分值。"""
    model = ConicalShadowModel()
    flux = _combine(model, [1.0, 0.6], [0.01, 0.01], [0.0, 0.05])
    assert flux == pytest.approx(0.6)


# --------------------------- 构造与半径表 ---------------------------


def test_default_bodies_is_earth_only() -> None:
    """默认遮挡体仅地球。"""
    model = ConicalShadowModel()
    assert model.bodies == ("EARTH",)


def test_bodies_normalized_uppercase() -> None:
    """遮挡体名大写规范化。"""
    model = ConicalShadowModel(bodies=["earth", "Moon"])
    assert model.bodies == ("EARTH", "MOON")


def test_radii_table_has_earth_and_moon() -> None:
    """半径表含地球与月球（验收：支持地球和月球两个遮挡天体）。"""
    model = ConicalShadowModel(bodies=["EARTH", "MOON"])
    assert model.body_radius("EARTH") == pytest.approx(6378.1363)
    assert model.body_radius("MOON") == pytest.approx(1737.4)


def test_unknown_body_without_override_raises() -> None:
    """未知遮挡体未提供 radii 覆盖 → ValueError。"""
    with pytest.raises(ValueError, match="MARS"):
        ConicalShadowModel(bodies=["MARS"])


def test_unknown_body_with_radii_override_ok() -> None:
    """未知遮挡体通过 radii 覆盖可构造。"""
    model = ConicalShadowModel(bodies=["MARS"], radii={"MARS": 3389.5})
    assert model.body_radius("MARS") == pytest.approx(3389.5)
