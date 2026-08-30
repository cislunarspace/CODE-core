"""共振梯（Table 1 / Table 2 全表）逐位复现测试。

陷阱（ADR 0041）：
- T☾ 解析派生（27.34460 天），Table 1 周期列才能逐位复现；
- Table 2 的 16 条月心外地球共振只能用 (GM☾/GM⊕)^{1/3} 因子复现；
- 1:3☾ 行周期表值 82.00 与自洽值 82.03 差 0.03 天，属论文表内舍入不一致，
  该行容差单独放宽。
"""

from __future__ import annotations

import pytest

from e2m2e.algorithm.spatiography.constants import PRIMER_DEFAULTS
from e2m2e.algorithm.spatiography.resonances import (
    PRIMER_RESONANCE_KINDS,
    resonance_centers,
)

pytestmark = pytest.mark.theory

# 论文 Table 1 行值：(label, a/a☾, T[days])
_TABLE1_INTERIOR_LUNAR = [
    ("5:1☾", 0.34, 5.47),
    ("4:1☾", 0.40, 6.84),
    ("3:1☾", 0.48, 9.11),
    ("5:2☾", 0.54, 10.94),
    ("2:1☾", 0.63, 13.67),
    ("5:3☾", 0.71, 16.41),
    ("3:2☾", 0.76, 18.23),
    ("4:3☾", 0.83, 20.51),
    ("5:4☾", 0.86, 21.88),
]
_TABLE1_EXTERIOR_LUNAR = [
    ("4:5☾", 1.16, 34.18),
    ("3:4☾", 1.21, 36.46),
    ("2:3☾", 1.31, 41.02),
    ("3:5☾", 1.41, 45.57),
    ("1:2☾", 1.59, 54.69),
    ("2:5☾", 1.84, 68.36),
    ("1:3☾", 2.08, 82.00),
    ("1:4☾", 2.52, 109.38),
    ("1:5☾", 2.92, 136.72),
]
_TABLE1_SOLAR = [
    ("5:1☉", 1.93, 73.05),
    ("4:1☉", 2.23, 91.31),
    ("3:1☉", 2.71, 121.75),
    ("5:2☉", 3.06, 146.10),
    ("2:1☉", 3.55, 182.63),
]

# 论文 Table 2 行值：(label, rho/R☾, T[days])
_TABLE2_SELENOCENTRIC = [
    ("8:1⊕", 12.73, 3.42),
    ("7:1⊕", 13.92, 3.91),
    ("6:1⊕", 15.43, 4.56),
    ("5:1⊕", 17.42, 5.47),
    ("9:2⊕", 18.69, 6.08),
    ("4:1⊕", 20.22, 6.84),
    ("7:2⊕", 22.10, 7.81),
    ("10:3⊕", 22.83, 8.20),
    ("3:1⊕", 24.49, 9.11),
    ("8:3⊕", 26.49, 10.25),
    ("5:2⊕", 27.65, 10.94),
    ("7:3⊕", 28.96, 11.72),
    ("9:4⊕", 29.67, 12.15),
    ("2:1⊕", 32.09, 13.67),
    ("9:5⊕", 34.42, 15.19),
    ("7:4⊕", 35.08, 15.63),
]


def _centers_of(kind: str) -> dict:
    return {center.label: center for center in resonance_centers(kind).centers}


@pytest.mark.parametrize(
    "kind, rows, pos_attr",
    [
        ("interior_lunar", _TABLE1_INTERIOR_LUNAR, "a_over_a_moon"),
        ("exterior_lunar", _TABLE1_EXTERIOR_LUNAR, "a_over_a_moon"),
        ("solar", _TABLE1_SOLAR, "a_over_a_moon"),
        ("exterior_terrestrial_selenocentric", _TABLE2_SELENOCENTRIC, "rho_over_moon_radius"),
    ],
)
def test_table_rows_reproduce_paper_values(kind, rows, pos_attr):
    """Table 1/2 行值逐位复现（位置按表列 2 位小数、周期按 0.02 天）。"""
    centers = _centers_of(kind)
    assert len(centers) >= len(rows)
    for label, pos_expected, t_expected in rows:
        center = centers[label]
        pos_actual = getattr(center, pos_attr)
        assert pos_actual == pytest.approx(pos_expected, abs=5e-3), label
        period_tol = 0.05 if label == "1:3☾" else 0.02
        assert center.period_days == pytest.approx(t_expected, abs=period_tol), label


def test_solar_ladder_centers_in_lunar_units():
    """日支在 a/a☾ 轴上的精确位置（表值 2 位小数的连续版本）。"""
    centers = _centers_of("solar")
    assert centers["5:1☉"].a_over_a_moon == pytest.approx(1.9253, abs=1e-3)
    assert centers["2:1☉"].a_over_a_moon == pytest.approx(3.5465, abs=1e-3)


def test_solar_secondary_6_to_1_is_marked():
    centers = _centers_of("solar")
    assert centers["6:1☉"].secondary is True
    assert centers["6:1☉"].a_over_a_moon == pytest.approx(1.705, abs=2e-3)
    assert centers["5:1☉"].secondary is False


def test_table2_requires_moon_earth_mass_factor_not_cr3bp_mu_bar():
    """陷阱③：Table 2 只能由 (GM☾/GM⊕)^{1/3} 因子逐位复现；论文式(126)字面
    mu_bar^{1/3}（mu_bar = GM☾/(GM⊕+GM☾)）系统性低 0.4%（12.68 vs 12.73）。"""
    c = PRIMER_DEFAULTS
    ours = resonance_centers("exterior_terrestrial_selenocentric")
    first = ours.centers[0]
    assert first.label == "8:1⊕"
    assert first.rho_over_moon_radius == pytest.approx(12.7348, abs=1e-3)
    naive = (
        (c.moon_gm / (c.earth_gm + c.moon_gm)) ** (1.0 / 3.0)
        * (1.0 / 8.0) ** (2.0 / 3.0)
        * c.moon_a_km
        / c.moon_radius_km
    )
    assert naive == pytest.approx(12.68, abs=5e-3)
    assert abs(naive - first.rho_over_moon_radius) > 0.03


def test_moon_period_column_is_consistent_with_derived_t_moon():
    """内月梯周期 = T☾·(k_b/k)，T☾ 解析派生；不得用 27.346 硬编码复现表值。"""
    c = PRIMER_DEFAULTS
    assert c.moon_period_days == pytest.approx(27.34460, rel=1e-5)
    centers = _centers_of("interior_lunar")
    for center in centers.values():
        expected = c.moon_period_days * (center.k_body / center.k)
        assert center.period_days == pytest.approx(expected, rel=1e-9)


def test_all_kind_concatenates_in_table_order():
    result = resonance_centers("all")
    labels = [center.label for center in result.centers]
    assert len(labels) == 9 + 9 + 6 + 16
    assert labels[0] == "5:1☾"
    assert labels[8] == "5:4☾"
    assert labels[-1] == "7:4⊕"
    assert all(center.kind in PRIMER_RESONANCE_KINDS for center in result.centers)


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="kind"):
        resonance_centers("bogus")
