"""Primer 常数节（constants.toml [primer]）加载与溯源测试。"""

from __future__ import annotations

import pytest

from e2m2e.data.constants import (
    PRIMER_CITATION,
    PRIMER_EARTH_GM,
    PRIMER_EARTH_GM_SOURCE,
    PRIMER_EARTH_J2,
    PRIMER_EARTH_J2_SOURCE,
    PRIMER_EARTH_REF_RADIUS_KM,
    PRIMER_EARTH_SIDEREAL_DAY_S,
    PRIMER_MOON_A_KM,
    PRIMER_MOON_A_KM_SOURCE,
    PRIMER_MOON_ECC,
    PRIMER_MOON_GM,
    PRIMER_MOON_INC_DEG,
    PRIMER_MOON_J2,
    PRIMER_SUN_A_AU,
    PRIMER_SUN_ECC,
    PRIMER_SUN_GM,
    ConstantSource,
)

pytestmark = pytest.mark.data


def test_primer_values_match_rosengren_2026_adopted_set():
    """[primer] 节 12 项常数 = 论文 §5 采用集（GM: SPICE；月根数: Simon 1994）。"""
    assert pytest.approx(1.08263552549e-3) == PRIMER_EARTH_J2
    assert pytest.approx(2.0322e-4) == PRIMER_MOON_J2
    assert pytest.approx(383397.7725) == PRIMER_MOON_A_KM
    assert pytest.approx(0.055545526) == PRIMER_MOON_ECC
    assert pytest.approx(5.15668983) == PRIMER_MOON_INC_DEG
    assert pytest.approx(1.0000010178) == PRIMER_SUN_A_AU
    assert pytest.approx(0.0167086342) == PRIMER_SUN_ECC
    assert pytest.approx(398600.4354360959) == PRIMER_EARTH_GM
    assert pytest.approx(4902.800066163796) == PRIMER_MOON_GM
    assert pytest.approx(1.327124400419393e11) == PRIMER_SUN_GM
    assert pytest.approx(6378.1363) == PRIMER_EARTH_REF_RADIUS_KM
    assert pytest.approx(86164.0905) == PRIMER_EARTH_SIDEREAL_DAY_S


def test_primer_sources_are_registered():
    """来源枚举已扩充 SIMON1994/GGM02/LEGNARO2024 并被 [primer] 节引用。"""
    assert ConstantSource.SIMON1994.value == "SIMON1994"
    assert ConstantSource.GGM02.value == "GGM02"
    assert ConstantSource.LEGNARO2024.value == "LEGNARO2024"
    assert PRIMER_EARTH_J2_SOURCE is ConstantSource.GGM02
    assert PRIMER_MOON_A_KM_SOURCE is ConstantSource.SIMON1994
    assert PRIMER_EARTH_GM_SOURCE is ConstantSource.NAIF


def test_primer_citation_names_the_primer():
    assert "Rosengren" in PRIMER_CITATION
    assert "Primer" in PRIMER_CITATION
