"""ConicalShadowModel 构造与半径表契约。

验证默认遮挡体、body 大写规范化、半径表与 radii 覆盖；阴影几何物理见
``physics/test_shadow.py``。
"""

from __future__ import annotations

import pytest

from e2m2e.algorithm.forces.shadow import ConicalShadowModel

pytestmark = pytest.mark.force


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
