"""标准原点（CelestialBodyOrigin / InertialOrigin）测试。

验证 body 属性大写规范化。
"""

from __future__ import annotations

from e2m2e.algorithm.coordinate.standard_origins import CelestialBodyOrigin


class TestCelestialBodyOriginBodyProperty:
    """body 只读属性暴露构造时传入的天体名。"""

    def test_body_returns_constructor_body_uppercased(self) -> None:
        """body 属性返回构造时传入的天体名（大写规范化）。"""
        origin = CelestialBodyOrigin(body="earth", spice=None)
        assert origin.body == "EARTH"
