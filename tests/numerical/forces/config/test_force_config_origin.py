"""力配置与传播原点的一致性守卫测试。

覆盖 ``ForceModel.from_config`` 对带天体参数的力的角色校验：中心项力必须取
传播原点，第三体引力与间接项必须取非原点天体。``FakeSystem`` 自带地心原点，
整个文件无需 SPICE。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from e2m2e.algorithm.forces import ForceModel
from tests.numerical.forces.conftest import FakeSystem

pytestmark = pytest.mark.force


def _entry(name: str, type_name: str, params: dict) -> dict:
    return {"name": name, "type": type_name, "params": params}


def _config(*entries: dict) -> dict:
    return {"version": 1, "forces": list(entries)}


class TestOriginConsistencyRejects:
    """与传播原点角色冲突的力配置在构建时被拒绝。"""

    def test_third_body_at_origin_rejected(self):
        """第三体引力不能以传播原点天体为摄动天体。"""
        config = _config(_entry("third_earth", "ThirdBodyGravity", {"body": "EARTH"}))
        with pytest.raises(ValueError, match="ThirdBodyGravity"):
            ForceModel.from_config(config, FakeSystem())

    def test_point_mass_not_at_origin_rejected(self):
        """点质量引力假定状态以该天体为原点，非原点天体不可用。"""
        config = _config(_entry("moon", "PointMassGravity", {"body": "MOON"}))
        with pytest.raises(ValueError, match="PointMassGravity"):
            ForceModel.from_config(config, FakeSystem())

    def test_indirect_term_at_origin_rejected(self):
        """间接项扣除的是摄动天体对原点的引力，不能取原点天体本身。"""
        config = _config(_entry("indirect_earth", "IndirectTerm", {"body": "EARTH"}))
        with pytest.raises(ValueError, match="IndirectTerm"):
            ForceModel.from_config(config, FakeSystem())

    def test_relativistic_central_body_not_at_origin_rejected(self):
        """相对论修正围绕 central_body 计算，必须等于传播原点。"""
        config = _config(_entry("rel", "RelativisticCorrection", {"central_body": "MOON"}))
        with pytest.raises(ValueError, match="RelativisticCorrection"):
            ForceModel.from_config(config, FakeSystem())

    def test_drag_atmosphere_not_at_origin_rejected(self):
        """大气阻力用原点天体的密度模型与相对速度，非原点天体不可用。"""
        config = _config(
            _entry(
                "drag",
                "DragModel",
                {
                    "atmosphere": {"type": "ExponentialAtmosphere", "params": {}},
                    "area": 5.0,
                    "mass": 1000.0,
                    "body": "MOON",
                },
            )
        )
        with pytest.raises(ValueError, match="DragModel"):
            ForceModel.from_config(config, FakeSystem())

    def test_drag_default_body_rejected_away_from_earth(self):
        """DragModel 缺省天体为 EARTH，月心传播下同样被拒绝。"""
        system = FakeSystem()
        system.origin = "MOON"
        config = _config(
            _entry(
                "drag",
                "DragModel",
                {
                    "atmosphere": {"type": "ExponentialAtmosphere", "params": {}},
                    "area": 5.0,
                    "mass": 1000.0,
                },
            )
        )
        with pytest.raises(ValueError, match="DragModel"):
            ForceModel.from_config(config, system)

    def test_error_message_names_entry_and_origin(self):
        """错误消息能定位到条目并说明传播原点。"""
        config = _config(_entry("third_earth", "ThirdBodyGravity", {"body": "EARTH"}))
        with pytest.raises(ValueError) as excinfo:
            ForceModel.from_config(config, FakeSystem())
        message = str(excinfo.value)
        assert "third_earth" in message
        assert "EARTH" in message


class TestOriginConsistencyAccepts:
    """角色与传播原点一致的力配置可以正常加载。"""

    def test_center_and_perturbers_at_earth(self):
        """地心点质量加非原点第三体与间接项可以加载。"""
        config = _config(
            _entry("earth", "PointMassGravity", {"body": "EARTH"}),
            _entry("moon", "ThirdBodyGravity", {"body": "MOON"}),
            _entry("moon_indirect", "IndirectTerm", {"body": "MOON"}),
            _entry("sun", "ThirdBodyGravity", {"body": "SUN"}),
        )
        assert isinstance(ForceModel.from_config(config, FakeSystem()), ForceModel)

    def test_relativistic_primary_body_unconstrained(self):
        """primary_body 不参与原点一致性，只约束 central_body。"""
        config = _config(
            _entry(
                "rel",
                "RelativisticCorrection",
                {"central_body": "EARTH", "primary_body": "MOON"},
            )
        )
        assert isinstance(ForceModel.from_config(config, FakeSystem()), ForceModel)

    def test_drag_at_origin_accepted(self):
        """大气阻力取传播原点天体时可以加载。"""
        config = _config(
            _entry(
                "drag",
                "DragModel",
                {
                    "atmosphere": {"type": "ExponentialAtmosphere", "params": {}},
                    "area": 5.0,
                    "mass": 1000.0,
                    "body": "EARTH",
                },
            )
        )
        assert isinstance(ForceModel.from_config(config, FakeSystem()), ForceModel)

    def test_gravity_field_non_origin_body_accepted(self):
        """球谐重力场可以模拟非原点天体，两种角色都合法。"""
        system = FakeSystem()
        system.origin = "MOON"
        config = _config(
            _entry("moon", "PointMassGravity", {"body": "MOON"}),
            _entry("earth", "GravityField", {"body": "EARTH", "degree": 2, "order": 0}),
        )
        assert isinstance(ForceModel.from_config(config, system), ForceModel)

    def test_origin_comparison_ignores_case(self):
        """原点与天体的比较忽略大小写。"""
        system = FakeSystem()
        system.origin = "earth"
        config = _config(_entry("earth", "PointMassGravity", {"body": "EARTH"}))
        assert isinstance(ForceModel.from_config(config, system), ForceModel)

    def test_disabled_inconsistent_entry_accepted(self):
        """禁用的条目不参与动力学，不触发守卫。"""
        config = _config(
            {
                **_entry("third_earth", "ThirdBodyGravity", {"body": "EARTH"}),
                "enabled": False,
            }
        )
        assert isinstance(ForceModel.from_config(config, FakeSystem()), ForceModel)

    def test_system_without_origin_skips_guard(self):
        """没有 origin 属性的系统不做原点一致性校验。"""
        system = SimpleNamespace(coordinate_system=object())
        config = _config(_entry("third_earth", "ThirdBodyGravity", {"body": "EARTH"}))
        assert isinstance(ForceModel.from_config(config, system), ForceModel)
