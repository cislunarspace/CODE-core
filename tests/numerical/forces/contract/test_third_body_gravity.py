"""ThirdBodyGravity 定义与序列化契约。

覆盖构造接口（body 大写、无 origin 参数）、``to_rust_spec`` 序列化、
``_name_or_id`` 异常收窄与 PhysicalModel 子类关系。物理对照见
``physics/test_third_body_gravity.py``。
"""

from __future__ import annotations

import pytest

from e2m2e.algorithm.forces import PhysicalModel, ThirdBodyGravity

pytestmark = pytest.mark.force


class TestThirdBodyGravityInterface:
    """测试类的基本构造与接口。"""

    def test_is_physical_model(self):
        """ThirdBodyGravity 应是 PhysicalModel 子类。"""
        assert issubclass(ThirdBodyGravity, PhysicalModel)

    def test_body_uppercased(self):
        """body 应被存为大写。"""
        force = ThirdBodyGravity(body="moon")
        assert force.body == "MOON"

    @pytest.mark.spice
    def test_to_rust_spec_serializes_body_and_mu(self, spice_eph_system):
        """to_rust_spec 返回 ("third_body", body, mu)。

        body 字段是 NAIF ID 字符串：spiceypy 可用且天体在 boddef 注册时
        （MOON→"301"），否则原样名字。环境固定 spiceypy>=8.1.0，故 MOON
        解析为 "301"。
        """
        force = ThirdBodyGravity("MOON")
        spec = force.to_rust_spec(spice_eph_system)
        assert spec[0] == "third_body"
        assert spec[1] == "301"
        assert spec[2] == pytest.approx(spice_eph_system.get_gm("MOON"))

    def test_no_origin_parameter_exposed(self):
        """构造函数不应暴露 origin 参数（接口约定）。"""
        import inspect

        sig = inspect.signature(ThirdBodyGravity.__init__)
        assert "origin" not in sig.parameters


class TestNameOrIdNarrowedExcept:
    """``_name_or_id`` 异常收窄。

    若对 ``bods2c`` 用宽泛的 ``except Exception``，编程错误（TypeError 等）会
    被一并吞掉并静默返回原名，真正的 bug 被藏进"名字未注册"的合理路径。
    收窄为只 catch spiceypy 错误（``SpiceyError``），编程错误上抛。
    """

    def test_registered_body_returns_naif_id(self):
        """bods2c 可识别的天体（MOON）返回 NAIF ID 字符串。"""
        assert ThirdBodyGravity._name_or_id("MOON") == "301"

    def test_unregistered_name_returns_original(self):
        """未注册名字（spiceypy 抛 SpiceyError）原样返回。"""
        assert ThirdBodyGravity._name_or_id("NOT_A_REAL_BODY_XYZ") == "NOT_A_REAL_BODY_XYZ"

    def test_programming_error_not_swallowed(self, monkeypatch):
        """编程错误（非 spiceypy 异常）不被吞（收窄 except）。"""
        import spiceypy

        def boom(name):
            raise ValueError("boom: 内部编程错误")

        monkeypatch.setattr(spiceypy, "bods2c", boom)
        with pytest.raises(ValueError, match="boom"):
            ThirdBodyGravity._name_or_id("MOON")
