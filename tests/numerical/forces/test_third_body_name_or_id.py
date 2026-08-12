"""``ThirdBodyGravity._name_or_id`` 异常收窄测试（#352）。

修复前 ``bods2c`` 的 ``except Exception`` 把编程错误（TypeError 等）一并
吞掉并静默返回原名——真正的 bug 被藏进"名字未注册"的合理路径。收窄后
只 catch spiceypy 错误（``SpiceyError``），编程错误上抛。
"""

from __future__ import annotations

import pytest

from e2m2e.algorithm.forces.third_body_gravity import ThirdBodyGravity

pytestmark = pytest.mark.force


class TestNameOrIdNarrowedExcept:
    def test_registered_body_returns_naif_id(self):
        """bods2c 可识别的天体（MOON）返回 NAIF ID 字符串。"""
        assert ThirdBodyGravity._name_or_id("MOON") == "301"

    def test_unregistered_name_returns_original(self):
        """未注册名字（spiceypy 抛 SpiceyError）原样返回。"""
        assert ThirdBodyGravity._name_or_id("NOT_A_REAL_BODY_XYZ") == "NOT_A_REAL_BODY_XYZ"

    def test_programming_error_not_swallowed(self, monkeypatch):
        """编程错误（非 spiceypy 异常）不再被吞（#352：收窄 except）。"""
        import spiceypy

        def boom(name):
            raise ValueError("boom: 内部编程错误")

        monkeypatch.setattr(spiceypy, "bods2c", boom)
        with pytest.raises(ValueError, match="boom"):
            ThirdBodyGravity._name_or_id("MOON")
