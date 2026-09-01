"""native 符号守卫的行为测试（#603）。

符号存在与缺失两种情况下谓词分别判真/判假（用 monkeypatch 模拟，不改
真实扩展）；skip 的落点是 pytest 的 skipif 机制本身。文案口径与
spice_ext 的报错一致，由 skip reason 断言钉住。
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest
from kernel_helpers import native_symbols_available, requires_native_symbols

pytestmark = pytest.mark.aux


@pytest.fixture
def fake_facade(monkeypatch):
    """往 sys.modules 放一个带指定符号的 integrators 门面替身。"""
    module = ModuleType("e2m2e.integrators")
    monkeypatch.setitem(sys.modules, "e2m2e.integrators", module)
    return module


def test_symbols_present_reads_true(fake_facade):
    fake_facade.propagate_geocentric_fate_map_py = object()
    assert native_symbols_available("propagate_geocentric_fate_map_py")


def test_missing_symbol_reads_false(fake_facade):
    fake_facade.other_symbol = object()
    assert not native_symbols_available("propagate_geocentric_fate_map_py")


def test_unimportable_facade_reads_false(monkeypatch):
    monkeypatch.setitem(sys.modules, "e2m2e.integrators", None)  # import 即 ImportError
    assert not native_symbols_available("any_symbol")


def test_skip_reason_cites_make_dev(fake_facade):
    """skip reason 复用 spice_ext 的 make dev 口径（#603：不另造表述）。"""
    fake_facade.other_symbol = object()
    mark = requires_native_symbols("propagate_geocentric_fate_map_py")
    assert mark.mark.name == "skipif"
    reason = mark.mark.kwargs["reason"]
    assert "make dev" in reason
    assert "propagate_geocentric_fate_map_py" in reason
