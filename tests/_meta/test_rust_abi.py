"""Python↔Rust ABI 版本戳测试。"""

from __future__ import annotations

import pytest

from e2m2e import integrators as gw

pytestmark = [pytest.mark.aux, pytest.mark.spice]

_py_abi_version = pytest.importorskip(
    "e2m2e._integrators", reason="compiled Rust extension not available"
)._py_abi_version


class TestRustAbiVersion:
    """真实 Rust 扩展的 ABI 版本契约。"""

    def test_rust_reports_abi_version(self):
        """Rust 扩展暴露 ``_py_abi_version()``，并返回正整数。"""
        v = _py_abi_version()
        assert isinstance(v, int)
        assert v >= 1

    def test_divergence_check(self):
        """Rust ABI 不得低于 Python 所需最低版本。"""
        actual = _py_abi_version()
        try:
            from e2m2e._rust_abi import _ABI_VERSION

            min_required = _ABI_VERSION
        except ImportError:
            min_required = gw._MIN_REQUIRED_RUST_ABI
        assert actual >= min_required, (
            f"ABI 散度：Rust 报 v{actual}，Python 要求 >= v{min_required}。"
            "请运行 make dev 重建扩展。"
        )

    def test_generated_abi_matches_rust(self):
        """构建期生成的 ABI 版本与 Rust 报告值一致。"""
        try:
            from e2m2e._rust_abi import _ABI_VERSION
        except ImportError:
            pytest.skip("_rust_abi.py 未生成（请先运行 make dev）")
        assert _py_abi_version() == _ABI_VERSION

    def test_stale_binary_raises_runtime_error(self, monkeypatch):
        """过期二进制在首次使用时提示用 ``make dev`` 重建。"""
        monkeypatch.setattr(gw, "_MIN_REQUIRED_RUST_ABI", 9999)
        monkeypatch.setattr(gw, "_abi_ok", False)

        with pytest.raises(RuntimeError, match="make dev"):
            gw._check_rust_abi()

    def test_import_time_check_catches_stale_binary(self, monkeypatch):
        """已加载的过期扩展在 ABI 检查时明确失败。"""
        import types

        monkeypatch.setitem(
            __import__("sys").modules,
            "e2m2e._integrators",
            types.SimpleNamespace(_py_abi_version=lambda: 0),
        )
        monkeypatch.setattr(gw, "_abi_ok", False)
        monkeypatch.setattr(gw, "_MIN_REQUIRED_RUST_ABI", 1)

        with pytest.raises(RuntimeError, match="make dev"):
            gw._check_rust_abi()
