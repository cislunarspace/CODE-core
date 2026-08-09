"""Python↔Rust ABI 版本戳散度与行为测试。

验证 Rust 侧 ``_py_abi_version()`` 与 Python 侧 ``_MIN_REQUIRED_RUST_ABI``
一致（CI 散度兜底），以及过期/缺失二进制下的正确失败模式。

本文件标 ``pytest.mark.spice``：需编译扩展（非内核）；无二进制时整文件跳过。
行为测试（monkeypatch 注入，不依赖真实二进制）仍在此文件内，但因模块级
``importorskip`` 会一并跳过；如需无二进制环境运行行为测试，请移至
``tests/data/test_rust_signature_guard.py``（该文件已有等价的缺失/过期测试）。
"""

from __future__ import annotations

import types

import pytest

from e2m2e import integrators as gw

pytestmark = pytest.mark.spice

# 模块级 gate：无编译扩展时跳过全文件（CI 有 .pyd，不跳过）
_py_abi_version = pytest.importorskip(
    "e2m2e._integrators", reason="compiled Rust extension not available"
)._py_abi_version


class TestRustAbiVersion:
    """ABI 版本戳散度与行为契约。"""

    def test_rust_reports_abi_version(self):
        """Rust 扩展暴露 _py_abi_version()，返回正整数。"""
        v = _py_abi_version()
        assert isinstance(v, int)
        assert v >= 1

    def test_divergence_check(self):
        """Rust ABI 版本 >= Python 所需最低版本（防只改一边）。"""
        actual = _py_abi_version()
        # 优先使用 build.rs 生成的 _ABI_VERSION，回落到 gw._MIN_REQUIRED_RUST_ABI
        try:
            from e2m2e._rust_abi import _ABI_VERSION

            min_required = _ABI_VERSION
        except ImportError:
            min_required = gw._MIN_REQUIRED_RUST_ABI
        assert actual >= min_required, (
            f"ABI 散度：Rust 报 v{actual}，Python 要求 >= v{min_required}。"
            "请 bump crates/e2m2e-integrators/abi-version.txt 后重新构建。"
        )

    def test_generated_abi_matches_rust(self):
        """build.rs 生成的 _rust_abi._ABI_VERSION 与 Rust 报告的版本一致。"""
        try:
            from e2m2e._rust_abi import _ABI_VERSION
        except ImportError:
            pytest.skip("_rust_abi.py not generated (run maturin develop first)")
        assert _py_abi_version() == _ABI_VERSION

    def test_stale_binary_raises_runtime_error(self, monkeypatch):
        """过期二进制（模拟 version < min）→ RuntimeError + maturin 提示。"""
        monkeypatch.setattr(gw, "_MIN_REQUIRED_RUST_ABI", 9999)
        # Reset the cache so the check runs again
        monkeypatch.setattr(gw, "_abi_ok", False)

        with pytest.raises(RuntimeError, match="maturin"):
            gw._check_rust_abi()

    def test_absent_extension_degrades_silently(self, monkeypatch):
        """扩展缺失 → 静默跳过（_abi_ok = True，不抛）。"""
        import sys

        monkeypatch.setitem(sys.modules, "e2m2e._integrators", None)
        monkeypatch.setattr(gw, "_abi_ok", False)
        gw._check_rust_abi()
        assert gw._abi_ok is True

    def test_import_time_check_catches_stale_binary(self, monkeypatch):
        """当 _integrators 已加载但版本过期时，__init__ 的 import-time 校验报错。"""
        import sys

        # Simulate: extension loaded (in sys.modules) with a stale version
        monkeypatch.setitem(
            sys.modules,
            "e2m2e._integrators",
            types.SimpleNamespace(_py_abi_version=lambda: 0),  # stale
        )
        monkeypatch.setattr(gw, "_abi_ok", False)
        monkeypatch.setattr(gw, "_MIN_REQUIRED_RUST_ABI", 1)
        # Re-trigger the check logic directly
        with pytest.raises(RuntimeError, match="maturin"):
            gw._check_rust_abi()
