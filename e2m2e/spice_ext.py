"""Thin shared-kernel binding for the Rust extension's SPICE surface.

The data layer must drive the Rust-side cspice instance directly: it keeps
its own kernel pool, separate from the Python spiceypy instance (dual-instance
bridge, ADR 0016), so kernel load/unload and the ephemeris-cache switches have
to reach ``spice_furnsh``/``spice_unload``/``enable_ephem_cache``/
``disable_ephem_cache`` without routing through the numerics facade
``e2m2e.integrators`` (which itself reaches the algorithm layer). This
package-root leaf is that legal channel: importable by every layer, importing
no layer itself (ADR 0039). It also owns the extension ABI gate shared with
the facade—``_check_rust_abi``/``_abi_ok``—and a ``require_rust_extension``
checking its own module-level symbols; the facade keeps a thin shell with the
same behaviour, checking facade globals.
"""
# ruff: noqa: F821, F822

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from e2m2e.exceptions import RustExtensionUnavailableError

# 扩展符号在运行时逐个装载；静态类型检查将其视为动态对象。
if TYPE_CHECKING:
    disable_ephem_cache: Any
    enable_ephem_cache: Any
    spice_furnsh: Any
    spice_unload: Any

__all__ = [
    "disable_ephem_cache",
    "enable_ephem_cache",
    "require_rust_extension",
    "spice_furnsh",
    "spice_unload",
]

# ---- Python↔Rust ABI 版本校验 ----
# 单一来源：crates/e2m2e-integrators/abi-version.txt
# build.rs 在 maturin develop 时生成 e2m2e/_rust_abi.py；未构建时回落到硬编码默认值。
try:
    from e2m2e._rust_abi import _ABI_VERSION as _MIN_REQUIRED_RUST_ABI
except ImportError:
    _MIN_REQUIRED_RUST_ABI: int = 1  # type: ignore[no-redef]  # 构建前/无扩展时的安全默认值

_abi_ok: bool = False  # 进程级一次性缓存


def _check_rust_abi() -> None:
    """校验 Rust 扩展 ABI 版本；过期或缺失即报，结果进程级缓存。

    在首次使用 Rust 扩展符号时调用（惰性）。扩展不存在时抛
    :class:`RustExtensionUnavailableError` （带 ``make dev`` 指引），
    不静默降级。过期二进制抛 ``RuntimeError``。
    """
    global _abi_ok
    if _abi_ok:
        return
    try:
        rust_extension = importlib.import_module("e2m2e._integrators")
    except ImportError as exc:
        raise RustExtensionUnavailableError(
            "e2m2e._integrators 不可用（Rust 扩展未构建）。请先构建：make dev"
        ) from exc
    _py_abi_version = getattr(rust_extension, "_py_abi_version", None)
    if _py_abi_version is None:
        raise RustExtensionUnavailableError(
            "e2m2e._integrators 缺少所需符号：_py_abi_version。请先构建：make dev"
        )
    actual = _py_abi_version()
    if actual < _MIN_REQUIRED_RUST_ABI:
        raise RuntimeError(
            f"e2m2e._integrators 编译产物过期（ABI v{actual} < 所需 v{_MIN_REQUIRED_RUST_ABI}）。"
            "请重建 Rust 扩展：make dev"
        )
    _abi_ok = True


# ---- 扩展符号装载（与 integrators 门面同一 sys.modules 条目，符号对象同一） ----

_SPICE_SYMBOLS = (
    "disable_ephem_cache",
    "enable_ephem_cache",
    "spice_furnsh",
    "spice_unload",
)

try:
    _rust_extension: Any = importlib.import_module("e2m2e._integrators")
except ImportError:
    _rust_extension = None

for _symbol in _SPICE_SYMBOLS:
    globals()[_symbol] = getattr(_rust_extension, _symbol, None)


def _ensure_symbols(namespace: dict[str, Any], required_symbols: tuple[str, ...]) -> None:
    """检查命名空间内的扩展符号非 None；缺失即抛带 ``make dev`` 指引的异常。

    门面与本模块各自的 ``require_rust_extension`` 共用，防止两份报错文案漂移
    （文案被 ``test_rust_signature_guard`` 的 ``match="make dev"`` 钉住）。
    """
    missing = [name for name in required_symbols if namespace.get(name) is None]
    if missing:
        raise RustExtensionUnavailableError(
            "e2m2e._integrators 缺少所需符号："
            + ", ".join(missing)
            + "。spice 是默认且唯一支持的 feature；请用 make dev 重建扩展。"
        )


def require_rust_extension(*required_symbols: str) -> None:
    """确保 Rust 扩展可用且指定的模块级符号存在；否则抛 ``RustExtensionUnavailableError``。

    在使用 Rust 扩展符号的每个入口调用。扩展未构建、构建不含 spice
    feature、或符号缺失时，抛带 ``make dev`` 指引的
    :class:`RustExtensionUnavailableError`，不允许静默回退到 Python/scipy。
    ``required_symbols`` 是 ``e2m2e.spice_ext`` 模块级
    符号名；扩展缺失时符号为 ``None``。

    Example:
        >>> require_rust_extension("spice_furnsh", "enable_ephem_cache")
    """
    _check_rust_abi()
    _ensure_symbols(globals(), required_symbols)
