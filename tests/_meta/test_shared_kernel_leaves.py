"""共享内核叶（ADR 0039）的架构元测试。

re-export 必须保持对象身份——旧路径是永久稳定别名，不是复制。
spice_ext 与 integrators 门面装载的是同一扩展模块的同一批符号对象。
"""

from __future__ import annotations

import pytest

from e2m2e import integrators, spice_ext, status
from e2m2e.algorithm.results import CAUSE_STATUS, ResultStatus
from e2m2e.data.templates import ConvergenceState, FailureCause

pytestmark = pytest.mark.aux

_SPICE_SYMBOLS = (
    "disable_ephem_cache",
    "enable_ephem_cache",
    "spice_furnsh",
    "spice_unload",
)


def test_status_reexports_preserve_identity():
    """旧路径（data.templates / algorithm.results）与 e2m2e.status 是同一批对象。"""
    assert status.ConvergenceState is ConvergenceState
    assert status.FailureCause is FailureCause
    assert status.ResultStatus is ResultStatus
    assert status.CAUSE_STATUS is CAUSE_STATUS
    # 防"复制而非 re-export"回归：定义点必须是共享内核叶本身。
    assert ResultStatus.__module__ == "e2m2e.status"
    assert ConvergenceState.__module__ == "e2m2e.status"


def test_spice_ext_symbols_match_facade():
    """spice_ext 与 integrators 门面暴露同一扩展符号（扩展缺失时两侧同为 None）。"""
    for name in _SPICE_SYMBOLS:
        assert getattr(spice_ext, name) is getattr(integrators, name), name


def test_extension_gates_share_abi_core():
    """两道符号门是各自模块的薄壳，但共享同一 ABI 校验核心与进程缓存。"""
    assert integrators.require_rust_extension is not spice_ext.require_rust_extension
    assert integrators._check_rust_abi is spice_ext._check_rust_abi
