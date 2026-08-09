"""Python（spiceypy）与 Rust（cspice-sys）双 CSPICE 实例一致性测试。

两个实例相互独立（静态链接、内核池/名字表互不共享）。``SPICEManager.load_kernel``
是唯一在两侧同时 furnsh + boddef 的入口；本测试经诊断 API ``spice_spkezr``
从 Python 直接查询 Rust 实例，对比两侧结果，守护同步不被破坏（issue #334）。

- ``test_spkezr_dual_consistency_by_id``：用 NAIF ID 查询，守内核**数据**同步。
- ``test_spkezr_dual_consistency_by_name``：用名字查询，守 **boddef** 同步
  （两侧都须把 "MARS" 解析成质心 ID 4，否则 de440 不含本体段会报错）。
- ``test_rust_query_no_kernel_clear_error``：Rust 实例无内核时，诊断 API 应抛
  项目语境错误（含"无内核加载"、不含裸 CSPICE 码）。用子进程隔离 Rust 全局
  状态——主进程的其他测试可能已 furnsh 过 Rust 侧且无法 unload。
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
import spiceypy

from e2m2e.integrators import reset_ephem_ffi_call_count, spice_spkezr

pytestmark = [pytest.mark.l2, pytest.mark.spice]


@pytest.fixture(autouse=True)
def _reset_ffi_count():
    """每个测试前后清零 Rust FFI 调用计数，避免污染其他"零 cspice"测试。"""
    reset_ephem_ffi_call_count()
    yield
    reset_ephem_ffi_call_count()


def test_spkezr_dual_consistency_by_id(spice_manager):
    """两侧 spkezr 用 NAIF ID 查询应数值一致（守内核数据同步）。"""
    et = 0.0  # J2000，de440 覆盖范围内
    py_state, _py_lt = spiceypy.spkezr("399", et, "J2000", "NONE", "10")
    rust_state, _rust_lt = spice_spkezr("399", et, "J2000", "NONE", "10")
    np.testing.assert_allclose(py_state, rust_state, atol=1e-9)


def test_spkezr_dual_consistency_by_name(spice_manager):
    """两侧 spkezr 用名字查询应数值一致（守 boddef 同步：MARS→质心 4）。"""
    et = 0.0
    py_state, _py_lt = spiceypy.spkezr("MARS", et, "J2000", "NONE", "10")
    rust_state, _rust_lt = spice_spkezr("MARS", et, "J2000", "NONE", "10")
    np.testing.assert_allclose(py_state, rust_state, atol=1e-9)


def test_rust_query_no_kernel_clear_error():
    """Rust CSPICE 实例无内核时，spice_spkezr 抛项目语境错误（子进程隔离）。"""
    # 子进程：全新 Python 解释器，Rust CSPICE 全局状态干净（未 furnsh）。
    # 主进程内 Rust 侧可能已被其他测试 furnsh 且无 unload 包装，无法清理。
    code = (
        "from e2m2e.integrators import spice_spkezr\n"
        "try:\n"
        "    spice_spkezr('399', 0.0, 'J2000', 'NONE', '10')\n"
        "    print('NO_ERROR')\n"
        "except RuntimeError as e:\n"
        "    print('ERROR:', str(e))\n"
        "except Exception as e:\n"
        "    print('OTHER:', type(e).__name__, str(e))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    output = result.stdout + result.stderr
    assert "ERROR:" in output, f"期望 RuntimeError（项目语境错误），实际输出: {output!r}"
    assert "无内核加载" in output, f"错误信息应含'无内核加载'，实际: {output!r}"
    # 不含裸 CSPICE 内部错误码（ADR 0020：项目语境错误优先于内部码翻译）
    assert "SPKINSUFFDATA" not in output, f"不应含裸 CSPICE 码，实际: {output!r}"
