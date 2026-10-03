"""CR3BP 多项式流（DA 传播）绑定的行为测试（issue #786）。

覆盖三类事实：返回契约与零偏差求值一致；一阶截断的线性映射与既有 STM
一致；前置校验错误路径抛 ValueError。DA 上下文是进程级全局状态，每个
用例开头各自 da_init_py（同时重置线程截断阶），不跨用例持对象。
"""

from __future__ import annotations

import numpy as np
import pytest
from kernel_helpers import requires_native_symbols

from e2m2e.integrators import (
    da_init_py,
    propagate_cr3bp_da_py,
    propagate_cr3bp_stm_py,
)

pytestmark = [
    pytest.mark.integrator,
    requires_native_symbols("propagate_cr3bp_da_py", "da_init_py"),
]

# DE421 校准地月质量比与北族 L2 NRHO 折叠种子，出处同
# crates/e2m2e-integrators/src/family_generation/nrho.rs（种子周期在该 μ 下标定）。
MU = 0.012150585350562453
SEED = [1.128103754424342, 0.0, 0.17883236940616654, 0.0, -0.22553424464298827, 0.0]


def test_result_contract_and_zero_deviation():
    """dict 四键齐全；flows 为 6 个 Da；零偏差求值等于名义状态。"""
    da_init_py(2, 6)
    result = propagate_cr3bp_da_py(MU, (0.0, 0.5), [0.5], SEED, 2, 1e-3)
    assert set(result) == {"times", "states", "flows", "n_steps"}
    assert result["times"] == [0.5]
    assert isinstance(result["n_steps"], int) and result["n_steps"] == 500
    flows = result["flows"][-1]
    assert len(flows) == 6
    for poly, nominal in zip(flows, result["states"][-1], strict=True):
        assert poly.eval([0.0] * 6) == nominal


def test_first_order_matches_stm():
    """一阶截断的线性映射与既有 STM 传播一致（筛选级 1e-8）。"""
    da_init_py(1, 6)
    arc = 0.5
    result = propagate_cr3bp_da_py(MU, (0.0, arc), [arc], SEED, 1, 1e-3)
    stm = propagate_cr3bp_stm_py(MU, (0.0, arc), [arc], SEED, 1e-12, 1e-12, None, None)
    linear = np.array([poly.linear() for poly in result["flows"][-1]])
    reference = np.array(stm["stm"][-1]).reshape(6, 6)
    rel_err = np.abs(linear - reference) / np.maximum(np.abs(reference), 1.0)
    assert rel_err.max() < 1e-8, f"一阶线性映射对 STM 的最大相对误差 {rel_err.max():.3e} 超过 1e-8"


def test_validation_errors():
    """前置校验错误路径统一抛 ValueError 并给出修复指引。"""
    da_init_py(2, 6)
    with pytest.raises(ValueError, match="initial_state"):
        propagate_cr3bp_da_py(MU, (0.0, 0.5), [0.5], SEED[:5], 2, 1e-3)
    with pytest.raises(ValueError, match="t_eval"):
        propagate_cr3bp_da_py(MU, (0.0, 0.5), [], SEED, 2, 1e-3)
    with pytest.raises(ValueError, match="截断阶"):
        propagate_cr3bp_da_py(MU, (0.0, 0.5), [0.5], SEED, 3, 1e-3)
    da_init_py(2, 3)
    with pytest.raises(ValueError, match="变量数"):
        propagate_cr3bp_da_py(MU, (0.0, 0.5), [0.5], SEED, 2, 1e-3)
