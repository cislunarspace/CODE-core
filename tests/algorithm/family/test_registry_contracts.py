"""轨道族注册表契约测试。

原各族的端到端收敛/物理不变量测试（axial/dpo/spo/lpo/horseshoe/nrho，
直接调 design_* 生成真轨道）因耗时过长从 pytest 移除（ADR 0021 修订
#420：默认测试时间上界靠缩小问题规模保证，端到端验证由实际使用反馈）。
本文件保留快速注册表契约：条目存在、可调用、指向正确的 design_* 函数。

关联：#534（测试耗时排查）、#536（修正器容差根因）。
"""

from __future__ import annotations

import pytest

from e2m2e.algorithm.family import registry
from e2m2e.algorithm.family.cr3bp_orbits import (
    design_axial,
    design_dpo,
    design_nrho,
)

pytestmark = pytest.mark.orchestration


#: 直绑 design_* 的条目（registry[key] is func）
_DIRECT_CASES = [
    ("AXIAL", design_axial),
    ("DPO", design_dpo),
    ("NRHO", design_nrho),
]

#: 绑定平动点的 lambda 包装条目（只验存在与可调用；原各族的
#: 端到端行为验证随重计算测试移除，见文件 docstring）
_LAMBDA_KEYS = [
    "L4_SPO",
    "L5_SPO",
    "L4_LPO",
    "L5_LPO",
    "L4_HORSESHOE",
    "L5_HORSESHOE",
]


@pytest.mark.parametrize("key, func", _DIRECT_CASES)
def test_registry_entry_is_bound_design_function(key: str, func) -> None:
    """直绑条目存在、可调用且指向正确的 design_* 函数。"""
    assert key in registry
    assert callable(registry[key])
    assert registry[key] is func


@pytest.mark.parametrize("key", _LAMBDA_KEYS)
def test_registry_wrapped_entry_exists_and_callable(key: str) -> None:
    """平动点绑定包装条目存在且可调用。"""
    assert key in registry
    assert callable(registry[key])
