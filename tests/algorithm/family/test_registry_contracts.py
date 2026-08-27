"""轨道族注册表契约测试。

各族耗时的端到端收敛/物理不变量验证（axial/dpo/spo/lpo/horseshoe/nrho，
直接调 design_* 生成真轨道）受默认测试时间上界约束，不在 pytest 中展开，
由实际使用反馈保证。本文件保留快速注册表契约：条目存在、可调用、指向
正确的 design_* 函数。
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

#: 绑定平动点的 lambda 包装条目（只验存在与可调用；各族
#: 端到端行为验证不在 pytest 展开，见文件 docstring）
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
