"""e2m2e 动力学模块

包含轨道力学的核心组件：传播、力模型、坐标变换和 SPICE 绑定。

子模块:
    propagation: 传播相关（动力学系统、轨道、积分）
    coord_trans: 坐标变换（Synodic↔J2000、ITRF、IAU）
    spice: SPICE 绑定（星历查询、内核管理）
"""

from __future__ import annotations

import importlib

# 延迟导入 SPICE 相关模块，避免强制加载 spiceypy
_LAZY_SPICE_EXPORTS: dict[str, str] = {
    "SPICEManager": "e2m2e.dynamics.spice.spice",
    "EphemerisSystem": "e2m2e.dynamics.propagation.ephemeris_system",
    "EphemerisDynamics": "e2m2e.dynamics.propagation.ephemeris_dynamics",
}


def __getattr__(name: str) -> object:
    """按需延迟导入 SPICE/星历相关符号。"""
    module_name = _LAZY_SPICE_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'e2m2e.dynamics' has no attribute '{name}'")

    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """确保 dir(e2m2e.dynamics) 包含延迟导出的公开符号。"""
    return sorted(set(__all__) | set(globals().keys()) | set(_LAZY_SPICE_EXPORTS.keys()))


__all__ = [
    # 子模块
    "propagation",
    "coord_trans",
    "spice",
    # 延迟导入的符号
    "SPICEManager",
    "EphemerisSystem",
    "EphemerisDynamics",
]
