"""轨道预报：给定初值与力模型的高精度数值外推。

单段能力（不建独立编排器，ADR 0011）：配 ForceModel + 调 propagate + 输出
EphemerisTable。单文件模块（不是目录）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..data.types import EphemerisTable

__all__ = ["propagate_orbit"]


def propagate_orbit(
    initial_state: Any,
    epoch: Any,
    duration: float,
    force_config: dict[str, Any] | None = None,
    output_step: float = 3600.0,
    **kwargs,
) -> EphemerisTable:
    """高精度轨道预报。

    配 ForceModel 并传播，输出通用星历表容器（UTC + GCRS 位置/速度）。

    Args:
        initial_state: 初值（GCRS，km, km/s，形状 (6,)）。
        epoch: 起始历元 UTC（ISO 字符串或 ``[年,月,日,时,分,秒]``）。
        duration: 预报时长（秒）。
        force_config: 力模型配置（缺省用空力模型）。
        output_step: 输出间隔（秒）。
        kwargs: 传给 ForceModel 的额外配置（如 system/spice 等）。

    Returns:
        预报星历表。

    Raises:
        ValueError: 初值形状或时长非法。
        NotImplementedError: force_config 缺失且无默认力模型可用（当前占位）。
    """
    state = np.asarray(initial_state, dtype=float)
    if state.shape != (6,):
        raise ValueError(f"initial_state 应为 (6,)，实际 {state.shape}")
    if float(duration) <= 0:
        raise ValueError(f"duration 必须为正数，当前 {duration}")

    # 薄封装：力模型配置 → 传播 → EphemerisTable。当前为骨架，能力在规划中
    # （FR4 全链路接入 dfh 编排后落地）。
    raise NotImplementedError(
        "propagate_orbit 实现未完成（能力在规划中）：待接入 ForceModel + EphemerisSystem"
    )
