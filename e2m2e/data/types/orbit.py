"""Orbit 数据容器：单条轨道的状态序列与时间序列。

纯数据容器（ADR 0011）：states（形状 (n_points, 6)）+ times + metadata + 可选
``system`` 绑定（解释单位/坐标系）+ 手动设 ``period``（不自动算，需要时由调用方设）。

实现状态：骨架。待从 ``core/orbit.py`` 迁入（保留现 API）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...data.types.state import State

__all__ = ["Orbit"]


@dataclass
class Orbit:
    """单条轨道：状态随时间演化的轨迹容器。

    Attributes:
        states: 状态序列（形状 (n_points, 6)）。
        times: 时间序列（形状 (n_points,)）。
        system: 绑定的动力学系统（解释坐标系与单位），可选。
        period: 轨道周期，可选（不自动算，需要时由调用方设）。
        family_type: 轨道族类型标签，可选。
        parameters: 生成参数（如振幅/平动点），可选。
        metadata: 附加元数据（created/source/description 等），可选。
    """

    states: np.ndarray
    times: np.ndarray
    system: Any = None
    period: float | None = None
    family_type: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        states = np.asarray(self.states, dtype=float)
        times = np.asarray(self.times, dtype=float)
        if states.ndim != 2 or states.shape[1] != 6:
            raise ValueError(f"states 应为 (n_points, 6)，实际 {states.shape}")
        if times.ndim != 1 or len(times) != len(states):
            raise ValueError(f"times 长度应等于 states 行数，实际 {len(times)} vs {len(states)}")
        self.states = states
        self.times = times

    @property
    def state0(self) -> State:
        """首状态。"""
        return self.states[0]
