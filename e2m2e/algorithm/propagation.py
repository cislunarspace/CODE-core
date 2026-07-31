"""轨道预报：给定初值与力模型的高精度数值外推。

单段能力（不建独立编排器，ADR 0011）：配 ForceModel + 调 propagate + 输出
EphemerisTable。单文件模块（不是目录）。

实现状态：骨架。完整实现待从 ``core/forces/force_model.py`` 的 ForceModel.propagate
薄封装。
"""

from __future__ import annotations

from typing import Any

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

    实现状态：骨架。完整实现待迁入。

    Args:
        initial_state: 初值（GCRS，km, km/s）。
        epoch: 起始历元 UTC。
        duration: 预报时长（秒或天，单位由约定）。
        force_config: 力模型配置。
        output_step: 输出间隔（秒）。

    Returns:
        预报星历表。

    Raises:
        NotImplementedError: 实现未完成（骨架）。
    """
    raise NotImplementedError("propagate_orbit 实现未完成（能力在规划中）")
