"""转移轨道设计。

按数学类型组织（ADR 0011）：脉冲路径（lambert/three_body_lambert/multi_impulse）、
自然动力学路径（low_energy/manifold，覆盖引力辅助数学内核）、低推力路径
（low_thrust/）、任务层（search/optimize/porkchop）。``transfer_orbit.py`` 是
编排器：接收 transfer_type（HMN/LGA/WSB/小推力），按枚举选路径组合底层数学模块。

实现状态：骨架。数学模块待从 ``transfer/`` 迁入；transfer_orbit.py 编排器为
新类型。未实现（对外承诺能力）：LGA/WSB 引力辅助弹道搜索，占位抛
``NotImplementedError``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["TransferDesignResult", "transfer_orbit"]


@dataclass
class TransferDesignResult:
    """转移轨道设计结果。

    Attributes:
        transfer_type: 转移类型（"HMN"/"LGA"/"WSB"/"low_thrust"）。
        delta_v: 总 Δv（km/s）。
        trajectory: 转移轨迹。
        details: 设计细节（弹道参数汇总）。
    """

    transfer_type: str
    delta_v: float
    trajectory: Any
    details: dict[str, Any] = field(default_factory=dict)


def transfer_orbit(
    transfer_type: str,
    *,
    target_ephemeris: Any = None,
    tli_params: Any = None,
    tof_range: tuple[float, float] | None = None,
    **kwargs,
) -> TransferDesignResult:
    """端到端转移轨道设计。

    Args:
        transfer_type: "HMN"（直接）/ "LGA"（月球引力辅助）/ "WSB"（太阳引力辅助）/
            "low_thrust"（小推力）。
        target_ephemeris: 目标轨道星历（FR1 产物）。
        tli_params: 地球停泊轨道参数（TLI 高度/倾角/航迹角）。
        tof_range: 飞行时间范围（天）。

    Raises:
        NotImplementedError: 实现未完成（骨架）；或 transfer_type 对应能力未实现。
    """
    raise NotImplementedError(
        f"transfer_orbit('{transfer_type}') 实现未完成（能力在规划中）"
    )
