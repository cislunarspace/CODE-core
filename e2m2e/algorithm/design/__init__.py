"""任务轨道设计（三段编排）。

回答"一个任务参数怎么变成一条可用的标称轨道"。三段编排（ADR 0011）：
family（初猜）→ ephemeris_correction（CR3BP→星历修正）→ propagation（高精度预报）。

六类轨道：DRO / NRHO / Halo / Lissajous / L4 / L5。

实现状态：骨架。完整实现待从 ``dfh/design_orbit.py`` 迁入（编排逻辑留这里，
api/ 只做 Pydantic 校验 + 薄调用）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["OrbitDesignResult", "design_orbit", "DesignNotConvergedError"]


class DesignNotConvergedError(RuntimeError):
    """星历修正未收敛。"""


@dataclass
class OrbitDesignResult:
    """任务轨道设计结果。

    Attributes:
        orbit_type: 轨道类型（"DRO"/"NRHO"/"Halo"/"Lissajous"/"L4"/"L5"）。
        epoch_utc: 起始历元 UTC（ISO 字符串）。
        duration_day: 维持时间（天）。
        initial_state: 历元时刻惯性系状态（km, km/s）。
        ephemeris: 标称星历（通用星历容器）。
        cr3bp_orbit: CR3BP 参考轨道（可选）。
        correction: 星历修正结果（可选）。
        force_config: 力模型配置字典。
    """

    orbit_type: str
    epoch_utc: str
    duration_day: float
    initial_state: Any
    ephemeris: Any
    cr3bp_orbit: Any = None
    correction: Any = None
    force_config: dict[str, Any] = field(default_factory=dict)


def design_orbit(
    orbit_type: str,
    *,
    amplitude: float | None = None,
    phase: float | None = None,
    collinear_point: int | None = None,
    north_south: int | None = None,
    perilune_height: float | None = None,
    amplitude_in: float | None = None,
    amplitude_out: float | None = None,
    phase_in: float | None = None,
    phase_out: float | None = None,
    epoch: Any = (2024, 1, 1, 0, 0, 0.0),
    duration: float = 1.0,
    output_step: float = 3600.0,
    force_config: dict[str, Any] | None = None,
    correction_method: str = "two_level",
    verbose: bool = False,
) -> OrbitDesignResult:
    """端到端设计六类标称轨道。

    实现状态：骨架。完整实现待从 ``dfh/design_orbit.py`` 迁入。无参调用将抛
    ``NotImplementedError``。

    Args:
        orbit_type: "DRO"/"NRHO"/"Halo"/"Lissajous"/"L4"/"L5"。
        amplitude: 振幅（km）。DRO 1737~110000 默认 10000；Halo ±73000 默认 30000。
        phase: 初始相位（周期份额 0~1）。
        collinear_point: 共线平动点 1/2/3。
        north_south: 1=北 / 2=南（NRHO）。
        perilune_height: 近月点高度（km，100~10000，NRHO）。
        amplitude_in / amplitude_out: 面内/面外振幅（km，Lissajous/L4/L5）。
        phase_in / phase_out: 面内/面外相位（0~1）。
        epoch: 起始历元 UTC。
        duration: 维持时间（年，0 < d ≤ 20）。
        output_step: 星历输出间隔（秒）。
        force_config: 力模型配置（缺省用默认标称配置）。
        correction_method: 星历修正方法（"standard"/"two_level"/"homotopy"）。
        verbose: 显示修正进度。

    Returns:
        OrbitDesignResult。

    Raises:
        NotImplementedError: 实现未完成（骨架）。
    """
    raise NotImplementedError(
        "design_orbit 实现未完成（待从 dfh/design_orbit.py 迁入），能力在规划中"
    )
