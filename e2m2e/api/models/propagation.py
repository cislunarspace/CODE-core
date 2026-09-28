"""轨道预报（orbit_propagation）请求/响应模型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .shared import ResultResponse, _ApiModel

__all__ = [
    "PropagationRequest",
    "PropagationResponse",
]


class PropagationRequest(_ApiModel):
    """轨道预报输入（对齐 algorithm/propagation 的 propagate_orbit 参数）。"""

    initial_state: list[float] = Field(
        min_length=6, max_length=6, description="初值（GCRS，km, km/s，长度 6）"
    )
    epoch: Any = Field(description="起始历元 UTC（ISO 字符串或 [年,月,日,时,分,秒]）")
    duration: float = Field(gt=0.0, description="预报时长（秒）")
    force_config: dict[str, Any] | None = Field(
        default=None, description="力模型配置（缺省用默认三体力模型）"
    )
    output_step: float = Field(default=3600.0, gt=0.0, description="输出间隔（秒）")
    direction: Literal["forward", "backward"] = Field(
        default="forward",
        description="传播方向：forward 自 epoch 正向预报，backward 自 epoch 反向"
        "回溯；duration 恒为正的时长幅值",
    )


class PropagationResponse(ResultResponse):
    """轨道预报输出。"""

    epoch_utc: str
    duration_sec: float
    direction: str = Field(description="实际传播方向回显（forward/backward）")
    output_step: float
    n_points: int
    time_sec: list[float]
    times_jd_tdb: list[float]
    position_km: list[list[float]]
    velocity_km_s: list[list[float]]
    final_state: list[float]
