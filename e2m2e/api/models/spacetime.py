"""时空坐标转换（spacetime_transform）请求/响应模型。"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .shared import ResultResponse, _ApiModel

__all__ = [
    "SpacetimeTransformRequest",
    "SpacetimeTransformResponse",
]


class SpacetimeTransformRequest(_ApiModel):
    """时空坐标转换输入。"""

    states: list[list[float]] = Field(description="状态列表，每项 [x,y,z,vx,vy,vz]")
    times: list[float] = Field(
        description="每个状态的时间值：GCRS↔EBCRS 用 JD_TDB；synodic/EPPR 转换用相对 "
        "et0_jd 的无量纲时间（0 = et0_jd 参考历元）"
    )
    transform_type: str = Field(
        description="synodic_to_j2000/j2000_to_synodic/j2000_to_eppr/eppr_to_j2000/"
        "gcrs_to_ebcrs/ebcrs_to_gcrs"
    )
    et0_jd: float = Field(description="参考历元 JD_TDB")
    ephemeris_path: str | None = Field(default=None, description="历表路径（GCRS↔EBCRS 必需）")


class SpacetimeTransformResponse(ResultResponse):
    """时空坐标转换输出。"""

    states: list[list[float]]
    times: list[float]
    transform_type: str
    details: dict[str, Any]
