"""行星际 MGA 链网格搜索（mission_architecture_search）请求/响应模型。"""

from __future__ import annotations

import math

from pydantic import ConfigDict, Field, model_validator

from .shared import ResultResponse, _ApiModel

__all__ = [
    "MissionArchitectureSearchRequest",
    "MissionArchitectureSearchResponse",
    "MgaFlybyInfo",
    "MgaChainCandidate",
]


class MissionArchitectureSearchRequest(_ApiModel):
    """行星际 MGA 链网格搜索输入（#723，ADR 0054）。

    纯网格搜索：发射窗口按 ``launch_window_step_days`` 进给，逐 leg TOF 区间按
    ``leg_tof_step_days`` 进给，两者笛卡尔积的全部组合逐格求解。中间天体即飞越
    体（无动力 V∞ 旋转）。
    """

    body_sequence: list[str] = Field(
        min_length=2,
        description="天体序列（SPICE 天体名，首项出发体，中间天体为飞越体，如 EVEJ）",
    )
    launch_window: list[str | float] = Field(
        min_length=2,
        max_length=2,
        description=(
            "发射窗口两端点 [start, end]（UTC ISO 字符串或 JD_TDB 浮点数；"
            "网格含 start、按步长进给）"
        ),
    )
    launch_window_step_days: float = Field(
        default=1.0, gt=0.0, description="发射历元网格步长（天）"
    )
    leg_tof_ranges: list[list[float]] = Field(
        min_length=1,
        description="逐 leg 飞行时间范围 [[min, max], ...]（天；长度 = 天体序列长度 − 1）",
    )
    leg_tof_step_days: float = Field(
        default=1.0, gt=0.0, description="逐 leg TOF 网格步长（天，全 leg 共用）"
    )
    min_flyby_pericenter_km: list[float] = Field(
        default_factory=list,
        description=(
            "逐飞越天体最小近心点半径 (km)，长度 = 天体序列长度 − 2（两体序列为"
            "空表；体半径倍数换算由调用方负责）"
        ),
    )
    v_inf_match_tol_km_s: float = Field(
        default=0.1,
        ge=0.0,
        description=("无动力 flyby 的 |v∞in|=|v∞out| 等模筛选容差 (km/s)，超差格剔除"),
    )
    top_n: int = Field(default=5, ge=1, le=50, description="返回候选数上限（按总 ΔV 升序）")

    @model_validator(mode="after")
    def _validate_grid_consistency(self) -> MissionArchitectureSearchRequest:
        """校验网格维度一致性与端点可解析性（违反 → INVALID_PARAMS）。"""
        n_bodies = len(self.body_sequence)
        n_legs = n_bodies - 1
        if len(self.leg_tof_ranges) != n_legs:
            raise ValueError(
                f"leg_tof_ranges 长度须为 {n_legs}（天体序列长度 − 1），"
                f"得到 {len(self.leg_tof_ranges)}"
            )
        for k, leg_range in enumerate(self.leg_tof_ranges):
            if len(leg_range) != 2:
                raise ValueError(f"leg {k} 的 TOF 范围须为 [min, max] 两个数，得到 {leg_range}")
            lower, upper = leg_range
            if not (math.isfinite(lower) and math.isfinite(upper)):
                raise ValueError(f"leg {k} 的 TOF 范围须为有限值，得到 {leg_range}")
            if not lower < upper:
                raise ValueError(f"leg {k} 的 TOF 范围须满足 min < max，得到 {leg_range}")
        if len(self.min_flyby_pericenter_km) != n_bodies - 2:
            raise ValueError(
                f"min_flyby_pericenter_km 长度须为 {n_bodies - 2}（天体序列长度 − 2），"
                f"得到 {len(self.min_flyby_pericenter_km)}"
            )
        if any(
            not math.isfinite(radius) or radius <= 0.0 for radius in self.min_flyby_pericenter_km
        ):
            raise ValueError(
                f"min_flyby_pericenter_km 须全为正的有限值，得到 {self.min_flyby_pericenter_km}"
            )
        numeric: list[float | None] = []
        for label, endpoint in zip(("start", "end"), self.launch_window, strict=True):
            if isinstance(endpoint, str):
                if not endpoint.strip():
                    raise ValueError(f"launch_window 的 {label} 为空字符串")
                numeric.append(None)
                continue
            value = float(endpoint)
            if not math.isfinite(value):
                raise ValueError(f"launch_window 的 {label} 须为有限 JD_TDB，得到 {endpoint!r}")
            numeric.append(value)
        start, end = numeric
        if start is not None and end is not None and not start < end:
            raise ValueError(
                f"launch_window 须满足 start < end（JD_TDB），得到 {self.launch_window}"
            )
        return self


class MgaFlybyInfo(_ApiModel):
    """单次无动力飞越的搜索结果（日心 Tisserand 为借力前后诊断值）。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    body: str = Field(description="飞越天体名")
    v_inf_km_s: float = Field(description="进入/离开双曲段 V∞ 模 (km/s)")
    turn_angle_deg: float = Field(description="flyby 转角 δ（度）")
    pericenter_radius_km: float | None = Field(
        default=None,
        description="近心点半径 (km)；δ=0（零退化）时为 null，避免 JSON 非有限值（#698 口径）",
    )
    tisserand_before: float = Field(
        description="借力前日心 Tisserand 参数（参考体 = 该飞越天体半长轴）"
    )
    tisserand_after: float = Field(
        description="借力后日心 Tisserand 参数（无动力下名义不变，供调用方自筛）"
    )


class MgaChainCandidate(_ApiModel):
    """一条可行 MGA 链候选（按总 ΔV 升序排列）。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    launch_epoch_jd_tdb: float = Field(description="发射历元（JD_TDB）")
    leg_tofs_days: list[float] = Field(description="逐 leg 飞行时间（天），长度 = 天体数 − 1")
    departure_v_inf_km_s: float = Field(description="出发双曲剩余速度模 (km/s)")
    arrival_v_inf_km_s: float = Field(description="到达双曲剩余速度模 (km/s)")
    total_delta_v_km_s: float = Field(
        description="总 ΔV = 出发 v∞ + 到达 v∞ (km/s)（MGA 快筛惯例，无中途冲量）"
    )
    flybys: list[MgaFlybyInfo] = Field(
        default_factory=list, description="逐中间天体的飞越评估，长度 = 天体数 − 2"
    )


class MissionArchitectureSearchResponse(ResultResponse):
    """行星际 MGA 链网格搜索输出（#723）。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    candidates: list[MgaChainCandidate] = Field(
        default_factory=list,
        description=(
            "可行候选（按总 ΔV 升序，至多 top_n 条）：无候选时为空表，"
            "状态三元组给出原因（Lambert 无解 / flyby 约束全剔）"
        ),
    )
