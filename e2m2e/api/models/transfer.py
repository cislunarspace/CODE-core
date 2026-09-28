"""转移轨道设计（transfer_design）请求/响应模型。"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from .shared import ResultResponse, _ApiModel

__all__ = [
    "BplaneTarget",
    "DepartureAsymptote",
    "TransferDesignRequest",
    "ManeuverEvent",
    "TransferCandidate",
    "BplaneInfo",
    "TransferDesignResponse",
]


class BplaneTarget(_ApiModel):
    """PCN 到达模式目标：月心 B-plane（Vallado 定义，#635）。

    B 平面坐标：``bdot_t_km`` 沿 T̂ = normalize(ẑ×Ŝ)、``bdot_r_km`` 沿
    R̂ = Ŝ×T̂（Ŝ 为入射渐近线速度方向）；``perilune_alt_km`` 为月面以上
    近月点高度。
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    perilune_alt_km: float = Field(gt=0.0, description="目标近月点高度 (km，月面以上)")
    bdot_t_km: float = Field(description="目标 B·T (km)")
    bdot_r_km: float = Field(default=0.0, description="目标 B·R (km)，默认 0")


class DepartureAsymptote(_ApiModel):
    """地球出发双曲渐近线参数化（TLI 设计入口，#635）。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    rha_deg: float = Field(ge=0.0, lt=360.0, description="渐近线赤经 (deg)")
    dha_deg: float = Field(ge=-90.0, le=90.0, description="渐近线赤纬 (deg)")
    c3_km2_s2: float = Field(gt=0.0, description="C3 能量 (km²/s²)，双曲出发须 >0")


class TransferDesignRequest(_ApiModel):
    """转移轨道设计输入（对齐 algorithm/transfer 的 transfer_orbit 参数）。"""

    transfer_type: str = Field(description="HMN/LGA/WSB/low_thrust/PCN（patched-conic 目标参数化）")
    tli_epoch: Any = Field(description="TLI 历元（UTC ISO 字符串或 JD_TDB 浮点数）")
    parking_alt_km: float = Field(default=200.0, gt=0.0, description="地球停泊轨道高度 (km)")
    incl_deg: float = Field(
        default=28.5, ge=0.0, le=180.0, description="轨道倾角 (度；PCN 路径不参与构造)"
    )
    flight_path_deg: float = Field(
        default=0.0, ge=0.0, le=0.0, description="航迹角 (度，仅支持 0；PCN 路径不参与构造)"
    )
    target_ephemeris: Any = Field(
        default=None,
        description=(
            "目标星历（EphemerisTable/NominalOrbit/ndarray），LGA/WSB 必需。"
            "坐标系契约按转移类型区分：LGA/WSB 要求会合旋转系（synodic）物理单位"
            "（km, km/s）状态，编排器直接无量纲化，不做惯性系→旋转系转换，"
            "orbit_propagation/design_orbit 产出的惯性星历必须先经 "
            "spacetime_transform(j2000_to_synodic) 转换后再传入，否则目标态几何全错；"
            "HMN/low_thrust 按地心惯性系 km/km/s 状态解释；PCN 不使用（月球取 CR3BP"
            "圆型理想化几何）。"
        ),
    )
    target_orbit_radius_km: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "目标轨道半径 (km)，HMN 必需；地心距——从地心量起的圆轨道半径，"
            "非月心高度（环月取 ≈384400）"
        ),
    )
    tof_range: list[float] | None = Field(
        default=None,
        min_length=2,
        max_length=2,
        description=(
            "飞行时间范围 [min, max]（天；须为有限数对且 min < max）：HMN 作 "
            "Lambert 扫描窗口、WSB 作搜索窗口、PCN 覆盖 "
            "PcnSearchParams.tof_range_days（到达与出发两种模式共用的 tof 搜索网格）；"
            "LGA/low_thrust 不使用"
        ),
    )
    lga_search_params: Any = Field(default=None, description="LGA 搜索参数（LgaSearchParams 实例）")
    wsb_search_params: Any = Field(default=None, description="WSB 搜索参数（WsbSearchParams 实例）")
    engine_config: dict[str, Any] | None = Field(
        default=None,
        description="推进配置（low_thrust 必需）：{'t_max': 最大推力 N, 'isp': 比冲 s}",
    )
    initial_mass: float | None = Field(
        default=None, gt=0.0, description="初始质量 (kg)，low_thrust 必需"
    )
    n_segments: int = Field(default=10, gt=0, description="求解器段数（low_thrust，默认 10）")
    target_oe: list[float] | None = Field(
        default=None,
        min_length=3,
        max_length=3,
        description="Q-law 目标 [a_T (km), e_T, i_T (弧度)]（low_thrust 可选）",
    )
    solver_method: str = Field(
        default="shooting", description="求解方法 shooting/collocation（low_thrust）"
    )
    duration_days: float = Field(
        default=30.0, gt=0.0, description="飞行时间（天）（low_thrust，默认 30.0）"
    )
    departure_state: list[float] | None = Field(
        default=None,
        min_length=6,
        max_length=6,
        description="出发状态 [x,y,z,vx,vy,vz]（地心惯性系，km, km/s）（low_thrust 可选）",
    )
    target_state: list[float] | None = Field(
        default=None,
        min_length=6,
        max_length=6,
        description="目标末态 [x,y,z,vx,vy,vz]（地心惯性系，km, km/s）（low_thrust 可选）",
    )
    top_n: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description=(
            "top-N 可行解契约（#583，ADR 0040 增补；可选）：返回至多 N 个"
            "可行候选（按上报 Δv 升序，选中解标记），推荐值 DEFAULT_TOP_N"
            "=5。缺省 None 不开启，行为与单解契约逐字段一致"
        ),
    )
    bplane_target: BplaneTarget | None = Field(
        default=None,
        description=(
            "PCN 到达模式目标：月心 B-plane（#635；与 departure_asymptote 二选一）。"
            "给定近月点高度与 B·T/B·R，打靶求解出发渐近线使其命中"
        ),
    )
    departure_asymptote: DepartureAsymptote | None = Field(
        default=None,
        description=(
            "PCN 出发模式渐近线（#635；与 bplane_target 二选一）。"
            "给定出发双曲渐近线，无迭代解算月心 B-plane 与 LOI 脉冲"
        ),
    )

    @field_validator("tof_range")
    @classmethod
    def _validate_tof_range(cls, value: list[float] | None) -> list[float] | None:
        """``tof_range`` 须为有限数对 ``[min, max]`` 且 ``min < max``（#698）。

        形状、有限性与序在请求边界拒绝（映射 ``INVALID_PARAMS``），不把单元素
        列表留给编排器的 ``tof_range[1]`` 索引（``IndexError`` → ``TRANSFER_FAILED``），
        也不让反向/非有限窗口退化成「无交集」的求解结果。
        """
        if value is None:
            return None
        finite = all(math.isfinite(item) for item in value)
        if len(value) != 2 or not finite or not value[0] < value[1]:
            raise ValueError(f"tof_range 须为有限数对 [min, max] 且 min < max，当前 {value!r}")
        return value


class ManeuverEvent(_ApiModel):
    """单次机动事件（#575 契约；与 transfer catalog record 的 details 块共用 schema）。

    ``kind`` 为开放枚举（departure/perilune/arrival/…）；``t_sec`` 为 TLI
    起算秒（与 trajectory_times 同基准）；非脉冲事件（perilune 旗标）的
    ``dv_km_s`` 为 0.0。
    """

    kind: str = Field(description="事件类别：departure/perilune/arrival/…（开放枚举）")
    t_sec: float = Field(ge=0.0, description="TLI 起算秒（t=0 为出发脉冲）")
    dv_km_s: float = Field(ge=0.0, description="该次机动脉冲大小；非脉冲事件为 0.0")
    note: str | None = Field(default=None, description="可选人类可读注记")


class TransferCandidate(_ApiModel):
    """top-N 可行解契约的单个候选（#583，ADR 0040 增补）。"""

    delta_v_km_s: float = Field(
        description=(
            "该候选的总 Δv (km/s)。选中解与顶层 delta_v 同口径；"
            "未精化候选（refined=False）为搜索网格估计（精化前）"
        )
    )
    tli_epoch: float | str | None = Field(
        description="出发历元（UTC 字符串或 JD_TDB 浮点）；low_thrust 出发态直传路径为 None"
    )
    tof_sec: float = Field(description="飞行时间 (秒)，TLI 起算")
    trajectory: list[list[float]] | None = Field(
        default=None,
        description=(
            "轨迹快照 (n, 6)，数据系由 state_frame 标注（LGA/WSB 未精化"
            "候选为候选自由飞行弧，选中解与顶层 trajectory 同一条）；"
            "快照组装失败为 None"
        ),
    )
    trajectory_times: list[float] | None = Field(
        default=None,
        description=(
            "快照时刻 (n,) 秒，TLI 起算，与 trajectory 逐行对齐；无时刻（low_thrust）为 None"
        ),
    )
    # 字面量与 e2m2e/algorithm/transfer 的 STATE_FRAME_* 常量同源（同顶层
    # state_frame 字段，改动须两侧同步；此处不导入以保持 models 层轻依赖）
    state_frame: Literal["synodic_barycentric_km", "force_model_state"] = Field(
        description="快照的数据系标签（ADR 0040 state_frame 词汇，同顶层字段）"
    )
    selected: bool = Field(
        description=("是否为选中解（与默认路径最优解一致的那一个）；恰有一个候选为 True")
    )
    refined: bool = Field(
        description=(
            "Δv 口径：True 为打靶精化后数值（与顶层同口径），False 为搜索"
            "网格估计（精化前）；HMN/low_thrust 无搜索-精化两级，恒为 True"
        )
    )


class BplaneInfo(_ApiModel):
    """达成的月心 B-plane 参数（PCN 响应出口字段，#635）。

    ``perilune_alt_km`` 为近月点半径 − 月球平均半径；交会解（``CONVERGED``）为月面
    以上正值，撞月解（``COLLISION``，失败路径同样回显几何）为负值。
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    v_inf_km_s: float = Field(description="月心到达剩余速度 v∞ (km/s)")
    c3_km2_s2: float = Field(description="月心到达 C3 = v∞² (km²/s²)")
    rha_deg: float = Field(description="月心到达渐近线赤经 (deg)")
    dha_deg: float = Field(description="月心到达渐近线赤纬 (deg)")
    bdot_r_km: float = Field(description="达成 B·R (km)")
    bdot_t_km: float = Field(description="达成 B·T (km)")
    b_mag_km: float = Field(description="瞄准距离 |B| (km)")
    theta_deg: float = Field(description="B 矢量角 atan2(B·R, B·T) (deg)")
    perilune_alt_km: float = Field(
        description="近月点高度 = 近月点半径 − 月球平均半径 (km)：交会解为月面以上正值，撞月解为负"
    )


def _sanitize_nonfinite(value: Any) -> Any:
    """递归把非有限浮点（``nan``/``inf``）替换为 ``None``（#698）。

    ``details`` 是自由字段，后端在缺几何/零结果时用 ``nan``/``inf`` 占位；
    ``model_dump(mode="json")`` 仍保留它们，信封的 ``json.dumps`` 于是写出非法
    JSON 记号（``NaN``/``Infinity``），故响应构造期统一清洗。口径与 catalog 写侧
    ``catalog_ingest._sanitize_value`` 一致：``None`` 表示缺位。递归覆盖
    dict/list/tuple（与 ``Any`` 字段的常见嵌套形态一致）。
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _sanitize_nonfinite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_nonfinite(item) for item in value]
    return value


class TransferDesignResponse(ResultResponse):
    """转移轨道设计输出。"""

    transfer_type: str
    delta_v: float
    trajectory: list[list[float]] | None = Field(
        default=None,
        description=(
            "转移轨迹 (n, 6)：地月会合旋转系、质心原点、物理单位 km / km/s"
            "（ADR 0040；HMN 为两体几何的相位对齐显示约定，low_thrust 暂为"
            "力模型状态系的已知不一致）"
        ),
    )
    trajectory_times: list[float] | None = Field(
        default=None,
        description="轨迹时刻 (n,) 秒，TLI 起算（t=0 为出发脉冲），与 trajectory 逐行对齐",
    )
    trajectory_gcrs_km: list[list[float]] | None = Field(
        default=None,
        description=(
            "惯性几何段 (n, 6)（#584，ADR 0040 增补）：地心原点、不旋转轴"
            "（GCRS 约定）物理 km / km/s，与 trajectory 逐行对齐、共享"
            " trajectory_times（时刻不双份）；HMN 为两体弧构造系原样，"
            "LGA/WSB 为会合几何旋回惯性（θ₀=0 理想化方位，无星历语义），"
            "PCN 为原生 GCRS 两段弧拼接（月亮段逐行加月球惯性位置），"
            "low_thrust 与零结果为 None。段的数据系即词汇值 gcrs_km"
        ),
    )
    # 字面量与 e2m2e/algorithm/transfer 的 STATE_FRAME_* 常量同源，改动须两侧同步；
    # 此处不导入以保持 models 层轻依赖（schema 出口）。
    state_frame: Literal["synodic_barycentric_km", "force_model_state"] = Field(
        description=(
            "trajectory 的数据系标签（ADR 0040 增补）：synodic_barycentric_km"
            " = 地月会合旋转系质心原点物理 km/km/s（HMN/LGA/WSB/PCN）；"
            "force_model_state = 力模型状态系（low_thrust，已知不一致）。"
            "词汇 gcrs_km 已由并行惯性段 trajectory_gcrs_km 启用（#584），"
            "synodic_barycentric_nd 待后续批次接入"
        ),
    )
    maneuver_events: list[ManeuverEvent] = Field(
        default_factory=list,
        description=(
            "结构化机动事件列表（#575，按 t_sec 升序）：HMN 为 departure/"
            "arrival 两条（到达点即近月点）；LGA/WSB 含 perilune 旗标"
            "（dv_km_s=0）；PCN 为 departure/arrival 两条（arrival 即近月点"
            "圆化 LOI 单脉冲）；low_thrust 连续推进恒为空；搜索零结果恒为空。"
            "旧 details Δv 字段保留一个版本后废弃"
        ),
    )
    candidates: list[TransferCandidate] | None = Field(
        default=None,
        description=(
            "top-N 可行解候选（#583，ADR 0040 增补；opt-in 参数 top_n 开启时"
            "才下发）：按上报 Δv 升序，恰一个 selected=True 且与顶层结果同"
            "口径；未精化候选的 Δv 为网格估计（refined=False）。默认（不开）"
            "为 None；搜索零结果不携带"
        ),
    )
    bplane: BplaneInfo | None = Field(
        default=None,
        description=("达成的月心 B-plane 参数（#635；仅 PCN 路径填充，其余为 None）"),
    )
    departure_asymptote: DepartureAsymptote | None = Field(
        default=None,
        description=("实际使用的出发双曲渐近线（#635；仅 PCN 填充，与请求同 schema 回显实际值）"),
    )
    details: dict[str, Any]
    record_id: str | None = Field(
        default=None,
        description="产物自动入库的记录 id（ADR 0031，#574）；库关闭或无轨迹产物时为 None",
    )

    @model_validator(mode="after")
    def _sanitize_details(self) -> TransferDesignResponse:
        """``details`` 出口统一清洗非有限值（#698）。

        PCN 等后端在缺几何/零结果时以 ``nan``/``inf`` 占位；不清洗会让
        ``model_dump(mode="json")`` 保留它们，信封的 ``json.dumps`` 写出
        ``NaN``/``Infinity`` 记号。清洗只作用于这个自由字段，显式定型的顶层字段
        （如零结果的 ``delta_v=inf``）保持契约不变。
        """
        self.details = _sanitize_nonfinite(self.details)
        return self
