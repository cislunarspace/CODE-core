"""轨道库（catalog）请求/响应模型与扫描网格维度表。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from pydantic import ConfigDict, Field, model_validator

from e2m2e.status import ConvergenceState, FailureCause

from .shared import ResultResponse, _ApiModel

__all__ = [
    "CatalogQueryRequest",
    "CatalogRecordSummary",
    "CatalogQueryResponse",
    "CatalogGetRequest",
    "CatalogRecordResponse",
    "CatalogDeleteRequest",
    "CatalogDeleteResponse",
    "CatalogTagRequest",
    "CatalogTagResponse",
    "CatalogTerminologyResponse",
    "CatalogExportRequest",
    "CatalogExportResponse",
    "CatalogSweepRequest",
    "CatalogSweepPointOutcome",
    "CatalogSweepResponse",
]

# ---------------------------------------------------------------------------
# 轨道库 catalog（ADR 0031）：查询/读取/删除/标注/提升/导出/批量生成。
# 分类六维度可组合过滤；区间维度（jacobi/amplitude）与记录的 [min, max]
# 包络做相交匹配。
# ---------------------------------------------------------------------------


class CatalogQueryRequest(_ApiModel):
    """轨道库多维查询过滤；各维度可独立组合（逻辑与）。"""

    orbit_family: str | None = Field(
        default=None,
        description="轨道族（dro/halo/nrho/lissajous/dpo/axial/spo/lpo/horseshoe/elfo 等）",
    )
    family_id: str | None = Field(
        default=None,
        description="生成批次标识（ADR 0045）：整族查询的句柄——返回该次生成的全部成员记录",
    )
    libration_point: int | None = Field(default=None, ge=1, le=5, description="平动点编号 1–5")
    jacobi_min: float | None = Field(default=None, description="Jacobi 常数区间下界")
    jacobi_max: float | None = Field(default=None, description="Jacobi 常数区间上界")
    amplitude_min_km: float | None = Field(default=None, ge=0.0, description="主振幅区间下界（km）")
    amplitude_max_km: float | None = Field(default=None, ge=0.0, description="主振幅区间上界（km）")
    has_cr3bp: bool | None = Field(default=None, description="是否含 CR3BP 段")
    has_ephemeris: bool | None = Field(default=None, description="是否含星历段")
    status: ConvergenceState | None = Field(
        default=None, description="按结果状态筛（如筛掉软失败产物）"
    )
    tags: list[str] | None = Field(default=None, description="按标签筛，命中任一即匹配")
    transfer_type: str | None = Field(
        default=None,
        description="转移类型等值过滤（HMN/LGA/WSB/low_thrust/PCN；#574 transfer record）",
    )
    delta_v_min_km_s: float | None = Field(
        default=None, ge=0.0, description="转移总 Δv 区间下界（km/s）"
    )
    delta_v_max_km_s: float | None = Field(
        default=None, ge=0.0, description="转移总 Δv 区间上界（km/s）"
    )
    tli_epoch_min: float | None = Field(
        default=None,
        description=(
            "TLI 历元区间下界（JD_TDB 数值）。仅数值历元入索引；UTC 字符串历元记录不匹配区间过滤"
        ),
    )
    tli_epoch_max: float | None = Field(
        default=None,
        description="TLI 历元区间上界（JD_TDB 数值）",
    )

    @model_validator(mode="after")
    def _validate_ranges(self) -> CatalogQueryRequest:
        if (
            self.jacobi_min is not None
            and self.jacobi_max is not None
            and self.jacobi_min > self.jacobi_max
        ):
            raise ValueError(
                f"jacobi_min 不得大于 jacobi_max：{self.jacobi_min} > {self.jacobi_max}"
            )
        if (
            self.amplitude_min_km is not None
            and self.amplitude_max_km is not None
            and self.amplitude_min_km > self.amplitude_max_km
        ):
            raise ValueError(
                f"amplitude_min_km 不得大于 amplitude_max_km："
                f"{self.amplitude_min_km} > {self.amplitude_max_km}"
            )
        if (
            self.delta_v_min_km_s is not None
            and self.delta_v_max_km_s is not None
            and self.delta_v_min_km_s > self.delta_v_max_km_s
        ):
            raise ValueError(
                f"delta_v_min_km_s 不得大于 delta_v_max_km_s："
                f"{self.delta_v_min_km_s} > {self.delta_v_max_km_s}"
            )
        if (
            self.tli_epoch_min is not None
            and self.tli_epoch_max is not None
            and self.tli_epoch_min > self.tli_epoch_max
        ):
            raise ValueError(
                f"tli_epoch_min 不得大于 tli_epoch_max：{self.tli_epoch_min} > {self.tli_epoch_max}"
            )
        return self


class CatalogRecordSummary(_ApiModel):
    """记录摘要：浏览大量记录时轻量，不含数组段与请求快照。"""

    record_id: str
    created_at: str
    source_tool: str
    source_record_id: str | None
    orbit_family: str | None
    libration_point: int | None
    jacobi: list[float] | None = Field(
        description="记录 Jacobi 包络 [min, max]；无 CR3BP 段为 None"
    )
    amplitude: list[float] | None = Field(
        description="主振幅包络 [min, max]（km）；无 CR3BP 段为 None"
    )
    has_cr3bp: bool
    has_ephemeris: bool
    taxonomy_labels: list[str] | None = Field(
        default=None,
        description="分类学标签（ADR 0042，多标签规范字符串）；未打标为 None",
    )
    transfer_type: str | None = Field(
        default=None, description="转移类型（HMN/LGA/WSB/low_thrust/PCN）；非 transfer 记录为 None"
    )
    delta_v_km_s: float | None = Field(
        default=None, description="转移总 Δv（km/s）；非 transfer 记录为 None"
    )
    tli_epoch: float | None = Field(
        default=None, description="TLI 历元（JD_TDB 数值）；UTC 字符串历元或非 transfer 记录为 None"
    )
    status: ConvergenceState
    cause: FailureCause
    message: str
    family_id: str | None = Field(
        default=None,
        description="生成批次标识（ADR 0045，族标签之一）；整族经 catalog_query(family_id=…) 查询；"
        "单条设计/受控/转移记录为 None",
    )
    member_index: int | None = Field(
        default=None, description="族内成员序号（自 0 起，延续走行顺序）；非族成员记录为 None"
    )
    tags: list[str]
    note: str


class CatalogQueryResponse(ResultResponse):
    """catalog_query 输出。"""

    records: list[CatalogRecordSummary]


class CatalogGetRequest(_ApiModel):
    """按 record_id 取完整记录。"""

    record_id: str = Field(min_length=1)


class CatalogRecordResponse(CatalogRecordSummary):
    """完整记录：元数据全文 + 数组段（numpy 值，键含 ``/`` 段前缀）。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    scalars: dict[str, Any] = Field(description="任务标量（历元、时长、mu、迭代次数等）")
    request: dict[str, Any] = Field(description="原始请求快照（族成员各自携带同一份生成请求）")
    details: dict[str, Any] | None = Field(
        default=None,
        description="设计细节块（transfer 记录：后端 details + maneuver_events；其余为 None）",
    )
    arrays: dict[str, Any] = Field(
        description="数组段：cr3bp/ 与 eph/ 前缀的 numpy 数组；一条记录只载一条轨迹（ADR 0045）"
    )

    def to_ephemeris_table(self) -> Any | None:
        """把星历段重建为 ``EphemerisTable`` 实例（供接续计算）；无星历段返回 None。"""
        from e2m2e.data.catalog import ephemeris_from_arrays

        if not self.has_ephemeris:
            return None
        return ephemeris_from_arrays(self.arrays)

    def to_orbit(self) -> Any | None:
        """把单轨道 CR3BP 段重建为 ``Orbit``；纯星历/转移记录返回 None。"""
        import numpy as np

        from e2m2e.data.types.orbit import Orbit

        states = self.arrays.get("cr3bp/states")
        times = self.arrays.get("cr3bp/times")
        if states is None or times is None:
            return None
        return Orbit(states=np.asarray(states), times=np.asarray(times))


class CatalogDeleteRequest(_ApiModel):
    """按 record_id 删除记录。"""

    record_id: str = Field(min_length=1)


class CatalogDeleteResponse(ResultResponse):
    """catalog_delete 输出。"""

    record_id: str
    deleted: bool


class CatalogTagRequest(_ApiModel):
    """写教学标注（随 JSON 记录走）；``tags`` 整体替换，``note=None`` 保留原注释。"""

    record_id: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list, description="标签列表（整体替换）")
    note: str | None = Field(default=None, description="自由文本注释；None 保留原注释")


class CatalogTagResponse(ResultResponse):
    """catalog_tag 输出：更新后的记录摘要。"""

    record: CatalogRecordSummary


class CatalogTerminologyResponse(ResultResponse):
    """catalog_terminology 输出：调用方渲染结果所需的全部闭值集（ADR 0044）。

    无参数；包版本即术语版本（清单随发布冻结，调用方每会话取一次、
    升级后刷新，未知标签按可读规范串原样渲染）。
    """

    taxonomy_labels: dict[str, dict[str, Any]] = Field(
        description="分类学标签图例：规范字符串 → 结构化字段"
        "（category/family/libration_point/hemisphere/resonance_p/resonance_q，ADR 0042）"
    )
    orbit_families: list[str] = Field(description="记录侧 orbit_family 闭值集（族名清单）")
    transfer_types: list[str] = Field(description="转移类型闭值集（HMN/LGA/WSB/low_thrust/PCN）")


class CatalogExportRequest(CatalogQueryRequest):
    """子集打包导出：过滤条件同 catalog_query，外加目标路径。"""

    dest: str = Field(
        min_length=1,
        description="目标路径；以 .zip 结尾产出 zip 包，否则产出目录（records/ + manifest.json）",
    )


class CatalogExportResponse(ResultResponse):
    """catalog_export 输出。"""

    dest: str
    record_ids: list[str]
    exported_count: int


#: catalog_sweep 各族可作主参数维度的请求字段（条件取值域的单一来源，
#: ADR 0014 决策 8）：LISSAJOUS 只走二维振幅网格（能量窗口不适用），
#: 其余族一维振幅/近月点维度与能量窗口二选一。
_SWEEP_GRID_DIMENSIONS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "HALO": frozenset({"max_amplitudes_km", "jacobi_windows"}),
        "NRHO": frozenset({"perilune_heights_max_km", "jacobi_windows"}),
        "AXIAL": frozenset({"max_amplitudes_km", "jacobi_windows"}),
        "LISSAJOUS": frozenset({"amplitude_ins_km", "amplitude_outs_km"}),
        "SPO": frozenset({"max_amplitudes_km", "jacobi_windows"}),
        "LPO": frozenset({"max_amplitudes_km", "jacobi_windows"}),
        "HORSESHOE": frozenset({"max_amplitudes_km", "jacobi_windows"}),
    }
)


class CatalogSweepRequest(_ApiModel):
    """参数空间扫描批量生成并入库（编排复用 ADR 0029 的 Rust 族生成）。

    扫描网格 = 族 × 平动点 × 主参数维度。主参数维度三选一（同传报错）：

    - 一维主参数：HALO/AXIAL/SPO/LPO/HORSESHOE 扫 ``max_amplitudes_km``，
      NRHO 扫 ``perilune_heights_max_km``；
    - 能量（Jacobi）窗口：``jacobi_windows``——同一（族、平动点）只走
      一次延拓 trace（族延拓范围取各族默认），各窗口成员分别成记录，
      记录 jacobi 包络落在窗口内；窗口零成员时该点无记录、结局可查；
    - LISSAJOUS 二维振幅网格：``amplitude_ins_km`` × ``amplitude_outs_km``
      笛卡尔积逐点采样（相位取请求默认值）；能量窗口不适用于 LISSAJOUS
      （其族生成是参数采样而非延拓 trace）。

    部分参数点失败时已产出的记录保留（ADR 0020 软失败语义）。
    """

    orbit_types: list[str] = Field(
        min_length=1, description="族集合：HALO/NRHO/AXIAL/LISSAJOUS/SPO/LPO/HORSESHOE"
    )
    libration_points: list[int] | None = Field(
        default=None, description="平动点集合；缺省按各族默认（共线 L2、三角 L4）"
    )
    max_amplitudes_km: list[float] | None = Field(
        default=None,
        description="族振幅上限网格（km）；HALO/AXIAL 带符号区分北/南（上/下）族；"
        "SPO/LPO/HORSESHOE 的下限取各族默认",
    )
    perilune_heights_max_km: list[float] | None = Field(
        default=None, description="近月点高度上限网格（km），仅 NRHO 用"
    )
    jacobi_windows: list[list[float]] | None = Field(
        default=None,
        min_length=1,
        description="能量（Jacobi）窗口网格 [[min, max], ...]，边界包含；"
        "与 max_amplitudes_km/perilune_heights_max_km 互斥；LISSAJOUS 不适用",
    )
    amplitude_ins_km: list[float] | None = Field(
        default=None,
        min_length=1,
        description="LISSAJOUS 面内振幅网格（km）；须与 amplitude_outs_km 同给，与其他网格维度互斥",
    )
    amplitude_outs_km: list[float] | None = Field(
        default=None,
        min_length=1,
        description="LISSAJOUS 面外振幅网格（km）；须与 amplitude_ins_km 同给",
    )
    n_orbits: int = Field(default=20, ge=1, description="每点族成员数量上限")

    @classmethod
    def supported_grid_dimensions(cls, orbit_type: str) -> tuple[str, ...]:
        """返回该族可作为扫描主参数维度的请求字段（条件取值域公开）。

        GUI/CLI/MCP 不得解析错误文本或维护本地副本（ADR 0014 决策 8）；
        Facade 的网格展开与校验共用本接口。LISSAJOUS 的两个振幅字段是
        同一维度（须同给）；其余族的 ``jacobi_windows`` 与一维振幅字段
        互斥，由请求级校验器拒绝同传。
        """
        if not isinstance(orbit_type, str):
            raise ValueError(f"orbit_type 必须为字符串，当前 {orbit_type!r}")
        try:
            return tuple(sorted(_SWEEP_GRID_DIMENSIONS[orbit_type.upper()]))
        except KeyError as exc:
            raise ValueError(f"不支持的 orbit_type: {orbit_type!r}") from exc

    @model_validator(mode="after")
    def _validate_grid_dimensions(self) -> CatalogSweepRequest:
        """主参数维度互斥与能量窗口取值域（ADR 0014 决策 8 同源规则）。"""
        amplitude_grid = (
            self.max_amplitudes_km is not None or self.perilune_heights_max_km is not None
        )
        lissajous_grid = self.amplitude_ins_km is not None or self.amplitude_outs_km is not None
        given = [
            name
            for name, present in (
                ("max_amplitudes_km/perilune_heights_max_km", amplitude_grid),
                ("jacobi_windows", self.jacobi_windows is not None),
                ("amplitude_ins_km/amplitude_outs_km", lissajous_grid),
            )
            if present
        ]
        if len(given) > 1:
            raise ValueError(f"扫描主参数维度互斥：{' 与 '.join(given)} 同传；一次调用只选一个维度")
        if (self.amplitude_ins_km is None) != (self.amplitude_outs_km is None):
            raise ValueError(
                "LISSAJOUS 二维振幅网格须同时给出 amplitude_ins_km 与 amplitude_outs_km"
            )
        for window in self.jacobi_windows or ():
            valid_length = len(window) == 2
            finite = all(math.isfinite(value) for value in window)
            if not valid_length or not finite or not window[0] < window[1]:
                raise ValueError(
                    f"jacobi_windows 每项须为有限数对 [min, max] 且 min < max，当前 {window!r}"
                )
        return self


class CatalogSweepPointOutcome(_ApiModel):
    """扫描单参数点的结局：成功（含软失败）保留 family_id，硬失败保留原因。"""

    orbit_type: str
    libration_point: int
    parameter_km: float | None = Field(
        default=None,
        description="网格点主参数值（振幅或近月点高度上限，km）；能量窗口与二维振幅点为 None",
    )
    jacobi_window: list[float] | None = Field(
        default=None, description="能量窗口点的 [min, max] Jacobi 窗口"
    )
    amplitudes_km: list[float] | None = Field(
        default=None, description="LISSAJOUS 二维网格点的 [面内, 面外] 振幅（km）"
    )
    status: ConvergenceState
    cause: FailureCause
    message: str
    family_id: str | None
    generated_members: int


class CatalogSweepResponse(ResultResponse):
    """catalog_sweep 输出。

    ``succeeded`` 为产出记录的参数点数（含软失败但有成员产出的点）；
    ``failed`` 为硬失败（无产出）参数点数；软失败且零成员的点两者都
    不计，其结局见 ``points`` 逐点状态。
    """

    points: list[CatalogSweepPointOutcome]
    family_ids: list[str] = Field(
        description="成功参数点的生成批次标识列表（成员记录逐条入库，ADR 0045）"
    )
    succeeded: int
    failed: int
