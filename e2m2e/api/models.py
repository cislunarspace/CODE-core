"""公开数据模型（Pydantic，全手写）。

输入/输出/错误模型精雕参数单位、默认值、取值域（ADR 0014）。Pydantic 只在
api/ 边界，算法层用 numpy/dataclass。每个 Facade 方法一个 Request/Response，
外加统一错误模型 ``OrbitError``。
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from e2m2e.algorithm.results import ResultStatus
from e2m2e.data.constants import SECONDS_PER_DAY
from e2m2e.data.templates import SEGMENTED_CORRECTION_ORBIT_TYPES, ConvergenceState, FailureCause
from e2m2e.data.templates.perturbations import DEFAULT_PERTURBATION
from e2m2e.data.templates.seed import _HALO_FOLD_Z0, CHAR_LENGTH_KM
from e2m2e.data.types.orbit import Orbit, OrbitFamily

__all__ = [
    "OrbitError",
    "NumericRange",
    "ResultResponse",
    "DesignOrbitRequest",
    "DesignOrbitResponse",
    "ControlOrbitRequest",
    "ControlOrbitResponse",
    "TransferDesignRequest",
    "TransferDesignResponse",
    "PropagationRequest",
    "PropagationResponse",
    "SpacetimeTransformRequest",
    "SpacetimeTransformResponse",
    "FamilyGenerationRequest",
    "FamilyGenerationResponse",
    "CatalogQueryRequest",
    "CatalogRecordSummary",
    "CatalogQueryResponse",
    "CatalogGetRequest",
    "CatalogRecordResponse",
    "CatalogDeleteRequest",
    "CatalogDeleteResponse",
    "CatalogTagRequest",
    "CatalogTagResponse",
    "CatalogPromoteRequest",
    "CatalogPromoteResponse",
    "CatalogExportRequest",
    "CatalogExportResponse",
    "CatalogSweepRequest",
    "CatalogSweepPointOutcome",
    "CatalogSweepResponse",
    "SpatiographyScalesRequest",
    "SpatiographyScalesResponse",
    "SpatiographyClassifyRequest",
    "SpatiographyClassifyResponse",
    "SpatiographyBoundariesRequest",
    "SpatiographyBoundariesResponse",
]


class OrbitError(Exception):
    """结构化错误（api/ 边界翻译，ADR 0014）。

    Attributes:
        code: 错误码（如 "NOT_IMPLEMENTED"/"NOT_CONVERGED"/"INVALID_PARAMS"）。
        message: 可读错误信息。
        details: 附加细节。
    """

    def __init__(
        self,
        code: str = "ERROR",
        message: str = "",
        details: dict[str, Any] | None = None,
        status: ConvergenceState = ConvergenceState.FAILED,
        cause: FailureCause = FailureCause.UNKNOWN,
    ) -> None:
        super().__init__(message)
        ResultStatus(status, cause, message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.status = status
        self.cause = cause

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class _ApiModel(BaseModel):
    """api 模型公共配置：允许任意内部字段，输出按声明序列化。"""

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class NumericRange:
    """数值参数的上下界、开闭区间及离散排除值。"""

    minimum: float | None = None
    maximum: float | None = None
    minimum_inclusive: bool = True
    maximum_inclusive: bool = True
    excluded_values: tuple[float, ...] = ()

    def contains(self, value: float) -> bool:
        """判断值是否落在此区间内且不属于排除值。"""
        if value in self.excluded_values:
            return False
        if self.minimum is not None and (
            value < self.minimum or (value == self.minimum and not self.minimum_inclusive)
        ):
            return False
        return not (
            self.maximum is not None
            and (value > self.maximum or (value == self.maximum and not self.maximum_inclusive))
        )

    def format_interval(self) -> str:
        """返回用于校验错误的紧凑区间表示。"""
        left = "[" if self.minimum_inclusive else "("
        right = "]" if self.maximum_inclusive else ")"
        return f"{left}{self.minimum}, {self.maximum}{right}"


def _range_map(**ranges: NumericRange) -> Mapping[str, NumericRange]:
    """构造不可变的按字段索引范围表。"""
    return MappingProxyType(ranges)


_GLOBAL_AMPLITUDE_OUT_RANGE = NumericRange(0.0, 76000.0, minimum_inclusive=False)


def _with_global_amplitude_out(
    ranges: Mapping[str, NumericRange],
) -> Mapping[str, NumericRange]:
    """为各类型范围补上共享字段 amplitude_out 的 API 上限。"""
    return MappingProxyType({"amplitude_out": _GLOBAL_AMPLITUDE_OUT_RANGE, **ranges})


_GLOBAL_AMPLITUDE_OUT_RANGES = _with_global_amplitude_out(_range_map())
_DRO_DPO_RANGES = _with_global_amplitude_out(_range_map(amplitude=NumericRange(1737.0, 110000.0)))
_SPO_RANGES = _with_global_amplitude_out(_range_map(amplitude=NumericRange(1737.0, 200000.0)))
_LPO_RANGES = _with_global_amplitude_out(_range_map(amplitude=NumericRange(1000.0, 110000.0)))
_HORSESHOE_RANGES = _with_global_amplitude_out(
    _range_map(amplitude=NumericRange(50000.0, 110000.0))
)

_ORBIT_TYPE_RANGES: Mapping[str, Mapping[str, NumericRange]] = MappingProxyType(
    {
        "DRO": _DRO_DPO_RANGES,
        "DPO": _DRO_DPO_RANGES,
        "HALO": _with_global_amplitude_out(_range_map(amplitude=NumericRange(-73000.0, 73000.0))),
        "NRHO": _with_global_amplitude_out(
            _range_map(perilune_height=NumericRange(100.0, 10000.0))
        ),
        "L4": _GLOBAL_AMPLITUDE_OUT_RANGES,
        "L5": _GLOBAL_AMPLITUDE_OUT_RANGES,
        "AXIAL": _with_global_amplitude_out(_range_map(amplitude=NumericRange(-60000.0, 60000.0))),
        "L4_SPO": _SPO_RANGES,
        "L5_SPO": _SPO_RANGES,
        "L4_LPO": _LPO_RANGES,
        "L5_LPO": _LPO_RANGES,
        "L4_HORSESHOE": _HORSESHOE_RANGES,
        "L5_HORSESHOE": _HORSESHOE_RANGES,
        "ELFO": _GLOBAL_AMPLITUDE_OUT_RANGES,
    }
)
_LISSAJOUS_L1_L2_RANGES = _range_map(
    amplitude_in=NumericRange(0.0, 7600.0, minimum_inclusive=False),
    amplitude_out=NumericRange(0.0, 7600.0, minimum_inclusive=False),
)
_LISSAJOUS_L3_RANGES = _range_map(
    amplitude_in=NumericRange(0.0, 100000.0, minimum_inclusive=False),
    amplitude_out=NumericRange(0.0, 100000.0, minimum_inclusive=False),
)


class DesignOrbitRequest(_ApiModel):
    """任务轨道设计输入。

    统一覆盖 CR3BP 周期轨道（DRO/NRHO/Halo/Lissajous/…）和 ELFO 冻结轨道。
    按 orbit_type 分派校验与默认值填充（``model_validator``）。
    duration 统一用秒。
    """

    orbit_type: str = Field(description="DRO/DPO/NRHO/HALO/LISSAJOUS/L4/L5/AXIAL/.../ELFO")
    # CR3BP 形状参数（字段约束为跨类型全局上下限；model_validator 内按类型收紧）
    amplitude: float | None = Field(default=None, ge=-110000.0, le=200000.0)
    phase: float | None = Field(default=None, ge=0.0, le=1.0)
    collinear_point: int | None = Field(default=None, ge=1, le=3)
    north_south: int | None = Field(default=None, ge=1, le=2)
    amplitude_in: float | None = Field(default=None, gt=0.0, le=100000.0)
    amplitude_out: float | None = Field(default=None, gt=0.0, le=100000.0)
    phase_in: float | None = Field(default=None, ge=0.0, le=1.0)
    phase_out: float | None = Field(default=None, ge=0.0, le=1.0)
    # 共享参数
    perilune_height: float | None = Field(default=None, gt=0.0, le=10000.0)
    # ELFO 形状参数
    inclination: float | None = Field(
        default=None, ge=0.0, le=180.0, description="倾角（度），ELFO 用"
    )
    arg_of_pericenter: float | None = Field(
        default=None, ge=0.0, lt=360.0, description="近月点幅角（度），ELFO 用，默认 270"
    )
    semi_major_axis: float | None = Field(
        default=None, gt=0.0, description="半长轴（km），ELFO 必填"
    )
    # 传播参数
    epoch: Any = Field(default=(2024, 1, 1, 0, 0, 0.0), description="[年,月,日,时,分,秒] 或 ISO")
    duration: float | None = Field(
        default=None, gt=0.0, description="传播时长（秒）；None 时按 orbit_type 填默认"
    )
    output_step: float = Field(default=3600.0, gt=0.0)
    perturbation: dict[str, int] | None = Field(default=None)
    dyb: list[float] | None = Field(default=None)
    earth_degree: int = Field(default=10, ge=2, le=120)
    moon_degree: int = Field(default=10, ge=2, le=120)
    # 修正参数
    correction_method: str = Field(
        default="two_level",
        description="星历修正方法：standard/two_level（稳定轨道，如 DRO）/segmented（"
        "不稳定轨道，全程分段打靶）。未显式指定时按族分派默认"
        "（HALO/NRHO/DPO → segmented，其余 CR3BP 族 → two_level）；"
        "显式传入与族冲突的值时告警并改写为 segmented",
    )
    correction_revolutions: int = Field(default=1, ge=1)

    @classmethod
    def valid_ranges(
        cls, orbit_type: str, *, collinear_point: int | None = None
    ) -> dict[str, NumericRange]:
        """返回指定轨道类型和上下文下适用的条件数值范围。"""
        if not isinstance(orbit_type, str):
            raise ValueError(f"orbit_type 必须为字符串，当前 {orbit_type!r}")
        selection = orbit_type.upper()
        if selection == "LISSAJOUS":
            point = 2 if collinear_point is None else collinear_point
            if point not in (1, 2, 3):
                raise ValueError(
                    f"LISSAJOUS collinear_point 必须为 1、2 或 3，当前 {collinear_point!r}"
                )
            ranges = _LISSAJOUS_L3_RANGES if point == 3 else _LISSAJOUS_L1_L2_RANGES
        else:
            try:
                ranges = _ORBIT_TYPE_RANGES[selection]
            except KeyError as exc:
                raise ValueError(f"不支持的 orbit_type: {orbit_type!r}") from exc
        return dict(ranges)

    def _validate_conditional_ranges(self, selection: str) -> None:
        """用公开范围接口校验已填充默认值的条件参数。"""
        for field, numeric_range in self.valid_ranges(
            selection, collinear_point=self.collinear_point
        ).items():
            value = getattr(self, field)
            if value is not None and not numeric_range.contains(value):
                raise ValueError(
                    f"{selection} {field} 应在 {numeric_range.format_interval()} km，实际 {value}"
                )

    @model_validator(mode="after")
    def _validate_orbit_type(self) -> DesignOrbitRequest:
        sel = self.orbit_type.upper()
        # duration 默认：CR3BP 1 年，ELFO 60 天
        if self.duration is None:
            self.duration = 5184000.0 if sel == "ELFO" else 31557600.0

        if sel == "ELFO":
            if self.semi_major_axis is None:
                raise ValueError("ELFO 必须提供 semi_major_axis")
            if self.inclination is None:
                self.inclination = 75.0
            if self.arg_of_pericenter is None:
                self.arg_of_pericenter = 270.0
            if self.perilune_height is None:
                self.perilune_height = 200.0
            self._validate_conditional_ranges(sel)
            return self

        # CR3BP 类型：按类型填默认值
        if sel == "DRO":
            if self.amplitude is None:
                self.amplitude = 10000.0
            if self.phase is None:
                self.phase = 0.5001
        elif sel == "DPO":
            if self.amplitude is None:
                self.amplitude = 20000.0
            if self.phase is None:
                self.phase = 0.5001
        elif sel == "HALO":
            if self.collinear_point is None:
                self.collinear_point = 2
            if self.amplitude is None:
                self.amplitude = 30000.0
            if self.phase is None:
                self.phase = 0.0
            if self.collinear_point not in (1, 2):
                raise ValueError(f"HALO collinear_point 必须为 1 或 2，当前 {self.collinear_point}")
        elif sel == "NRHO":
            if self.collinear_point is None:
                self.collinear_point = 2
            if self.north_south is None:
                self.north_south = 2
            if self.perilune_height is None:
                self.perilune_height = 5000.0
            if self.phase is None:
                self.phase = 0.5
            if self.collinear_point not in (1, 2):
                raise ValueError(f"NRHO collinear_point 必须为 1 或 2，当前 {self.collinear_point}")
        elif sel == "LISSAJOUS":
            if self.collinear_point is None:
                self.collinear_point = 2
            if self.amplitude_in is None:
                self.amplitude_in = 2500.0
            if self.amplitude_out is None:
                self.amplitude_out = 7500.0
            if self.phase_in is None:
                self.phase_in = 0.01
            if self.phase_out is None:
                self.phase_out = 0.55
        elif sel in ("L4", "L5"):
            if self.amplitude_in is None:
                self.amplitude_in = 8000.0
            if self.amplitude_out is None:
                self.amplitude_out = 6000.0
            if self.phase_in is None:
                self.phase_in = 0.0
            if self.phase_out is None:
                self.phase_out = 0.0
        elif sel == "AXIAL":
            if self.collinear_point is None:
                self.collinear_point = 2
            if self.amplitude is None:
                self.amplitude = 5000.0
            if self.phase is None:
                self.phase = 0.0
        elif sel in ("L4_SPO", "L5_SPO"):
            if self.amplitude is None:
                self.amplitude = 10000.0
            if self.phase is None:
                self.phase = 0.0
        elif sel in ("L4_LPO", "L5_LPO"):
            if self.amplitude is None:
                self.amplitude = 50000.0
            if self.phase is None:
                self.phase = 0.0
        elif sel in ("L4_HORSESHOE", "L5_HORSESHOE"):
            if self.amplitude is None:
                self.amplitude = 100000.0
            if self.phase is None:
                self.phase = 0.0
        else:
            raise ValueError(
                f"orbit_type 必须为 DRO/DPO/NRHO/HALO/LISSAJOUS/L4/L5/AXIAL"
                f"/L4_SPO/L5_SPO/L4_LPO/L5_LPO/L4_HORSESHOE/L5_HORSESHOE/ELFO，当前 {sel!r}"
            )
        self._dispatch_correction_method(sel)
        self._validate_conditional_ranges(sel)
        return self

    def _dispatch_correction_method(self, selection: str) -> None:
        """按族规范化星历修正方法：不稳定族强制 segmented。

        未显式指定时静默分派默认值；显式传入与族冲突的值时告警后改写
        （不拒绝，兼容既有调用方）。请求对象经此即为事实，算法层只做
        防御检查。
        """
        if selection not in SEGMENTED_CORRECTION_ORBIT_TYPES:
            return
        if self.correction_method == "segmented":
            return
        if "correction_method" in self.model_fields_set:
            warnings.warn(
                f"{selection} 属不稳定轨道族，星历修正只支持 segmented"
                f"（two_level/standard 自由外推必发散）："
                f"correction_method={self.correction_method!r} 已改写为 'segmented'",
                UserWarning,
                stacklevel=2,
            )
        self.correction_method = "segmented"


class ResultResponse(_ApiModel):
    """Facade 成功处理后的任务最终状态三元组。"""

    status: ConvergenceState
    cause: FailureCause
    message: str

    @model_validator(mode="after")
    def _validate_result_status(self) -> ResultResponse:
        ResultStatus(self.status, self.cause, self.message)
        return self


class FamilyGenerationResponse(_ApiModel, OrbitFamily):
    """轨道族生成响应。

    继承 ``OrbitFamily`` 保持既有成功返回的读取接口，同时由 Pydantic 在
    Facade 接缝直接承载状态三元组。算法层软失败的部分成员使用同一响应。
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    status: ConvergenceState
    cause: FailureCause
    message: str
    orbits: list[Orbit]
    family_type: str | None = None
    system: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    taxonomy_labels: list[str] = Field(
        default_factory=list,
        description="分类学标签（ADR 0042）：全体成员实测 primary 标签的去重集合"
        "（规范字符串）；lissajous 等不在分类学内的族为空表",
    )
    requested_members: int
    generated_members: int
    record_id: str | None = Field(
        default=None,
        description="产物自动入库的记录 id（ADR 0031）；库关闭或无成员产出时为 None",
    )

    def __iter__(self):
        """按 OrbitFamily 兼容语义迭代轨道成员。"""
        return iter(self.orbits)

    @model_validator(mode="after")
    def validate_result_contract(self) -> FamilyGenerationResponse:
        """校验状态组合和成员计数。"""
        ResultStatus(self.status, self.cause, self.message)
        if self.generated_members != len(self.orbits):
            raise ValueError("generated_members 必须等于 orbits 成员数")
        if self.generated_members > self.requested_members:
            raise ValueError("generated_members 不得超过 requested_members")
        return self


class DesignOrbitResponse(ResultResponse):
    """任务轨道设计输出。

    几何字段（``mu`` / ``states`` / ``times`` / ``ephemeris``）让下游
    （画图 / 落盘 / design→control 链式）可仅依赖 Facade，不必穿透 algorithm
    层。``states`` / ``times`` 为 CR3BP 参考周期轨道（无量纲会合系），
    ``ephemeris`` 为标称星历（GCRS km / 速度 m/s + 会合系，``EphemerisTable``
    全字段）。ELFO 场景下 CR3BP/修正字段为 None/默认值，漂移字段填充。
    """

    orbit_type: str
    epoch_utc: str
    duration_day: float
    initial_state: list[float]
    cr3bp_jacobi: float
    correction_iterations: int
    correction_method: str | None = Field(
        default=None,
        description="实际执行的星历修正方法；ELFO 场景（无星历修正）为 None",
    )
    force_config: dict[str, Any]
    mu: float | None = Field(
        default=None,
        description="CR3BP 质量比 μ = m₂/(m₁+m₂)；ELFO 场景为 None",
    )
    states: list[list[float]] = Field(
        default_factory=list,
        description="CR3BP 参考周期轨道状态序列 (n,6)，无量纲会合系",
    )
    taxonomy_labels: list[str] = Field(
        default_factory=list,
        description="分类学标签（ADR 0042）：对 CR3BP 参考轨道的实测多标签"
        "（规范字符串，如 halo_l2_northern）；ELFO 等无 CR3BP 场景为空表",
    )
    times: list[float] = Field(
        default_factory=list,
        description="CR3BP 参考周期轨道时间序列 (n,)，无量纲",
    )
    ephemeris: dict[str, Any] | None = Field(
        default=None,
        description="标称星历（EphemerisTable 全字段：UTC + GCRS km/m/s + 会合系）",
    )
    drift_e: float | None = Field(default=None, description="传播弧段 Δe（仅 ELFO）")
    drift_aop_deg: float | None = Field(default=None, description="传播弧段 Δω 度（仅 ELFO）")
    drift_rp_km: float | None = Field(default=None, description="传播弧段 Δrp km（仅 ELFO）")
    secular_aop_rate_deg_per_year: float | None = Field(
        default=None, description="ω 线性拟合年漂移率（仅 ELFO）"
    )
    record_id: str | None = Field(
        default=None,
        description="产物自动入库的记录 id（ADR 0031）；库关闭或无产物时为 None",
    )


class ControlOrbitRequest(_ApiModel):
    """轨道保持输入（对齐 algorithm/station_keeping 的 control_orbit 参数）。

    字段与算法层业务参数一一对应；运行时参数（spice/kernel_dir/n_workers/
    seed）由 Facade 注入，不进模型。默认值、单位与算法层签名一致。

    输入源二选一（ADR 0031）：``input_ephemeris`` 直接给星历，或
    ``input_record_id`` 引用库中记录（取其星历段，站保产物记录自动以
    ``source_record_id`` 指向该记录，谱系跨进程不断）。
    """

    input_ephemeris: Any = Field(
        default=None, description="标称轨道星历路径或 EphemerisTable；与 input_record_id 二选一"
    )
    input_record_id: str | None = Field(
        default=None,
        min_length=1,
        description="库中记录 id：取其星历段作为标称轨道；站保产物自动指向该记录",
    )
    # 控制策略
    control_mode: int = Field(
        default=1,
        ge=1,
        le=6,
        description="控制模式：1=目标点宽松、2=目标点严格、3=特征点、"
        "4=目标点宽松+角动量管理、5=目标点严格+角动量管理、6=特征点+角动量管理",
    )
    is_nrho: int = Field(default=0, ge=0, le=1, description="目标轨道是否 NRHO（1=是）")
    special_mode: int = Field(
        default=1,
        ge=1,
        le=2,
        description="特征点模式：1=Lissajous（ẋ=0）、2=Halo/NRHO（ẋ=0 且 ż=0）",
    )
    control_interval: float = Field(default=30.0, gt=0.0, description="控制时间间隔（天）")
    feedback_arc: float = Field(default=28.0, gt=0.0, description="目标点模式反馈弧段（天）")
    special_crossings: int = Field(default=3, ge=1, description="特征点目标穿越 x-z 平面次数")
    num_controls: int = Field(
        default=120, ge=1, le=10000, description="控制次数（总时间 = (N-1)·间隔）"
    )
    num_monte_carlo: int = Field(default=5, ge=1, le=1000, description="蒙特卡洛样本数（惯例 100）")
    output_step: float = Field(
        default=SECONDS_PER_DAY, gt=0.0, description="受控星历输出间隔（秒）"
    )
    # 测定轨误差
    position_accuracy: float = Field(default=1500.0, gt=0.0, description="测定轨位置 1-sigma（m）")
    velocity_accuracy: float = Field(default=0.002, gt=0.0, description="测定轨速度 1-sigma（m/s）")
    # 分段控制误差（《控制方案》式 5.40）
    thrust_angle_err: float = Field(default=0.333, ge=0.0, description="推力方向角 1-sigma（deg）")
    thrust_mean: float = Field(default=10.0, gt=0.0, description="推力中点值（m/s）")
    thrust_rel_err: float = Field(default=0.003, ge=0.0, description="推力相对 1-sigma（无量纲）")
    thrust_abs_err: float = Field(default=0.033, ge=0.0, description="推力绝对 1-sigma（m/s）")
    thrust_min: float = Field(default=0.1, gt=0.0, description="最小开机推力（m/s）")
    thrust_max: float = Field(default=100.0, gt=0.0, description="最大开机推力（m/s）")
    thrust_total: float = Field(default=1000.0, gt=0.0, description="累计推力上限（m/s）")
    srp_error_level: float = Field(
        default=0.10, ge=0.0, description="光压弧段随机误差量级（百分比/100）"
    )
    # 控制（理论）力模型：默认 2×2、球光压开、耦合项关
    perturbation: dict[str, int] | None = Field(
        default=None,
        description="无量纲摄动开关：sun_body/planets/earth_nonspherical/"
        "moon_nonspherical/atmosphere/relativity/tide/coupling 为 0=关、1=开；"
        "solar_radiation 另可为 2=ECOM（未实现）。缺省用控制力模型默认值。",
    )
    dyb: list[float] | None = Field(
        default=None,
        min_length=9,
        max_length=9,
        description="DYB 系数（9 元素）：元素 1=等效面质比（m²/kg），其余为相对比值",
    )
    earth_degree: int = Field(default=2, ge=2, le=120, description="地球引力场阶数（控制力模型）")
    moon_degree: int = Field(default=2, ge=2, le=120, description="月球引力场阶数（控制力模型）")
    # 真实（实际）力模型：默认 10×10
    real_perturbation: dict[str, int] | None = Field(
        default=None,
        description="无量纲摄动开关（键和值同 perturbation）；缺省继承控制力模型开关，"
        "阶数默认 10×10",
    )
    real_dyb: list[float] | None = Field(
        default=None,
        min_length=9,
        max_length=9,
        description="真实力模型 DYB 系数（9 元素）：元素 1=等效面质比（m²/kg），其余为相对比值",
    )
    real_earth_degree: int = Field(
        default=10, ge=2, le=120, description="地球引力场阶数（真实力模型）"
    )
    real_moon_degree: int = Field(
        default=10, ge=2, le=120, description="月球引力场阶数（真实力模型）"
    )
    # 角动量管理
    engine_layout: Any = Field(default=None, description="EngineLayout（角动量管理 4-6 必填）")
    momentum_interval: float = Field(default=5.0, gt=0.0, description="角动量卸载间隔（天）")
    srp_offset_m: list[float] | None = Field(
        default=None,
        min_length=3,
        max_length=3,
        description="SRP 压心相对质心偏移 [x,y,z]（m），逐元素：x/y/z 分量",
    )
    spacecraft_mass: float = Field(default=1000.0, gt=0.0, description="航天器质量（kg）")
    srp_torque: list[float] | None = Field(
        default=None,
        min_length=3,
        max_length=3,
        description="常值 SRP 力矩 [τx,τy,τz]（N·m），逐元素：x/y/z 分量力矩",
    )
    # TIGHT/SPECIAL 模式迭代参数
    tight_tolerance_km: float = Field(
        default=0.1, gt=0.0, description="TIGHT 模式位置重合容差（km）"
    )
    tight_max_iter: int = Field(default=6, ge=1, description="TIGHT 模式微分修正迭代上限")
    special_damping_factor: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="SPECIAL 模式牛顿迭代阻尼因子（<1 时启用回溯）",
    )

    @field_validator("perturbation", "real_perturbation")
    @classmethod
    def _validate_perturbation(cls, value: dict[str, int] | None) -> dict[str, int] | None:
        """在 API 边界校验摄动开关的键和值。"""
        if value is None:
            return None
        unknown = set(value) - set(DEFAULT_PERTURBATION)
        if unknown:
            raise ValueError(f"未知摄动开关字段: {sorted(unknown)}")
        for key, switch in value.items():
            allowed = (0, 1, 2) if key == "solar_radiation" else (0, 1)
            if switch not in allowed:
                raise ValueError(f"摄动开关 {key} 取值必须为 {allowed}，当前 {switch!r}")
        return value

    mu: float | None = Field(
        default=None,
        description="CR3BP 质量比 μ（透传到响应，画地月/L 点标注用）；算法层不产 mu",
    )

    @model_validator(mode="after")
    def _validate_input_source(self) -> ControlOrbitRequest:
        """输入源二选一：裸星历或库记录引用。"""
        if (self.input_ephemeris is None) == (self.input_record_id is None):
            raise ValueError("input_ephemeris 与 input_record_id 必须且只能提供一个")
        return self


class ControlOrbitResponse(ResultResponse):
    """轨道保持输出。

    几何字段（``controlled_ephemeris`` / ``mu``）：``controlled_ephemeris``
    为最后一次蒙特卡洛样本的受控真实轨道星历（``EphemerisTable`` 全字段；
    全失败时 ``None``）；``mu`` 由请求透传（算法层不产 mu）。
    """

    num_failed: int
    sk_statistic: dict[str, Any]
    maneuvers: dict[str, Any]
    controlled_ephemeris: dict[str, Any] | None = Field(
        default=None,
        description="受控星历（EphemerisTable 全字段）；所有蒙特卡洛样本失败时 None",
    )
    mu: float | None = Field(
        default=None,
        description="CR3BP 质量比 μ（请求透传，画地月/L 点标注用）",
    )
    record_id: str | None = Field(
        default=None,
        description="产物自动入库的记录 id（ADR 0031）；库关闭或无产物时为 None",
    )


class TransferDesignRequest(_ApiModel):
    """转移轨道设计输入（对齐 algorithm/transfer 的 transfer_orbit 参数）。"""

    transfer_type: str = Field(description="HMN/LGA/WSB/low_thrust")
    tli_epoch: Any = Field(description="TLI 历元（UTC ISO 字符串或 JD_TDB 浮点数）")
    parking_alt_km: float = Field(default=200.0, gt=0.0, description="地球停泊轨道高度 (km)")
    incl_deg: float = Field(default=28.5, ge=0.0, le=180.0, description="轨道倾角 (度)")
    flight_path_deg: float = Field(default=0.0, ge=0.0, le=0.0, description="航迹角 (度，仅支持 0)")
    target_ephemeris: Any = Field(
        default=None,
        description=(
            "目标星历（EphemerisTable/NominalOrbit/ndarray），LGA/WSB 必需。"
            "坐标系契约按转移类型区分：LGA/WSB 要求会合旋转系（synodic）物理单位"
            "（km, km/s）状态，编排器直接无量纲化，不做惯性系→旋转系转换，"
            "orbit_propagation/design_orbit 产出的惯性星历必须先经 "
            "spacetime_transform(j2000_to_synodic) 转换后再传入，否则目标态几何全错；"
            "HMN/low_thrust 按地心惯性系 km/km/s 状态解释。"
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
    tof_range: list[float] | None = Field(default=None, description="飞行时间范围 [min, max]（天）")
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
            "low_thrust 与零结果为 None。段的数据系即词汇值 gcrs_km"
        ),
    )
    # 字面量与 e2m2e/algorithm/transfer 的 STATE_FRAME_* 常量同源，改动须两侧同步；
    # 此处不导入以保持 models 层轻依赖（schema 出口）。
    state_frame: Literal["synodic_barycentric_km", "force_model_state"] = Field(
        description=(
            "trajectory 的数据系标签（ADR 0040 增补）：synodic_barycentric_km"
            " = 地月会合旋转系质心原点物理 km/km/s（HMN/LGA/WSB）；"
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
            "（dv_km_s=0）；low_thrust 连续推进恒为空；搜索零结果恒为空。"
            "旧 details Δv 字段保留一个版本后废弃"
        ),
    )
    details: dict[str, Any]
    record_id: str | None = Field(
        default=None,
        description="产物自动入库的记录 id（ADR 0031，#574）；库关闭或无轨迹产物时为 None",
    )


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


class PropagationResponse(ResultResponse):
    """轨道预报输出。"""

    epoch_utc: str
    duration_sec: float
    output_step: float
    n_points: int
    time_sec: list[float]
    times_jd_tdb: list[float]
    position_km: list[list[float]]
    velocity_km_s: list[list[float]]
    final_state: list[float]


class SpacetimeTransformRequest(_ApiModel):
    """时空坐标转换输入。"""

    states: list[list[float]] = Field(description="状态列表，每项 [x,y,z,vx,vy,vz]")
    times: list[float] = Field(
        description="每个状态的时间值：GCRS↔EBCRS 用 JD_TDB；会合系转换用"
        "无量纲会合时间 t_syn（0 = et0_jd 参考历元）"
    )
    transform_type: str = Field(
        description="synodic_to_j2000/j2000_to_synodic/gcrs_to_ebcrs/ebcrs_to_gcrs"
    )
    et0_jd: float = Field(description="参考历元 JD_TDB")
    ephemeris_path: str | None = Field(default=None, description="历表路径（GCRS↔EBCRS 必需）")


class SpacetimeTransformResponse(ResultResponse):
    """时空坐标转换输出。"""

    states: list[list[float]]
    times: list[float]
    transform_type: str
    details: dict[str, Any]


# ---------------------------------------------------------------------------
# 轨道族生成：公开平动点统一术语 libration_point（1=L1 … 5=L5）。
# 各族允许的平动点取值域与默认值；Halo 族振幅上限为固定 z0 延拓的
# 折叠点（同 seed._HALO_FOLD_Z0，按平动点区分）。
# ---------------------------------------------------------------------------

#: orbit_type → 允许的平动点取值域。DRO 是月心族不绑定平动点（空元组），
#: DPO 等其余无平动点族不进本模型。
_FAMILY_LIBRATION_POINT_RANGES: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {
        "HALO": (1, 2),
        "NRHO": (1, 2),
        "AXIAL": (1, 2),
        "LISSAJOUS": (1, 2, 3),
        "SPO": (4, 5),
        "LPO": (4, 5),
        "HORSESHOE": (4, 5),
        "DRO": (),
    }
)

#: orbit_type → libration_point 默认值（共线族取 L2，三角族取 L4）。
_FAMILY_DEFAULT_LIBRATION_POINT: Mapping[str, int] = MappingProxyType(
    {
        "HALO": 2,
        "NRHO": 2,
        "AXIAL": 2,
        "LISSAJOUS": 2,
        "SPO": 4,
        "LPO": 4,
        "HORSESHOE": 4,
    }
)

#: Halo 固定 z0 延拓的折叠点（km），按平动点：L1≈26908、L2≈57660。
_HALO_FOLD_KM: Mapping[int, float] = MappingProxyType(
    {lp: _HALO_FOLD_Z0[lp] * CHAR_LENGTH_KM for lp in (1, 2)}
)

#: Halo 族振幅上限默认值（km），按平动点取折叠点内的标定值。
_HALO_DEFAULT_MAX_AMPLITUDE_KM: Mapping[int, float] = MappingProxyType({1: 25000.0, 2: 30000.0})

_FAMILY_COMMON_FIELDS = frozenset({"orbit_type", "libration_point", "n_orbits"})
_FAMILY_SPECIFIC_FIELDS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "HALO": frozenset({"max_amplitude_km", "sampling_mode"}),
        "NRHO": frozenset(
            {
                "north_south",
                "perilune_height_max_km",
                "continuation_direction",
                "sampling_mode",
            }
        ),
        "AXIAL": frozenset({"max_amplitude_km", "continuation_direction", "sampling_mode"}),
        "LISSAJOUS": frozenset(
            {
                "amplitude_in_km",
                "amplitude_out_km",
                "phase_in",
                "phase_out",
                "sampling_mode",
            }
        ),
        "SPO": frozenset(
            {
                "min_amplitude_km",
                "max_amplitude_km",
                "continuation_direction",
                "sampling_mode",
                "match_tolerance_km",
            }
        ),
        "LPO": frozenset(
            {
                "min_amplitude_km",
                "max_amplitude_km",
                "continuation_direction",
                "sampling_mode",
                "match_tolerance_km",
            }
        ),
        "HORSESHOE": frozenset(
            {
                "min_amplitude_km",
                "max_amplitude_km",
                "continuation_direction",
                "sampling_mode",
                "match_tolerance_km",
            }
        ),
        "DRO": frozenset({"min_amplitude_km", "max_amplitude_km", "sampling_mode"}),
    }
)
_FAMILY_OPTION_VALUES: Mapping[str, Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "HALO": MappingProxyType({"sampling_mode": ("natural-z0",)}),
        "NRHO": MappingProxyType(
            {
                "continuation_direction": ("toward-moon",),
                "sampling_mode": ("halo-segment",),
            }
        ),
        "AXIAL": MappingProxyType(
            {
                "continuation_direction": ("increase-amplitude",),
                "sampling_mode": ("fixed-vz0",),
            }
        ),
        "LISSAJOUS": MappingProxyType({"sampling_mode": ("linear-amplitudes",)}),
        "SPO": MappingProxyType(
            {
                "continuation_direction": ("decrease-x0", "increase-x0"),
                "sampling_mode": ("full-period-pal",),
            }
        ),
        "LPO": MappingProxyType(
            {
                "continuation_direction": ("decrease-x0", "increase-x0"),
                "sampling_mode": ("full-period-pal",),
            }
        ),
        "HORSESHOE": MappingProxyType(
            {
                "continuation_direction": ("decrease-x0", "increase-x0"),
                "sampling_mode": ("full-period-pal",),
            }
        ),
        "DRO": MappingProxyType({"sampling_mode": ("natural-x0",)}),
    }
)


class FamilyGenerationRequest(_ApiModel):
    """轨道族生成输入（二档 Facade.orbit_family_generation）。

    公开参数用平动点统一术语（``libration_point``，1=L1 … 5=L5），由
    Facade 映射到各族算法参数。按 ``orbit_type`` 分派校验取值域
    （与 ``DesignOrbitRequest`` 同构）：共线族（Halo/NRHO/Axial）仅
    L1/L2，Lissajous 支持 L1/L2/L3，三角族（SPO/LPO/Horseshoe）仅
    L4/L5，DRO 是月心族不绑定平动点（请求不得携带 ``libration_point``）。
    八族均已实现：周期族返回严格周期成员，Lissajous
    返回拟周期有界轨迹的参数采样（族上显式标注 quasi-periodic）。

    按族适用的字段：

    - HALO/AXIAL：``max_amplitude_km`` （带符号，区分北/南或上/下族）
    - NRHO：``north_south``、``perilune_height_max_km``、``continuation_direction``
    - LISSAJOUS：``amplitude_in_km``、``amplitude_out_km``、``phase_in``、``phase_out``
    - SPO/LPO/HORSESHOE：振幅上下限、延拓方向与 ``match_tolerance_km``
    - DRO：振幅上下限（距月心距离 min/max 均值，km）

    ``sampling_mode`` 显式登记各族固定的首版采样规则；传入其他规则会
    结构化拒绝，而不是静默改用默认算法。
    """

    orbit_type: str = Field(description="HALO/NRHO/AXIAL/LISSAJOUS/SPO/LPO/HORSESHOE/DRO")
    libration_point: int | None = Field(
        default=None, ge=1, le=5, description="平动点编号：1=L1 … 5=L5；缺省按族填默认；DRO 不适用"
    )
    max_amplitude_km: float | None = Field(
        default=None,
        description="族振幅上限（km）；HALO/AXIAL 带符号区分北/南（上/下）族，"
        "SPO/LPO/HORSESHOE 为正值（距 L4/L5 径向距离 min/max 均值），"
        "DRO 为距月心距离 min/max 均值",
    )
    min_amplitude_km: float | None = Field(
        default=None,
        description="族振幅下限（km），仅 SPO/LPO/HORSESHOE/DRO 用",
    )
    north_south: int | None = Field(
        default=None, ge=1, le=2, description="北/南族：1=北，2=南；仅 NRHO 用"
    )
    perilune_height_max_km: float | None = Field(
        default=None,
        description="族成员近月点高度上限（km），仅 NRHO 用",
    )
    amplitude_in_km: float | None = Field(
        default=None, gt=0.0, description="面内振幅上限（km），仅 LISSAJOUS 用"
    )
    amplitude_out_km: float | None = Field(
        default=None, gt=0.0, description="面外振幅上限（km），仅 LISSAJOUS 用"
    )
    phase_in: float | None = Field(
        default=None, ge=0.0, le=1.0, description="面内初始相位（0~1），仅 LISSAJOUS 用"
    )
    phase_out: float | None = Field(
        default=None, ge=0.0, le=1.0, description="面外初始相位（0~1），仅 LISSAJOUS 用"
    )
    continuation_direction: str | None = Field(
        default=None,
        description="延拓方向：NRHO=toward-moon、AXIAL=increase-amplitude、"
        "SPO/LPO/HORSESHOE=decrease-x0 或 increase-x0",
    )
    sampling_mode: str | None = Field(
        default=None,
        description="采样规则：natural-z0/halo-segment/fixed-vz0/"
        "linear-amplitudes/full-period-pal；按族固定",
    )
    match_tolerance_km: float | None = Field(
        default=None,
        gt=0.0,
        description="振幅上限匹配容差（km），仅 SPO/LPO/HORSESHOE 用",
    )
    n_orbits: int = Field(
        default=50,
        ge=1,
        description="族成员数量上限；延拓族包含种子或首个命中范围的成员",
    )

    @classmethod
    def valid_ranges(
        cls, orbit_type: str, *, libration_point: int | None = None
    ) -> dict[str, NumericRange]:
        """返回指定族和上下文下适用的条件数值范围。"""
        if not isinstance(orbit_type, str):
            raise ValueError(f"orbit_type 必须为字符串，当前 {orbit_type!r}")
        selection = orbit_type.upper()
        if selection not in _FAMILY_LIBRATION_POINT_RANGES:
            raise ValueError(f"不支持的 orbit_type: {orbit_type!r}")
        allowed = _FAMILY_LIBRATION_POINT_RANGES[selection]
        ranges: dict[str, NumericRange] = {}
        point: int | None = None
        if allowed:
            ranges["libration_point"] = NumericRange(min(allowed), max(allowed))
            point = (
                _FAMILY_DEFAULT_LIBRATION_POINT[selection]
                if libration_point is None
                else libration_point
            )
            if point not in allowed:
                raise ValueError(
                    f"{selection} libration_point 必须为 {sorted(allowed)}，当前 {point!r}"
                )
        elif libration_point is not None:
            # 月心族（DRO）不绑定平动点，显式携带即为跨族字段
            raise ValueError(f"{selection} 不绑定平动点，请求不得携带 libration_point")
        if selection == "HALO":
            assert point is not None  # HALO 必有平动点
            fold_km = _HALO_FOLD_KM[point]
            ranges["max_amplitude_km"] = NumericRange(-fold_km, fold_km, excluded_values=(0.0,))
        elif selection == "NRHO":
            ranges["north_south"] = NumericRange(1, 2)
            ranges["perilune_height_max_km"] = NumericRange(1000.0, 40000.0)
        elif selection == "AXIAL":
            ranges["max_amplitude_km"] = NumericRange(-60000.0, 60000.0, excluded_values=(0.0,))
        elif selection == "LISSAJOUS":
            # 与 DesignOrbitRequest 的 Lissajous 振幅包络一致
            amp_range = (
                NumericRange(0.0, 100000.0, minimum_inclusive=False)
                if point == 3
                else NumericRange(0.0, 7600.0, minimum_inclusive=False)
            )
            ranges["amplitude_in_km"] = amp_range
            ranges["amplitude_out_km"] = amp_range
            ranges["phase_in"] = NumericRange(0.0, 1.0)
            ranges["phase_out"] = NumericRange(0.0, 1.0)
        elif selection == "DRO":
            # 与 DesignOrbitRequest 的 DRO 振幅包络一致（月面以上）
            amp_range = NumericRange(1737.0, 110000.0)
            ranges["min_amplitude_km"] = amp_range
            ranges["max_amplitude_km"] = amp_range
        else:  # SPO/LPO/HORSESHOE：声明范围不得超出可达包络
            if selection == "SPO":
                amp_range = NumericRange(1737.0, 75000.0)
            elif selection == "LPO":
                amp_range = NumericRange(1000.0, 110000.0)
            else:
                amp_range = NumericRange(50000.0, 110000.0)
            ranges["min_amplitude_km"] = amp_range
            ranges["max_amplitude_km"] = amp_range
            ranges["match_tolerance_km"] = NumericRange(0.0, 5000.0, minimum_inclusive=False)
        return ranges

    @classmethod
    def valid_options(cls, orbit_type: str) -> dict[str, tuple[str, ...]]:
        """返回指定族的公开离散选项（延拓方向与采样规则）。"""
        if not isinstance(orbit_type, str):
            raise ValueError(f"orbit_type 必须为字符串，当前 {orbit_type!r}")
        selection = orbit_type.upper()
        try:
            return dict(_FAMILY_OPTION_VALUES[selection])
        except KeyError as exc:
            raise ValueError(f"不支持的 orbit_type: {orbit_type!r}") from exc

    @model_validator(mode="after")
    def _validate_orbit_type(self) -> FamilyGenerationRequest:
        sel = self.orbit_type.upper()
        if sel not in _FAMILY_LIBRATION_POINT_RANGES:
            raise ValueError(
                f"orbit_type 必须为 {'/'.join(_FAMILY_LIBRATION_POINT_RANGES)}，当前 {sel!r}"
            )
        invalid_fields = (
            self.model_fields_set - _FAMILY_COMMON_FIELDS - _FAMILY_SPECIFIC_FIELDS[sel]
        )
        if invalid_fields:
            names = ", ".join(sorted(invalid_fields))
            raise ValueError(f"{sel} 不适用字段：{names}")
        if _FAMILY_LIBRATION_POINT_RANGES[sel] and self.libration_point is None:
            # 月心族（DRO）不填平动点默认值；显式携带由 valid_ranges 拒绝
            self.libration_point = _FAMILY_DEFAULT_LIBRATION_POINT[sel]
        # 先经公开范围接口校验平动点取值域（越界在此被拒）
        ranges = self.valid_ranges(sel, libration_point=self.libration_point)
        options = self.valid_options(sel)
        direction_options = options.get("continuation_direction")
        if direction_options is not None:
            if self.continuation_direction is None:
                self.continuation_direction = direction_options[0]
            elif self.continuation_direction not in direction_options:
                raise ValueError(
                    f"{sel} continuation_direction 必须为 {direction_options}，"
                    f"当前 {self.continuation_direction!r}"
                )
        sampling_options = options["sampling_mode"]
        if self.sampling_mode is None:
            self.sampling_mode = sampling_options[0]
        elif self.sampling_mode not in sampling_options:
            raise ValueError(
                f"{sel} sampling_mode 必须为 {sampling_options}，当前 {self.sampling_mode!r}"
            )
        # 按族填默认值
        if sel == "HALO":
            assert self.libration_point is not None  # HALO 必有平动点
            if self.max_amplitude_km is None:
                self.max_amplitude_km = _HALO_DEFAULT_MAX_AMPLITUDE_KM[self.libration_point]
            if self.max_amplitude_km == 0:
                raise ValueError("max_amplitude_km 不能为 0")
        elif sel == "NRHO":
            if self.north_south is None:
                self.north_south = 2
            if self.perilune_height_max_km is None:
                self.perilune_height_max_km = 20000.0
        elif sel == "AXIAL":
            if self.max_amplitude_km is None:
                self.max_amplitude_km = 10000.0
            if self.max_amplitude_km == 0:
                raise ValueError("max_amplitude_km 不能为 0")
        elif sel == "LISSAJOUS":
            if self.amplitude_in_km is None:
                self.amplitude_in_km = 2500.0
            if self.amplitude_out_km is None:
                self.amplitude_out_km = 7500.0
            if self.phase_in is None:
                self.phase_in = 0.01
            if self.phase_out is None:
                self.phase_out = 0.55
        elif sel == "DRO":
            if self.min_amplitude_km is None:
                self.min_amplitude_km = 2000.0
            if self.max_amplitude_km is None:
                self.max_amplitude_km = 60000.0
        else:  # SPO/LPO/HORSESHOE
            if self.min_amplitude_km is None:
                self.min_amplitude_km = 50000.0 if sel == "HORSESHOE" else 2000.0
            if self.max_amplitude_km is None:
                self.max_amplitude_km = 110000.0 if sel in ("LPO", "HORSESHOE") else 60000.0
            if self.match_tolerance_km is None:
                self.match_tolerance_km = 50.0 if sel == "HORSESHOE" else 20.0
        # 用公开范围接口校验取值（ADR 0014 决策 8：校验器与接口共用一份规则）
        for field, numeric_range in ranges.items():
            value = getattr(self, field)
            if value is not None and not numeric_range.contains(value):
                raise ValueError(
                    f"{sel} {field} 应在 {numeric_range.format_interval()}，实际 {value}"
                )
        if sel in ("SPO", "LPO", "HORSESHOE", "DRO"):
            assert self.min_amplitude_km is not None
            assert self.max_amplitude_km is not None
            if self.min_amplitude_km >= self.max_amplitude_km:
                raise ValueError(
                    f"{sel} min_amplitude_km 必须小于 max_amplitude_km，"
                    f"当前 {self.min_amplitude_km} >= {self.max_amplitude_km}"
                )
        return self


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
        description="转移类型等值过滤（HMN/LGA/WSB/low_thrust；#574 transfer record）",
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
        default=None, description="转移类型（HMN/LGA/WSB/low_thrust）；非 transfer 记录为 None"
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
    member_count: int = Field(description="族成员数；单轨道记录为 1，纯星历/转移记录为 0")
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
    request: dict[str, Any] = Field(description="原始请求快照")
    members: list[dict[str, Any]] = Field(description="族成员参数表；非族记录为空")
    details: dict[str, Any] | None = Field(
        default=None,
        description="设计细节块（transfer 记录：后端 details + maneuver_events；其余为 None）",
    )
    arrays: dict[str, Any] = Field(
        description="数组段：cr3bp/ 与 eph/ 前缀的 numpy 数组（含族成员 cr3bp/members/）"
    )

    def to_ephemeris_table(self) -> Any | None:
        """把星历段重建为 ``EphemerisTable`` 实例（供接续计算）；无星历段返回 None。"""
        from e2m2e.data.catalog import ephemeris_from_arrays

        if not self.has_ephemeris:
            return None
        return ephemeris_from_arrays(self.arrays)

    def to_orbit(self) -> Any | None:
        """把单轨道 CR3BP 段重建为 ``Orbit``；族记录与纯星历记录返回 None。"""
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


class CatalogPromoteRequest(_ApiModel):
    """把族成员提升为独立记录（``source_record_id`` 指向所属族）。"""

    record_id: str = Field(min_length=1, description="族记录 id")
    member_index: int = Field(ge=0, description="族内成员序号（自 0 起）")


class CatalogPromoteResponse(ResultResponse):
    """catalog_promote 输出：提升出的独立记录。"""

    record: CatalogRecordResponse


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
    """扫描单参数点的结局：成功（含软失败）保留 record_id，硬失败保留原因。"""

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
    record_id: str | None
    generated_members: int


class CatalogSweepResponse(ResultResponse):
    """catalog_sweep 输出。

    ``succeeded`` 为产出记录的参数点数（含软失败但有成员产出的点）；
    ``failed`` 为硬失败（无产出）参数点数；软失败且零成员的点两者都
    不计，其结局见 ``points`` 逐点状态。
    """

    points: list[CatalogSweepPointOutcome]
    record_ids: list[str]
    succeeded: int
    failed: int


class SpatiographyScalesRequest(_ApiModel):
    """分区解析尺度计算输入（spatiography，ADR 0041）。"""

    system: Literal["earth_moon"] = Field(
        default="earth_moon",
        description="天体系统；当前仅支持地月（Primer §5 口径，Simon 1994 月根数）",
    )
    elements: list[str] = Field(
        default_factory=list,
        description="要计算的量（scales 键名）子集；空 = 全部。含 laplace_radius_geolunar、"
        "hill_radius_moon、soi_laplace_moon、battin_moon_earthward_km 等",
    )


class SpatiographyScalesResponse(ResultResponse):
    """分区解析尺度输出（Rosengren et al. 2026 §5 闭式边界，物理单位 km）。"""

    scales: dict[str, float]
    libration_points_km: dict[str, list[float]] = Field(
        description="L1–L5 精确解位置（会合系质心原点，km，z=0）；精确求根口径"
        "（论文表值 57868/64347 km 为级数近似注记）"
    )
    jacobi_criticals: dict[str, float] = Field(
        description="平动点处临界 Jacobi 值 C1..C5（Parker 约定，无量纲）"
    )
    resonance_ladder: list[dict[str, Any]] = Field(
        description="共振名义中心阶梯（Table 1/2 全表）：label/k/k_body/a_km/"
        "a_over_a_moon 或 rho_over_moon_radius/period_days/secondary"
    )
    constants_used: dict[str, float] = Field(description="Primer 常数集（数值）")
    citation: str = Field(description="常数集出处")
    details: dict[str, Any]


class SpatiographyClassifyRequest(_ApiModel):
    """分区区域分类输入。"""

    states: list[list[float]] = Field(
        min_length=1,
        description="状态列表，每项 [x,y,z,vx,vy,vz]；坐标系与单位由 frame 声明",
    )
    frame: Literal["synodic_barycentric_km", "synodic_barycentric_nd"] = Field(
        description="状态的数据系标签（ADR 0040 state_frame 词汇，本工具首批启用"
        " synodic_barycentric_nd）：synodic_barycentric_km = 地月会合旋转系、质心原点、"
        "物理单位 km/km/s；synodic_barycentric_nd = 同系无量纲（长度 a☾、速度 a☾·n，"
        "Primer 常数口径）"
    )
    reference: Literal["table1", "table4"] = Field(
        default="table1",
        description="分区口径：table1 = 论文 Table 1 五省语义；table4 = 附录 B 六"
        "制图带（deliberate-overlap，相邻区端部有意重叠）",
    )
    include_overlaps: bool = Field(
        default=True,
        description="重叠带返回多标签（False 时按优先序取主标签）",
    )


class SpatiographyClassifyResponse(ResultResponse):
    """分区区域分类输出。"""

    zone_ids: list[list[int]] = Field(
        description="逐状态区域 id 列表（重叠带多值、升序），名称见 legend"
    )
    legend: dict[str, str] = Field(
        description="区域 id → 名称（terrestrial / cislunar_inner_secular /"
        " cislunar_outer_resonant / circumlunar / translunar / heliocentric；"
        "cislunar 为狭义带级名，非伞式）"
    )
    diagnostics: list[dict[str, Any]] = Field(
        description="逐状态诊断：r_geocentric_km / rho_selenocentric_km /"
        " a_geocentric_km / a_over_a_moon / jacobi_constant / topology_case /"
        " open_necks"
    )
    details: dict[str, Any]


class SpatiographyBoundariesRequest(_ApiModel):
    """分区边界几何输入（可视化数据层）。"""

    kind: Literal["synodic_planar", "ae_curves"] = Field(
        default="synodic_planar",
        description="synodic_planar = 地月会合旋转系（质心原点、z=0 平面）边界圆族"
        "与 Battin 非对称曲线、L1–L5；ae_curves = 地心 osculating (a,e) 根数平面"
        "走廊曲线族（掠地线/Hill 远点线/月 Hill 相遇走廊/GEO 穿越线/共振竖线/"
        "Tisserand 等值线；crossing diagnostics 而非物理面）",
    )
    boundary_set: list[str] | None = Field(
        default=None,
        description="元素/曲线族名子集；空 = 全部（名称见 kind 对应支持清单）",
    )
    resolution: int = Field(
        default=720,
        ge=8,
        le=4096,
        description="曲线离散点数（synodic_planar 为闭合曲线点数；ae_curves 为每条曲线采样点数）",
    )


class SpatiographyBoundariesResponse(ResultResponse):
    """分区边界几何输出（前端只做归一与绘制，不做数值计算）。"""

    elements: list[dict[str, Any]] = Field(
        description="边界元素：kind=circle（center_km/radius_km/points_km）|"
        " polyline（center_km/points_km）| point（center_km，会合系质心原点 km）|"
        " curve_ae（points_ae=[a_km,e]）| vertical_ae（a_km）"
    )
    state_frame: Literal["synodic_barycentric_km", "element_space_ae"] = Field(
        description="几何的数据系标签：synodic_planar 输出为 synodic_barycentric_km"
        "（地月会合旋转系、质心原点、物理 km）；ae_curves 输出在根数空间，标签为"
        " element_space_ae（横轴 a 单位 km、纵轴 e 无量纲，ADR 0041 登记的新词汇）"
    )
    details: dict[str, Any]
