"""公开数据模型（Pydantic，全手写）。

输入/输出/错误模型精雕参数单位、默认值、取值域（ADR 0014）。Pydantic 只在
api/ 边界，算法层用 numpy/dataclass。每个 Facade 方法一个 Request/Response，
外加统一错误模型 ``OrbitError``。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from e2m2e.algorithm.results import ResultStatus
from e2m2e.data.constants import SECONDS_PER_DAY
from e2m2e.data.templates import ConvergenceState, FailureCause
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
        description="星历修正方法：standard/two_level（稳定轨道，DRO 等）/segmented（"
        "不稳定轨道，全程分段打靶）。Halo/NRHO 强制走 segmented（two_level 自由外推"
        "对不稳定轨道必发散），传入值会被覆盖",
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
        self._validate_conditional_ranges(sel)
        return self


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
    requested_members: int
    generated_members: int

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

    几何字段（``mu`` / ``states`` / ``times`` / ``ephemeris``，#312）让下游
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
    force_config: dict[str, Any]
    mu: float | None = Field(
        default=None,
        description="CR3BP 质量比 μ = m₂/(m₁+m₂)；ELFO 场景为 None",
    )
    states: list[list[float]] = Field(
        default_factory=list,
        description="CR3BP 参考周期轨道状态序列 (n,6)，无量纲会合系",
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


class ControlOrbitRequest(_ApiModel):
    """轨道保持输入（对齐 algorithm/station_keeping 的 control_orbit 参数）。

    字段与算法层业务参数一一对应；运行时参数（spice/kernel_dir/n_workers/
    seed）由 Facade 注入，不进模型。默认值、单位与算法层签名一致。
    """

    input_ephemeris: Any = Field(description="标称轨道星历路径或 EphemerisTable")
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


class ControlOrbitResponse(ResultResponse):
    """轨道保持输出。

    几何字段（``controlled_ephemeris`` / ``mu``，#312）：``controlled_ephemeris``
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


class TransferDesignRequest(_ApiModel):
    """转移轨道设计输入（对齐 algorithm/transfer 的 transfer_orbit 参数）。"""

    transfer_type: str = Field(description="HMN/LGA/WSB/low_thrust")
    tli_epoch: Any = Field(description="TLI 历元（UTC ISO 字符串或 JD_TDB 浮点数）")
    parking_alt_km: float = Field(default=200.0, gt=0.0, description="地球停泊轨道高度 (km)")
    incl_deg: float = Field(default=28.5, ge=0.0, le=180.0, description="轨道倾角 (度)")
    flight_path_deg: float = Field(default=0.0, ge=0.0, le=0.0, description="航迹角 (度，仅支持 0)")
    target_ephemeris: Any = Field(
        default=None, description="目标星历（EphemerisTable/NominalOrbit/ndarray）"
    )
    target_orbit_radius_km: float | None = Field(
        default=None, gt=0.0, description="目标轨道半径 (km)，HMN 必需"
    )
    tof_range: list[float] | None = Field(default=None, description="飞行时间范围 [min, max]（天）")
    lga_search_params: Any = Field(default=None, description="LGA 搜索参数（LgaSearchParams 实例）")
    wsb_search_params: Any = Field(default=None, description="WSB 搜索参数（WsbSearchParams 实例）")


class TransferDesignResponse(ResultResponse):
    """转移轨道设计输出。"""

    transfer_type: str
    delta_v: float
    trajectory: list[list[float]] | None
    details: dict[str, Any]


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
    times: list[float] = Field(description="每个状态的 JD_TDB 时间值")
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
# 轨道族生成（#411）：公开平动点统一术语 libration_point（1=L1 … 5=L5）。
# 各族允许的平动点取值域与默认值；Halo 族振幅上限为固定 z0 延拓的
# 折叠点（同 seed._HALO_FOLD_Z0，按平动点区分）。
# ---------------------------------------------------------------------------

#: orbit_type → 允许的平动点取值域。DRO/DPO 等无平动点概念，不进本模型。
_FAMILY_LIBRATION_POINT_RANGES: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {
        "HALO": (1, 2),
        "NRHO": (1, 2),
        "AXIAL": (1, 2),
        "LISSAJOUS": (1, 2, 3),
        "SPO": (4, 5),
        "LPO": (4, 5),
        "HORSESHOE": (4, 5),
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
    }
)


class FamilyGenerationRequest(_ApiModel):
    """轨道族生成输入（二档 Facade.orbit_family_generation）。

    公开参数用平动点统一术语（``libration_point``，1=L1 … 5=L5），由
    Facade 映射到各族算法参数。按 ``orbit_type`` 分派校验取值域
    （与 ``DesignOrbitRequest`` 同构）：共线族（Halo/NRHO/Axial）仅
    L1/L2，Lissajous 支持 L1/L2/L3，三角族（SPO/LPO/Horseshoe）仅
    L4/L5。七族均已实现（#428）：周期族返回严格周期成员，Lissajous
    返回拟周期有界轨迹的参数采样（族上显式标注 quasi-periodic）。

    按族适用的字段：

    - HALO/AXIAL：``max_amplitude_km`` （带符号，区分北/南或上/下族）
    - NRHO：``north_south``、``perilune_height_max_km``、``continuation_direction``
    - LISSAJOUS：``amplitude_in_km``、``amplitude_out_km``、``phase_in``、``phase_out``
    - SPO/LPO/HORSESHOE：振幅上下限、延拓方向与 ``match_tolerance_km``

    ``sampling_mode`` 显式登记各族固定的首版采样规则；传入其他规则会
    结构化拒绝，而不是静默改用默认算法。
    """

    orbit_type: str = Field(description="HALO/NRHO/AXIAL/LISSAJOUS/SPO/LPO/HORSESHOE")
    libration_point: int | None = Field(
        default=None, ge=1, le=5, description="平动点编号：1=L1 … 5=L5；缺省按族填默认"
    )
    max_amplitude_km: float | None = Field(
        default=None,
        description="族振幅上限（km）；HALO/AXIAL 带符号区分北/南（上/下）族，"
        "SPO/LPO/HORSESHOE 为正值（距 L4/L5 径向距离 min/max 均值）",
    )
    min_amplitude_km: float | None = Field(
        default=None,
        description="族振幅下限（km），仅 SPO/LPO/HORSESHOE 用",
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
        ranges: dict[str, NumericRange] = {
            "libration_point": NumericRange(min(allowed), max(allowed))
        }
        point = (
            _FAMILY_DEFAULT_LIBRATION_POINT[selection]
            if libration_point is None
            else libration_point
        )
        if point not in allowed:
            raise ValueError(
                f"{selection} libration_point 必须为 {sorted(allowed)}，当前 {point!r}"
            )
        if selection == "HALO":
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
        else:  # SPO/LPO/HORSESHOE：声明范围不得超出可达包络（#435 标定）
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
        if self.libration_point is None:
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
        if sel in ("SPO", "LPO", "HORSESHOE"):
            assert self.min_amplitude_km is not None
            assert self.max_amplitude_km is not None
            if self.min_amplitude_km >= self.max_amplitude_km:
                raise ValueError(
                    f"{sel} min_amplitude_km 必须小于 max_amplitude_km，"
                    f"当前 {self.min_amplitude_km} >= {self.max_amplitude_km}"
                )
        return self
