"""公开数据模型（Pydantic，全手写）。

输入/输出/错误模型精雕参数单位、默认值、取值域（ADR 0014）。Pydantic 只在
api/ 边界，算法层用 numpy/dataclass。每个 Facade 方法一个 Request/Response，
外加统一错误模型 ``OrbitError``。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from e2m2e.algorithm.results import ResultStatus
from e2m2e.data.constants import SECONDS_PER_DAY
from e2m2e.data.templates import ConvergenceState, FailureCause
from e2m2e.data.templates.perturbations import DEFAULT_PERTURBATION

__all__ = [
    "OrbitError",
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
    amplitude_out: float | None = Field(default=None, gt=0.0, le=76000.0)
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
    correction_velocity_tolerance: float = Field(default=0.1, gt=0.0)

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
            return self

        # CR3BP 类型：按类型填默认值
        if sel == "DRO":
            if self.amplitude is None:
                self.amplitude = 10000.0
            if self.phase is None:
                self.phase = 0.5001
            if not 1737.0 <= self.amplitude <= 110000.0:
                raise ValueError(f"DRO amplitude 应在 1737~110000 km，实际 {self.amplitude}")
        elif sel == "DPO":
            if self.amplitude is None:
                self.amplitude = 20000.0
            if self.phase is None:
                self.phase = 0.5001
            if not 1737.0 <= self.amplitude <= 110000.0:
                raise ValueError(f"DPO amplitude 应在 1737~110000 km，实际 {self.amplitude}")
        elif sel == "HALO":
            if self.collinear_point is None:
                self.collinear_point = 2
            if self.amplitude is None:
                self.amplitude = 30000.0
            if self.phase is None:
                self.phase = 0.0
            if self.collinear_point not in (1, 2):
                raise ValueError(f"HALO collinear_point 必须为 1 或 2，当前 {self.collinear_point}")
            if abs(self.amplitude) > 73000.0:
                raise ValueError(f"HALO amplitude 应在 ±73000 km，实际 {self.amplitude}")
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
            if not 100.0 <= self.perilune_height <= 10000.0:
                raise ValueError(
                    f"NRHO perilune_height 应在 100~10000 km，实际 {self.perilune_height}"
                )
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
            limit = 100000.0 if self.collinear_point == 3 else 7600.0
            if not 0.0 < self.amplitude_in <= limit:
                raise ValueError(
                    f"LISSAJOUS amplitude_in 应在 (0, {limit}] km，实际 {self.amplitude_in}"
                )
            if not 0.0 < self.amplitude_out <= limit:
                raise ValueError(
                    f"LISSAJOUS amplitude_out 应在 (0, {limit}] km，实际 {self.amplitude_out}"
                )
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
            if abs(self.amplitude) > 60000.0:
                raise ValueError(f"AXIAL amplitude 应在 ±60000 km，实际 {self.amplitude}")
        elif sel in ("L4_SPO", "L5_SPO"):
            if self.amplitude is None:
                self.amplitude = 10000.0
            if self.phase is None:
                self.phase = 0.0
            if not 1737.0 <= self.amplitude <= 200000.0:
                raise ValueError(f"{sel} amplitude 应在 1737~200000 km，实际 {self.amplitude}")
        elif sel in ("L4_LPO", "L5_LPO"):
            if self.amplitude is None:
                self.amplitude = 50000.0
            if self.phase is None:
                self.phase = 0.0
            if not 1000.0 <= self.amplitude <= 200000.0:
                raise ValueError(f"{sel} amplitude 应在 1000~200000 km，实际 {self.amplitude}")
        elif sel in ("L4_HORSESHOE", "L5_HORSESHOE"):
            if self.amplitude is None:
                self.amplitude = 150000.0
            if self.phase is None:
                self.phase = 0.0
            if not 50000.0 <= self.amplitude <= 200000.0:
                raise ValueError(f"{sel} amplitude 应在 50000~200000 km，实际 {self.amplitude}")
        else:
            raise ValueError(
                f"orbit_type 必须为 DRO/DPO/NRHO/HALO/LISSAJOUS/L4/L5/AXIAL"
                f"/L4_SPO/L5_SPO/L4_LPO/L5_LPO/L4_HORSESHOE/L5_HORSESHOE/ELFO，当前 {sel!r}"
            )
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
