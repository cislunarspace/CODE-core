"""轨道保持（control_orbit）请求/响应模型。"""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator, model_validator

from e2m2e.data.constants import SECONDS_PER_DAY
from e2m2e.data.templates.perturbations import DEFAULT_PERTURBATION

from .shared import ResultResponse, _ApiModel

__all__ = [
    "ControlOrbitRequest",
    "ControlOrbitResponse",
]


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
