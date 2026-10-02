"""任务轨道设计（design_orbit）请求/响应模型与条件值域表。"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from pydantic import Field, model_validator

from e2m2e.data.templates import (
    RO_SUPPORTED_RESONANCES,
    SEGMENTED_CORRECTION_ORBIT_TYPES,
)

from .ranges import NumericRange, _range_map, _with_global_amplitude_out
from .shared import ResultResponse, _ApiModel

__all__ = [
    "DesignOrbitRequest",
    "DesignOrbitResponse",
]

_GLOBAL_AMPLITUDE_OUT_RANGES = _with_global_amplitude_out(_range_map())
_DRO_DPO_RANGES = _with_global_amplitude_out(_range_map(amplitude=NumericRange(1737.0, 110000.0)))
_SPO_RANGES = _with_global_amplitude_out(_range_map(amplitude=NumericRange(1737.0, 200000.0)))
_LPO_RANGES = _with_global_amplitude_out(_range_map(amplitude=NumericRange(1000.0, 110000.0)))
#: RO（共振轨道）：五档精确成员分布在不同偏心族上，不能再用全比例
#: 交集表达共同可达域。当前范围采用五个锚点实测振幅的包络并留 5% 余量，
#: 设计时若指定振幅而族行走无法命中则诚实报错；resonance_p/q 为包围盒
#: 粗筛，合法对（RO_SUPPORTED_RESONANCES）由校验器裁决。
_RO_RANGES = _with_global_amplitude_out(
    _range_map(
        amplitude=NumericRange(145000.0, 340000.0),
        resonance_p=NumericRange(2, 4),
        resonance_q=NumericRange(1, 3),
    )
)
_HORSESHOE_RANGES = _with_global_amplitude_out(
    _range_map(amplitude=NumericRange(50000.0, 110000.0))
)

_ORBIT_TYPE_RANGES: Mapping[str, Mapping[str, NumericRange]] = MappingProxyType(
    {
        "DRO": _DRO_DPO_RANGES,
        "DPO": _DRO_DPO_RANGES,
        "NRHO": _with_global_amplitude_out(
            _range_map(perilune_height=NumericRange(100.0, 40000.0))
        ),
        "RO": _RO_RANGES,
        "L4": _GLOBAL_AMPLITUDE_OUT_RANGES,
        "L5": _GLOBAL_AMPLITUDE_OUT_RANGES,
        "AXIAL": _with_global_amplitude_out(_range_map(amplitude=NumericRange(-60000.0, 60000.0))),
        "LYAPUNOV": _with_global_amplitude_out(_range_map(amplitude=NumericRange(5000.0, 60000.0))),
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

#: design_orbit 条件字段的单位（与范围表同住；未列出的字段为无量纲或计数值）。
_DESIGN_ORBIT_FIELD_UNITS: Mapping[str, str] = MappingProxyType(
    {
        "amplitude": "km",
        "amplitude_in": "km",
        "amplitude_out": "km",
        "perilune_height": "km",
    }
)


class DesignOrbitRequest(_ApiModel):
    """任务轨道设计输入。

    统一覆盖 CR3BP 周期轨道（DRO/NRHO/Halo/Lissajous/…）和 ELFO 冻结轨道。
    按 orbit_type 分派校验与默认值填充（``model_validator``）。
    duration 统一用秒。
    """

    orbit_type: str = Field(
        description="DRO/DPO/NRHO/HALO/LYAPUNOV/LISSAJOUS/L4/L5/AXIAL/RO/.../ELFO"
    )
    # CR3BP 形状参数（字段约束为跨类型全局上下限；model_validator 内按类型收紧）
    amplitude: float | None = Field(default=None, ge=-110000.0, le=350000.0)
    resonance_p: int | None = Field(
        default=None, ge=1, description="共振比卫星侧整数（p:q = 卫星:月球），仅 RO 用"
    )
    resonance_q: int | None = Field(default=None, ge=1, description="共振比月球侧整数，仅 RO 用")
    phase: float | None = Field(default=None, ge=0.0, le=1.0)
    collinear_point: int | None = Field(default=None, ge=1, le=3)
    north_south: int | None = Field(default=None, ge=1, le=2)
    amplitude_in: float | None = Field(default=None, gt=0.0, le=100000.0)
    amplitude_out: float | None = Field(default=None, gt=0.0, le=100000.0)
    phase_in: float | None = Field(default=None, ge=0.0, le=1.0)
    phase_out: float | None = Field(default=None, ge=0.0, le=1.0)
    # 共享参数
    perilune_height: float | None = Field(default=None, gt=0.0, le=40000.0)
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
        "（HALO/NRHO/DPO/LYAPUNOV → segmented，其余 CR3BP 族 → two_level）；"
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
        if selection in ("LISSAJOUS", "HALO"):
            point = 2 if collinear_point is None else collinear_point
            if point not in (1, 2, 3) or (selection == "HALO" and point == 3):
                allowed = "1、2 或 3" if selection == "LISSAJOUS" else "1 或 2"
                raise ValueError(
                    f"{selection} collinear_point 必须为 {allowed}，当前 {collinear_point!r}"
                )
            if selection == "LISSAJOUS":
                ranges = _LISSAJOUS_L3_RANGES if point == 3 else _LISSAJOUS_L1_L2_RANGES
            else:
                # L1 设计域止于固定 z0 延拓安全上界 26 908 km（0.07×特征长度，
                # 与族生成侧上限同源，#643 探测确认；族折叠点约 32674 km 在其后，
                # 折叠点后经 NRHO 路径可达）；L2 放宽到 z0 折叠顶 ≈77 787 km 内
                # 的 77 000 km
                limit = 26_908.0 if point == 1 else 77_000.0
                ranges = _with_global_amplitude_out(
                    _range_map(amplitude=NumericRange(-limit, limit))
                )
        else:
            try:
                ranges = _ORBIT_TYPE_RANGES[selection]
            except KeyError as exc:
                raise ValueError(f"不支持的 orbit_type: {orbit_type!r}") from exc
        return dict(ranges)

    @classmethod
    def valid_range_contexts(cls) -> tuple[tuple[str, int | None], ...]:
        """全量导出条件值域用的 (orbit_type, collinear_point) 键集。

        无平动点条件的族 point 为 None；HALO 与 LISSAJOUS 逐点展开
        （HALO 为 L1/L2 两档独立域，LISSAJOUS 1/2 同表、3 独立）。
        """
        contexts = [(orbit_type, None) for orbit_type in _ORBIT_TYPE_RANGES]
        return (
            *contexts,
            ("HALO", 1),
            ("HALO", 2),
            ("LISSAJOUS", 1),
            ("LISSAJOUS", 2),
            ("LISSAJOUS", 3),
        )

    @classmethod
    def field_units(cls) -> Mapping[str, str]:
        """design_orbit 条件字段的单位表；未列出者为无量纲或计数值。"""
        return _DESIGN_ORBIT_FIELD_UNITS

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
        elif sel == "LYAPUNOV":
            if self.collinear_point is None:
                self.collinear_point = 2
            if self.amplitude is None:
                self.amplitude = 12000.0
            if self.phase is None:
                self.phase = 0.0
            if self.collinear_point not in (1, 2):
                raise ValueError(
                    f"LYAPUNOV collinear_point 必须为 1 或 2，当前 {self.collinear_point}"
                )
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
        elif sel == "RO":
            if self.resonance_p is None:
                self.resonance_p = 3
            if self.resonance_q is None:
                self.resonance_q = 1
            if (self.resonance_p, self.resonance_q) not in RO_SUPPORTED_RESONANCES:
                raise ValueError(
                    f"RO 共振比 {self.resonance_p}:{self.resonance_q} 不支持；"
                    f"支持 {'/'.join(f'{p}:{q}' for p, q in sorted(RO_SUPPORTED_RESONANCES))}"
                    "（顺行内共振，p:q = 卫星:月球）"
                )
            # amplitude 缺省保持 None：返回精确成员（会合系周期恰为
            # 2πq，即 q 个恒星月内惯性绕地 p 圈）。
            if self.phase is None:
                self.phase = 0.0
        else:
            raise ValueError(
                f"orbit_type 必须为 DRO/DPO/NRHO/HALO/LYAPUNOV/LISSAJOUS/L4/L5/AXIAL/RO"
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
