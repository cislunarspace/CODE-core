"""轨道族生成（orbit_family_generation）请求/响应模型与族条件值域表。"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from pydantic import ConfigDict, Field, model_validator

from e2m2e.data.templates import RO_SUPPORTED_RESONANCES
from e2m2e.data.templates.seed import _HALO_FOLD_Z0, CHAR_LENGTH_KM
from e2m2e.data.types.orbit import Orbit, OrbitFamily
from e2m2e.status import ConvergenceState, FailureCause, ResultStatus

from .ranges import NumericRange
from .shared import _ApiModel

__all__ = [
    "FamilyGenerationRequest",
    "FamilyGenerationResponse",
]

# ---------------------------------------------------------------------------
# 轨道族生成：公开平动点统一术语 libration_point（1=L1 … 5=L5）。
# 各族允许的平动点取值域与默认值；Halo 族振幅上限为固定 z0 延拓的
# 折叠点（同 seed._HALO_FOLD_Z0，按平动点区分）。
# ---------------------------------------------------------------------------

#: orbit_type → 允许的平动点取值域。DRO（月心族）与 RO（地心族）不绑定
#: 平动点（空元组），DPO 等其余无平动点族不进本模型。
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
        "RO": (),
    }
)

#: 族生成条件字段的单位（与范围表同住；未列出的字段为无量纲或计数值）。
_FAMILY_FIELD_UNITS: Mapping[str, str] = MappingProxyType(
    {
        "min_amplitude_km": "km",
        "max_amplitude_km": "km",
        "perilune_height_max_km": "km",
        "match_tolerance_km": "km",
        "amplitude_in_km": "km",
        "amplitude_out_km": "km",
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
        "RO": frozenset(
            {
                "resonance_p",
                "resonance_q",
                "min_amplitude_km",
                "max_amplitude_km",
                "sampling_mode",
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
        "DRO": MappingProxyType({"sampling_mode": ("natural-x0",)}),
        "RO": MappingProxyType({"sampling_mode": ("natural-x0",)}),
    }
)


class FamilyGenerationRequest(_ApiModel):
    """轨道族生成输入（二档 Facade.orbit_family_generation）。

    公开参数用平动点统一术语（``libration_point``，1=L1 … 5=L5），由
    Facade 映射到各族算法参数。按 ``orbit_type`` 分派校验取值域
    （与 ``DesignOrbitRequest`` 同构）：共线族（Halo/NRHO/Axial）仅
    L1/L2，Lissajous 支持 L1/L2/L3，三角族（SPO/LPO/Horseshoe）仅
    L4/L5，DRO 是月心族、RO 是地心族，均不绑定平动点（请求不得携带
    ``libration_point``）。九族均已实现：周期族返回严格周期成员，
    Lissajous 返回拟周期有界轨迹的参数采样（族上显式标注 quasi-periodic）。

    按族适用的字段：

    - HALO/AXIAL：``max_amplitude_km`` （带符号，区分北/南或上/下族）
    - NRHO：``north_south``、``perilune_height_max_km``、``continuation_direction``
    - LISSAJOUS：``amplitude_in_km``、``amplitude_out_km``、``phase_in``、``phase_out``
    - SPO/LPO/HORSESHOE：振幅上下限、延拓方向与 ``match_tolerance_km``
    - DRO：振幅上下限（距月心距离 min/max 均值，km）
    - RO：``resonance_p``/``resonance_q`` + 振幅上下限（距地心距离
      min/max 均值，km）

    ``sampling_mode`` 显式登记各族固定的首版采样规则；传入其他规则会
    结构化拒绝，而不是静默改用默认算法。
    """

    orbit_type: str = Field(description="HALO/NRHO/AXIAL/LISSAJOUS/SPO/LPO/HORSESHOE/DRO/RO")
    libration_point: int | None = Field(
        default=None, ge=1, le=5, description="平动点编号：1=L1 … 5=L5；缺省按族填默认；DRO/RO 不用"
    )
    max_amplitude_km: float | None = Field(
        default=None,
        description="族振幅上限（km）；HALO/AXIAL 带符号区分北/南（上/下）族，"
        "SPO/LPO/HORSESHOE 为正值（距 L4/L5 径向距离 min/max 均值），"
        "DRO 为距月心距离 min/max 均值",
    )
    min_amplitude_km: float | None = Field(
        default=None,
        description="族振幅下限（km），仅 SPO/LPO/HORSESHOE/DRO/RO 用",
    )
    resonance_p: int | None = Field(
        default=None, ge=1, description="共振比卫星侧整数（p:q = 卫星:月球），仅 RO 用"
    )
    resonance_q: int | None = Field(default=None, ge=1, description="共振比月球侧整数，仅 RO 用")
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
        elif selection == "RO":
            # 与 DesignOrbitRequest 的 RO 包络同源；不同共振比的锚点
            # 分布不再取交集，设计者对不可达窗口收到结构化失败。
            amp_range = NumericRange(145000.0, 340000.0)
            ranges["min_amplitude_km"] = amp_range
            ranges["max_amplitude_km"] = amp_range
            ranges["resonance_p"] = NumericRange(2, 4)
            ranges["resonance_q"] = NumericRange(1, 3)
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

    @classmethod
    def valid_range_contexts(cls) -> tuple[tuple[str, int | None], ...]:
        """全量导出条件值域用的 (orbit_type, libration_point) 键集。

        绑定平动点的族逐点展开；不绑平动点的族（DRO）point 为 None。
        """
        contexts: list[tuple[str, int | None]] = []
        for family, points in _FAMILY_LIBRATION_POINT_RANGES.items():
            if points:
                contexts.extend((family, point) for point in points)
            else:
                contexts.append((family, None))
        return tuple(contexts)

    @classmethod
    def field_units(cls) -> Mapping[str, str]:
        """族生成条件字段的单位表；未列出者为无量纲或计数值。"""
        return _FAMILY_FIELD_UNITS

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
            if self.min_amplitude_km is None:
                self.min_amplitude_km = 145000.0
            if self.max_amplitude_km is None:
                self.max_amplitude_km = 340000.0
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
        if sel in ("SPO", "LPO", "HORSESHOE", "DRO", "RO"):
            assert self.min_amplitude_km is not None
            assert self.max_amplitude_km is not None
            if self.min_amplitude_km >= self.max_amplitude_km:
                raise ValueError(
                    f"{sel} min_amplitude_km 必须小于 max_amplitude_km，"
                    f"当前 {self.min_amplitude_km} >= {self.max_amplitude_km}"
                )
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
    family_id: str | None = Field(
        default=None,
        description="生成批次标识（ADR 0045）：成员已逐条入库，整族经"
        " catalog_query(family_id=…) 查询；库关闭或无成员产出时为 None",
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
