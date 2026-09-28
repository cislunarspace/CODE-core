"""数值范围基础设施：design 与 family 共用的区间类型、构造器与序列化形式。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from pydantic import Field

from .shared import ResultResponse, _ApiModel

__all__ = [
    "NumericRange",
    "RangeSpec",
    "ValidRangesResponse",
]


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


class RangeSpec(_ApiModel):
    """NumericRange 的序列化形式（机器可读，ADR 0014 决策 8 请求侧）。

    unit 携带字段单位（如 km）；缺省为 None 表示无量纲量或计数值。
    单位属于字段而非区间，同一字段在各类型下的 unit 一致。
    """

    minimum: float | None = Field(default=None, description="下界；None 表示无下界")
    maximum: float | None = Field(default=None, description="上界；None 表示无上界")
    minimum_inclusive: bool = Field(default=True, description="下界是否闭区间")
    maximum_inclusive: bool = Field(default=True, description="上界是否闭区间")
    excluded_values: list[float] = Field(default_factory=list, description="区间内的离散排除值")
    unit: str | None = Field(default=None, description="数值单位（如 km）；缺省为无量纲或计数值")

    @classmethod
    def from_numeric_range(cls, numeric_range: NumericRange, unit: str | None = None) -> RangeSpec:
        """由校验侧 NumericRange 构造；只搬运数值，不复制判定逻辑。"""
        return cls(
            minimum=numeric_range.minimum,
            maximum=numeric_range.maximum,
            minimum_inclusive=numeric_range.minimum_inclusive,
            maximum_inclusive=numeric_range.maximum_inclusive,
            excluded_values=list(numeric_range.excluded_values),
            unit=unit,
        )


class ValidRangesResponse(ResultResponse):
    """valid_ranges 输出：请求侧条件值域全量清单（ADR 0014 决策 8 请求侧）。

    无参数；包版本即值域版本（清单随发布冻结，调用方每会话取一次、
    升级后刷新）。design_orbit 键为 orbit_type（LISSAJOUS 逐平动点拆
    LISSAJOUS_L1/L2/L3）；family_generation_ranges 键为 族_Ln（DRO 不绑
    平动点，键不带后缀）；family_generation_options 为族生成的离散选项。
    """

    design_orbit: dict[str, dict[str, RangeSpec]] = Field(
        description="design_orbit 条件数值范围：orbit_type → 请求字段 → 区间"
    )
    family_generation_ranges: dict[str, dict[str, RangeSpec]] = Field(
        description="族生成条件数值范围：族_Ln → 请求字段 → 区间（含 libration_point 取值约束）"
    )
    family_generation_options: dict[str, dict[str, list[str]]] = Field(
        description="族生成离散选项：族 → 请求字段 → 合法值（延拓方向、采样规则）"
    )
