"""catalog_sweep 参数扫描编排：逐点复用统一 Rust 族生成（ADR 0029/0031）。

网格展开（参数空间 → 参数点序列）在接口层完成（它持有请求模型的取值
域知识）；本模块只做逐点执行与失败语义：单点硬失败捕获为带标记的结
局，不中断扫描，已产出的结果保留（ADR 0020 软失败语义）。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..data.templates import ConvergenceState, FailureCause
from ..data.types.orbit import OrbitFamily
from .family import (
    design_axial_family,
    design_halo_family,
    design_horseshoe_family,
    design_lpo_family,
    design_nrho_family,
    design_spo_family,
)
from .results import FamilyGenerationResult

__all__ = ["FamilySweepPoint", "SweepPointResult", "run_family_sweep"]

#: orbit_type → 族生成入口（kwargs 调用形式）。
#: Facade.orbit_family_generation 另有位置参数形式的分派（既有调用契约，
#: 测试断言精确位置参数）；两处分派新增族时需同步。LISSAJOUS 不进扫描
#: （振幅为面内×面外二维）。
_ENTRIES = {
    "HALO": design_halo_family,
    "NRHO": design_nrho_family,
    "AXIAL": design_axial_family,
    "SPO": design_spo_family,
    "LPO": design_lpo_family,
    "HORSESHOE": design_horseshoe_family,
}


@dataclass(frozen=True)
class FamilySweepPoint:
    """扫描单参数点：族、平动点、成员上限与族特定参数（算法层术语）。"""

    orbit_type: str
    libration_point: int
    n_orbits: int
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SweepPointResult:
    """单参数点结局：成功（含软失败）带 ``result``，硬失败带原因。"""

    point: FamilySweepPoint
    status: ConvergenceState
    cause: FailureCause
    message: str
    result: FamilyGenerationResult | None

    def __post_init__(self) -> None:
        if self.result is not None:
            assert self.status is self.result.status
            assert self.cause is self.result.cause


def run_family_sweep(points: Iterable[FamilySweepPoint]) -> list[SweepPointResult]:
    """逐点执行族生成；单点失败不中断，顺序与输入一致。"""
    outcomes: list[SweepPointResult] = []
    for point in points:
        try:
            entry = _ENTRIES[point.orbit_type]
        except KeyError:
            outcomes.append(
                SweepPointResult(
                    point=point,
                    status=ConvergenceState.FAILED,
                    cause=FailureCause.INVALID_INPUT,
                    message=f"catalog_sweep 不支持族 {point.orbit_type!r}",
                    result=None,
                )
            )
            continue
        try:
            raw = entry(point.libration_point, n_orbits=point.n_orbits, **point.kwargs)
        except Exception as exc:
            outcomes.append(
                SweepPointResult(
                    point=point,
                    status=getattr(exc, "status", ConvergenceState.FAILED),
                    cause=getattr(exc, "cause", FailureCause.UNKNOWN),
                    message=str(exc),
                    result=None,
                )
            )
            continue
        if isinstance(raw, FamilyGenerationResult):
            result = raw
        else:  # Halo 旧契约：成功返回裸 OrbitFamily
            assert isinstance(raw, OrbitFamily)
            result = FamilyGenerationResult(
                status=ConvergenceState.CONVERGED,
                cause=FailureCause.NONE,
                message="轨道族生成完成",
                family=raw,
                requested_members=len(raw),
                generated_members=len(raw),
            )
        outcomes.append(
            SweepPointResult(
                point=point,
                status=result.status,
                cause=result.cause,
                message=result.message,
                result=result,
            )
        )
    return outcomes
