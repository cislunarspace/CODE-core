"""catalog_sweep 参数扫描编排：逐点复用统一 Rust 族生成（ADR 0029/0031）。

网格展开（参数空间 → 参数点序列）在接口层完成（它持有请求模型的取值
域知识）；本模块只做逐点执行与失败语义：单点硬失败捕获为带标记的结
局，不中断扫描，已产出的结果保留（ADR 0020 软失败语义）。

能量（Jacobi）窗口点按（族、平动点、生成参数）分组：同组共享一次
延拓 trace（Rust 单次调用返回各窗口结果），行为契约是 trace 只生成
一次，与共享机制的实现无关。
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
    design_lissajous_family,
    design_lpo_family,
    design_nrho_family,
    design_spo_family,
)
from .family.rust_generation import generate_rust_family_windows
from .results import FamilyGenerationResult

__all__ = ["FamilySweepPoint", "SweepPointResult", "run_family_sweep"]

#: orbit_type → 族生成入口（kwargs 调用形式）。
#: Facade.orbit_family_generation 另有位置参数形式的分派（既有调用契约，
#: 测试断言精确位置参数）；两处分派新增族时需同步。LISSAJOUS 走面内×
#: 面外二维振幅网格逐点采样（每次调用自成一点，无窗口分组）。
_ENTRIES = {
    "HALO": design_halo_family,
    "NRHO": design_nrho_family,
    "AXIAL": design_axial_family,
    "LISSAJOUS": design_lissajous_family,
    "SPO": design_spo_family,
    "LPO": design_lpo_family,
    "HORSESHOE": design_horseshoe_family,
}


@dataclass(frozen=True)
class FamilySweepPoint:
    """扫描单参数点：族、平动点、成员上限与族特定参数（算法层术语）。

    ``jacobi_window`` 非空时该点是能量窗口点：同一（族、平动点、
    ``n_orbits``、``kwargs``）的窗口点共享一次延拓 trace，各窗口的
    成员筛选在 Rust 单次调用内完成。
    """

    orbit_type: str
    libration_point: int
    n_orbits: int
    kwargs: dict[str, Any] = field(default_factory=dict)
    jacobi_window: tuple[float, float] | None = None


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
    """逐点执行族生成；单点失败不中断，顺序与输入一致。

    能量窗口点先按（族、平动点、成员上限、生成参数）分组，同组一次
    Rust 调用按窗口序返回各结果；其余点逐点独立执行。
    """
    indexed = list(points)
    outcomes: list[SweepPointResult | None] = [None] * len(indexed)
    for indices in _window_groups(indexed):
        _run_window_group(indexed, indices, outcomes)
    for index, point in enumerate(indexed):
        if outcomes[index] is None:
            outcomes[index] = _run_single_point(point)
    assert all(outcome is not None for outcome in outcomes)
    return [outcome for outcome in outcomes if outcome is not None]


def _window_groups(points: list[FamilySweepPoint]) -> list[list[int]]:
    """把能量窗口点按共享键分组，返回各组输入下标（保持首现顺序）。"""
    groups: dict[tuple, list[int]] = {}
    for index, point in enumerate(points):
        if point.jacobi_window is None:
            continue
        key = (
            point.orbit_type,
            point.libration_point,
            point.n_orbits,
            tuple(sorted(point.kwargs.items())),
        )
        groups.setdefault(key, []).append(index)
    return list(groups.values())


def _run_window_group(
    points: list[FamilySweepPoint],
    indices: list[int],
    outcomes: list[SweepPointResult | None],
) -> None:
    """同组窗口点共享一次延拓 trace，各窗口结果按点序回填。"""
    head = points[indices[0]]
    windows: list[tuple[float, float]] = []
    for index in indices:
        window = points[index].jacobi_window
        assert window is not None
        windows.append(window)
    try:
        results = generate_rust_family_windows(
            head.orbit_type.lower(),
            head.libration_point,
            head.n_orbits,
            windows,
            **head.kwargs,
        )
    except Exception as exc:
        for index in indices:
            outcomes[index] = SweepPointResult(
                point=points[index],
                status=getattr(exc, "status", ConvergenceState.FAILED),
                cause=getattr(exc, "cause", FailureCause.UNKNOWN),
                message=str(exc),
                result=None,
            )
        return
    if len(results) != len(indices):
        for index in indices:
            outcomes[index] = SweepPointResult(
                point=points[index],
                status=ConvergenceState.FAILED,
                cause=FailureCause.BACKEND_FAILURE,
                message=(
                    f"能量窗口批量生成返回 {len(results)} 条结果，与窗口数 {len(indices)} 不符"
                ),
                result=None,
            )
        return
    for index, result in zip(indices, results, strict=True):
        outcomes[index] = SweepPointResult(
            point=points[index],
            status=result.status,
            cause=result.cause,
            message=result.message,
            result=result,
        )


def _run_single_point(point: FamilySweepPoint) -> SweepPointResult:
    """逐点执行族生成；异常捕获为带标记的硬失败结局。"""
    try:
        entry = _ENTRIES[point.orbit_type]
    except KeyError:
        return SweepPointResult(
            point=point,
            status=ConvergenceState.FAILED,
            cause=FailureCause.INVALID_INPUT,
            message=f"catalog_sweep 不支持族 {point.orbit_type!r}",
            result=None,
        )
    try:
        raw = entry(point.libration_point, n_orbits=point.n_orbits, **point.kwargs)
    except Exception as exc:
        return SweepPointResult(
            point=point,
            status=getattr(exc, "status", ConvergenceState.FAILED),
            cause=getattr(exc, "cause", FailureCause.UNKNOWN),
            message=str(exc),
            result=None,
        )
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
    return SweepPointResult(
        point=point,
        status=result.status,
        cause=result.cause,
        message=result.message,
        result=result,
    )
