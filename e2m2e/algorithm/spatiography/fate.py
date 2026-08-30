"""命运（fate）诊断量与分类器（Primer §7.2，ADR 0041 Phase 3b）。

两层天图的第二层：MEGNO 之外的几何结局分类。论文 §7.2 只给两条硬规则：

1. **终端事件按几何结局归类，与 MEGNO 无关**（Earth reentry / moon
   impact 优先于一切变分判据）；
2. **非终端轨迹按变分特征细分**（bounded：stable quasiperiodic /
   sticky resident；escape：orderly / chaotic；变分特征在积分窗内
   未能干净分辨时落入两个 unclassified 类）。

论文未给出的自由参数（ADR 0041 Phase 3 增补登记，实现者定标）：

- **MEGNO 阈值带**：``|Ȳ − 2| ≤ ybar_ordered_band`` 记 ordered、
  ``Ȳ ≥ 2 + ybar_chaotic_excess`` 记 chaotic、之间记 unclassified。
  缺省 0.2 / 1.0，定标口径：#578 的共振位置交叉验证（SC 区低倾切片
  近全 Ȳ ≈ 2，论文 line 1419）。
- **撞月几何判据**：月面碰撞 ``r_sel ≤ R☾``（月心距触月面），区别于
  进月球 Hill 区（``r_sel ≤ ρ_H``，非终端、只计数）——三选一取月面
  碰撞，与 CR3BP ``collision_detection`` 的半径语义一致。
- **终端检查优先序**：reentry → impact → escape（再入带最常见且几何
  上最先触发，论文 Fig. 13 的黑色带）。

事件机制复用 ``Dynamics.propagate(events=..., backend=...)`` 的 scipy
语义（terminal/direction 属性）；穿越检测的步内定位由积分器完成，
与 ``manifold/sections.py`` 的 ``PoincareSection.event()`` 同款用法，
不与 ``station_keeping`` 的事件列表语义冲突（那边是控制时刻表，非
几何事件面）。
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ...status import ConvergenceState, FailureCause

if TYPE_CHECKING:
    from ..dynamics.cr3bp_system import CR3BP_System

__all__ = [
    "FATE_CLASSES",
    "FATE_THRESHOLDS_DEFAULT",
    "FateDiagnostics",
    "FateThresholds",
    "build_cr3bp_fate_events",
    "classify_fate",
    "extract_cr3bp_fate",
]

#: 8 类命运标签（论文 §7.2 清单）。
FATE_CLASSES: tuple[str, ...] = (
    "stable_quasiperiodic",
    "sticky_resident",
    "orderly_escape",
    "chaotic_escape",
    "earth_reentry",
    "moon_impact",
    "escape_unclassified",
    "bounded_unclassified",
)


@dataclass(frozen=True)
class FateThresholds:
    """MEGNO 有序/混沌阈值带（自由参数，ADR 0041 Phase 3 登记）。

    Attributes:
        ybar_ordered_band: |Ȳ − 2| ≤ band 判 ordered。
        ybar_chaotic_excess: Ȳ ≥ 2 + excess 判 chaotic。
            两带之间为 unclassified（几何结局已知、变分特征未分辨）。
    """

    ybar_ordered_band: float = 0.2
    ybar_chaotic_excess: float = 1.0


FATE_THRESHOLDS_DEFAULT = FateThresholds()


@dataclass(frozen=True)
class FateDiagnostics:
    """单条轨迹的命运诊断量（论文 §7.2 line 1407 清单）。

    Attributes:
        escaped: 是否越出地球 Hill 球（外边界判据）。
        t_escape_days: 首次逃逸时刻（越出 Hill 界），天；未逃逸为 None。
        earth_reentry: 是否再入（地心距 ≤ R⊕）。
        t_reentry_days: 再入时刻，天；未再入为 None。
        moon_impact: 是否撞月（月心距 ≤ R☾）。
        t_impact_days: 撞月时刻，天；未撞为 None。
        min_geocentric_km: 全程最小地心距，km。
        min_selenocentric_km: 全程最小月心距，km。
        moon_hill_entries: 进入月球 Hill 区的次数（非终端计数）。
        span_days: 积分窗长（实际传播的），天。
    """

    escaped: bool
    t_escape_days: float | None
    earth_reentry: bool
    t_reentry_days: float | None
    moon_impact: bool
    t_impact_days: float | None
    min_geocentric_km: float
    min_selenocentric_km: float
    moon_hill_entries: int
    span_days: float


def _event(fn: Callable[[float, np.ndarray], float], terminal: bool, direction: float):
    """按 scipy 语义给事件函数挂 terminal/direction 属性。"""
    fn.terminal = terminal  # type: ignore[attr-defined]
    fn.direction = direction  # type: ignore[attr-defined]
    return fn


def build_cr3bp_fate_events(
    system: CR3BP_System,
    *,
    earth_radius_km: float | None = None,
    moon_radius_km: float | None = None,
    hill_earth_km: float | None = None,
    hill_moon_km: float | None = None,
    escape_over_hill_earth: bool = True,
    count_moon_hill: bool = True,
) -> list[Callable[[float, np.ndarray], float]]:
    """CR3BP 会合系命运事件函数组（供 ``Dynamics.propagate(events=...)``）。

    事件面（无量纲会合系；月球固定于 x = 1−μ）：

    - 再入：``|r_⊕| − R⊕``，terminal、下行；
    - 撞月：``|r_☾| − R☾``，terminal、下行（判据=月面碰撞，见模块
      docstring 自由参数）；
    - 逃逸：``(r_H)^⊕ − |r_⊕|``，terminal、下行（越出地球 Hill 球）；
    - 月 Hill 进入：``ρ_H − |r_☾|``，非终端、上行（只计数）。

    Args:
        system: CR3BP 系统（特征长度换算到无量纲会合系）。
        earth_radius_km / moon_radius_km: 几何半径；None = 天体登记表
            （IAU2015 平均半径）。
        hill_earth_km: 地球 Hill 界；None = spatiography 口径
            （Primer 常数，式 111）。
        hill_moon_km: 月球 Hill 界；None = 同上（式 110 近似式）。
        escape_over_hill_earth: 是否启用地球 Hill 逃逸终端事件。
        count_moon_hill: 是否启用月球 Hill 进入计数事件。

    Returns:
        事件函数列表（顺序即优先序：reentry、impact、escape、hill）。
    """
    from ...data.constants import EARTH, MOON
    from .scales import hill_radius_earth, hill_radius_moon

    length_km = system.characteristic_length
    if length_km is None:
        raise ValueError("CR3BP 系统未初始化特征尺度（characteristic_length），无法构造命运事件")
    mu = system.mu
    r_moon_nd = 1.0 - mu

    if earth_radius_km is None:
        earth_radius_km = EARTH.require_mean_radius_km()
    if moon_radius_km is None:
        moon_radius_km = MOON.require_mean_radius_km()
    if hill_earth_km is None:
        hill_earth_km = hill_radius_earth()
    if hill_moon_km is None:
        hill_moon_km = hill_radius_moon()
    earth_r_nd = earth_radius_km / length_km
    moon_r_nd = moon_radius_km / length_km
    hill_earth_nd = hill_earth_km / length_km
    hill_moon_nd = hill_moon_km / length_km

    def g_reentry(t: float, state: np.ndarray) -> float:
        return math.hypot(state[0], math.hypot(state[1], state[2])) - earth_r_nd

    def g_impact(t: float, state: np.ndarray) -> float:
        dx = state[0] - r_moon_nd
        return math.hypot(dx, math.hypot(state[1], state[2])) - moon_r_nd

    def g_escape(t: float, state: np.ndarray) -> float:
        return hill_earth_nd - math.hypot(state[0], math.hypot(state[1], state[2]))

    def g_moon_hill(t: float, state: np.ndarray) -> float:
        dx = state[0] - r_moon_nd
        return hill_moon_nd - math.hypot(dx, math.hypot(state[1], state[2]))

    events = [
        _event(g_reentry, terminal=True, direction=-1),
        _event(g_impact, terminal=True, direction=-1),
    ]
    if escape_over_hill_earth:
        events.append(_event(g_escape, terminal=True, direction=-1))
    if count_moon_hill:
        events.append(_event(g_moon_hill, terminal=False, direction=1))
    return events


def extract_cr3bp_fate(
    result: dict[str, Any],
    system: CR3BP_System,
    *,
    with_escape: bool = True,
    with_moon_hill: bool = True,
) -> FateDiagnostics:
    """从 ``Dynamics.propagate`` 结果提取命运诊断量（物理单位）。

    Args:
        result: ``CR3BP_Dynamics.propagate(events=..., backend="scipy")``
            的返回（含 t_events/y_events；states 为 (n, 6) 无量纲）。
        system: CR3BP 系统（特征尺度换算 km / 天）。
        with_escape / with_moon_hill: 与事件组构造时的开关一致。

    Returns:
        :class:`FateDiagnostics`（地心/月心距在输出采样点上取最小，
        终端时刻来自积分器步内定位）。
    """
    length_km = system.characteristic_length
    time_days = system.characteristic_time / 86400.0 if system.characteristic_time else None
    if length_km is None or time_days is None:
        raise ValueError("CR3BP 系统未初始化特征尺度，无法换算物理单位")
    mu = system.mu
    states = np.asarray(result["states"], dtype=float)
    times = np.asarray(result["time"], dtype=float)
    r_geo = np.hypot(states[:, 0], np.hypot(states[:, 1], states[:, 2]))
    r_sel = np.hypot(states[:, 0] - (1.0 - mu), np.hypot(states[:, 1], states[:, 2]))

    n_events = 2 + int(with_escape) + int(with_moon_hill)
    t_events = result.get("t_events", [None] * n_events)

    def _first(idx: int) -> float | None:
        if idx >= len(t_events) or t_events[idx] is None or len(t_events[idx]) == 0:
            return None
        return float(t_events[idx][0]) * time_days

    escaped = _first(2) is not None if with_escape else False
    return FateDiagnostics(
        escaped=escaped,
        t_escape_days=_first(2) if with_escape else None,
        earth_reentry=_first(0) is not None,
        t_reentry_days=_first(0),
        moon_impact=_first(1) is not None,
        t_impact_days=_first(1),
        min_geocentric_km=float(r_geo.min() * length_km),
        min_selenocentric_km=float(r_sel.min() * length_km),
        moon_hill_entries=(len(t_events[3]) if with_moon_hill and t_events[3] is not None else 0),
        span_days=float((times[-1] - times[0]) * time_days),
    )


def classify_fate(
    diagnostics: FateDiagnostics,
    megno_ybar: float,
    thresholds: FateThresholds = FATE_THRESHOLDS_DEFAULT,
) -> tuple[str, ConvergenceState, FailureCause, str]:
    """八类命运分类器（§7.2 决策树；自由参数见 :class:`FateThresholds`）。

    决策序（登记于 ADR 0041）：

    1. 终端几何结局优先、与 MEGNO 无关：reentry → impact；
    2. 逃逸（越出地球 Hill 界）：ordered → orderly_escape、chaotic →
       chaotic_escape、未分辨 → escape_unclassified；
    3. 有界：ordered → stable_quasiperiodic、chaotic → sticky_resident
       （有界混沌=粘滞驻留）、未分辨 → bounded_unclassified。

    Returns:
        (类别名, status, cause, message)——类别在 :data:`FATE_CLASSES`。
    """
    if not math.isfinite(megno_ybar):
        return (
            "bounded_unclassified" if not diagnostics.escaped else "escape_unclassified",
            ConvergenceState.CONVERGED,
            FailureCause.NONE,
            "MEGNO 非有限值，变分特征未分辨",
        )

    ordered = abs(megno_ybar - 2.0) <= thresholds.ybar_ordered_band
    chaotic = megno_ybar >= 2.0 + thresholds.ybar_chaotic_excess

    if diagnostics.earth_reentry:
        label = "earth_reentry"
        message = "终端事件：再入（几何结局，与 MEGNO 无关）"
    elif diagnostics.moon_impact:
        label = "moon_impact"
        message = "终端事件：撞月（几何结局，与 MEGNO 无关）"
    elif diagnostics.escaped:
        if ordered:
            label = "orderly_escape"
            message = "规则逃逸：越出地球 Hill 界且 Ȳ 保持正则带"
        elif chaotic:
            label = "chaotic_escape"
            message = "混沌逃逸：越出地球 Hill 界且 Ȳ 强增长"
        else:
            label = "escape_unclassified"
            message = "逃逸但变分特征在窗内未干净分辨"
    else:
        if ordered:
            label = "stable_quasiperiodic"
            message = "有界正则：Ȳ 在正则带内"
        elif chaotic:
            label = "sticky_resident"
            message = "粘滞驻留：有界但变分增长（有界混沌）"
        else:
            label = "bounded_unclassified"
            message = "有界但变分特征在窗内未干净分辨"

    return label, ConvergenceState.CONVERGED, FailureCause.NONE, message
