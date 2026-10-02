#!/usr/bin/env python
"""平面 DRO 族垂直临界点标定（#689 三维 DRO 族的种子来源）。

用法（诊断/基线脚本约定：直接用虚拟环境解释器，不要走 `uv run`）：

    .venv/bin/python scripts/calibrate_dro3d_vertical_critical.py

流程：

1. 从平面 DRO 标准种子出发，沿 x0 双向链式延拓铺族（种子本身已是族上的大
   振幅成员，垂直临界点可能落在振幅增大侧或减小侧，两侧都扫）；
2. 逐成员由 monodromy 的辛块直接读出三条稳定性指数 ν（#687 删除
   `e2m2e.algorithm.stability` 后本脚本自含，不做特征值配对）：面外对
   ``ν_out = M[2,2] + M[5,5]``（即 ``2·vt``）；面内对由 x/y/vx/vy 的 4×4
   辛块 ``A_in`` 满足 ``ν² − tr(A_in)·ν + (m_in − 2) = 0``（``m_in`` 为该块
   特征多项式中间系数 ``(tr² − tr A_in²)/2``），两根含平凡对 ``ν ≡ 2``；
   沿 x0 跟踪这三条 ν 轨迹（平凡对逐成员按值剔除，缺口两侧不跨接），相邻
   成员间的乘子位移超 ``|Δν| > 0.3·max(1, |ν|)`` 的区间判为跳支、不报穿越；
3. 对每个候选穿越用二分精化（分点修正失败依次试 1/2 → 1/4 → 3/4 分点），
   并对 SADDLE_NODE 候选用库内 ``_vertical_trace`` 复算 z-vz 块半迹
   ``vt = 0.5·(M[2,2] + M[5,5])``，打印 ``|vt − 1|``（定义级交叉验证：ν 跨
   +2 ⟺ vt 过 +1），并以 ``|vt − 1| ≤ _VT_TOL`` 加门控；未过门控的
   SADDLE_NODE 候选标注为「疑似面内对，非垂直临界」，不计入结论；
4. 取族曲线上距种子最近的垂直临界点（即从种子出发的第一个），打印可直接粘贴
   进 ``crates/e2m2e-integrators/src/family_generation/dro3d.rs`` 的常量块。

单方向延拓在修正失败时退回上一成功成员并把步长减半重试（下限与生产族行走
``_walk_family`` 同为 1e-4），并记录该方向的停止原因（域边界 / 修正失败 / 成员
预算耗尽）。域内无垂直临界穿越时以退出码 1 结束，结论按停止原因与扫描是否干净
分三档——两侧都走到族参数域边界且无失败点、无跳支区间判为物理阻塞；至少一侧在
到达域端点前停止但扫描干净时，结论是「已覆盖域内无穿越」（覆盖未及域端点，逐侧
停止原因见输出）；存在失败点或跳支区间时结论不可得出。三档都列出停止原因与失败/
跳支计数，且都须把证据发到 #689 请求裁决，不得静默改用其他分岔点。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.algorithm.family.axial_initial_guess import _vertical_trace
from e2m2e.algorithm.family.orbits.dro import _correct_dro
from e2m2e.algorithm.family.orbits.walk import _moon_distance_minmax, _z_amplitude_max
from e2m2e.data.constants import Datum
from e2m2e.data.templates.seed import _DRO_SEED_X0
from e2m2e.data.types.orbit import Orbit

#: 垂直临界判据 ``|vt − 1|`` 容差；与
#: ``e2m2e/algorithm/family/axial_initial_guess.py`` 的
#: ``abs(vt_mid - 1.0) < 1e-4`` 同口径
_VT_TOL = 1e-4

#: 延拓步长下限；与 ``e2m2e/algorithm/family/orbits/walk.py::_walk_family``
#: 的 ``step *= 0.5; if step < 1e-4`` 退半步重试下限一致
_MIN_WALK_STEP = 1e-4

#: 平凡乘子对判据（自治 Hamilton 流的能量/时间平移方向，ν ≡ 2）；数值噪声
#: 量级 ~1e-11，与已删模块 `_drop_trivial_pair` 同阈值口径。
_TRIVIAL_NU_TOL = 1e-6

#: 判据的 |Im ν| 容差（区分实对与离圆复四元组），与 #688 同口径。
_NS_IMAG_TOL = 1e-6

#: 二分精化的收敛阈值（族参数区间宽 / |判据|），与 #688 同口径。
_PARAMETER_TOL = 1e-6
_INDICATOR_TOL = 1e-6
_MAX_REFINE_ITER = 50

#: 跳支区间的 |Δν| 阈值（按 ν 量级缩放：``|Δν| > 0.3·max(1, |ν|)``），与
#: #688 同口径；共线族的面内指数量级可达 1e3 且逐步变化数十，绝对阈值会把
#: 平滑延拓的每个区间都误判为跳支、连真实穿越一并屏蔽。
_JUMP_THRESHOLD = 0.3

#: 逐类型的实轴分岔判据零点偏移：判据 = Re ν + offset，过零即分岔。
_CRITERION_OFFSETS: dict[Kind, float] = {
    "saddle_node": -2.0,
    "period_doubling": 2.0,
}

Kind = Literal["saddle_node", "period_doubling", "torus"]


def _period(orbit: Orbit) -> float:
    """周期（周期轨道必有 period；运行时守卫）。"""
    assert orbit.period is not None
    return float(orbit.period)


def _char_length(dynamics: CR3BP_Dynamics) -> float:
    """特征长度（DU→km；系统必有缩放）。"""
    length = dynamics.system.characteristic_length
    assert length is not None
    return float(length)


def _moon_amplitude_km(dynamics: CR3BP_Dynamics, orbit: Orbit) -> float:
    """一个周期内距月心距离 min/max 均值（km），与 ``design_dro`` 同定义。"""
    minimum, maximum = _moon_distance_minmax(dynamics, orbit)
    return 0.5 * (minimum + maximum) * _char_length(dynamics)


def _z_amplitude_km(dynamics: CR3BP_Dynamics, orbit: Orbit) -> float:
    """一个周期内 ``max|z|``（km）；采样点数用库函数默认值（1000）。"""
    return _z_amplitude_max(dynamics, orbit) * _char_length(dynamics)


def _in_plane_nus(monodromy: np.ndarray) -> list[complex]:
    """面内两根：x/y/vx/vy 的 4×4 辛块解 ``ν² − tr(A_in)·ν + (m_in − 2) = 0``。

    平面成员的 x/y/vx/vy 子块是与 z/vz 解耦的 4×4 辛块，其 ν 谱为
    {平凡对 2, 面内对}；``m_in`` 取特征多项式中间系数
    ``(tr(A_in)² − tr(A_in²))/2``，不取行列式（辛矩阵行列式恒为 1）。
    """
    indices = (0, 1, 3, 4)
    block = monodromy[np.ix_(indices, indices)]
    trace = float(np.trace(block))
    middle = 0.5 * (trace * trace - float(np.trace(block @ block)))
    discriminant = trace * trace - 4.0 * (middle - 2.0)
    if discriminant >= 0.0:
        root = float(np.sqrt(discriminant))
        return [complex(0.5 * (trace + root)), complex(0.5 * (trace - root))]
    root = float(np.sqrt(-discriminant))
    return [complex(0.5 * trace, 0.5 * root), complex(0.5 * trace, -0.5 * root)]


def _nus(dynamics: CR3BP_Dynamics, orbit: Orbit) -> list[complex]:
    """三条稳定性指数 ν：面内两根（含平凡对 ν=2）+ 面外 ν_out = M[2,2] + M[5,5]。"""
    assert orbit.period is not None
    monodromy = np.asarray(
        dynamics.compute_state_transition_matrix(orbit.states[0], orbit.period), dtype=float
    )
    vertical = complex(monodromy[2, 2] + monodromy[5, 5])
    return [*_in_plane_nus(monodromy), vertical]


def _member_nus(dynamics: CR3BP_Dynamics, orbit: Orbit) -> list[complex] | None:
    """成员 → 剔除平凡对后的 ν 轨迹集合；单值矩阵传播失败时返回 None。"""
    try:
        return _drop_trivial_pair(_nus(dynamics, orbit))
    except Exception:  # noqa: BLE001 - 诊断脚本：单成员失败即记 failure，不中断
        return None


def _drop_trivial_pair(nus: list[complex]) -> list[complex]:
    """剔除平凡对（ν ≡ 2）。

    平凡对的 ν 恒为 2，而真实分岔对**穿过** ν = 2：按值跟踪时两者在临界成员上
    互换次序，逐成员剔除平凡对才不会让跟踪在临界处交换身份而吞掉穿越。
    """
    if len(nus) < 2:
        return list(nus)
    nearest = min(range(len(nus)), key=lambda index: abs(nus[index] - 2.0))
    if abs(nus[nearest] - 2.0) > _TRIVIAL_NU_TOL:
        return list(nus)
    return [nu for index, nu in enumerate(nus) if index != nearest]


def _indicator(kind: Kind, nu: complex, ns_imag_tol: float) -> float:
    """分岔判据函数：穿越判据的标量形式（过零即分岔）。"""
    offset = _CRITERION_OFFSETS.get(kind)
    if offset is not None:
        return nu.real + offset
    return abs(nu.imag) - ns_imag_tol


def _crossing_kind(
    nu_lo: complex,
    nu_hi: complex,
    ns_imag_tol: float,
    *,
    nu_before: complex | None = None,
) -> Kind | None:
    """单区间、单条 ν 轨迹上的穿越判据（变号或离圆 onset）。

    实轴穿越分两种：判据在区间两端**严格异号**；或判据恰为零的**左端点**成员
    （网格节点恰好落在临界参数上）——后者只在该节点确为符号变化点时报出，即前一个
    成员（``nu_before``）与右端点的判据异号；``nu_before`` 缺省或也为零时不报。
    因此零判据节点若只是被触及而未穿过（前后同号）不报点，真穿越也只有一个区间
    认领。族参数域端点的零判据成员在扫描外的端点认领里处理（见 ``_scan``）。
    """
    if abs(nu_lo.imag) <= ns_imag_tol and abs(nu_hi.imag) <= ns_imag_tol:
        for kind, offset in _CRITERION_OFFSETS.items():
            lower = nu_lo.real + offset
            upper = nu_hi.real + offset
            if (lower < 0.0 < upper) or (upper < 0.0 < lower):
                return kind
            if lower == 0.0 and upper != 0.0 and nu_before is not None:
                before = nu_before.real + offset
                if before != 0.0 and (before < 0.0) != (upper < 0.0):
                    return kind
    if abs(nu_lo.imag) <= ns_imag_tol < abs(nu_hi.imag):
        return "torus"
    return None


def _nearest_nu(nus: list[complex], reference: complex) -> complex:
    """在成员的三条 ν 中认领最接近参考值的跟踪对。"""
    return min(nus, key=lambda nu: abs(nu - reference))


@dataclass(frozen=True)
class _Point:
    """族级分岔点（沿族参数定位的单个分岔穿越）。"""

    parameter: float
    kind: Kind
    orbit: Orbit
    residual: float


@dataclass(frozen=True)
class _Jump:
    """族成员间的跳支区间：乘子位移超阈，不产生分岔点。"""

    parameter_lo: float
    parameter_hi: float
    max_displacement: float


@dataclass(frozen=True)
class _Failure:
    """族成员分析失败记录（该成员两侧的区间都不参与配对与判据）。"""

    parameter: float
    message: str


@dataclass(frozen=True)
class _Scan:
    """沿族参数的扫描结果（与 #688 扫描器同结构的自含版本）。"""

    points: tuple[_Point, ...]
    branch_jumps: tuple[_Jump, ...]
    failures: tuple[_Failure, ...]


def _track(
    ordered: list[tuple[float, Orbit]], dynamics: CR3BP_Dynamics
) -> tuple[list[dict[int, complex]], list[list[int] | None], list[_Failure]]:
    """逐成员算 ν → 相邻成员间按最小位移分配保序配对，得到 ν 轨迹。

    成员分析失败的缺口两侧不跨接（后续成员另起新轨迹）；未分配到的槽位另起一条
    新轨迹，绝不复用既有轨迹号。
    """
    failures: list[_Failure] = []
    member_nus: list[list[complex] | None] = []
    for parameter, orbit in ordered:
        nus = _member_nus(dynamics, orbit)
        if nus is None:
            failures.append(_Failure(parameter=parameter, message="单值矩阵传播或 ν 求解失败"))
        member_nus.append(nus)

    tracks: list[dict[int, complex]] = []
    track_of: list[list[int] | None] = []
    previous_slots: list[int] | None = None
    for index, slot_nus in enumerate(member_nus):
        if slot_nus is None:
            track_of.append(None)
            previous_slots = None
            continue
        if previous_slots is None:
            slots = list(range(len(tracks), len(tracks) + len(slot_nus)))
            for nu in slot_nus:
                tracks.append({index: nu})
        else:
            # 贪心最近匹配：槽位数 ≤ 3 且轨迹间量级分离（面内 O(1)~1e3、
            # 面外 O(0.1)~1），与 #688 的最优分配（linear_sum_assignment）同解。
            slots = [-1] * len(slot_nus)
            remaining = set(range(len(slot_nus)))
            for row in sorted(previous_slots, key=lambda slot: tracks[slot][index - 1].real):
                reference = tracks[row][index - 1]
                column = min(remaining, key=lambda item: abs(slot_nus[item] - reference))
                remaining.discard(column)
                tracks[row][index] = slot_nus[column]
                slots[column] = row
            for column in sorted(remaining):
                tracks.append({index: slot_nus[column]})
                slots[column] = len(tracks) - 1
        track_of.append(slots)
        previous_slots = slots
    return tracks, track_of, failures


def _refine(
    *,
    kind: Kind,
    interval: tuple[float, float],
    nus: tuple[complex, complex],
    walker: _DroWalker,
    dynamics: CR3BP_Dynamics,
) -> tuple[float, Orbit, float] | None:
    """对穿越区间二分精化，返回 (族参数, 轨道, |判据|)。

    分点修正失败时依次试 1/4、3/4 分点（``_walk_family`` 先例，绕开共振
    缝隙）；三点全失败返回 None，由调用方记入失败清单并放弃该点。
    """
    lo, hi = interval
    nu_lo, nu_hi = nus
    ind_lo = _indicator(kind, nu_lo, _NS_IMAG_TOL)
    best: tuple[float, Orbit, float] | None = None
    for _ in range(_MAX_REFINE_ITER):
        sampled = False
        for fraction in (0.5, 0.25, 0.75):
            p_try = lo + fraction * (hi - lo)
            try:
                orbit_try = walker(p_try)
                candidates = _member_nus(dynamics, orbit_try)
                if candidates is None:
                    raise ValueError("ν 求解失败")
                nu_try = _nearest_nu(candidates, 0.5 * (nu_lo + nu_hi))
            except Exception:  # noqa: BLE001 - 诊断脚本：分点失败即换分点
                continue
            sampled = True
            ind_try = _indicator(kind, nu_try, _NS_IMAG_TOL)
            residual = abs(ind_try)
            if best is None or residual < best[2]:
                best = (p_try, orbit_try, residual)
            if residual <= _INDICATOR_TOL or (hi - lo) <= _PARAMETER_TOL:
                return best
            if (ind_try > 0.0) == (ind_lo > 0.0):
                lo, ind_lo, nu_lo = p_try, ind_try, nu_try
            else:
                hi, nu_hi = p_try, nu_try
            break
        if not sampled:
            return None
    return best


def _scan(
    ordered: list[tuple[float, Orbit]], dynamics: CR3BP_Dynamics, walker: _DroWalker
) -> _Scan:
    """沿族参数扫描分岔穿越（#688 扫描器的自含版本，判据与阈值同口径）。"""
    tracks, track_of, failures = _track(ordered, dynamics)
    params = [parameter for parameter, _ in ordered]
    detections: list[tuple[int, Kind, complex, complex]] = []
    branch_jumps: list[_Jump] = []
    for index in range(len(params) - 1):
        slots_lo = track_of[index]
        slots_hi = track_of[index + 1]
        if slots_lo is None or slots_hi is None:
            continue
        shared = [slot for slot in dict.fromkeys(slots_lo) if slot in set(slots_hi)]
        displacements = [
            (slot, abs(tracks[slot][index + 1] - tracks[slot][index])) for slot in shared
        ]
        jumps = [
            (slot, displacement)
            for slot, displacement in displacements
            if displacement
            > _JUMP_THRESHOLD * max(1.0, abs(tracks[slot][index]), abs(tracks[slot][index + 1]))
        ]
        if jumps:
            branch_jumps.append(
                _Jump(
                    parameter_lo=params[index],
                    parameter_hi=params[index + 1],
                    max_displacement=float(max(displacement for _, displacement in jumps)),
                )
            )
            continue
        for slot in shared:
            nu_lo = tracks[slot][index]
            nu_hi = tracks[slot][index + 1]
            kind = _crossing_kind(nu_lo, nu_hi, _NS_IMAG_TOL, nu_before=tracks[slot].get(index - 1))
            if kind is not None:
                detections.append((index, kind, nu_lo, nu_hi))

    points: list[_Point] = []
    for index, kind, nu_lo, nu_hi in detections:
        p_lo, p_hi = params[index], params[index + 1]
        refined = _refine(
            kind=kind,
            interval=(p_lo, p_hi),
            nus=(nu_lo, nu_hi),
            walker=walker,
            dynamics=dynamics,
        )
        if refined is None:
            failures.append(
                _Failure(
                    parameter=0.5 * (p_lo + p_hi),
                    message=f"分岔精化区间 [{p_lo}, {p_hi}] 内成员修正均失败",
                )
            )
            continue
        parameter, orbit, residual = refined
        points.append(_Point(parameter=parameter, kind=kind, orbit=orbit, residual=residual))

    points.extend(_claim_domain_endpoint(ordered, dynamics))
    points.sort(key=lambda point: point.parameter)
    # 同参数、同类型的重复检测（同一条穿越被两次认领）只保留残差最小的一条。
    deduped: list[_Point] = []
    for point in points:
        previous = deduped[-1] if deduped else None
        if (
            previous is not None
            and previous.parameter == point.parameter
            and previous.kind == point.kind
        ):
            if point.residual < previous.residual:
                deduped[-1] = point
            continue
        deduped.append(point)
    return _Scan(points=tuple(deduped), branch_jumps=tuple(branch_jumps), failures=tuple(failures))


def _claim_domain_endpoint(
    ordered: list[tuple[float, Orbit]], dynamics: CR3BP_Dynamics
) -> list[_Point]:
    """认领族参数域端点上的零判据成员。

    域首端成员没有前一个成员可比、域末端只有左邻，按"判据恰为零的左端点"规则
    都不被区间认领；端点成员本身落在临界参数上时判据恰为零，这里单独认领，避免
    端点上的穿越被漏掉。认领走未剔平凡对的完整 ν 集合：端点上的垂直临界正是面外
    ``ν_out`` 恰为 +2，而 ``_drop_trivial_pair`` 会把它当作最接近 2 的那条剔掉，
    只看跟踪轨迹这一档永远认不到。实轴判据另按区间口径补 ``|Im ν|`` 门控，避免
    浮点舍入恰好等于 2.0 的平凡根误报。
    """
    claimed: list[_Point] = []
    for parameter, orbit in (ordered[0], ordered[-1]):
        try:
            nus = _nus(dynamics, orbit)
        except Exception:  # noqa: BLE001 - 诊断脚本：端点 ν 求解失败即不认领
            continue
        for nu in nus:
            if abs(nu.imag) > _NS_IMAG_TOL:
                continue
            for kind in _CRITERION_OFFSETS:
                if _indicator(kind, nu, _NS_IMAG_TOL) == 0.0:
                    claimed.append(
                        _Point(parameter=parameter, kind=kind, orbit=orbit, residual=0.0)
                    )
    return claimed


class _DroWalker:
    """链式初猜的平面 DRO 族修正器（最近已知成员作 Newton 初猜）。"""

    def __init__(self, dynamics: CR3BP_Dynamics, seed_x0: float) -> None:
        self.dynamics = dynamics
        self.known: list[tuple[float, Orbit]] = [(seed_x0, _correct_dro(dynamics, seed_x0, None))]

    def __call__(self, x0: float) -> Orbit:
        guess = min(self.known, key=lambda item: abs(item[0] - x0))[1]
        orbit = _correct_dro(self.dynamics, x0, guess)
        self.known.append((x0, orbit))
        return orbit

    def seed(self, seed_x0: float) -> Orbit:
        return next(orbit for x0, orbit in self.known if x0 == seed_x0)


class _WalkResult(NamedTuple):
    """单方向延拓结果：成员序列（含种子）、停止原因、是否走到族参数域边界。"""

    members: list[tuple[float, Orbit]]
    stop_reason: str
    at_domain_edge: bool


def _walk(
    walker: _DroWalker,
    step: float,
    count: int,
    dynamics: CR3BP_Dynamics,
) -> _WalkResult:
    """自种子沿 ``step``（带符号）定步延拓，返回成员序列与停止原因。

    修正失败时退回上一成功成员并把步长减半重试（下限 ``_MIN_WALK_STEP``，
    与生产族行走 ``_walk_family`` 同口径）——从失败点原步长继续会在域边界处
    一路失败；``x0`` 越出族参数域、步长减到下限仍失败、成员预算耗尽三者任一
    都停止该方向并记录原因。
    """
    x0 = _DRO_SEED_X0
    members: list[tuple[float, Orbit]] = [(x0, walker(x0))]
    for _ in range(count - 1):
        while True:
            x_try = x0 + step
            if not 0.0 < x_try < 1.0 - dynamics.system.mu:
                return _WalkResult(
                    members,
                    f"x0={x_try:.6f} 越出族参数域（0 < x0 < {1.0 - dynamics.system.mu:.6f}），"
                    f"止于 x0={x0:.6f}（{len(members)} 条成员）",
                    True,
                )
            try:
                orbit = walker(x_try)
            except Exception as exc:  # noqa: BLE001 - 诊断脚本：失败即退半步重试
                step *= 0.5
                if abs(step) < _MIN_WALK_STEP:
                    return _WalkResult(
                        members,
                        f"在 x0={x_try:.6f} 修正失败（{type(exc).__name__}: {exc}），"
                        f"步长已退到下限 {_MIN_WALK_STEP:.1e} 以下，"
                        f"止于 x0={x0:.6f}（{len(members)} 条成员）",
                        False,
                    )
                continue
            x0 = x_try
            members.append((x0, orbit))
            break
    return _WalkResult(
        members,
        f"成员预算耗尽（{len(members)} 条），止于 x0={x0:.6f}",
        False,
    )


def _print_trace(label: str, members: list[tuple[float, Orbit]], dynamics: CR3BP_Dynamics) -> None:
    amplitudes = [_moon_amplitude_km(dynamics, orbit) for _, orbit in members]
    print(
        f"# {label}：x0 ∈ [{members[0][0]:.6f}, {members[-1][0]:.6f}]、{len(members)} 条成员、"
        f"振幅 {amplitudes[0]:.1f} → {amplitudes[-1]:.1f} km、"
        f"vt {_vertical_trace(dynamics, members[0][1]):+.6f} → "
        f"{_vertical_trace(dynamics, members[-1][1]):+.6f}"
    )
    print("#   x0          振幅(km)  max|z|(km)      vt       |vt-1|   ν（剔平凡对后）")
    for (x0, orbit), amplitude in zip(members, amplitudes, strict=True):
        vt = float(_vertical_trace(dynamics, orbit))
        nus = _member_nus(dynamics, orbit)
        # ν 求解失败时显式标注，不得静默少打一条
        nus_text = (
            " ".join(f"{nu.real:+.6f}{nu.imag:+.6f}i" for nu in nus)
            if nus is not None
            else "<ν 求解失败>"
        )
        print(
            f"    {x0:.6f}  {amplitude:10.1f}  {_z_amplitude_km(dynamics, orbit):9.3e}"
            f"  {vt:+.6f}  {abs(vt - 1.0):.3e}  {nus_text}"
        )


def _rust_block(orbit: Orbit, amplitude_km: float) -> str:
    state = np.asarray(orbit.states[0], dtype=float)
    lines = [
        "pub(crate) const SEED_DRO3D_STATE: [f64; 6] = [",
        *(f"    {value:.17g}," for value in state),
        "];",
        f"pub(crate) const SEED_DRO3D_PERIOD: f64 = {_period(orbit):.17g};",
        f"pub(crate) const SEED_DRO3D_AMPLITUDE_KM: f64 = {amplitude_km:.1f};",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="平面 DRO 族垂直临界点标定（#689）")
    parser.add_argument("--step", type=float, default=0.005, help="x0 延拓步长（无量纲，须 > 0）")
    parser.add_argument(
        "--max-members", type=int, default=90, help="单方向延拓成员上限（含种子，须 ≥ 2）"
    )
    args = parser.parse_args(argv)
    if not args.step > 0.0:
        parser.error(f"--step 须为正数（得到 {args.step}）")
    if args.max_members < 2:
        parser.error(f"--max-members 须 ≥ 2（得到 {args.max_members}）")

    system = CR3BP_System(
        mu=Datum.DE421.mu, primary="Earth", secondary="Moon"
    )._with_default_scales()
    dynamics = CR3BP_Dynamics(system)
    print(
        f"# μ = {system.mu!r}（DE421 地月）、DU = {_char_length(dynamics):.1f} km、"
        f"月心 x = {1.0 - system.mu:.6f}"
    )

    walker = _DroWalker(dynamics, _DRO_SEED_X0)
    seed = walker.seed(_DRO_SEED_X0)
    print(
        f"# 种子 x0={_DRO_SEED_X0:.6f}：振幅 {_moon_amplitude_km(dynamics, seed):.1f} km、"
        f"max|z| {_z_amplitude_km(dynamics, seed):.3e} km、"
        f"vt {_vertical_trace(dynamics, seed):+.6f}、T {_period(seed):.6f}"
    )

    step = args.step
    decreasing = _walk(walker, -step, args.max_members, dynamics)
    _print_trace("x0 减小侧（振幅增大侧）", decreasing.members, dynamics)
    print(f"# x0 减小侧延拓停止：{decreasing.stop_reason}")
    increasing = _walk(walker, +step, args.max_members, dynamics)
    _print_trace("x0 增大侧（振幅减小侧）", increasing.members, dynamics)
    print(f"# x0 增大侧延拓停止：{increasing.stop_reason}")
    members = decreasing.members + increasing.members

    # 扫描：parameters 须严格递增，按 x0 升序送入；x0 相同的成员（种子两侧
    # 延拓各出现一次）合流
    combined = dict(members)
    ordered = sorted(combined.items())
    scan = _scan(ordered, dynamics, walker)
    print(
        f"# 扫描（{len(ordered)} 成员）：{len(scan.points)} 个候选穿越、"
        f"{len(scan.branch_jumps)} 个跳支区间、{len(scan.failures)} 个失败点"
    )
    for failure in scan.failures:
        print(f"#   失败 x0={failure.parameter:.6f}: {failure.message}")
    for jump in scan.branch_jumps:
        print(
            f"#   跳支 [{jump.parameter_lo:.6f}, {jump.parameter_hi:.6f}]"
            f" Δν_max={jump.max_displacement:.6f}"
        )

    vertical: list[tuple[_Point, float, float]] = []  # (点, 振幅, vt)
    gated_out = 0
    for point in scan.points:
        amplitude = _moon_amplitude_km(dynamics, point.orbit)
        vt = float(_vertical_trace(dynamics, point.orbit))
        print(
            f"# 候选 {point.kind:14s} x0={point.parameter:.9f}"
            f" 振幅={amplitude:10.1f} km max|z|={_z_amplitude_km(dynamics, point.orbit):.3e} km"
            f" 残差={point.residual:.3e} |vt-1|={abs(vt - 1.0):.3e} vt={vt:+.9f}"
            f" T={_period(point.orbit):.9f}"
        )
        if point.kind != "saddle_node":
            continue
        if abs(vt - 1.0) <= _VT_TOL:
            vertical.append((point, amplitude, vt))
        else:
            # ν 跨 +2 但 vt 不过 1：疑似面内对（非垂直临界），不得计入结论
            gated_out += 1
            print(
                f"#   ↑ ν=+2 候选（疑似面内对，非垂直临界）："
                f"|vt-1|={abs(vt - 1.0):.3e} > {_VT_TOL:.1e}"
            )

    if gated_out:
        print(f"# 另有 {gated_out} 个 ν=+2 候选未通过 |vt-1| ≤ {_VT_TOL:.1e} 门控（疑似面内对）")

    if not vertical:
        clean = not scan.failures and not scan.branch_jumps
        print("# 未在已覆盖的平面 DRO 族参数域内发现 ν 跨 +2（垂直临界）穿越。", file=sys.stderr)
        if decreasing.at_domain_edge and increasing.at_domain_edge and clean:
            print(
                "# 两侧延拓均走到族参数域边界、扫描无失败点与跳支区间：属物理阻塞，"
                "须把本输出发到 #689 请求裁决。",
                file=sys.stderr,
            )
        elif clean:
            print(
                "# 延拓在到达族参数域端点前停止（逐侧原因见下：可能成员预算耗尽，"
                "也可能修正失败退到步长下限），扫描无失败点与跳支区间：结论是"
                "「已覆盖域内无穿越」，覆盖未及族参数域端点；须把本输出发到 #689"
                " 请求裁决。",
                file=sys.stderr,
            )
        else:
            print(
                "# 延拓停止原因或扫描结果不满足上述两档（失败点/跳支区间见下），"
                "结论不可得出；须把本输出发到 #689 请求裁决。",
                file=sys.stderr,
            )
        print(f"#   x0 减小侧停止原因：{decreasing.stop_reason}", file=sys.stderr)
        print(f"#   x0 增大侧停止原因：{increasing.stop_reason}", file=sys.stderr)
        print(
            f"#   扫描失败点 {len(scan.failures)} 个、跳支区间 {len(scan.branch_jumps)} 个",
            file=sys.stderr,
        )
        return 1

    # 距种子最近者即"从种子出发沿族曲线遇到的第一个垂直临界点"
    point, amplitude, vt = min(vertical, key=lambda item: abs(item[0].parameter - _DRO_SEED_X0))
    on_growing_side = point.parameter < _DRO_SEED_X0
    side = "x0 减小侧（振幅增大侧）" if on_growing_side else "x0 增大侧（振幅减小侧）"
    print(
        f"\n# 选定垂直临界点（{side}）：x0={point.parameter:.9f}、振幅={amplitude:.1f} km、"
        f"vt={vt:+.9f}（|vt-1|={abs(vt - 1.0):.3e}）、残差={point.residual:.3e}、"
        f"T={_period(point.orbit):.9f}"
    )
    print("\n# 粘贴进 crates/e2m2e-integrators/src/family_generation/dro3d.rs：")
    print(_rust_block(point.orbit, amplitude))
    return 0


if __name__ == "__main__":
    sys.exit(main())
