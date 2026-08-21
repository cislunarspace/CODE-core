"""离散推力工况与连续油门映射（#501，ADR 0032 决策 5）。

低推力配点/打靶求解器（``LowThrustCollocation`` / ``LowThrustShooting``）
的决策变量是连续油门 ``δ ∈ [0,1]`` 加方向角 ``(θ₁, θ₂)``；真实推力器
只有若干固定档位，且每次点火不短于最短弧时长。本模块提供：

- :class:`ThrustArc` / :class:`ThrustArcSequence`：离散工况弧段数据模型
  （自 geo-nrho ``thrust_arcs.py`` 迁入，档位与任务常数参数化）；
- :func:`sequence_from_controls`：连续控制序列 → 离散弧段序列，
  经合并处理最短弧约束（段密于最短弧不报错）；
- :func:`controls_from_sequence`：逆展开回均匀段控制，供重传播验证。

映射只生成离散弧段序列，不声称映射后轨迹仍命中终端约束；验证由调用方
重传播完成（不满足时触发重新传播或以其为初值的离散配点）。

质量流率口径与 Rust 力模型一致：``g0 = 9.81 m/s²``
（见 e2m2e-forces ``augmented_state.rs``）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from e2m2e.algorithm.transfer.lowthrust_shooting import EngineConfig

__all__ = [
    "DEFAULT_THRUST_LEVELS",
    "G0_MPS2",
    "ThrustArc",
    "ThrustArcSequence",
    "angles_from_direction",
    "controls_from_sequence",
    "direction_from_angles",
    "level_from_throttle",
    "sequence_from_controls",
]

#: 默认档位（油门比）：滑行 / 60% / 全推
DEFAULT_THRUST_LEVELS: tuple[float, ...] = (0.0, 0.6, 1.0)

#: 标准重力加速度（m/s²），与 Rust 力模型质量流率口径一致
G0_MPS2 = 9.81

#: 弧段拼接的时间容差（秒）：浮点累积误差的吸收带
_TIME_TOL_S = 1e-9


@dataclass(frozen=True)
class ThrustArc:
    """一条固定工况弧。

    ``direction`` 对滑行弧也保留但不参与动力学；对点火弧（throttle > 0）
    要求是三维惯性系单位向量。``t_start``/``t_end`` 使用同一时间轴。

    Args:
        t_start: 弧起始时刻（s）。
        t_end: 弧结束时刻（s）。
        throttle: 工况油门比，须为档位集合中的一员（由映射保证）。
        direction: 推力方向，惯性系，``(3,)``。
    """

    t_start: float
    t_end: float
    throttle: float
    direction: npt.NDArray[np.floating]

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start

    @property
    def is_burn(self) -> bool:
        return self.throttle > 0.0

    def validate(self, *, min_duration_s: float, levels: tuple[float, ...] | None = None) -> None:
        if not np.isfinite(self.t_start) or not np.isfinite(self.t_end):
            raise ValueError("弧时间必须有限")
        if self.t_end <= self.t_start:
            raise ValueError("弧结束时间必须晚于起始时间")
        if self.duration_s < min_duration_s - _TIME_TOL_S:
            raise ValueError(f"弧长 {self.duration_s:.3f}s 低于最短弧 {min_duration_s:.3f}s")
        if levels is not None and self.throttle not in levels:
            raise ValueError(f"油门 {self.throttle!r} 不在档位集合 {levels!r} 内")
        direction = np.asarray(self.direction, dtype=float)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("弧方向必须为有限三维向量")
        if self.is_burn and not np.isclose(np.linalg.norm(direction), 1.0, atol=1e-9):
            raise ValueError(f"点火弧方向必须为单位向量，当前模长 {np.linalg.norm(direction):.12g}")


@dataclass(frozen=True)
class ThrustArcSequence:
    """连续覆盖一个任务时间窗的离散弧段序列。"""

    arcs: tuple[ThrustArc, ...]

    def validate(self, *, min_duration_s: float, levels: tuple[float, ...] | None = None) -> None:
        if not self.arcs:
            raise ValueError("弧段序列不能为空")
        for index, arc in enumerate(self.arcs):
            arc.validate(min_duration_s=min_duration_s, levels=levels)
            if index and abs(self.arcs[index - 1].t_end - arc.t_start) > _TIME_TOL_S:
                raise ValueError(f"弧 {index - 1} 与弧 {index} 不连续")

    @property
    def t_start(self) -> float:
        return self.arcs[0].t_start

    @property
    def t_end(self) -> float:
        return self.arcs[-1].t_end

    @property
    def total_duration_s(self) -> float:
        return self.t_end - self.t_start

    def fuel_kg(self, engine: EngineConfig) -> float:
        """按推进配置累计燃料消耗（kg）：``ṁ = δ·T_max / (Isp·g0)``。"""
        mass_flow_full = engine.t_max / (engine.isp * G0_MPS2)
        return sum(arc.throttle * arc.duration_s * mass_flow_full for arc in self.arcs)


def level_from_throttle(throttle: float, levels: tuple[float, ...]) -> float:
    """把连续油门映射到最近的离散档位。"""
    if not np.isfinite(throttle):
        raise ValueError("throttle 必须有限")
    _check_levels(levels)
    return min(levels, key=lambda level: abs(level - float(throttle)))


def direction_from_angles(theta1: float, theta2: float) -> npt.NDArray[np.floating]:
    """(θ₁, θ₂) 方向角 → 惯性系单位向量（与低推力求解器控制口径一致）。"""
    return np.array(
        [
            np.cos(theta1) * np.cos(theta2),
            np.sin(theta1) * np.cos(theta2),
            np.sin(theta2),
        ]
    )


def angles_from_direction(direction: npt.ArrayLike) -> tuple[float, float]:
    """惯性系单位向量 → (θ₁, θ₂) 方向角，:func:`direction_from_angles` 的逆。"""
    d = np.asarray(direction, dtype=float).reshape(-1)
    if d.size != 3 or not np.all(np.isfinite(d)):
        raise ValueError("direction 须为有限三维向量")
    norm = float(np.linalg.norm(d))
    if norm < 1e-12:
        raise ValueError("零向量无方向角")
    d = d / norm
    return float(np.arctan2(d[1], d[0])), float(np.arcsin(np.clip(d[2], -1.0, 1.0)))


def sequence_from_controls(
    times: npt.ArrayLike,
    controls: npt.ArrayLike,
    *,
    levels: tuple[float, ...] = DEFAULT_THRUST_LEVELS,
    min_duration_s: float,
) -> ThrustArcSequence:
    """将分段 ``(throttle, θ₁, θ₂)`` 控制序列映射为离散弧段序列。

    流程：逐段取最近档位 → 同档相邻段合并为弧 → 贪心合并短弧直到所有弧
    满足最短弧约束。短弧并入档位更近的邻居（等距取时长更长者，再等距取
    左侧）；合并弧的档位按时长加权平均油门取最近档位，方向按时长加权
    平均方向重新归一化（加权方向近零的对冲情形取较长弧方向）。

    整个时间窗短于最短弧时，单弧序列豁免最短弧约束（无可合并对象）。

    Args:
        times: 段边界时刻（s），形状 ``(N+1,)``，严格递增；第 k 段覆盖
            ``[times[k], times[k+1]]``。均匀分段（打靶/配点求解器的节点
            口径）传 ``np.linspace(t0, tf, N+1)`` 即可，非均匀时间节点
            同样接受。
        controls: 控制数组，形状 ``(N, 3)``，每行 ``(throttle, θ₁, θ₂)``。
        levels: 档位集合（油门比，含 0.0），默认 0/60/100%。
        min_duration_s: 最短弧时长（s）。

    Returns:
        :class:`ThrustArcSequence`，连续覆盖 ``[times[0], times[-1]]``。
    """
    controls_np = np.asarray(controls, dtype=float)
    if controls_np.ndim != 2 or controls_np.shape[1] != 3 or controls_np.shape[0] == 0:
        raise ValueError("controls 形状须为 (N, 3) 且 N > 0")
    if not np.all(np.isfinite(controls_np)):
        raise ValueError("controls 含非有限值")
    times_np = np.asarray(times, dtype=float).reshape(-1)
    if times_np.size != controls_np.shape[0] + 1:
        raise ValueError(
            f"times 长度须为段数+1（{controls_np.shape[0] + 1}），当前 {times_np.size}"
        )
    if not np.all(np.isfinite(times_np)) or not np.all(np.diff(times_np) > 0):
        raise ValueError("times 须有限且严格递增")
    if not np.isfinite(min_duration_s) or min_duration_s <= 0:
        raise ValueError("min_duration_s 必须为正")
    _check_levels(levels)

    snapped = [level_from_throttle(row[0], levels) for row in controls_np]
    directions = [direction_from_angles(row[1], row[2]) for row in controls_np]

    arcs = _runs_to_arcs(times_np, snapped, directions)
    arcs = _merge_short_arcs(arcs, levels, min_duration_s)

    sequence = ThrustArcSequence(tuple(arcs))
    sequence.validate(
        min_duration_s=min(min_duration_s, float(times_np[-1] - times_np[0])), levels=levels
    )
    return sequence


def controls_from_sequence(
    sequence: ThrustArcSequence, n_segments: int
) -> npt.NDArray[np.floating]:
    """把离散弧段序列展开回均匀分段的 ``(throttle, θ₁, θ₂)`` 控制数组。

    每段取其中点所在弧的档位与方向；用于以 ``LowThrustShooting`` 等
    均匀分段传播器重传播验证映射结果。

    Args:
        sequence: 离散弧段序列。
        n_segments: 展开段数 ``N ≥ 1``。

    Returns:
        形状 ``(N, 3)`` 的控制数组。
    """
    if n_segments < 1:
        raise ValueError(f"n_segments 须 ≥ 1，当前 {n_segments}")
    dt = sequence.total_duration_s / n_segments
    out = np.empty((n_segments, 3))
    for k in range(n_segments):
        midpoint = sequence.t_start + (k + 0.5) * dt
        arc = next(
            (a for a in sequence.arcs if a.t_start - _TIME_TOL_S <= midpoint < a.t_end),
            sequence.arcs[-1],
        )
        theta1, theta2 = angles_from_direction(arc.direction)
        out[k] = (arc.throttle, theta1, theta2)
    return out


def _check_levels(levels: tuple[float, ...]) -> None:
    if not levels or any(not np.isfinite(level) or level < 0.0 or level > 1.0 for level in levels):
        raise ValueError("levels 须为 [0,1] 内的有限油门比")
    if len(set(levels)) != len(levels):
        raise ValueError("levels 含重复档位")


def _runs_to_arcs(
    times: npt.NDArray[np.floating],
    snapped: list[float],
    directions: list[npt.NDArray[np.floating]],
) -> list[ThrustArc]:
    """同档相邻段合并为初始弧（尚未处理最短弧约束）。"""
    arcs: list[ThrustArc] = []
    start = 0
    for k in range(1, len(snapped) + 1):
        if k == len(snapped) or snapped[k] != snapped[start]:
            arcs.append(
                _fuse(
                    float(times[start]),
                    float(times[k]),
                    snapped[start],
                    directions[start:k],
                    np.diff(times[start : k + 1]),
                    None,
                )
            )
            start = k
    return arcs


def _fuse(
    t_start: float,
    t_end: float,
    throttle: float,
    directions: list[npt.NDArray[np.floating]],
    weights: npt.ArrayLike,
    fallback: npt.NDArray[np.floating] | None,
) -> ThrustArc:
    """构造弧：方向为各段方向按时长加权平均后归一化；对冲近零时退回 fallback。"""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    mean = np.tensordot(w, np.asarray(directions, dtype=float), axes=(0, 0))
    norm = float(np.linalg.norm(mean))
    if norm < 1e-12:
        if fallback is None or throttle == 0.0:
            direction = np.asarray(directions[0], dtype=float)
        else:
            direction = np.asarray(fallback, dtype=float)
    else:
        direction = mean / norm
    return ThrustArc(float(t_start), float(t_end), float(throttle), direction)


def _merge_short_arcs(
    arcs: list[ThrustArc], levels: tuple[float, ...], min_duration_s: float
) -> list[ThrustArc]:
    """贪心合并短弧：最短弧并入档位更近的邻居，直到全部满足最短弧或只剩一弧。"""
    arcs = list(arcs)
    while len(arcs) > 1:
        short = [i for i, a in enumerate(arcs) if a.duration_s < min_duration_s - _TIME_TOL_S]
        if not short:
            break
        i = min(short, key=lambda idx: arcs[idx].duration_s)
        neighbor = _pick_neighbor(arcs, i)
        merged = _merge_pair(arcs[i], arcs[neighbor], levels)
        arcs = [a for k, a in enumerate(arcs) if k not in (i, neighbor)]
        insert_at = min(i, neighbor)
        arcs.insert(insert_at, merged)
    return arcs


def _pick_neighbor(arcs: list[ThrustArc], i: int) -> int:
    """选合并邻居：档位差更小者；等距取时长更长者；再等距取左侧。"""
    candidates = [j for j in (i - 1, i + 1) if 0 <= j < len(arcs)]
    return min(
        candidates,
        key=lambda j: (abs(arcs[j].throttle - arcs[i].throttle), -arcs[j].duration_s, j),
    )


def _merge_pair(a: ThrustArc, b: ThrustArc, levels: tuple[float, ...]) -> ThrustArc:
    """合并两弧：档位取时长加权平均油门的最近档位，方向按时长加权平均。"""
    left, right = (a, b) if a.t_start <= b.t_start else (b, a)
    total = a.duration_s + b.duration_s
    weighted_throttle = (a.throttle * a.duration_s + b.throttle * b.duration_s) / total
    throttle = level_from_throttle(weighted_throttle, levels)
    longer = a if a.duration_s >= b.duration_s else b
    return _fuse(
        left.t_start,
        right.t_end,
        throttle,
        [a.direction, b.direction],
        [a.duration_s, b.duration_s],
        np.asarray(longer.direction, dtype=float),
    )
