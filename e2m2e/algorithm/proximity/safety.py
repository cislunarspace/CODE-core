"""保持点与安全分析（主题 3）。

Bucchioni 2022 保持点 + 被动安全校验 + Chan 最大碰撞概率。

核心概念：
- **保持点（Keeping Point）**：相对轨道上的安全驻留位置，自由漂移
  一圈后仍在安全域内
- **安全域（Safety Region）**：球/椭球/逼近锥，分 keep-out（禁区）
  和 approach（逼近区）两种语义
- **被动安全**：自由漂移轨迹不违背安全域
- **碰撞概率**：Chan 公式，协方差等比例放大使联合 PDF 等值线
  切于组合球时的最大 Pc
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt


@dataclass
class SafetyRegion:
    """安全域（球或逼近锥）。

    Attributes:
        kind: ``"keep_out"``（禁区，不可进入）或 ``"approach"``（逼近区，
            可进入但需监控）
        shape: ``"sphere"``（球）或 ``"cone"``（逼近锥）
        center: 中心位置，形状 ``(3,)``，km（LVLH 系）
        radius: 半径（球）或锥底半径（锥），km
        cone_axis: 锥轴方向（仅 shape="cone" 时用），形状 ``(3,)``
        cone_half_angle: 锥半顶角（rad，仅 shape="cone" 时用）
    """

    kind: Literal["keep_out", "approach"]
    shape: Literal["sphere", "cone"]
    center: np.ndarray
    radius: float
    cone_axis: np.ndarray | None = None
    cone_half_angle: float | None = None

    def contains(self, point: npt.ArrayLike) -> bool:
        """判断点是否在安全域内。"""
        point = np.asarray(point, dtype=float)
        d = point - self.center
        dist = np.linalg.norm(d)
        if self.shape == "sphere":
            return dist <= self.radius
        # cone
        if self.cone_axis is None or self.cone_half_angle is None:
            raise ValueError("cone 安全域需 cone_axis 和 cone_half_angle")
        axis = self.cone_axis / np.linalg.norm(self.cone_axis)
        axial = np.dot(d, axis)
        radial = np.linalg.norm(d - axial * axis)
        # 锥内：径向距离 <= 轴向距离 * tan(半顶角)，且轴向距离在 [0, 深度] 内
        if axial < 0:
            return False
        return radial <= axial * np.tan(self.cone_half_angle)

    def distance_to(self, point: npt.ArrayLike) -> float:
        """点到安全域边界的有向距离（正=外部，负=内部）。"""
        point = np.asarray(point, dtype=float)
        d = point - self.center
        dist = np.linalg.norm(d)
        if self.shape == "sphere":
            return dist - self.radius
        # cone：近似用球距离
        return dist - self.radius


@dataclass
class SafetyReport:
    """被动安全校验报告。

    Attributes:
        safe: 是否全程安全
        violation_intervals: 违背区间列表 ``[(t_start, t_end), ...]``
        min_distance: 最小距离（到安全域边界），km
        collision_probability: 最大碰撞概率（Chan 公式）
    """

    safe: bool
    violation_intervals: list[tuple[float, float]]
    min_distance: float
    collision_probability: float


@dataclass
class KeepingPoint:
    """保持点。

    Attributes:
        position: 相对位置（LVLH 系），形状 ``(3,)``，km
        epoch: 参考历元
        dwell_time: 驻留时间，s
        safety_region: 关联安全域
    """

    position: np.ndarray
    epoch: float
    dwell_time: float
    safety_region: SafetyRegion


def check_passive_safety(
    traj_times: npt.ArrayLike,
    traj_positions: npt.ArrayLike,
    region: SafetyRegion,
    *,
    cov: np.ndarray | None = None,
    r_chaser: float = 5.0,
    r_target: float = 5.0,
) -> SafetyReport:
    """校验自由漂移轨迹的被动安全性。

    逐点检查轨迹是否进入 keep-out 安全域。若提供协方差，同时计算
    Chan 最大碰撞概率。

    Args:
        traj_times: 轨迹时间序列，形状 ``(n,)``
        traj_positions: 轨迹位置序列，形状 ``(n, 3)``，km（LVLH 系）
        region: 安全域
        cov: 相对位置协方差，形状 ``(3, 3)``，km²。None 时不算碰撞概率
        r_chaser: 追逐器半径，km
        r_target: 目标器半径，km

    Returns:
        :class:`SafetyReport`
    """
    traj_times = np.asarray(traj_times, dtype=float)
    traj_positions = np.asarray(traj_positions, dtype=float)

    violation_intervals: list[tuple[float, float]] = []
    min_dist = np.inf
    in_violation = False
    v_start = 0.0

    for i, pos in enumerate(traj_positions):
        dist = region.distance_to(pos)
        min_dist = min(min_dist, dist)
        is_inside = region.contains(pos)

        if region.kind == "keep_out" and is_inside and not in_violation:
            in_violation = True
            v_start = float(traj_times[i])
        elif in_violation and not is_inside:
            in_violation = False
            violation_intervals.append((v_start, float(traj_times[i])))

    if in_violation:
        violation_intervals.append((v_start, float(traj_times[-1])))

    safe = len(violation_intervals) == 0
    pc = 0.0
    if cov is not None and not safe:
        # 用最小距离处的位置估算碰撞概率
        pc = max_collision_probability(min_dist, cov, r_chaser, r_target)

    return SafetyReport(
        safe=safe,
        violation_intervals=violation_intervals,
        min_distance=float(min_dist),
        collision_probability=pc,
    )


def max_collision_probability(
    d: float,
    cov: np.ndarray,
    r_chaser: float,
    r_target: float,
) -> float:
    """Chan 最大碰撞概率公式。

    最大碰撞概率 = 协方差等比例放大使联合 PDF 等值线切于组合球时的 Pc。
    组合球半径 R = r_chaser + r_target，相对距离 d，协方差 Σ。

    公式（Chan 2008, 简化球形近似）：

        Pc_max = (R / (2√π σ))³ exp(−d² / (4σ²))

    其中 σ² 为协方差最大特征值（最不利方向）。

    Args:
        d: 相对距离，km
        cov: 相对位置协方差，形状 ``(3, 3)``，km²
        r_chaser: 追逐器半径，km
        r_target: 目标器半径，km

    Returns:
        最大碰撞概率（0~1）
    """
    cov = np.asarray(cov, dtype=float)
    R = r_chaser + r_target
    # 最大特征值（最不利方向）
    eigvals = np.linalg.eigvalsh(cov)
    sigma2 = float(np.max(eigvals))
    if sigma2 <= 0:
        return 0.0
    sigma = np.sqrt(sigma2)
    # Chan 公式（球形近似）
    pc = (R / (2.0 * np.sqrt(np.pi) * sigma)) ** 3 * np.exp(-d * d / (4.0 * sigma2))
    return float(min(pc, 1.0))
