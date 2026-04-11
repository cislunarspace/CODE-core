"""转移轨道搜索配置模块

将 TransferSearch 的搜索参数提取为独立的 dataclass，
便于复用、序列化和类型检查。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SearchConfig:
    """TransferSearch 网格搜索配置

    所有参数默认为 None，使用前须显式赋值或通过 ``configure_search()`` / ``search()`` 设置。

    Attributes:
        alpha_min: α（切向速度比）下界。推荐值 ∈ (0, 1.0]
        alpha_max: α（切向速度比）上界。推荐值 ∈ [1.0, 3.0]
        n_alpha: α 方向网格点数。推荐值 ∈ [51, 2001]；Cui et al. 2025 Table 3 为 1001
        n_departure: 出发点采样数量。推荐值 ∈ [50, 500]，典型值 200
        max_transfer_time: 最大转移时间（CR3BP 无量纲时间）。推荐值 ∈ [5.0, 30.0]
        intersection_threshold: 相交判定距离阈值（无量纲距离）。推荐值 ∈ [1e-4, 1e-2]
        min_distance_threshold: 候选解最小距离阈值（无量纲距离）。默认 100 km / 地月距离
        collision_earth_radius: 地球碰撞检测半径（无量纲距离）。200 km ≈ 0.0005
        collision_moon_radius: 月球碰撞检测半径（无量纲距离）。100 km ≈ 0.00026
        integration_dt: 积分时间步长（CR3BP 无量纲时间）。推荐值 ∈ [1e-4, 0.1]
        alpha_range: 优化阶段 α 搜索范围。推荐值 (0.5, 2.5)
        transfer_time_range: 优化阶段转移时间范围。推荐值 (1.0, 30.0)
        t_ins_range: 优化阶段插入时间范围。推荐值 (0.0, 10.0)
        velocity_angle_tolerance: 速度平行性容差（弧度）。推荐值 1e-6
    """

    # --- 搜索阶段参数 ---
    alpha_min: Optional[float] = None
    alpha_max: Optional[float] = None
    n_alpha: Optional[int] = None
    n_departure: Optional[int] = None
    max_transfer_time: Optional[float] = None
    intersection_threshold: Optional[float] = None
    min_distance_threshold: Optional[float] = None
    collision_earth_radius: Optional[float] = None
    collision_moon_radius: Optional[float] = None
    integration_dt: Optional[float] = None

    # --- 优化阶段参数 ---
    alpha_range: Optional[Tuple[float, float]] = None
    transfer_time_range: Optional[Tuple[float, float]] = None
    t_ins_range: Optional[Tuple[float, float]] = None
    velocity_angle_tolerance: Optional[float] = None
