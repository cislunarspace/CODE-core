"""转移轨道脉冲代价计算模块。

提供 ΔV 出发/插入脉冲及总代价的计算。
"""

import numpy as np
from typing import NamedTuple


class TransferCost(NamedTuple):
    """转移代价三元组（无量纲速度单位）。

    Attributes:
        dv1: 出发脉冲大小。
        dv2: 插入脉冲大小。
        total: dv1 + dv2。
    """

    dv1: float
    dv2: float
    total: float


def compute_transfer_cost(
    departure_state: np.ndarray,
    initial_velocity: np.ndarray,
    final_velocity: np.ndarray,
    insertion_velocity: np.ndarray,
) -> TransferCost:
    """计算两脉冲转移代价 Δv₁ + Δv₂。

    Args:
        departure_state: 出发点六维状态 [x,y,z,vx,vy,vz]。
        initial_velocity: 出发注入速度（调整后）[vx,vy,vz]。
        final_velocity: 转移轨迹末端速度 [vx,vy,vz]。
        insertion_velocity: 目标轨道上插入点的速度 [vx,vy,vz]。

    Returns:
        TransferCost 含 dv1、dv2 及 total。
    """
    # departure_state[3:] 即出发点的原始速度；与注入速度之差给出出发脉冲
    dv1 = float(np.linalg.norm(initial_velocity - departure_state[3:]))
    # 转移末端速度与目标轨道插入速度之差给出插入脉冲
    dv2 = float(np.linalg.norm(final_velocity - insertion_velocity))
    return TransferCost(dv1=dv1, dv2=dv2, total=dv1 + dv2)
