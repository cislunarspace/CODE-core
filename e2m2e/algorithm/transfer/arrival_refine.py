"""LGA/WSB 到达段精化共用打靶段。

perilune → target 的 ThreeBodyLambert 打靶在 LGA 与 WSB 两条间接转移线上
逐段同构；本模块承担公共打靶段，候选 dataclass 重建留在各自模块的薄包装
（lga._refine_lga_candidate / wsb._refine_wsb_candidate），模块级名字不变
（测试 monkeypatch 缝）。LGA 的“精化未改进保留网格候选”守卫在 lga 包装内：
若放进本 helper，守卫拒绝与打靶失败将都坍缩为 None，包装无法区分“返回原
候选”与“MAX_ITERATIONS 重建”，行为会变（#747 要求零变化）。
"""

from __future__ import annotations

import logging

import numpy as np

from ...exceptions import PropagationFailure
from ...status import ConvergenceState
from ..dynamics import CR3BP_Dynamics, CR3BP_System
from .config import TransferArc

__all__ = ["refine_arrival_leg"]

logger = logging.getLogger(__name__)


def refine_arrival_leg(
    system: CR3BP_System,
    dynamics: CR3BP_Dynamics,
    perilune_state: np.ndarray,
    perilune_time_dim: float,
    arrival_time_dim: float,
    target_state: np.ndarray,
) -> tuple[TransferArc, float] | None:
    """打靶精化到达段（perilune → target），返回 (弧, 到达 Δv) 或 None。

    ThreeBodyLambert 在物理单位（km / km/s / s）下解 perilune → target
    的到达段；返回的 ``dv_arr`` 已按特征速度换算回 CR3BP 无量纲。

    Args:
        system: CR3BP 系统（须已设置特征时间与特征速度）。
        dynamics: 打靶传播所用动力学。
        perilune_state: 近月点状态（无量纲）。
        perilune_time_dim: 近月点时刻（无量纲）。
        arrival_time_dim: 到达时刻（无量纲，须晚于近月点）。
        target_state: 目标状态（无量纲）。

    Returns:
        ``(到达段弧, 到达 Δv 无量纲)``；打靶未收敛或失败时返回 None
        （由调用方决定保留原候选还是重建失败状态）。

    Raises:
        ValueError: ``system.characteristic_time`` 未设置（打靶前的
            前置条件，直接上抛）。
    """
    from .terminal import StateTerminal
    from .three_body_lambert import ThreeBodyLambert

    char_time = system.characteristic_time
    if char_time is None:
        raise ValueError("system.characteristic_time must be set")

    try:
        shooter = ThreeBodyLambert(dynamics)

        peri_phys = system.dimensionless_to_physical(perilune_state)
        # 到达段的 tof：近月点 → 目标的剩余时间（非出发→到达的总时间）
        tof_arrival = (arrival_time_dim - perilune_time_dim) * char_time
        if tof_arrival <= 0.0:
            raise ValueError(
                f"到达段剩余时间非正：arrival_time_dim={arrival_time_dim}, "
                f"perilune_time_dim={perilune_time_dim}, tof_arrival={tof_arrival}"
            )

        target_phys = system.dimensionless_to_physical(target_state)

        arrival_leg = shooter.solve(
            StateTerminal(peri_phys, 0.0),
            StateTerminal(target_phys, tof_arrival),
            tof_arrival,
            guess="lambert",
        )

        if arrival_leg.status is ConvergenceState.CONVERGED:
            # ThreeBodyLambert 解为物理单位 (km/s)，换算回无量纲
            v_arrival_shot = arrival_leg.arcs[-1].states[-1][3:]
            vu_km_s = system.characteristic_velocity
            if vu_km_s is None or vu_km_s <= 0.0:
                raise ValueError("system.characteristic_velocity must be set")
            dv_arr = float(np.linalg.norm(v_arrival_shot - target_phys[3:])) / vu_km_s
            return arrival_leg.arcs[0], dv_arr
    except (RuntimeError, ValueError, np.linalg.LinAlgError, PropagationFailure):
        # PropagationFailure：打靶内部传播失败（退化候选几何可触发），
        # 与其他打靶失败同义——保留原始候选，不让编排器崩（#566）。
        logger.debug("ThreeBodyLambert 打靶失败，保留原始候选", exc_info=True)

    return None
