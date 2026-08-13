"""流形拼接与低能转移流水线模块。

低能转移初猜生成（郑越、赵敏 2023 流程的产品化）：

1. 出发轨道不稳定流形与目标轨道稳定流形各自传播到同一庞加莱截面；
2. :func:`patch_manifolds` 把两管穿越点两两配对，按加权拼接代价升序给出候选；
3. :func:`design_low_energy_transfer` 取最优候选，以 :class:`ThreeBodyLambert`
   打靶把拼接点之后的弧段闭合到目标轨道（CR3BP 微分修正）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ...data.types.orbit import Orbit
from ..dynamics import CR3BP_Dynamics, CR3BP_System
from ..manifold import InvariantManifold, ManifoldKind, ManifoldTube, PoincareSection
from .config import TransferArc, TransferSolution
from .terminal import OrbitTerminal, StateTerminal
from .three_body_lambert import ThreeBodyLambert

logger = logging.getLogger(__name__)


@dataclass
class PatchCandidate:
    """流形管拼接候选（同一截面上两个穿越点的配对）。

    Attributes:
        i_a: 管 A 穿越点索引
        i_b: 管 B 穿越点索引
        state_a: 管 A 穿越态（无量纲六维向量）
        state_b: 管 B 穿越态（无量纲）
        delta_r: 位置差（无量纲）
        delta_v: 速度差（无量纲）
        cost: 加权拼接代价，w_r·位置差 + w_v·速度差
    """

    i_a: int
    i_b: int
    state_a: np.ndarray
    state_b: np.ndarray
    delta_r: float
    delta_v: float
    cost: float


def patch_manifolds(
    m_a: ManifoldTube,
    m_b: ManifoldTube,
    section: PoincareSection,
    weights: tuple[float, float] = (1.0, 1.0),
) -> list[PatchCandidate]:
    """两流形管在同一截面的穿越点两两配对，按拼接代价升序输出。

    Args:
        m_a: 管 A（如出发轨道不稳定流形）
        m_b: 管 B（如目标轨道稳定流形）
        section: 庞加莱截面
        weights: ``(w_r, w_v)`` 位置/速度差权重（无量纲量纲下的相对权重）

    Returns:
        按 ``cost`` 升序排列的候选列表；任一管无穿越时返回空列表
    """
    w_r, w_v = float(weights[0]), float(weights[1])
    crossings_a = section.crossings(m_a)
    crossings_b = section.crossings(m_b)

    candidates: list[PatchCandidate] = []
    for i, state_a in enumerate(crossings_a.states):
        for j, state_b in enumerate(crossings_b.states):
            delta_r = float(np.linalg.norm(state_a[:3] - state_b[:3]))
            delta_v = float(np.linalg.norm(state_a[3:] - state_b[3:]))
            candidates.append(
                PatchCandidate(
                    i_a=i,
                    i_b=j,
                    state_a=np.array(state_a, copy=True),
                    state_b=np.array(state_b, copy=True),
                    delta_r=delta_r,
                    delta_v=delta_v,
                    cost=w_r * delta_r + w_v * delta_v,
                )
            )

    candidates.sort(key=lambda c: c.cost)
    return candidates


# 种子扰动幅度对应的物理长度 (km)，除以特征长度得无量纲 ε
_SEED_OFFSET_KM = 50.0


def design_low_energy_transfer(
    departure: OrbitTerminal,
    target: Orbit,
    epoch=None,
    model: Literal["cr3bp"] = "cr3bp",
) -> TransferSolution:
    """低能转移流水线：流形拼接初猜 + CR3BP 打靶闭合。

    流程：出发轨道不稳定流形 + 目标轨道稳定流形（± 分支都试）→
    次天体近拱点截面拼接取最优候选 → 出发弧（流形弧，精确）+
    :class:`ThreeBodyLambert` 把拼接点之后闭合到目标轨道。

    脉冲构成：出发脉冲（上出发流形）+ 拼接脉冲（截面处）+ 到达脉冲（入目标流形）。

    Args:
        departure: 出发轨道终端
        target: 目标周期轨道（须关联 system 且 period 已知）
        epoch: 参考历元（预留；星历转换未接入，见下）
        model: 动力学模型，当前仅支持 ``"cr3bp"``

    Returns:
        :class:`TransferSolution`，两段弧（物理单位）

    Raises:
        ValueError: model 不支持，或两流形管在截面上无穿越点

    TODO: 星历转换（CR3BP 闭合解 → 星历模型）未接入；接入时复用设计链路
        ``e2m2e.algorithm.design`` 的 Rust 多重打靶修正，epoch 参数即为其入口。
    """
    if model != "cr3bp":
        raise ValueError(f"当前仅支持 model='cr3bp'，得到 {model!r}")
    if epoch is not None:
        logger.info("epoch 参数暂未使用：星历转换未接入（见模块 TODO）")

    system = target.system
    if not isinstance(system, CR3BP_System) or not system.is_initialized:
        raise ValueError("target 须关联已初始化特征尺度的 CR3BP system")
    if departure.orbit.period is None or target.period is None:
        raise ValueError("出发与目标轨道的 period 须已知")
    # is_initialized 保证三个特征尺度均已设置
    assert system.characteristic_length is not None
    assert system.characteristic_time is not None
    assert system.characteristic_velocity is not None
    dynamics = CR3BP_Dynamics(system)
    section = PoincareSection.periapsis(system.secondary_body, system)
    epsilon = _SEED_OFFSET_KM / system.characteristic_length
    t_span = 2.0 * max(float(departure.orbit.period), float(target.period))

    # ± 分支四种组合全局取最优
    best: PatchCandidate | None = None
    best_tubes: tuple[ManifoldTube, ManifoldTube] | None = None
    for branch_a in ("+", "-"):
        tube_a = InvariantManifold(
            departure.orbit, ManifoldKind.UNSTABLE, branch_a, epsilon
        ).propagate(t_span, section=section)
        for branch_b in ("+", "-"):
            tube_b = InvariantManifold(target, ManifoldKind.STABLE, branch_b, epsilon).propagate(
                t_span, section=section
            )
            candidates = patch_manifolds(tube_a, tube_b, section)
            if candidates and (best is None or candidates[0].cost < best.cost):
                best = candidates[0]
                best_tubes = (tube_a, tube_b)

    if best is None or best_tubes is None:
        raise ValueError("两流形管在近拱点截面上均无穿越点，无法拼接")
    tube_a, tube_b = best_tubes

    # 出发弧：不稳定流形弧（种子 → 截面，正向积分，时间递增）
    idx_a = section.crossings(tube_a).trajectory_index[best.i_a]
    arc_a = tube_a.trajectories[idx_a]

    # 目标弧：稳定流形弧（种子近目标轨道 → 截面，反向积分），
    # 闭合终点取该弧种子，飞行时间取弧长
    idx_b = section.crossings(tube_b).trajectory_index[best.i_b]
    arc_b = tube_b.trajectories[idx_b]
    tof_b_s = float(abs(arc_b.times[-1] - arc_b.times[0])) * system.characteristic_time

    # 打靶闭合：从拼接点位置射向目标流形种子。初猜取目标侧速度
    # （该速度沿稳定流形弧本就到达种子，Newton 只需桥接 Δr/Δv 缝隙），
    # 即 term0 = (state_a 位置, state_b 速度)，guess="orbit"。
    guess_state = np.concatenate([best.state_a[:3], best.state_b[3:]])
    seed_b_phys = system.dimensionless_to_physical(arc_b.states[0])
    shooter = ThreeBodyLambert(dynamics)
    leg2 = shooter.solve(
        StateTerminal(system.dimensionless_to_physical(guess_state), 0.0),
        StateTerminal(seed_b_phys, tof_b_s),
        tof_b_s,
        guess="orbit",
    )

    # 出发脉冲：上不稳定流形种子的脉冲（种子 = 相位态 + ε·v̂，
    # 相位时间 = 轨道首点时刻 + 索引/弧数 × 周期）
    phase_time = float(departure.orbit.times[0]) + idx_a / len(tube_a.trajectories) * float(
        departure.orbit.period
    )
    orbit_state = dynamics.propagate_orbit_state_at_time(departure.orbit, phase_time)
    delta_v1 = float(
        np.linalg.norm(arc_a.states[0][3:] - orbit_state[3:]) * system.characteristic_velocity
    )

    states_a = np.array([system.dimensionless_to_physical(s) for s in arc_a.states])
    times_a = np.asarray(arc_a.times, dtype=float) * system.characteristic_time
    times_a = times_a - times_a[0]

    leg2_arc = leg2.arcs[0]
    states_b = np.vstack([leg2_arc.states[1:], [seed_b_phys]])
    times_b = leg2_arc.times[1:] + times_a[-1]
    times_b = np.append(times_b, times_a[-1] + tof_b_s)

    # 拼接脉冲 = 闭合后的出发速度 − 出发侧穿越速度（物理单位）
    state_a_phys = system.dimensionless_to_physical(best.state_a)
    delta_v_patch = float(np.linalg.norm(leg2_arc.states[0][3:] - state_a_phys[3:]))
    total = delta_v1 + delta_v_patch + leg2.arrival_delta_v
    message = leg2.message or (
        f"拼接代价 {best.cost:.4e}（无量纲）：|Δr|={best.delta_r:.4e}, |Δv|={best.delta_v:.4e}"
    )
    return TransferSolution(
        arcs=(
            TransferArc(states=states_a, times=times_a, delta_v=delta_v1),
            TransferArc(states=states_b, times=times_b, delta_v=delta_v_patch),
        ),
        arrival_delta_v=leg2.arrival_delta_v,
        total_delta_v=total,
        transfer_time=float(times_a[-1] + tof_b_s),
        status=leg2.status,
        cause=leg2.cause,
        n_iter=leg2.n_iter,
        message=message,
    )
