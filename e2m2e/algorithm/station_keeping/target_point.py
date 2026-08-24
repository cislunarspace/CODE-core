"""基于目标轨道的目标点控制（《控制方案.md》§1.3-1.4）。

两种模式都以标称轨道（FR1 设计产物）为目标参照，在控制节点施加 Δv 使
受控轨道贴近标称：

- **严格控制** （§1.3，式 5.35）：受控轨道外推至目标节点时刻的位置与
  标称轨道节点严格重合，微分修正式迭代求解。目标节点取控制时刻后
  ``feedback_arc`` 天处的标称轨道节点（1 月内距当前最远的节点的
  DFH 参数化）。
- **宽松控制** （§1.4，式 5.36）：不要求重合，最小化
  ``J = ΔvᵀQΔv + ΣpᵢᵀRᵢpᵢ + ΣvᵢᵀSᵢvᵢ`` （位置/速度偏差加权），
  用标称轨道 STM 线性传递偏差，解析最优（默认单节点反馈，Q=R=I、
  S=1e-2·I）。

STM 在 GCRS 惯性系传播，偏差 ``p₀/v₀`` 也在惯性系计算（与标称轨道同为
GCRS 表达，无需转会合系）。控制量输出 GCRS 速度增量（km/s）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from ...data.constants import SECONDS_PER_DAY
from .special_point import StmPropagator

__all__ = ["NominalOrbitView", "StrictTargetPointLaw", "LooseTargetPointLaw"]


class NominalOrbitView(Protocol):
    """标称轨道视图：任意时刻的标称状态（GCRS，km, km/s）。"""

    def state_at(self, t: float) -> npt.NDArray[np.floating]:
        """t（秒）时刻的标称状态（6 维）。"""
        ...


@dataclass
class StrictTargetPointLaw:
    """目标点严格控制律（§1.3）。

    Attributes:
        feedback_arc_days: 反馈弧段长度（天），即目标节点相对控制时刻的
            时间偏移（DFH 默认 28 天，即1 月内最远节点）
        tolerance_km: 位置重合容差（km）
        max_iter: 微分修正迭代上限
    """

    feedback_arc_days: float = 28.0
    tolerance_km: float = 0.1  # #280: 比测定轨精度（1.5 km 1σ）小一个量级
    max_iter: int = 6  # #280: NRHO 非线性效应需多次迭代（原值 2 不足）

    def compute_maneuver(
        self,
        state0: npt.ArrayLike,
        t0: float,
        *,
        propagator: StmPropagator,
        nominal: NominalOrbitView,
    ) -> npt.NDArray[np.floating]:
        """计算控制量 Δv（km/s）。

        Args:
            state0: 控制时刻状态（GCRS，km, km/s；通常为测量轨道状态）
            t0: 控制时刻（秒）
            propagator: 带 STM 传播器
            nominal: 标称轨道视图

        Returns:
            Δv 矢量（km/s）
        """
        state0 = np.asarray(state0, dtype=float)
        t_j = t0 + self.feedback_arc_days * SECONDS_PER_DAY
        r_target = nominal.state_at(t_j)[:3]

        # 用 STM 一次传播取线性初值 Δv₀ = -B⁻¹·dr_free（自由外推偏差），省去
        # 从 Δv=0 起的多轮冷启动传播（NRHO 上 2-3 次收敛）
        res = propagator.propagate_with_stm(state0, t0, np.array([t_j]))
        stm = np.asarray(res["stm"])[0]
        b = stm[:3, 3:]  # B 块：∂r(t_j)/∂v₀
        dr_free = np.asarray(res["states"])[0, :3] - r_target
        dv_init, *_ = np.linalg.lstsq(b, -dr_free, rcond=None)
        v = state0[3:] + dv_init

        # 微分修正迭代精化（式 5.35，非线性收敛）
        for _ in range(self.max_iter):
            state_v = state0.copy()
            state_v[3:] = v
            res = propagator.propagate_with_stm(state_v, t0, np.array([t_j]))
            r_pred = np.asarray(res["states"])[0, :3]
            dr = r_pred - r_target
            if float(np.linalg.norm(dr)) < self.tolerance_km:
                break
            b = np.asarray(res["stm"])[0, :3, 3:]  # B 块：∂r(t_j)/∂v₀
            # 式 5.35 的微分修正：Δv = -B⁻¹·dr（B 奇异时取最小范数解）
            dv, *_ = np.linalg.lstsq(b, -dr, rcond=None)
            v = v + dv

        return v - state0[3:]


@dataclass
class LooseTargetPointLaw:
    """目标点宽松控制律（§1.4，单节点反馈解析解，式 5.36）。

    偏差取**当前偏差**：变轨前测量状态与目标（标称）轨道在控制时刻的
    状态差 p₀/v₀（式 5.36 原文语义）；Φ(t_k, t₀) 分块 A/B/C/D 由 STM
    传播给出，控制后的目标节点偏差经线性传递 p_k = Φ·(p₀, v₀+Δv)。
    最小化 J = ΔvᵀQΔv + pᵀRp + vᵀSv 得解析最优：

    Δv* = -(Q + BᵀRB + DᵀSD)⁻¹·[(BᵀRB + DᵀSD)·v₀ + (BᵀRA + DᵀSC)·p₀]

    对 NRHO 这类强不稳定轨道，A 块（位置-位置）放大巨大，BᵀRA·p₀ 项
    主导控制量（实测 p₀~90 km 时 Δv 达 m/s 量级，与历史标定样本一致）。

    Attributes:
        feedback_arc_days: 反馈弧段长度（天），目标节点时间偏移
        q: 控制量权重矩阵 Q = q·I
        r: 目标节点位置偏差权重 R = r·I
        s: 目标节点速度偏差权重 S = s·I（默认 1e-2，速度偏差权重取小）
    """

    feedback_arc_days: float = 28.0
    q: float = 1.0
    r: float = 1.0
    s: float = 1e-2

    def compute_maneuver(
        self,
        state0: npt.ArrayLike,
        t0: float,
        *,
        propagator: StmPropagator,
        nominal: NominalOrbitView,
    ) -> npt.NDArray[np.floating]:
        """计算控制量 Δv（km/s）。

        单节点反馈（N=1，式 5.36 的解析解）。p₀/v₀ 为控制时刻测量状态
        相对标称的偏差；A/B/C/D 为 Φ(t_j, t₀) 的四块。
        """
        state0 = np.asarray(state0, dtype=float)
        t_j = t0 + self.feedback_arc_days * SECONDS_PER_DAY

        res = propagator.propagate_with_stm(state0, t0, np.array([t_j]))
        stm = np.asarray(res["stm"])[0]
        a, b, c, d = stm[:3, :3], stm[:3, 3:6], stm[3:6, :3], stm[3:6, 3:6]

        x_nom = nominal.state_at(t0)
        p0 = state0[:3] - x_nom[:3]
        v0 = state0[3:] - x_nom[3:]

        q = self.q * np.eye(3)
        rmat = self.r * np.eye(3)
        smat = self.s * np.eye(3)

        bt_r_b = b.T @ rmat @ b
        dt_s_d = d.T @ smat @ d
        lhs = q + bt_r_b + dt_s_d
        rhs = (bt_r_b + dt_s_d) @ v0 + (b.T @ rmat @ a + d.T @ smat @ c) @ p0
        return -np.linalg.solve(lhs, rhs)
