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

Floquet 投影变体（#843，Gómez vol I §8.2.3 方法学）例外：它在 CR3BP
会合系无量纲口径下工作，偏差分解到 Floquet 模态坐标，最小范数 Δv 清零
漂移分量。模态量来源（注入表或在线数值投影）显式注入，既有严格、宽松
两律零改动。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt

from ...data.constants import SECONDS_PER_DAY
from ...data.types.trajectory import NominalOrbit
from ..proximity import (
    FloquetModeError,
    FloquetModeSet,
    invert_modal_basis,
    modal_basis_at,
)
from .special_point import StmPropagator

if TYPE_CHECKING:
    from ..dynamics import CR3BP_Dynamics

__all__ = [
    "SynodicNominalView",
    "NominalOrbitView",
    "StrictTargetPointLaw",
    "LooseTargetPointLaw",
    "FloquetModalSource",
    "FloquetTargetPointLaw",
    "InjectedModalSource",
    "OnlineModalSource",
]


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
    tolerance_km: float = 0.1  # 比测定轨精度（1.5 km 1σ）小一个量级
    max_iter: int = 6  # NRHO 非线性效应需多次迭代

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
    主导控制量（实测 p₀~90 km 时 Δv 达 m/s 量级）。

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


class SynodicNominalView(Protocol):
    """会合系标称轨道视图：任意时刻的标称状态（CR3BP 会合系无量纲，TU）。

    与 ``NominalOrbitView`` 结构相同（仅 ``state_at``），口径只体现在
    题名与文档：结构化类型检查分辨不了数值口径，两协议互相可赋值，
    GCRS km 视图传入本律不会报错、只会得到无意义的 Δv。题名单列是
    为了在签名与文档层面把口径说清，混用由调用方负责。
    """

    def state_at(self, t: float) -> npt.NDArray[np.floating]:
        """t（TU）时刻的标称状态（6 维，会合系无量纲）。"""
        ...


class FloquetModalSource(Protocol):
    """模态量来源：任意时刻的模态对偶基（6×6，行序同模态基列序）。

    两种实现：``InjectedModalSource`` 查注入 ``NominalOrbit`` 的对偶基
    表（生产路径），``OnlineModalSource`` 按需传播重算（等价验证）。
    两者在任意时刻（含跨整圈）都给出 B(t)⁻¹，跨圈语义一致。
    """

    def dual_basis_at(self, t: float) -> npt.NDArray[np.floating]:
        """t 时刻的模态对偶基 B(t)⁻¹（CR3BP 会合系无量纲）。"""
        ...


@dataclass
class InjectedModalSource:
    """注入表来源：``NominalOrbit`` 模态表的模 T 查表。

    表与历元跨一个周期（``build_nominal_orbit`` 的产物形状），查表按
    相位回卷到圈内，相位以首历元为原点，表历元整体平移整数周期不改
    变查表结果。对偶基第 5 行（漂移对偶行）T 周期，回卷逐历元成立；
    切向行（第 0 行）跨圈按整圈数乘带符号每圈增长系数修正（系数取漂
    移列一周期增量在单位切向上的投影，见 ``dual_basis`` 的周期语义），
    跨圈查表与在线重算一致。

    Attributes:
        orbit: 已注入 ``floquet_basis`` 与 ``projection_factors`` 的标称
            轨道容器（后者载对偶基表）。

    Raises:
        FloquetModeError: 字段未注入、形状不符、历元少于两个、含非有
            限值或非严格递增。
    """

    orbit: NominalOrbit

    def __post_init__(self) -> None:
        epochs = np.asarray(self.orbit.epochs, dtype=float)
        dual = self.orbit.projection_factors
        if dual is None:
            raise FloquetModeError("projection_factors 未注入，无法按注入表取模态量")
        dual = np.asarray(dual, dtype=float)
        basis = self.orbit.floquet_basis
        if basis is None:
            raise FloquetModeError("floquet_basis 未注入，跨圈切向行修正需要模态基表")
        basis = np.asarray(basis, dtype=float)
        if dual.ndim != 3 or dual.shape[1:] != (6, 6):
            raise FloquetModeError("projection_factors 须为逐历元 (n, 6, 6) 对偶基表")
        if basis.shape != dual.shape:
            raise FloquetModeError("floquet_basis 须与 projection_factors 逐历元同形")
        if epochs.ndim != 1 or epochs.shape[0] != dual.shape[0]:
            raise FloquetModeError("epochs 须为一维且与对偶基表逐历元对齐（跨一个周期）")
        if epochs.shape[0] < 2:
            raise FloquetModeError("epochs 至少两个历元（单点表无回卷跨度）")
        if not (
            np.all(np.isfinite(epochs)) and np.all(np.isfinite(dual)) and np.all(np.isfinite(basis))
        ):
            raise FloquetModeError("epochs 与两张模态表须全部为有限值")
        if np.any(np.diff(epochs) <= 0.0):
            raise FloquetModeError("epochs 须严格递增")
        self._origin = float(epochs[0])
        self._phases = epochs - epochs[0]
        self._dual = dual
        # 漂移列一周期沿单位切向的带符号增量（漂移方向归一化的取向
        # 决定符号），即切向行的每圈修正系数。
        growth = basis[-1][:, 5] - basis[0][:, 5]
        self._lap_shift = float(growth @ basis[0][:, 0])

    def dual_basis_at(self, t: float) -> npt.NDArray[np.floating]:
        """相位回卷插值查表，切向行按整圈数修正。"""
        span = float(self._phases[-1])
        elapsed = float(t) - self._origin
        phase = elapsed % span
        laps = math.floor(elapsed / span)
        dual_at = np.array(
            [
                [np.interp(phase, self._phases, self._dual[:, i, j]) for j in range(6)]
                for i in range(6)
            ]
        )
        if laps:
            dual_at[0] = dual_at[0] - laps * self._lap_shift * dual_at[5]
        return dual_at


@dataclass
class OnlineModalSource:
    """在线来源：按需传播 STM 重构模态基并求逆。

    逐次查询一次整段传播，生产控制序列应走注入表。本路径用于注入表的
    等价验证与稀疏查询，时刻须不早于模态集相位原点。

    Attributes:
        modes: Floquet 模态基础解集。
        dynamics: 构造该模态集的 CR3BP 动力学。
    """

    modes: FloquetModeSet
    dynamics: CR3BP_Dynamics

    def dual_basis_at(self, t: float) -> npt.NDArray[np.floating]:
        return invert_modal_basis(modal_basis_at(self.modes, float(t), self.dynamics))


@dataclass
class FloquetTargetPointLaw:
    """Floquet 投影目标点律（Gómez vol I §8.2.3 方法学，漂移分量清零）。

    把控制时刻偏差 δx 分解到 Floquet 模态坐标，取唯一无界的漂移分量
    α_drift = n(t)·δx（n 为漂移对偶行，即有界子空间法向量的逐历元取
    值），施加最小范数速度增量使机动后偏差的漂移坐标归零（投回有界
    子空间）：

        Δv = −α_drift·n_v/‖n_v‖²，n_v = n[3:]

    面内与法向中心分量有界振荡、切向分量是相位偏移，均不修正。模态量
    经 ``modal`` 来源显式注入（注入表或在线数值投影），不靠属性嗅探。
    ``nominal`` 与 ``modal`` 须同为 CR3BP 会合系无量纲、时间同基准
    （TU），标称视图用 :class:`SynodicNominalView` 口径。Gómez 方法学
    的开/关判据与死区属调用方在此解析解之上组合。
    """

    def compute_maneuver(
        self,
        state0: npt.ArrayLike,
        t0: float,
        *,
        nominal: SynodicNominalView,
        modal: FloquetModalSource,
    ) -> npt.NDArray[np.floating]:
        """计算控制量 Δv（与输入状态同单位制）。

        Args:
            state0: 控制时刻测量状态（与标称视图同口径）。
            t0: 控制时刻（与标称视图、模态量来源同时间基准）。
            nominal: 标称轨道视图（任意时刻标称状态，会合系无量纲口径）。
            modal: 模态量来源（任意时刻模态对偶基）。

        Returns:
            3 维速度增量，使机动后偏差的漂移模态坐标归零。

        Raises:
            FloquetModeError: 漂移对偶行的速度分块相对全行退化（相对范数
                低于 1e-8）且漂移坐标非零，速度增量无法改变漂移坐标。
        """
        measured = np.asarray(state0, dtype=float)
        deviation = measured - np.asarray(nominal.state_at(t0), dtype=float)
        normal = np.asarray(modal.dual_basis_at(t0), dtype=float)[5]
        alpha_drift = float(normal @ deviation)
        n_v = normal[3:]
        denom = float(n_v @ n_v)
        if denom <= 1e-16 * float(normal @ normal) or denom == 0.0:
            if alpha_drift == 0.0:
                return np.zeros(3)
            raise FloquetModeError("漂移对偶行的速度分块相对全行退化，无最小范数解")
        return -(alpha_drift / denom) * n_v
