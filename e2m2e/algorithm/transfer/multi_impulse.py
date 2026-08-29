"""多脉冲转移框架与 Lawden 主矢量检验。

多脉冲 NLP 采用节点参数化：两端为固定 :class:`StateTerminal` （位置、速度、时刻
均固定），中途脉冲节点以 ``(t_i, r_i)`` 为决策变量，相邻节点间的弧段由 Lambert
封闭——二体 :func:`solve_lambert` （快，默认，用于初扫）或
:class:`ThreeBodyLambert` 打靶（准，用于精修）。各节点的脉冲是封闭结果的输出
（进出弧速度差），不是独立变量。这是 Prussing《Optimal Spacecraft Trajectories》
第 5 章与宝音贺西等 (2025) 内层封闭/外层 NLP 结构的标准形式。

主矢量检验的公式出处：

- Prussing《Optimal Spacecraft Trajectories》(SIAM, 2019) 第 3 章：Lawden 主矢量
  p(t) 的最优性必要条件——全程 ``|p(t)| ≤ 1``；脉冲点 ``|p| = 1`` 且脉冲方向与 p 共线。
- 同书第 4 章：给定脉冲转移的主矢量由端点横截条件 p(t0) = Δv̂₀、p(tf) = Δv̂_f
  确定。协态 z = [−ṗ; p] 沿轨迹按 z(t) = Φ(t, t0)^{-T} z(t0) 携载（Φ 为状态转移
  矩阵，固定脉冲不改变变分映射，故 Φ 在脉冲点连续、可跨弧链接）；先由 p(tf) 条件
  解出 ṗ(t0)（3×3 线性系统），再逐采样点携载得 p(t) 曲线。
- Lion & Handelsman (1968)：``|p|`` 在弧内超过 1 是转移非最优的判据，在峰值处插入
  中途脉冲可降低总 ΔV；Grossi et al. (2024) 用它驱动地月三脉冲转移设计。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp
from scipy.optimize import Bounds, OptimizeResult, minimize

from ...data.templates import ConvergenceState, FailureCause
from ...exceptions import PropagationFailure
from ..results import scipy_slsqp_status
from .config import TransferArc, TransferSolution
from .lambert import solve_lambert
from .terminal import StateTerminal
from .three_body_lambert import ThreeBodyLambert

if TYPE_CHECKING:
    from ..dynamics import CR3BP_Dynamics

logger = logging.getLogger(__name__)

Closure = Literal["two_body", "three_body"]

# 目标函数在弧段封闭失败（Lambert 不收敛、打靶发散、几何退化）时返回的罚值。
# 正常 ΔV 量级为 km/s，1e3 足以把这类点逐出搜索域。
_CLOSURE_PENALTY = 1e3

# SLSQP 单次优化相对初猜的改善低于该值（km/s）视为"卡住"：目标函数在
# 零脉冲初猜（双脉冲弧上的点，目标值恰为双脉冲成本）附近梯度极小，SLSQP
# 收敛判据可能提前触发、一步即停。触发后从微扰初猜重试；正常收敛
# 的改善为 0.1 量级。
_STALLED_IMPROVEMENT = 1e-6

# 二体传播（含 STM）的积分容差；主矢量对 STM 精度敏感，取紧容差
_TB_RTOL = 1e-11
_TB_ATOL = 1e-12

# 零脉冲判据（km/s）：低于该值的方向无定义，主矢量端点条件无法构造
_ZERO_IMPULSE_EPS = 1e-9


@dataclass(frozen=True)
class Impulse:
    """脉冲节点：给定时刻的速度增量。

    Attributes:
        time: 脉冲时刻，s（与终端时刻同坐标）
        delta_v: 速度增量，形状 ``(3,)``，km/s
    """

    time: float
    delta_v: np.ndarray


@dataclass(frozen=True)
class CoastArc:
    """无动力弧段规划节点：自 terminal0 起飞行 tof。

    Attributes:
        terminal0: 弧段起点终端（状态为脉冲后的出发状态）
        tof: 弧段飞行时间，s
        dynamics: 弧段封闭动力学标识，``"two_body"`` 或 ``"three_body"``
    """

    terminal0: StateTerminal
    tof: float
    dynamics: str = "two_body"


@dataclass(frozen=True)
class PrimerVectorReport:
    """主矢量检验报告。

    Attributes:
        times: 采样时刻，形状 ``(n,)``，s（与终端时刻同坐标）
        primer: 主矢量 p(t)，形状 ``(n, 3)``，无量纲
        primer_magnitude: ``|p(t)|``，形状 ``(n,)``
        max_magnitude: 全程最大 ``|p|``
        lawden_satisfied: Lawden 必要条件是否满足（弧内 ``|p|≤1`` 且中途脉冲
            与 p 共线、``|p|=1``，均在容差内）
        impulse_times: 各脉冲时刻（含出发与到达脉冲），形状 ``(k,)``，s
        impulse_magnitudes: 脉冲点 ``|p|``，形状 ``(k,)``
        impulse_alignment_cosines: 脉冲方向与 p 的夹角余弦，形状 ``(k,)``；
            零脉冲方向无定义，记为 NaN
        suggested_insertion_time: 建议的中途脉冲插入时刻（弧内 ``|p|>1`` 的峰值处），
            无违例时为 None
        suggested_insertion_position: 建议插入点位置，形状 ``(3,)``，km
        message: 附加说明（端点携载误差、违例细节等）
    """

    times: np.ndarray
    primer: np.ndarray
    primer_magnitude: np.ndarray
    max_magnitude: float
    lawden_satisfied: bool
    impulse_times: np.ndarray
    impulse_magnitudes: np.ndarray
    impulse_alignment_cosines: np.ndarray
    suggested_insertion_time: float | None
    suggested_insertion_position: np.ndarray | None
    message: str = ""


def _two_body_rhs_stm(t: float, y: np.ndarray, mu: float) -> np.ndarray:
    """二体运动方程 + 变分方程（42 维增广状态）。

    重力梯度 G = μ(3rrᵀ/r⁵ − I/r³)，雅可比 A = [[0, I], [G, 0]]。
    """
    del t  # 自治系统
    r = y[:3]
    rn = float(np.linalg.norm(r))
    accel = -mu * r / rn**3
    grav_grad = mu * (3.0 * np.outer(r, r) / rn**5 - np.eye(3) / rn**3)
    jac = np.zeros((6, 6))
    jac[:3, 3:] = np.eye(3)
    jac[3:, :3] = grav_grad
    dphi = jac @ y[6:].reshape(6, 6)
    return np.concatenate([y[3:6], accel, dphi.ravel()])


def _propagate_two_body(
    state0: npt.ArrayLike,
    t_eval: npt.ArrayLike,
    mu: float,
    *,
    with_stm: bool,
) -> dict[str, np.ndarray]:
    """二体传播器（模块内部用）：主矢量检验与弧段采样的底座。

    项目 Python 层无二体传播器（CR3BP_Dynamics 只含三体模型），此处用
    scipy solve_ivp 实现最小版本；Lambert 封闭本身不需要它。
    """
    state0_arr = np.asarray(state0, dtype=float)
    times = np.asarray(t_eval, dtype=float)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        del t
        return np.concatenate([y[3:6], -mu * y[:3] / np.linalg.norm(y[:3]) ** 3])

    if with_stm:
        y0 = np.concatenate([state0_arr, np.eye(6).ravel()])
        rhs = partial(_two_body_rhs_stm, mu=mu)
    else:
        y0 = state0_arr

    sol = solve_ivp(
        rhs,
        (times[0], times[-1]),
        y0,
        method="DOP853",
        t_eval=times,
        rtol=_TB_RTOL,
        atol=_TB_ATOL,
    )
    out = {"time": sol.t, "states": sol.y[:6].T}
    if with_stm:
        out["stm"] = sol.y[6:].T.reshape(-1, 6, 6)
    return out


def propagate_two_body(
    state0: npt.ArrayLike,
    t_eval: npt.ArrayLike,
    mu: float,
) -> dict[str, np.ndarray]:
    """二体数值传播（公开封装，HMN 转移弧采样用）。

    Args:
        state0: 出发状态 (6,)，km / km/s，与中心天体固连的惯性系。
        t_eval: 采样时刻（秒，单调）；首末时刻定义积分区间。
        mu: 中心天体 GM，km³/s²。

    Returns:
        ``{"time": (n,), "states": (n, 6)}``（DOP853，同内部精度常量）。
    """
    return _propagate_two_body(state0, t_eval, mu, with_stm=False)


class MultiImpulseTransfer:
    """固定端点间的多脉冲转移规划器。

    Attributes:
        term0: 出发终端（位置、速度、时刻固定）
        term1: 到达终端（位置、速度、时刻固定）
        mu: 二体中心天体 GM，km³/s²（二体封闭必填）
        dynamics: CR3BP_Dynamics（三体封闭必填）
        closure: 默认弧段封闭方式
        legs: 当前方案的脉冲/弧段序列；初始为单弧，:meth:`optimize` 后刷新
    """

    def __init__(
        self,
        term0: StateTerminal,
        term1: StateTerminal,
        *,
        mu: float | None = None,
        dynamics: CR3BP_Dynamics | None = None,
    ) -> None:
        if mu is None and dynamics is None:
            raise ValueError("须指定二体 mu 或三体 dynamics 至少其一")
        if term1.time <= term0.time:
            raise ValueError(f"到达时刻须晚于出发时刻，得到 {term0.time} → {term1.time}")
        self.term0 = term0
        self.term1 = term1
        self.mu = mu
        self.dynamics = dynamics
        self.closure: Closure = "three_body" if mu is None else "two_body"
        self.legs: list[Impulse | CoastArc] = [
            CoastArc(term0, term1.time - term0.time, self.closure)
        ]
        self._shooter = ThreeBodyLambert(dynamics) if dynamics is not None else None

    # ---- 多脉冲优化 ----

    def optimize(
        self,
        n_impulses: int,
        bounds: dict[str, tuple] | None = None,
        *,
        backend: str = "scipy",
        closure: Closure | None = None,
        x0: npt.ArrayLike | None = None,
        min_dt: float = 1e-3,
        verbose: bool = False,
    ) -> TransferSolution:
        """优化 n_impulses 脉冲转移，最小化总 ΔV。

        决策变量为各中途脉冲节点的时刻与位置 ``[t_1..t_m, r_1..r_m]``
        （m = n_impulses − 2；时刻为相对 term0.time 的秒数，位置为 km），
        弧段按 closure 封闭，脉冲为封闭速度差。n_impulses=2 时无自由变量，
        直接封闭单弧返回。

        Args:
            n_impulses: 脉冲总数（含出发与到达脉冲），≥ 2
            bounds: 中途节点界限 ``{"t": (lo, hi), "r": (lo3, hi3)}``；
                ``"t"`` 为相对 term0.time 的秒数区间，``"r"`` 为位置分量盒
                （km），对所有中途节点统一适用；None 时用默认界限
            backend: NLP 后端，当前仅支持 ``"scipy"`` （SLSQP）
            closure: 弧段封闭方式；None 时取实例默认
            x0: 初猜决策向量；None 时中途节点在时间上均布、位置线性插值
            min_dt: 相邻脉冲最小间隔，s
            verbose: 是否打印 SLSQP 迭代信息

        Returns:
            :class:`TransferSolution`，弧段数 = n_impulses − 1

        Note:
            SLSQP 对零脉冲初猜（目标值恰为双脉冲成本的弧上点）数值敏感：目标函数
            在该平坦走廊上梯度极小，收敛判据可能提前触发、一步即停。首次
            优化相对初猜改善不足时，自动从微扰初猜重试，取总 ΔV 最小者。
        """
        if backend != "scipy":
            raise ValueError(f"当前仅支持 backend='scipy'，得到 {backend!r}")
        if n_impulses < 2:
            raise ValueError(f"n_impulses 须 ≥ 2，得到 {n_impulses}")
        closure = self._resolve_closure(closure)
        n_mid = n_impulses - 2
        if n_mid == 0:
            return self._build_solution(
                np.empty(0),
                closure,
                status=ConvergenceState.CONVERGED,
                cause=FailureCause.NONE,
                message="双脉冲封闭完成",
            )

        lb, ub = self._resolve_bounds(n_mid, bounds, min_dt)
        y0 = np.asarray(x0, dtype=float) if x0 is not None else self._default_x0(n_mid, closure)
        if y0.shape != (4 * n_mid,):
            raise ValueError(f"x0 形状须为 ({4 * n_mid},)，得到 {y0.shape}")

        # 中途脉冲时刻递增约束：t_{i+1} − t_i ≥ min_dt
        constraints = [
            {"type": "ineq", "fun": lambda y, i=i: y[i + 1] - y[i] - min_dt}
            for i in range(n_mid - 1)
        ]

        def _solve(
            start: np.ndarray,
        ) -> tuple[OptimizeResult, ConvergenceState, FailureCause]:
            result = minimize(
                lambda y: self._total_dv(y, closure),
                start,
                method="SLSQP",
                bounds=Bounds(lb, ub),
                constraints=constraints,
                options={"ftol": 1e-12, "maxiter": 500, "disp": verbose},
            )
            status, cause = scipy_slsqp_status(bool(result.success), int(result.status))
            return result, status, cause

        # SLSQP 对零脉冲初猜数值敏感：目标函数在双脉冲弧附近梯度极小，
        # 收敛判据可能提前触发、一步即停。首次优化改善不足时从微扰初猜重试，
        # 取总 ΔV 最小者。
        candidates = [_solve(y0)]
        if self._total_dv(y0, closure) - candidates[0][0].fun < _STALLED_IMPROVEMENT:
            logger.info("多脉冲优化相对初猜无改善（零脉冲平坦区），微扰初猜重试")
            for start in self._stalled_retries(y0, n_mid):
                candidates.append(_solve(start))
        result, status, cause = min(candidates, key=lambda c: c[0].fun)
        return self._build_solution(
            result.x,
            closure,
            status=status,
            cause=cause,
            n_iter=int(result.nit),
            message=str(result.message),
        )

    # ---- 主矢量检验 ----

    def check_primer_vector(
        self,
        solution: TransferSolution,
        n_samples: int = 100,
        tol: float = 1e-3,
        closure: Closure | None = None,
    ) -> PrimerVectorReport:
        """对给定转移解做主矢量检验（Lawden 必要条件）。

        端点横截条件 p(t0) = Δv̂₀、p(tf) = Δv̂_f，协态 z = [−ṗ; p] 经
        z(t) = Φ(t, t0)^{-T} z(t0) 携载（公式出处见模块 docstring）。

        Args:
            solution: 待检验的转移解（弧段初态须为脉冲后状态）
            n_samples: 每弧采样点数
            tol: Lawden 判据容差（``|p|≤1+tol``、脉冲点 ``|p|∈1±tol``、
                共线 ``|cos|≥1−tol``）
            closure: 弧段重传播所用动力学；None 时取实例默认，须与产生
                solution 时的封闭方式一致

        Returns:
            :class:`PrimerVectorReport`

        Raises:
            ValueError: 出发或到达脉冲过小，无法确定主矢量端点方向
        """
        closure = self._resolve_closure(closure)
        arcs = solution.arcs

        # 1. 各脉冲矢量与时刻（出发脉冲 + 弧间脉冲 + 到达脉冲）
        impulse_dvs = [arcs[0].states[0][3:] - self.term0.state[3:]]
        impulse_times = [float(arcs[0].times[0])]
        for k in range(1, len(arcs)):
            prev, nxt = arcs[k - 1], arcs[k]
            impulse_dvs.append(nxt.states[0][3:] - prev.states[-1][3:])
            impulse_times.append(float(nxt.times[0]))
        impulse_dvs.append(self.term1.state[3:] - arcs[-1].states[-1][3:])
        impulse_times.append(float(arcs[-1].times[-1]))
        dvs = np.asarray(impulse_dvs, dtype=float)
        dv_mags = np.linalg.norm(dvs, axis=1)
        if dv_mags[0] < _ZERO_IMPULSE_EPS or dv_mags[-1] < _ZERO_IMPULSE_EPS:
            raise ValueError("出发/到达脉冲过小，无法确定主矢量端点方向")
        p0 = dvs[0] / dv_mags[0]
        pf = dvs[-1] / dv_mags[-1]

        # 2. 逐弧传播（含 STM），链接全程 Φ(t, t0)；固定脉冲的变分映射为
        #    单位阵，故 Φ 在脉冲点连续，可直接相乘
        all_times: list[float] = []
        all_states: list[np.ndarray] = []
        all_phi: list[np.ndarray] = []
        junction_idx: list[int] = []  # 脉冲点取值下标（每弧首点 + 末弧末点）
        junction_all: list[int] = []  # 脉冲点全部下标（含弧尾重复样本）
        phi_chain = np.eye(6)
        for arc in arcs:
            t_eval = np.linspace(arc.times[0], arc.times[-1], n_samples)
            res = self._propagate_arc(arc.states[0], t_eval, closure, with_stm=True)
            junction_idx.append(len(all_times))
            for k in range(len(t_eval)):
                all_times.append(float(t_eval[k]))
                all_states.append(res["states"][k])
                all_phi.append(res["stm"][k] @ phi_chain)
            junction_all.append(len(all_times) - 1)
            phi_chain = res["stm"][-1] @ phi_chain
        junction_idx.append(len(all_times) - 1)
        junction_all.extend(junction_idx)

        # 3. 由 p(tf) = pf 解 ṗ(t0)：z(tf) = Φ^{-T} z(t0)，z = [−ṗ; p]
        jac = np.linalg.inv(phi_chain).T
        j21, j22 = jac[3:6, :3], jac[3:6, 3:6]
        pdot0 = np.linalg.solve(j21, j22 @ p0 - pf)
        z0 = np.concatenate([-pdot0, p0])

        # 4. 逐点携载协态，得 p(t) 曲线
        primer = np.array([(np.linalg.inv(phi).T @ z0)[3:6] for phi in all_phi])
        primer_mag = np.linalg.norm(primer, axis=1)
        times = np.asarray(all_times)
        states = np.asarray(all_states)

        # 5. 脉冲点判定（出发/到达点 |p|=1、方向对齐由构造保证，仍有数值误差）
        imp_mag = primer_mag[junction_idx]
        with np.errstate(invalid="ignore"):
            imp_cos = np.where(
                dv_mags > _ZERO_IMPULSE_EPS,
                np.einsum("ij,ij->i", primer[junction_idx], dvs)
                / (imp_mag * np.maximum(dv_mags, 1e-300)),
                np.nan,
            )

        # 6. 弧内违例与插入建议（排除脉冲点样本，含弧尾重复样本）
        interior_mask = np.ones(len(times), dtype=bool)
        interior_mask[junction_all] = False
        interior_mag = primer_mag[interior_mask]
        peak_local = int(np.argmax(interior_mag))
        peak_mag = float(interior_mag[peak_local])
        peak_idx = int(np.flatnonzero(interior_mask)[peak_local])

        mid_ok = True
        for k in range(1, len(junction_idx) - 1):
            if not (abs(imp_mag[k] - 1.0) <= tol and abs(imp_cos[k]) >= 1.0 - tol):
                mid_ok = False
        lawden_ok = peak_mag <= 1.0 + tol and mid_ok

        suggestion_time: float | None = None
        suggestion_pos: np.ndarray | None = None
        if peak_mag > 1.0 + tol:
            suggestion_time = float(times[peak_idx])
            suggestion_pos = np.array(states[peak_idx][:3], copy=True)

        pf_err = float(np.linalg.norm((jac @ z0)[3:6] - pf))
        message = f"端点携载误差 {pf_err:.2e}；弧内峰值 |p|={peak_mag:.4f}" + (
            "" if lawden_ok else "，Lawden 必要条件不满足"
        )
        return PrimerVectorReport(
            times=times,
            primer=primer,
            primer_magnitude=primer_mag,
            max_magnitude=float(np.max(primer_mag)),
            lawden_satisfied=lawden_ok,
            impulse_times=np.asarray(impulse_times),
            impulse_magnitudes=imp_mag,
            impulse_alignment_cosines=imp_cos,
            suggested_insertion_time=suggestion_time,
            suggested_insertion_position=suggestion_pos,
            message=message,
        )

    # ---- 内部实现 ----

    def _resolve_closure(self, closure: Closure | None) -> Closure:
        """解析封闭方式并校验所需动力学齐备。"""
        resolved: Closure = closure if closure is not None else self.closure
        if resolved == "two_body" and self.mu is None:
            raise ValueError("二体封闭需要 mu")
        if resolved == "three_body" and self._shooter is None:
            raise ValueError("三体封闭需要 dynamics")
        return resolved

    @property
    def _tof(self) -> float:
        return self.term1.time - self.term0.time

    def _resolve_bounds(
        self, n_mid: int, bounds: dict[str, tuple] | None, min_dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """组装决策变量上下界：[t 部分（s），r 部分（km，逐分量盒）]。"""
        span = 3.0 * max(
            float(np.linalg.norm(self.term0.state[:3])),
            float(np.linalg.norm(self.term1.state[:3])),
        )
        bounds = bounds or {}
        t_lo, t_hi = bounds.get("t", (min_dt, self._tof - min_dt))
        r_lo_raw, r_hi_raw = bounds.get("r", (np.full(3, -span), np.full(3, span)))
        r_lo, r_hi = np.asarray(r_lo_raw, dtype=float), np.asarray(r_hi_raw, dtype=float)
        if r_lo.shape != (3,) or r_hi.shape != (3,):
            raise ValueError(f"bounds['r'] 上下界须为长度 3 向量，得到 {r_lo.shape}、{r_hi.shape}")
        lb = np.concatenate([np.full(n_mid, t_lo), np.tile(r_lo, n_mid)])
        ub = np.concatenate([np.full(n_mid, t_hi), np.tile(r_hi, n_mid)])
        return lb, ub

    def _default_x0(self, n_mid: int, closure: Closure) -> np.ndarray:
        """默认初猜：中途节点取双脉冲封闭弧上的等间隔点。

        该初猜对应零中途脉冲解（目标值即双脉冲成本），SLSQP 从可行邻域
        出发只降不升；比端点直线插值（可能穿过中心天体附近的退化几何）
        稳健。封闭/传播失败（Lambert 无解、打靶未收敛、传播失败）时回退
        为端点线性插值，并记录警告，不静默退化。
        """
        fracs = (np.arange(n_mid) + 1.0) / (n_mid + 1.0)
        try:
            legs = self._close_velocities(np.empty(0), closure)
            state0 = np.concatenate([self.term0.state[:3], legs[0][0]])
            t_eval = self.term0.time + np.concatenate([[0.0], fracs * self._tof])
            res = self._propagate_arc(state0, t_eval, closure, with_stm=False)
            positions = res["states"][1:, :3]
        except (RuntimeError, ValueError, PropagationFailure) as e:
            # 预期失败（Lambert 无解 / 打靶未收敛 / 传播失败）：回退线性插值
            # 初猜并标记；编程错误（如参数形状）不被吞
            logger.warning("双脉冲初猜失败，回退端点线性插值：%s", e)
            r0, rf = self.term0.state[:3], self.term1.state[:3]
            positions = np.array([r0 + f * (rf - r0) for f in fracs])
        return np.concatenate([fracs * self._tof, positions.ravel()])

    def _stalled_retries(self, y0: np.ndarray, n_mid: int) -> list[np.ndarray]:
        """SLSQP 卡在零脉冲平坦区时的微扰重试初猜。

        时刻分量整体缩放（保持递增顺序）、位置分量小幅缩放，扰动后的起点
        脱离双脉冲弧上的平坦走廊，通常能找到下降方向。
        """
        retries = []
        for factor in (0.5, 1.5, 2.0):
            p = y0.copy()
            p[:n_mid] *= factor
            retries.append(p)
        for factor in (0.9, 1.1):
            p = y0.copy()
            p[n_mid:] *= factor
            retries.append(p)
        return retries

    def _node_positions(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """把决策向量解码为节点时刻（相对 s）与位置序列（含两端点）。"""
        n_mid = y.size // 4
        times = np.concatenate([[0.0], y[:n_mid], [self._tof]])
        positions = np.vstack(
            [self.term0.state[:3], y[n_mid:].reshape(n_mid, 3), self.term1.state[:3]]
        )
        return times, positions

    def _close_velocities(
        self, y: np.ndarray, closure: Closure
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """封闭全部弧段：返回每弧的 (出发速度, 到达速度)，物理单位。"""
        times, positions = self._node_positions(y)
        legs = []
        for k in range(len(times) - 1):
            dt = float(times[k + 1] - times[k])
            r_a, r_b = positions[k], positions[k + 1]
            if closure == "two_body":
                assert self.mu is not None
                lam = solve_lambert(r_a, r_b, dt, self.mu)
                legs.append((lam.v0, lam.vf))
            else:
                assert self._shooter is not None
                shot = self._shooter.solve(
                    StateTerminal(np.concatenate([r_a, np.zeros(3)]), 0.0),
                    StateTerminal(np.concatenate([r_b, np.zeros(3)]), dt),
                    dt,
                )
                if shot.status is not ConvergenceState.CONVERGED:
                    raise RuntimeError(f"三体打靶未收敛（弧 {k}）：{shot.message}")
                legs.append((shot.arcs[0].states[0][3:], shot.arcs[0].states[-1][3:]))
        return legs

    def _total_dv(self, y: np.ndarray, closure: Closure) -> float:
        """目标函数：总 ΔV = 出发脉冲 + 弧间脉冲 + 到达脉冲。"""
        try:
            legs = self._close_velocities(y, closure)
        except Exception:  # Lambert 不收敛 / 打靶发散 / 退化几何：罚出搜索域
            return _CLOSURE_PENALTY
        dv = float(np.linalg.norm(legs[0][0] - self.term0.state[3:]))
        dv += sum(float(np.linalg.norm(legs[k][0] - legs[k - 1][1])) for k in range(1, len(legs)))
        dv += float(np.linalg.norm(self.term1.state[3:] - legs[-1][1]))
        return dv

    def _build_solution(
        self,
        y: np.ndarray,
        closure: Closure,
        *,
        status: ConvergenceState,
        cause: FailureCause,
        n_iter: int = 0,
        message: str = "",
    ) -> TransferSolution:
        """由决策向量组装 TransferSolution（传播弧段采样）并刷新 legs。"""
        legs = self._close_velocities(y, closure)
        times, positions = self._node_positions(y)

        arcs: list[TransferArc] = []
        impulse_dvs: list[np.ndarray] = []
        for k, (v_dep, _v_arr) in enumerate(legs):
            state0 = np.concatenate([positions[k], v_dep])
            t_eval = np.linspace(self.term0.time + times[k], self.term0.time + times[k + 1], 50)
            res = self._propagate_arc(state0, t_eval, closure, with_stm=False)
            dv_in = v_dep - self.term0.state[3:] if k == 0 else v_dep - legs[k - 1][1]
            impulse_dvs.append(np.asarray(dv_in, dtype=float))
            arcs.append(
                TransferArc(
                    states=res["states"],
                    times=np.asarray(res["time"], dtype=float),
                    delta_v=float(np.linalg.norm(dv_in)),
                )
            )
        arrival_dv = self.term1.state[3:] - legs[-1][1]
        arrival_dv_mag = float(np.linalg.norm(arrival_dv))
        total = sum(arc.delta_v for arc in arcs) + arrival_dv_mag

        # 刷新 legs：Impulse 与 CoastArc 交替（首末端脉冲均记录）
        self.legs = []
        for k, arc in enumerate(arcs):
            self.legs.append(Impulse(time=float(arc.times[0]), delta_v=impulse_dvs[k]))
            self.legs.append(
                CoastArc(
                    StateTerminal(arc.states[0], float(arc.times[0])),
                    float(arc.times[-1] - arc.times[0]),
                    closure,
                )
            )
        self.legs.append(Impulse(time=float(arcs[-1].times[-1]), delta_v=np.asarray(arrival_dv)))

        return TransferSolution(
            arcs=tuple(arcs),
            arrival_delta_v=arrival_dv_mag,
            total_delta_v=total,
            transfer_time=self._tof,
            status=status,
            cause=cause,
            n_iter=n_iter,
            message=message,
        )

    def _propagate_arc(
        self,
        state0: np.ndarray,
        t_eval: np.ndarray,
        closure: Closure,
        *,
        with_stm: bool,
    ) -> dict[str, np.ndarray]:
        """按封闭动力学传播单弧；三体时内部转无量纲再转回物理单位。"""
        if closure == "two_body":
            assert self.mu is not None
            return _propagate_two_body(state0, t_eval, self.mu, with_stm=with_stm)

        assert self.dynamics is not None
        system = self.dynamics.system
        tu = system.characteristic_time
        assert tu is not None  # ThreeBodyLambert 构造时已校验 system 初始化
        x0 = system.physical_to_dimensionless(state0)
        t_eval_dim = (t_eval - t_eval[0]) / tu
        res = self.dynamics.propagate(
            x0, (0.0, float(t_eval_dim[-1])), t_eval=t_eval_dim, with_stm=with_stm
        )
        states = np.array([system.dimensionless_to_physical(s) for s in res["states"]])
        out = {"time": t_eval, "states": states}
        if with_stm:
            out["stm"] = res["stm"]
        return out
