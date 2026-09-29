"""Sims-Flanagan 预设计：段中冲量直接法 NLP（单 leg MVP #740；多 leg + flyby 链 #741）。

## 方法概述

把固定飞行时间 ``[0, tof]`` 均分为 ``n`` 段，每段**中点**作用一个冲量决策变量
``ΔVᵢ ∈ R³``（Sims-Flanagan 段中冲量转录）。前向 pass 自出发态经前 ``m = n//2``
段 conic 档 Kepler 档传播（:func:`e2m2e.integrators.propagate_kepler_py`）接到匹配
节点，后向 pass 自到达态经后 ``n − m`` 段反向传播到同一节点；两段轨迹在匹配点的
位置/速度连续性（6 维）是等式约束。段可行域 ``|ΔVᵢ| ≤ T_max(r̄ᵢ)/m̄ᵢ·Δt``：SEP
模式下可用功率随日心距平方反比衰减（消费 :mod:`e2m2e.algorithm.transfer.sep`），
常推力模式上限与日心距无关。目标为平滑化 L1 ``Σ‖ΔVᵢ‖``（min-fuel；固定 ``m₀``
下与最大化末态质量逐位等价——火箭方程 ``m_end = m₀·exp(−Σ‖ΔVᵢ‖/c)`` 是决策
向量的严格单调变换）。

**质量连续性恒等式**：冲量前质量取 ``m̄ₖ = m₀·exp(−Σ_{j<k}‖ΔVⱼ‖/c)``，前向与
后向 pass 用同一公式（后向锚定 ``m_end·exp(Σ_{j≥k}‖ΔVⱼ‖/c)`` 代入 ``m_end`` 后
与之恒等）。因此两 pass 在匹配点的质量由构造自动相等——7 维连续性中的质量维
是恒等式，独立等式约束只有 6 维。

## 求解

SLSQP + 全解析雅可比：目标梯度逐段闭式；等式约束雅可比消费 ``propagate_kepler_py``
的 f & g 解析 STM 做前向/后向灵敏度递推（#739 内核的立项消费方）；不等式约束
雅可比含 SEP 推力-日心距梯度链 ``dT/dr``。``use_analytic_jac=False`` 时退回
SLSQP 数值差分。约束按问题尺度归一（位置除以边界距离尺度、速度除以边界速度
尺度、段界约束除以 ``cap²``）后交给 SLSQP：同一可行集，日心公里量纲下良态。

成功解再过一道**后验闸门**：重算归一化匹配点残差与段界约束，越界即改判
``INFEASIBLE/CONSTRAINT_VIOLATION``（ADR 0020 红线：不谎报 success）。

## 双档结构（ADR 0050）

``backend`` 为必填关键字参数：``"conic"``（二体封闭解，:func:`propagate_kepler_py`）
或 ``"ephemeris"``（星历 N 体数值传播，#727）均已实现；其余取值 ``ValueError``。
ephemeris 档必填 ``epoch_et_s``（出发历元 SPICE et 秒，TDB past J2000）与
``ephemeris_system``（:class:`~e2m2e.algorithm.dynamics.EphemerisSystem` 实例，
``bodies`` 须含 ``origin``——中心引力项由 origin 提供）；段中冲量、匹配点约束、
成本函数、SLSQP 求解与软失败三元组结构两档共用，仅半段传播内核与 TOF 灵敏度
链 RHS 换为 ``EphemerisDynamics``（Rust ``propagate_with_stm_py``，含 STM、
支持后向）。星历档节点状态仍是约束数据；leg 的起始绝对历元取出发历元加
**前序 leg 的决策 TOF 前缀和**（``min_time``/``weighted`` 下 TOF 决策变量
移动 leg 时序时保证 leg 时序连续：leg j 的时间窗自 leg j−1 实际末端开始），
节点状态本身**不**随历元重查星历（保持 conic 档"节点状态是数据"语义；逐
leg 重定目标归 #726 裁决）。注意解析 ∂/∂TOF 链只覆盖本 leg 时长项；前序
leg TOF 平移本 leg 时间窗的星历灵敏度项不在链内——时间不变的退化系下
精确，真实 N 体下为近似（跨 leg 耦合归 #726）。

## 多 leg 链与节点（#741）

:class:`SimsFlanaganMultiLegProblem` 把上述转录推广到多 leg 链：每 leg 独立
前向/后向 pass 至**本 leg 匹配点**（leg 间经节点锚定解耦，位置匹配由锚定
结构性满足）。节点（:class:`SimsFlanaganNode`）分两类：

- **rendezvous**：位置+速度匹配，锚定到固定节点状态（conic 档语义：节点
  状态是数据，不随 TOF 决策变量变化）；
- **flyby**（仅中间节点）：位置匹配 + V∞ 旋转——之后的 leg 前向锚定
  ``[r_B; v_B + v∞_out]``、之前的 leg 后向锚定 ``[r_B; v_B + v∞_in]``（v∞
  是决策变量），等模 ``|v∞_in| = |v∞_out|`` 为等式约束、转角
  ``δ ≤ δ_max(r_p_min, v_eff, μ_B)`` 为不等式约束；转角与近心点闭式复用
  :func:`.mga.flyby_turn_angle` / :func:`.mga.flyby_pericenter_radius`。

质量链 ``m̄ₖ = m₀·exp(−Σ_{j<k}‖ΔVⱼ‖/c)`` 按**全部 leg 段的全局时序**递推
（flyby 不改质量）。逐 leg TOF 可选入决策变量（``tof_bounds_s`` 给定时），
解析 ``∂/∂TOF`` 链在每次半段传播后累加 ``RHS(x_out)·(±1/(2n))``
（``RHS(x) = [v; −μr/|r|³]``，后向 pass 带负号）。

## 成本函数集（① 仓内定义性公式；``c = Isp·g₀/1000`` km/s，``t_ref = Σ 名义 TOF``）

- ``min_fuel``：``J = Σᵢ√(‖ΔVᵢ‖² + ε²)``（平滑 L1；固定 ``m₀`` 下与最大化
  末态质量、最小化初始质量逐位等价）；
- ``min_time``：``J = Σ_leg TOF/t_ref``（TOF 必须为决策变量）；
- ``weighted``：``J = w_dv·Σᵢ√(‖ΔVᵢ‖²+ε²)/c + w_t·Σ_leg TOF/t_ref``
  （两项无量纲：ΔV/c 为质量分数指数、时间归一）；
- ``gtoc``：``J = −Σ_{j∈交会节点（含终点）} exp(−p_j/c)``，``p_j`` 为节点
  ``j`` 之前全部段的 ``Σ‖ΔVᵢ‖``（GTOC4 多交会口径，**本仓定义**）。

⑦ 注记：MALTO（AIAA 2006-6746）的目标集合枚举仅作人工对照出处，原文未
取得、口径未复核；上述 GTOC 目标为本仓定义性公式（用户裁决），无结果性
数值断言。多 leg 的 ①/②/③/④ 类 oracle 见
``tests/algorithm/transfer/test_sims_flanagan_multileg.py``。

## oracle 口径（ADR 0055 决策 1）

⑦ 类结果性数值（Sims & Flanagan 1999 报告的行星际任务算例收敛 ΔV）**非断言
人工对照**，一律禁入 CI 断言；本文档不复制具体数值（原文表值为准，未逐值复核
前不落数字）。文献出处：Sims, J. A. & Flanagan, S. N. (1999),
"Preliminary Design of Low-Thrust Interplanetary Missions", AAS/AIAA
Astrodynamics Specialist Conference, AAS 99-409。① 带 / ② 不变量 / ③ 退化 /
④ 机制类 oracle 见 ``tests/algorithm/transfer/test_sims_flanagan.py``。

单位约定：距离 km、速度 km/s、质量 kg、推力 N、功率 W、时间 s。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, minimize

from e2m2e.data.constants import AU_KM
from e2m2e.integrators import propagate_kepler_py, require_rust_extension

from ...status import ConvergenceState, FailureCause, ResultStatus
from ..dynamics import EphemerisDynamics, EphemerisSystem
from ..results import scipy_slsqp_status
from .mga import flyby_pericenter_radius, flyby_turn_angle
from .sep import (
    edelbaum_delta_v,
    edelbaum_delta_v_inclined,
    sep_available_power,
    thrust_from_power,
)
from .thrust_arcs import G0_MPS2

__all__ = [
    "SimsFlanaganLegSolution",
    "SimsFlanaganFlybyResult",
    "SimsFlanaganMultiLegProblem",
    "SimsFlanaganMultiLegSolution",
    "SimsFlanaganNode",
    "SimsFlanaganProblem",
    "SimsFlanaganPropulsion",
    "SimsFlanaganSolution",
]

#: 冲量对 6 维状态的映射矩阵 [[0₃], [I₃]]（冲量只改速度分量）。
_B_IMPULSE: npt.NDArray[np.floating] = np.zeros((6, 3))
_B_IMPULSE[3:, :] = np.eye(3)

#: 平滑 L1 目标的正则化参数（km/s）：|ΔV| ≫ ε 时目标梯度 ≈ û。
_EPS_KM_S = 1e-9

#: û = ΔV/|ΔV| 的退化阈值（km/s）：低于阈值取零向量（雅可比连续化）。
_UHAT_EPS_KM_S = 1e-12

#: 后验闸门阈值：归一化匹配点残差与段界相对越界裕度（SLSQP 谎报 success 的
#: 防线，取宽于 SLSQP 自身收敛精度的量级，只拦“明显不可行却报成功”）。
_POST_EQ_TOL = 1e-6
_POST_INEQ_RATIO_TOL = 1e-6

#: 决策变量盒上限相对“段界物理上限”的慷慨倍数：段可行域 |ΔVₖ| ≤ capₖ 本就是
#: 硬约束，盒取其数倍只防 SLSQP 线搜索漫游进近抛物线能量带（闭式 Kepler 内核
#: 在 |E| ≈ 0 处 Newton 发散），最优解处盒不起约束作用。
_CAP_BOX_FACTOR = 3.0

#: 线搜索试探点传播失败（近抛物线等病态能量）时的惩罚值：目标罚大数、等式
#: 约给大残差、不等式给深负值，让 SLSQP 的 L1 罚函数自动回退步长。
_PENALTY_OBJECTIVE = 1e9
_PENALTY_EQ = 1e3
_PENALTY_INEQ = -1e3


def _propagate_half(
    x: npt.NDArray[np.floating],
    t_start: float | None,
    half: float,
    *,
    mu_km3_s2: float,
    dyn: EphemerisDynamics | None,
    with_stm: bool,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating] | None]:
    """半段传播内核：conic 封闭解 / 星历 N 体数值传播（#727 双档）。

    conic 档忽略 ``t_start``（二体时间不变）；星历档在绝对历元
    ``[t_start, t_start + half]`` 上数值传播（``half < 0`` 即后向）。
    返回 ``(x_out, Φ(6,6) 或 None)``。
    """
    if dyn is None:
        out = propagate_kepler_py(x.tolist(), [half], mu_km3_s2, with_stm=with_stm)
        x_out = np.asarray(out["states"][0], dtype=float)
        phi = np.asarray(out["stm"][0], dtype=float).reshape(6, 6) if with_stm else None
    else:
        if t_start is None:
            raise ValueError("ephemeris 档传播必须提供绝对历元 t_start")
        out = dyn.propagate(
            x, (t_start, t_start + half), t_eval=[t_start + half], with_stm=with_stm
        )
        x_out = np.asarray(out["states"][-1], dtype=float)
        phi = np.asarray(out["stm"][-1], dtype=float) if with_stm else None
    return x_out, phi


def _tier_rhs(
    t: float | None,
    x: npt.NDArray[np.floating],
    *,
    mu_km3_s2: float,
    dyn: EphemerisDynamics | None,
) -> npt.NDArray[np.floating]:
    """TOF 灵敏度链 RHS：conic 二体 ``[v; −μr/|r|³]`` / 星历 ``equations_of_motion(t, x)``。"""
    if dyn is None:
        r = x[:3]
        return np.concatenate([x[3:], -mu_km3_s2 * r / float(np.linalg.norm(r)) ** 3])
    if t is None:
        raise ValueError("ephemeris 档 RHS 必须提供绝对历元 t")
    return np.asarray(dyn.equations_of_motion(t, x), dtype=float)


def _plane_change_deg(
    r_dep: npt.NDArray[np.floating],
    v_dep: npt.NDArray[np.floating],
    r_arr: npt.NDArray[np.floating],
    v_arr: npt.NDArray[np.floating],
) -> float:
    """两轨道面法向（r̂×v̂ 归一化）夹角（度）；任一法向退化按 0 处理。"""
    h_dep = np.cross(r_dep, v_dep)
    h_arr = np.cross(r_arr, v_arr)
    h_dep_mag = float(np.linalg.norm(h_dep))
    h_arr_mag = float(np.linalg.norm(h_arr))
    if h_dep_mag < 1e-12 or h_arr_mag < 1e-12:
        return 0.0
    cos_inc = float(np.clip(np.dot(h_dep / h_dep_mag, h_arr / h_arr_mag), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_inc)))


def _edelbaum_leg_guess(
    departure_state: npt.NDArray[np.floating],
    arrival_state: npt.NDArray[np.floating],
    n: int,
) -> npt.NDArray[np.floating]:
    """单 leg Edelbaum 闭式初猜：总 ΔV 均分 n 段，方向沿出发速度（抬升取 +）。

    单 leg MVP 与多 leg 链的每 leg 共用；多 leg 对 flyby 起点的 leg 以
    ``[r_B; v_B + v∞_默认]`` 作出发态调用。
    """
    r_dep, v_dep = departure_state[:3], departure_state[3:]
    r_arr, v_arr = arrival_state[:3], arrival_state[3:]
    v_dep_mag = float(np.linalg.norm(v_dep))
    v_arr_mag = float(np.linalg.norm(v_arr))
    inc_deg = _plane_change_deg(r_dep, v_dep, r_arr, v_arr)
    if inc_deg > 0.0:
        dv_total = edelbaum_delta_v_inclined(v_dep_mag, v_arr_mag, inc_deg)
    else:
        dv_total = edelbaum_delta_v(v_dep_mag, v_arr_mag)
    guess = np.zeros(3 * n)
    # ΔV 近零或出发速度退化时不给方向，退化为全零初猜。
    if dv_total < _UHAT_EPS_KM_S or v_dep_mag < _UHAT_EPS_KM_S:
        return guess
    # 抬升（末速更小，如外推）取顺行 +；下降取逆行 −。
    sign = 1.0 if v_arr_mag < v_dep_mag else -1.0
    direction = sign * v_dep / v_dep_mag
    return np.tile(direction * (dv_total / n), n)


@dataclass(frozen=True)
class SimsFlanaganPropulsion:
    """推进配置：常推力与 SEP 功率受限二选一（段可行域上界的来源）。

    Attributes:
        isp_s: 比冲（s），必须为正。
        t_max_n: 常推力模式的最大推力（N）；与 ``p0_w`` 恰一非空。
        p0_w: SEP 模式 1 AU 处的推进阵列功率（W）；与 ``t_max_n`` 恰一非空。
        p_bus_w: 平台常耗功率（W），仅 SEP 模式有意义。
        efficiency: 功率-推力映射效率 ``η ∈ (0, 1]``。
        r_helio_km: SEP 功率评估的固定日心距（km）；``None`` 表示用段中点瞬时
            |r|（日心问题语义）。非日心问题（如地心测试）必须显式传常数，
            避免 |r| 被误读为日心距。
    """

    isp_s: float
    t_max_n: float | None = None
    p0_w: float | None = None
    p_bus_w: float = 0.0
    efficiency: float = 1.0
    r_helio_km: float | None = None

    @classmethod
    def constant(cls, t_max_n: float, isp_s: float) -> SimsFlanaganPropulsion:
        """常推力模式（推力上限与日心距无关）。"""
        return cls(isp_s=isp_s, t_max_n=float(t_max_n))

    @classmethod
    def sep(
        cls,
        p0_w: float,
        isp_s: float,
        *,
        p_bus_w: float = 0.0,
        efficiency: float = 1.0,
        r_helio_km: float | None = None,
    ) -> SimsFlanaganPropulsion:
        """SEP 功率受限模式（P(r) = P₀·(1AU/r)² − P_bus，负值截 0）。"""
        return cls(
            isp_s=isp_s,
            p0_w=float(p0_w),
            p_bus_w=float(p_bus_w),
            efficiency=float(efficiency),
            r_helio_km=r_helio_km,
        )

    def validate(self) -> None:
        """校验配置一致性；不满足即 ``ValueError``。"""
        if not np.isfinite(self.isp_s) or self.isp_s <= 0.0:
            raise ValueError(f"isp_s 必须为正的有限数，得到 {self.isp_s!r}")
        t_max = self.t_max_n
        p0 = self.p0_w
        if (t_max is None) == (p0 is None):
            raise ValueError("t_max_n 与 p0_w 必须恰好指定一个（常推力/SEP 二选一）")
        if t_max is not None:
            if not np.isfinite(t_max) or t_max < 0.0:
                raise ValueError(f"t_max_n 必须为非负有限数，得到 {t_max!r}")
        elif p0 is not None:
            if not np.isfinite(p0) or p0 <= 0.0:
                raise ValueError(f"p0_w 必须为正的有限数，得到 {p0!r}")
            if not np.isfinite(self.p_bus_w) or self.p_bus_w < 0.0:
                raise ValueError(f"p_bus_w 必须为非负有限数，得到 {self.p_bus_w!r}")
        if not np.isfinite(self.efficiency) or not 0.0 < self.efficiency <= 1.0:
            raise ValueError(f"efficiency 必须在 (0, 1]，得到 {self.efficiency!r}")
        if self.r_helio_km is not None and (
            not np.isfinite(self.r_helio_km) or self.r_helio_km <= 0.0
        ):
            raise ValueError(f"r_helio_km 必须为正的有限数或 None，得到 {self.r_helio_km!r}")

    def _sep_inputs(self) -> tuple[float, float]:
        """SEP 模式参数对（``validate`` 保证 SEP 模式下非空；常推力模式不调用）。"""
        if self.p0_w is None:
            raise ValueError("SEP 模式未配置 p0_w")
        return self.p0_w, self.p_bus_w

    def _sep_radius_km(self, r_km: float) -> float:
        """SEP 功率评估用的日心距：显式常数优先，否则用瞬时 |r|。"""
        return float(r_km) if self.r_helio_km is None else float(self.r_helio_km)

    def max_thrust_n(self, r_km: float) -> float:
        """段中点距离 ``r_km`` 处的可用最大推力（N）。"""
        t_max = self.t_max_n
        if t_max is not None:
            return float(t_max)
        p0, p_bus = self._sep_inputs()
        power = sep_available_power(p0, p_bus, self._sep_radius_km(r_km))
        return thrust_from_power(power, self.efficiency, self.isp_s)

    def _thrust_gradient_n_per_km(self, r_km: float) -> float:
        """``dT/dr``（N/km）：SEP 功率衰减正分支的解析梯度；常推力或截 0 分支为 0。

        正分支 ``T(r) = 2η/(Isp·g₀)·(P₀·AU²/r² − P_bus)`` 对 r 求导得
        ``−4η·P₀·AU²/(Isp·g₀·r³)``（P_bus 项导数为零）。``r_helio_km`` 显式
        固定时功率评估点不随决策变量变化（∂r_eff/∂ΔV ≡ 0），梯度恒为 0。
        """
        if self.t_max_n is not None or self.r_helio_km is not None:
            return 0.0
        p0, p_bus = self._sep_inputs()
        r_eff = self._sep_radius_km(r_km)
        if sep_available_power(p0, p_bus, r_eff) <= 0.0:
            return 0.0
        coef = 2.0 * self.efficiency / (self.isp_s * G0_MPS2)
        return float(coef * (-2.0 * p0 * AU_KM**2 / r_eff**3))


@dataclass
class SimsFlanaganSolution:
    """Sims-Flanagan 求解结果（软失败三元组随解携带，不抛异常）。

    Attributes:
        impulses_km_s: 各段中冲量 ``(n, 3)``（km/s），行序对应段 0..n-1。
        impulse_times_s: 冲量时刻 ``(n,)``（s），第 k 段中点 ``(k+0.5)·Δt``。
        node_times_s: 节点时刻 ``(n+1,)``（s）。
        forward_states: 前向 pass 节点状态 ``(m+1, 6)``，``[0]`` 为出发态、
            ``[-1]`` 为匹配节点。
        backward_states: 后向 pass 节点状态 ``(n-m+1, 6)``，``[0]`` 为匹配节点、
            ``[-1]`` 为到达态。
        delta_v_total_km_s: ``Σ‖ΔVᵢ‖``（km/s）。
        final_mass_kg: 末态质量（kg），``m₀·exp(−Σ‖ΔVᵢ‖/c)``。
        fuel_kg: 燃料消耗（kg），``m₀ − final_mass_kg``。
        matchpoint_residual: 匹配点连续性残差 ``(6,)``（前向 − 后向；位置 km、
            速度 km/s）。
        n_iter: SLSQP 迭代次数。
        status: 算法最终状态。
        cause: 算法最终原因码。
        message: 求解器状态消息（含后验闸门结论）。
    """

    impulses_km_s: npt.NDArray[np.floating]
    impulse_times_s: npt.NDArray[np.floating]
    node_times_s: npt.NDArray[np.floating]
    forward_states: npt.NDArray[np.floating]
    backward_states: npt.NDArray[np.floating]
    delta_v_total_km_s: float
    final_mass_kg: float
    fuel_kg: float
    matchpoint_residual: npt.NDArray[np.floating]
    n_iter: int
    status: ConvergenceState
    cause: FailureCause
    message: str

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)


@dataclass
class _Evaluation:
    """单次决策向量评估的全部产物（``SimsFlanaganProblem.solve`` 内部缓存用）。

    ``eq``/``ineq`` 是交给 SLSQP 的约束（等式按问题尺度归一，不等式原始
    量纲）；``eq_raw`` 是调用方口径的原始匹配点残差（位置 km、速度 km/s）。
    """

    dv: npt.NDArray[np.floating]
    objective: float
    obj_grad: npt.NDArray[np.floating] | None
    eq: npt.NDArray[np.floating]
    eq_raw: npt.NDArray[np.floating]
    eq_jac: npt.NDArray[np.floating] | None
    ineq: npt.NDArray[np.floating]
    ineq_jac: npt.NDArray[np.floating] | None
    forward_states: npt.NDArray[np.floating]
    backward_states: npt.NDArray[np.floating]
    cap_km_s: npt.NDArray[np.floating]
    total_dv_km_s: float
    final_mass_kg: float


class SimsFlanaganProblem:
    """单 leg 出发→到达交会的 Sims-Flanagan 预设计问题（conic / ephemeris 双档）。

    Args:
        departure_state: 出发状态 ``[r, v]``，``(6,)``，km / km/s。
        arrival_state: 到达状态 ``[r, v]``，``(6,)``，km / km/s。
        tof_s: 飞行时间（s），必须为正。
        propulsion: 推进配置（:class:`SimsFlanaganPropulsion`）。
        initial_mass_kg: 出发质量（kg），必须为正。
        mu_km3_s2: 中心天体引力常数（km³/s²），必须为正。
        backend: 保真度档位，必填关键字：``"conic"``（二体封闭解）或
            ``"ephemeris"``（星历 N 体数值传播，#727）；其余取值报错。
        epoch_et_s: ephemeris 档必填：出发历元 SPICE et 秒（TDB past
            J2000）；conic 档必须为 None。
        ephemeris_system: ephemeris 档必填：``EphemerisSystem`` 实例
            （``bodies`` 须含 ``origin``，中心引力项由 origin 提供）；conic
            档必须为 None。
    """

    def __init__(
        self,
        departure_state: npt.ArrayLike,
        arrival_state: npt.ArrayLike,
        tof_s: float,
        propulsion: SimsFlanaganPropulsion,
        initial_mass_kg: float,
        mu_km3_s2: float,
        *,
        backend: str,
        epoch_et_s: float | None = None,
        ephemeris_system: EphemerisSystem | None = None,
    ) -> None:
        departure = np.asarray(departure_state, dtype=float)
        arrival = np.asarray(arrival_state, dtype=float)
        if departure.shape != (6,):
            raise ValueError(f"departure_state 必须为 (6,) 状态向量，得到 shape {departure.shape}")
        if arrival.shape != (6,):
            raise ValueError(f"arrival_state 必须为 (6,) 状态向量，得到 shape {arrival.shape}")
        if not np.all(np.isfinite(departure)):
            raise ValueError("departure_state 含非有限分量")
        if not np.all(np.isfinite(arrival)):
            raise ValueError("arrival_state 含非有限分量")
        tof = float(tof_s)
        if not np.isfinite(tof) or tof <= 0.0:
            raise ValueError(f"tof_s 必须为正的有限数，得到 {tof_s!r}")
        m0 = float(initial_mass_kg)
        if not np.isfinite(m0) or m0 <= 0.0:
            raise ValueError(f"initial_mass_kg 必须为正的有限数，得到 {initial_mass_kg!r}")
        mu = float(mu_km3_s2)
        if not np.isfinite(mu) or mu <= 0.0:
            raise ValueError(f"mu_km3_s2 必须为正的有限数，得到 {mu_km3_s2!r}")
        propulsion.validate()
        if backend not in ("conic", "ephemeris"):
            raise ValueError(f"backend 必须为 'conic' 或 'ephemeris'，得到 {backend!r}")
        self._backend = backend
        self._epoch_et_s: float | None = None
        self._eph_dyn: EphemerisDynamics | None = None
        if backend == "ephemeris":
            if epoch_et_s is None:
                raise ValueError(
                    "backend='ephemeris' 必须提供 epoch_et_s（SPICE et 秒，TDB past J2000）"
                )
            if ephemeris_system is None:
                raise ValueError(
                    "backend='ephemeris' 必须提供 ephemeris_system（EphemerisSystem 实例）"
                )
            if not isinstance(ephemeris_system, EphemerisSystem):
                raise ValueError(
                    "ephemeris_system 必须为 EphemerisSystem 实例，得到 "
                    f"{type(ephemeris_system).__name__}"
                )
            if not np.isfinite(float(epoch_et_s)):
                raise ValueError(f"epoch_et_s 必须为有限数，得到 {epoch_et_s!r}")
            if ephemeris_system.origin not in ephemeris_system.bodies:
                raise ValueError(
                    f"ephemeris_system.bodies 须包含 origin 天体 {ephemeris_system.origin!r}"
                    "（中心引力项由 origin 提供）"
                )
            self._epoch_et_s = float(epoch_et_s)
            self._eph_dyn = EphemerisDynamics(system=ephemeris_system)
            # 60 s 上限是 LEO 短弧调优值；置 inf 后经 _get_max_step 变为逐半段
            # span/10，避免日量级段被强加 60 s 步长（性能）。容差保持默认 1e-12。
            self._eph_dyn.max_step = float("inf")
        elif epoch_et_s is not None or ephemeris_system is not None:
            raise ValueError(
                "backend='conic' 不接受 epoch_et_s/ephemeris_system"
                "（星历上下文仅 ephemeris 档使用）"
            )

        self._departure = departure.copy()
        self._arrival = arrival.copy()
        self._tof_s = tof
        self._propulsion = propulsion
        self._m0_kg = m0
        self._mu_km3_s2 = mu

        # 约束归一化尺度：边界状态幅值（退化为零时取 1，避免除零）。
        self._len_scale = max(
            float(np.linalg.norm(departure[:3])), float(np.linalg.norm(arrival[:3])), 1.0
        )
        self._vel_scale = max(
            float(np.linalg.norm(departure[3:])), float(np.linalg.norm(arrival[3:])), 1.0
        )

    # ---- 公开入口 ----

    def solve(
        self,
        n_segments: int,
        *,
        x0: npt.ArrayLike | None = None,
        guess: str | None = None,
        ftol: float = 1e-9,
        maxiter: int = 200,
        use_analytic_jac: bool = True,
        verbose: bool = False,
    ) -> SimsFlanaganSolution:
        """求解 min-fuel 交会 NLP（软失败，不抛异常）。

        Args:
            n_segments: 段数 ``n ≥ 2``；匹配点取 ``m = n // 2``。
            x0: 初猜冲量 ``(n, 3)``（km/s）；给定则优先生效。
            guess: ``"edelbaum"`` 用 Edelbaum 闭式生成初猜；``None`` 全零。
            ftol: SLSQP 目标容差。
            maxiter: SLSQP 最大迭代次数。
            use_analytic_jac: True（默认）用解析雅可比（f & g STM 灵敏度递推）；
                False 退回 SLSQP 数值差分。
            verbose: 是否打印 SLSQP 迭代信息。

        Returns:
            :class:`SimsFlanaganSolution`；收敛失败以后验闸门与 SLSQP 结束码
            翻译成的 ``(status, cause, message)`` 三元组表达。
        """
        require_rust_extension("propagate_kepler_py")

        n = int(n_segments)
        if n != n_segments or n < 2:
            raise ValueError(f"n_segments 必须为 ≥ 2 的整数，得到 {n_segments!r}")
        n_var = 3 * n
        x_start = self._initial_guess(n, x0=x0, guess=guess)
        dt = self._tof_s / n
        # 决策变量盒：速度慷慨上限与段界物理上限的数倍取小。段可行域
        # |ΔVₖ| ≤ capₖ 是硬约束，收紧盒只挡线搜索漫游（防近抛物线能量带的
        # 闭式 Kepler Newton 发散），最优解处盒不起约束作用。
        v_cap = 2.0 * max(
            float(np.linalg.norm(self._departure[3:])), float(np.linalg.norm(self._arrival[3:]))
        )
        r_ref = min(
            float(np.linalg.norm(self._departure[:3])), float(np.linalg.norm(self._arrival[:3]))
        )
        t_ref = self._propulsion.max_thrust_n(r_ref)
        cap_box = _CAP_BOX_FACTOR * t_ref / self._m0_kg * dt / 1000.0
        dv_cap = min(v_cap, cap_box) if cap_box > 0.0 else v_cap
        bounds = Bounds(np.full(n_var, -dv_cap), np.full(n_var, dv_cap))

        want_sens = bool(use_analytic_jac)
        cache: dict[bytes, _Evaluation] = {}

        def evaluated(x_flat: npt.NDArray[np.floating]) -> _Evaluation:
            key = np.ascontiguousarray(x_flat, dtype=float).tobytes()
            hit = cache.get(key)
            if hit is None:
                hit = self._evaluate(
                    np.asarray(x_flat, dtype=float), n_segments=n, with_sens=want_sens
                )
                if len(cache) > 32:
                    cache.clear()
                cache[key] = hit
            return hit

        eq_dict: dict[str, object] = {
            "type": "eq",
            "fun": lambda x: evaluated(x).eq,
        }
        ineq_dict: dict[str, object] = {
            "type": "ineq",
            "fun": lambda x: evaluated(x).ineq,
        }
        if want_sens:
            eq_dict["jac"] = lambda x: evaluated(x).eq_jac
            ineq_dict["jac"] = lambda x: evaluated(x).ineq_jac

        result = minimize(
            lambda x: evaluated(x).objective,
            x_start,
            method="SLSQP",
            jac=(lambda x: evaluated(x).obj_grad) if want_sens else None,
            bounds=bounds,
            constraints=[eq_dict, ineq_dict],
            options={"ftol": ftol, "maxiter": maxiter, "disp": verbose},
        )

        status, cause = scipy_slsqp_status(bool(result.success), int(result.status))
        message = str(result.message)
        final = evaluated(result.x)

        # 后验闸门（成功也要过）：归一化匹配点残差 / 段界相对越界超限即改判
        # 不可行（ADR 0020 红线）。段界越界取相对量 |ΔV|/cap（原始 g 随场景
        # 跨若干量级，绝对阈值无一致语义）。
        eq_max = float(np.max(np.abs(final.eq)))
        cap_safe = np.where(final.cap_km_s > 0.0, final.cap_km_s, 1.0)
        over_ratio = np.linalg.norm(final.dv, axis=1) / cap_safe
        over_max = float(np.max(over_ratio))
        if eq_max > _POST_EQ_TOL or over_max > 1.0 + _POST_INEQ_RATIO_TOL:
            raw_max = float(np.max(np.abs(final.eq_raw)))
            message = (
                f"{message}；后验闸门未过：归一化匹配点残差 max|c| = {eq_max:.3e}"
                f"（原始 {raw_max:.3e}，阈值 {_POST_EQ_TOL}），"
                f"段界相对越界 max|ΔV|/cap = {over_max:.12f}"
                f"（阈值 1 + {_POST_INEQ_RATIO_TOL}）"
            )
            status = ConvergenceState.INFEASIBLE
            cause = FailureCause.CONSTRAINT_VIOLATION

        dt = self._tof_s / n
        return SimsFlanaganSolution(
            impulses_km_s=final.dv.copy(),
            impulse_times_s=(np.arange(n) + 0.5) * dt,
            node_times_s=np.arange(n + 1) * dt,
            forward_states=final.forward_states.copy(),
            backward_states=final.backward_states.copy(),
            delta_v_total_km_s=final.total_dv_km_s,
            final_mass_kg=final.final_mass_kg,
            fuel_kg=self._m0_kg - final.final_mass_kg,
            matchpoint_residual=final.eq_raw.copy(),
            n_iter=int(result.nit),
            status=status,
            cause=cause,
            message=message,
        )

    # ---- 内部：初猜 ----

    def _initial_guess(
        self, n: int, *, x0: npt.ArrayLike | None, guess: str | None
    ) -> npt.NDArray[np.floating]:
        """初猜决策向量（(3n,)，km/s）：显式 x0 优先，其次 Edelbaum，否则全零。"""
        if x0 is not None:
            arr = np.asarray(x0, dtype=float)
            if arr.shape != (n, 3):
                raise ValueError(f"x0 必须为 (n, 3) = ({n}, 3)，得到 shape {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError("x0 含非有限分量")
            return arr.reshape(3 * n).copy()
        if guess is None:
            return np.zeros(3 * n)
        if guess != "edelbaum":
            raise ValueError(f"guess 必须为 'edelbaum' 或 None，得到 {guess!r}")
        return self._edelbaum_guess(n)

    def _edelbaum_guess(self, n: int) -> npt.NDArray[np.floating]:
        """Edelbaum 闭式初猜（委托模块级 :func:`_edelbaum_leg_guess`）。"""
        return _edelbaum_leg_guess(self._departure, self._arrival, n)

    # ---- 内部：段传播与灵敏度 ----

    def _forward_pass(
        self,
        dv: npt.NDArray[np.floating],
        n: int,
        m: int,
        dt: float,
        *,
        with_sens: bool,
        t_start: float | None = None,
    ) -> tuple[
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
    ]:
        """前向 pass：出发态经段 0..m-1 接龙（段中冲量）到匹配节点。

        ``t_start`` 为本 leg 起始绝对历元（ephemeris 档必填，conic 档忽略）。

        返回 ``(节点状态 (m+1,6), 段中点位置 (m,3), 段中点半径 (m,),
        ∂匹配节点/∂ΔV (6,3n), 段中点位置灵敏度 (m,3,3n))``。
        """
        half = 0.5 * dt
        x = self._departure
        states = [x.copy()]
        mid_pos = np.zeros((m, 3))
        r_mid = np.zeros(m)
        sens = np.zeros((6, 3 * n))
        s_mid_pos = np.zeros((m, 3, 3 * n))
        t = t_start
        for k in range(m):
            t_mid = None if t is None else t + half
            x_mid, phi1 = _propagate_half(
                x, t, half, mu_km3_s2=self._mu_km3_s2, dyn=self._eph_dyn, with_stm=with_sens
            )
            mid_pos[k] = x_mid[:3]
            r_mid[k] = float(np.linalg.norm(x_mid[:3]))
            if with_sens:
                assert phi1 is not None  # with_stm=True 时 _propagate_half 恒返回 STM
                s_mid = phi1 @ sens
                s_mid_pos[k] = s_mid[:3, :]
                # 冲量是段中点的状态跳变：∂x_mid⁺/∂ΔV_k = s_mid[:,k] + B（无 Φ1）。
                s_mid[:, 3 * k : 3 * k + 3] += _B_IMPULSE
            x_mid = x_mid + _B_IMPULSE @ dv[k]
            x, phi2 = _propagate_half(
                x_mid, t_mid, half, mu_km3_s2=self._mu_km3_s2, dyn=self._eph_dyn, with_stm=with_sens
            )
            if with_sens:
                assert phi2 is not None
                sens = phi2 @ s_mid
            states.append(x.copy())
            t = None if t_mid is None else t_mid + half
        return np.asarray(states), mid_pos, r_mid, sens, s_mid_pos

    def _backward_pass(
        self,
        dv: npt.NDArray[np.floating],
        n: int,
        m: int,
        dt: float,
        *,
        with_sens: bool,
        t_start: float | None = None,
    ) -> tuple[
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
    ]:
        """后向 pass：到达态经段 n-1..m 反向接龙（内核支持负 dt）到匹配节点。

        ``t_start`` 为本 leg 末端绝对历元（ephemeris 档必填，游标自它反向
        递减；conic 档忽略）。

        返回 ``(节点状态 (n-m+1,6) [匹配节点..到达], 段中点位置 (n-m,3),
        段中点半径 (n-m,), ∂匹配节点/∂ΔV (6,3n), 段中点位置灵敏度 (n-m,3,3n))``。
        段序数组（中点位置/半径/灵敏度）按**段升序** k = m..n-1 排列，与前向
        pass 拼接语义一致；冲量在后向路径上取 ``v_pre = v_post − ΔV_k``，
        灵敏度带负号。
        """
        half = -0.5 * dt
        x = self._arrival
        nodes = [x.copy()]
        nb = n - m
        mid_pos = np.zeros((nb, 3))
        r_mid = np.zeros(nb)
        sens = np.zeros((6, 3 * n))
        s_mid_pos = np.zeros((nb, 3, 3 * n))
        t = t_start
        for idx in range(nb):
            k = n - 1 - idx
            t_mid = None if t is None else t + half
            x_mid, phi1 = _propagate_half(
                x, t, half, mu_km3_s2=self._mu_km3_s2, dyn=self._eph_dyn, with_stm=with_sens
            )
            mid_pos[idx] = x_mid[:3]
            r_mid[idx] = float(np.linalg.norm(x_mid[:3]))
            if with_sens:
                assert phi1 is not None  # with_stm=True 时 _propagate_half 恒返回 STM
                s_mid = phi1 @ sens
                s_mid_pos[idx] = s_mid[:3, :]
                # 后向冲量是段中点的反向状态跳变：v_pre = v_post − ΔV_k（无 Φ1）。
                s_mid[:, 3 * k : 3 * k + 3] -= _B_IMPULSE
            x_mid = x_mid - _B_IMPULSE @ dv[k]
            x, phi2 = _propagate_half(
                x_mid, t_mid, half, mu_km3_s2=self._mu_km3_s2, dyn=self._eph_dyn, with_stm=with_sens
            )
            if with_sens:
                assert phi2 is not None
                sens = phi2 @ s_mid
            nodes.append(x.copy())
            t = None if t_mid is None else t_mid + half
        # 遍历按 k 降序；段序数组翻转为升序（与索引 k-m 的消费方语义对齐）。
        return (
            np.asarray(nodes[::-1]),
            mid_pos[::-1],
            r_mid[::-1],
            sens,
            s_mid_pos[::-1],
        )

    def _penalty_evaluation(
        self,
        dv: npt.NDArray[np.floating],
        n: int,
        m: int,
        with_sens: bool,
    ) -> _Evaluation:
        """传播失败试探点的深罚评估（目标大数、等式大残差、不等式深负）。"""
        c_kms = self._propulsion.isp_s * G0_MPS2 / 1000.0
        total_dv = float(np.sum(np.linalg.norm(dv, axis=1)))
        n_var = 3 * n
        return _Evaluation(
            dv=dv,
            objective=_PENALTY_OBJECTIVE,
            obj_grad=np.zeros(n_var) if with_sens else None,
            eq=np.full(6, _PENALTY_EQ),
            eq_raw=np.full(6, _PENALTY_EQ),
            eq_jac=np.zeros((6, n_var)) if with_sens else None,
            ineq=np.full(n, _PENALTY_INEQ),
            ineq_jac=np.zeros((n, n_var)) if with_sens else None,
            forward_states=np.zeros((m + 1, 6)),
            backward_states=np.zeros((n - m + 1, 6)),
            cap_km_s=np.zeros(n),
            total_dv_km_s=total_dv,
            final_mass_kg=self._m0_kg * float(np.exp(-total_dv / c_kms)),
        )

    def _evaluate(
        self, x_flat: npt.NDArray[np.floating], *, n_segments: int, with_sens: bool
    ) -> _Evaluation:
        """决策向量 → 目标/约束/雅可比/状态剖面的统一评估。

        等式约束按 ``(位置/长度尺度, 速度/速度尺度)`` 归一后交给 SLSQP（同一
        可行集，日心公里量纲下良态）；不等式约束保持原始量纲
        ``g_k = cap_k² − ‖ΔV_k‖²``（每条约束自洽，避免对依赖 ΔV 的 ``cap²``
        归一化引入商法则项）。
        """
        n = n_segments
        m = n // 2
        dt = self._tof_s / n
        c_kms = self._propulsion.isp_s * G0_MPS2 / 1000.0
        dv = np.asarray(x_flat, dtype=float).reshape(n, 3)
        dv_norm = np.linalg.norm(dv, axis=1)

        try:
            fwd_states, fwd_mid_pos, r_fwd, s_fwd, spos_fwd = self._forward_pass(
                dv, n, m, dt, with_sens=with_sens, t_start=self._epoch_et_s
            )
            bwd_states, bwd_mid_pos, r_bwd, s_bwd, spos_bwd = self._backward_pass(
                dv,
                n,
                m,
                dt,
                with_sens=with_sens,
                # 后向自 leg 末端历元出发（ephemeris 档；conic 档为 None）。
                t_start=None if self._epoch_et_s is None else self._epoch_et_s + self._tof_s,
            )
        except ValueError:
            # 线搜索试探点落在闭式 Kepler 的病态能量带（近抛物线 Newton 发散）：
            # 返回深罚评估让 SLSQP 的 L1 罚函数回退步长，而不是让整个求解崩溃。
            return self._penalty_evaluation(dv, n, m, with_sens)

        # 冲量前质量 m̄ₖ = m₀·exp(−Σ_{j<k}‖ΔVⱼ‖/c)：两 pass 同式（恒等式，见模块 docstring）。
        prefix = np.concatenate([[0.0], np.cumsum(dv_norm)[:-1]])
        m_bar = self._m0_kg * np.exp(-prefix / c_kms)
        r_mid = np.concatenate([r_fwd, r_bwd])
        thrust = np.array([self._propulsion.max_thrust_n(r) for r in r_mid])
        cap = thrust / m_bar * dt / 1000.0

        eq_raw = fwd_states[-1] - bwd_states[0]
        eq = np.concatenate([eq_raw[:3] / self._len_scale, eq_raw[3:] / self._vel_scale])
        ineq = cap**2 - dv_norm**2

        total_dv = float(np.sum(dv_norm))
        final_mass = self._m0_kg * float(np.exp(-total_dv / c_kms))
        objective = float(np.sum(np.sqrt(dv_norm**2 + _EPS_KM_S**2)))

        obj_grad = None
        eq_jac = None
        ineq_jac = None
        if with_sens:
            denom = np.sqrt(dv_norm**2 + _EPS_KM_S**2)
            obj_grad = (dv / denom[:, None]).reshape(3 * n)
            eq_jac = np.vstack([s_fwd[:3] / self._len_scale, s_fwd[3:] / self._vel_scale]) - (
                np.vstack([s_bwd[:3] / self._len_scale, s_bwd[3:] / self._vel_scale])
            )
            ineq_jac = self._ineq_jacobian(
                dv,
                dv_norm,
                cap,
                r_mid,
                np.concatenate([fwd_mid_pos, bwd_mid_pos]),
                m_bar,
                dt,
                c_kms,
                m,
                spos_fwd,
                spos_bwd,
            )

        return _Evaluation(
            dv=dv,
            objective=objective,
            obj_grad=obj_grad,
            eq=eq,
            eq_raw=eq_raw,
            eq_jac=eq_jac,
            ineq=ineq,
            ineq_jac=ineq_jac,
            forward_states=fwd_states,
            backward_states=bwd_states,
            cap_km_s=cap,
            total_dv_km_s=total_dv,
            final_mass_kg=final_mass,
        )

    def _ineq_jacobian(
        self,
        dv: npt.NDArray[np.floating],
        dv_norm: npt.NDArray[np.floating],
        cap: npt.NDArray[np.floating],
        r_mid: npt.NDArray[np.floating],
        mid_pos: npt.NDArray[np.floating],
        m_bar: npt.NDArray[np.floating],
        dt: float,
        c_kms: float,
        m: int,
        spos_fwd: npt.NDArray[np.floating],
        spos_bwd: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """段界约束 ``g_k = cap_k² − ‖ΔV_k‖²`` 的解析雅可比（原始量纲，(n, 3n)）。

        链式拆解（``scale = Δt/1000`` 把 N/kg·s 化为 km/s）::

            ∂cap_k/∂ΔV_j = scale·[ T'(r̄ₖ)·(r̂ₖ·Spos_k[:,j])/m̄ₖ
                                   + T(r̄ₖ)·û_j·[j<k]/(c·m̄ₖ) ]
            ∂g_k/∂ΔV_j   = 2·cap_k·∂cap_k/∂ΔV_j − 2·ΔV_k·δ_{kj}

        两支的支集不同，统一全 j 循环、由零支集自然截断：``m̄ₖ`` 链只在
        ``j < k`` 非零（``m̄ₖ = m₀·exp(−Σ_{j<k}‖ΔVⱼ‖/c)``，两 pass 同式）；
        ``T'`` 链的支集由 ``Spos_k``（该段**中点**位置灵敏度，段 k < m 取
        前向 pass、否则取后向 pass）决定——前向段支集 j < k、后向段支集
        j > k（后向中点跟随更晚段冲量的反向传播）。``û_j = ΔV_j/|ΔV_j|``
        在 |ΔV_j| ≤ 阈值时取零向量（雅可比连续化）。
        """
        n = dv.shape[0]
        jac = np.zeros((n, 3 * n))
        scale = dt / 1000.0
        for k in range(n):
            cap_k = cap[k]
            m_bar_k = m_bar[k]
            thrust_k = self._propulsion.max_thrust_n(r_mid[k])
            grad_t = self._propulsion._thrust_gradient_n_per_km(r_mid[k])
            spos_k = spos_fwd[k] if k < m else spos_bwd[k - m]
            rhat_k = mid_pos[k] / r_mid[k]
            dcap = np.zeros(3 * n)
            for j in range(n):
                block = slice(3 * j, 3 * j + 3)
                if grad_t != 0.0:
                    dcap[block] += scale * grad_t * (rhat_k @ spos_k[:, block]) / m_bar_k
                if j < k and dv_norm[j] > _UHAT_EPS_KM_S:
                    dcap[block] += scale * thrust_k * (dv[j] / dv_norm[j]) / (c_kms * m_bar_k)
            jac[k] = 2.0 * cap_k * dcap
            jac[k, 3 * k : 3 * k + 3] -= 2.0 * dv[k]
        return jac


# ---- 多 leg 链（rendezvous / flyby 节点，issue #741）----

#: flyby 节点 v∞ 决策变量的默认初猜幅值（km/s）：非零值规避 |v∞| = 0 处等模
#: 等式约束梯度（2v/vel² → 0）退化。
_VINF_GUESS_KM_S = 1e-3

#: 多 leg 成本函数名集合（定义见模块 docstring 与 ``solve`` docstring）。
_ML_COSTS = ("min_fuel", "min_time", "weighted", "gtoc")


@dataclass(frozen=True)
class SimsFlanaganNode:
    """多 leg 链的节点（conic 档语义：节点状态是**数据**，不随 TOF 决策变量变化）。

    节点历元由调用方保证与 ``state`` 一致（conic 档不查星历）。

    Attributes:
        kind: ``"rendezvous"``（位置+速度匹配）或 ``"flyby"``（位置匹配 +
            V∞ 旋转：等模等式约束 + 转角不等式约束）。flyby 只用于中间节点，
            末节点必须 rendezvous。
        state: 节点天体/目标 ``(6,)`` 日心状态 ``[r, v]``，km / km/s。
        mu_km3_s2: flyby 天体引力常数（km³/s²），flyby 必填、rendezvous 须 None。
        r_p_min_km: flyby 最小近心点半径（km），flyby 必填、rendezvous 须 None。

    Raises:
        ValueError: kind 非法、state 非 ``(6,)`` 有限、flyby 缺 ``mu_km3_s2`` /
            ``r_p_min_km`` 或非正、rendezvous 携带任一参数。
    """

    kind: str
    state: npt.ArrayLike
    mu_km3_s2: float | None = None
    r_p_min_km: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("rendezvous", "flyby"):
            raise ValueError(f"kind 必须为 'rendezvous' 或 'flyby'，得到 {self.kind!r}")
        arr = np.asarray(self.state, dtype=float)
        if arr.shape != (6,):
            raise ValueError(f"state 必须为 (6,) 状态向量，得到 shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("state 含非有限分量")
        object.__setattr__(self, "state", arr.copy())
        if self.kind == "flyby":
            for name, value in (("mu_km3_s2", self.mu_km3_s2), ("r_p_min_km", self.r_p_min_km)):
                if value is None:
                    raise ValueError(f"flyby 节点必须提供 {name}")
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError(f"flyby 节点 {name} 必须为正的有限数，得到 {value!r}")
        elif self.mu_km3_s2 is not None or self.r_p_min_km is not None:
            raise ValueError("rendezvous 节点不接受 mu_km3_s2/r_p_min_km（须为 None）")


@dataclass
class SimsFlanaganLegSolution:
    """多 leg 链中单 leg 的解剖面（字段语义与 :class:`SimsFlanaganSolution` 同构）。"""

    impulses_km_s: npt.NDArray[np.floating]
    impulse_times_s: npt.NDArray[np.floating]
    node_times_s: npt.NDArray[np.floating]
    forward_states: npt.NDArray[np.floating]
    backward_states: npt.NDArray[np.floating]
    matchpoint_residual: npt.NDArray[np.floating]


@dataclass
class SimsFlanaganFlybyResult:
    """flyby 节点的解报告（V∞ 等模约束下的进入/离开渐近线速度）。

    Attributes:
        v_inf_in_km_s: 进入 V∞ 向量 ``(3,)``（km/s）。
        v_inf_out_km_s: 离开 V∞ 向量 ``(3,)``（km/s）。
        v_inf_km_s: V∞ 幅值（km/s），取进入模（等模约束下与离开一致）。
        turn_angle_rad: 实现转角 δ（弧度）；``δ = 0`` 表示 V∞ 方向不变。
        pericenter_radius_km: 由 ``(δ, v_eff)`` 反解的近心点半径（km，
            :func:`.mga.flyby_pericenter_radius`）；``δ = 0`` 时为 ``inf``
            （零退化极限：任意远近心点都给零转角）。
    """

    v_inf_in_km_s: npt.NDArray[np.floating]
    v_inf_out_km_s: npt.NDArray[np.floating]
    v_inf_km_s: float
    turn_angle_rad: float
    pericenter_radius_km: float


@dataclass
class SimsFlanaganMultiLegSolution:
    """多 leg 链求解结果（软失败三元组随解携带，不抛异常）。

    Attributes:
        legs: 逐 leg 解剖面（长度 = 节点数）。
        flybys: 逐 flyby 节点的报告（仅 flyby 节点，按节点序）。
        leg_tofs_s: 优化后逐 leg 飞行时间 ``(L,)``（s）。
        delta_v_total_km_s: 全部段 ``Σ‖ΔVᵢ‖``（km/s）。
        final_mass_kg: 末态质量（kg），``m₀·exp(−Σ‖ΔVᵢ‖/c)``。
        fuel_kg: 燃料消耗（kg），``m₀ − final_mass_kg``。
        objective_value: 求解所用成本函数在解处的值。
        cost: 求解所用成本函数名。
        n_iter: SLSQP 迭代次数。
        status: 算法最终状态。
        cause: 算法最终原因码。
        message: 求解器状态消息（含后验闸门结论）。
    """

    legs: list[SimsFlanaganLegSolution]
    flybys: list[SimsFlanaganFlybyResult]
    leg_tofs_s: npt.NDArray[np.floating]
    delta_v_total_km_s: float
    final_mass_kg: float
    fuel_kg: float
    objective_value: float
    cost: str
    n_iter: int
    status: ConvergenceState
    cause: FailureCause
    message: str

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)


@dataclass(frozen=True)
class _FlybyEval:
    """单 flyby 节点约束的全部评估产物（值 + 相对 [v_in; v_out] 的局部梯度）。"""

    v_in: npt.NDArray[np.floating]
    v_out: npt.NDArray[np.floating]
    n_in: float
    n_out: float
    delta: float
    delta_max: float
    v_eff: float
    eq_val: float
    eq_grad: npt.NDArray[np.floating]
    ineq_val: float
    ineq_grad: npt.NDArray[np.floating]


@dataclass(frozen=True)
class _MultiLegLayout:
    """一次 ``solve`` 调用的决策变量布局与成本配置。"""

    n_list: tuple[int, ...]
    imp_offsets: tuple[int, ...]
    n_imp: int
    flyby_nodes: tuple[int, ...]
    vinf_off: int
    tof_off: int
    tof_free: bool
    n_var: int
    cost: str
    weights: tuple[float, float] | None


@dataclass
class _MultiLegEvaluation:
    """多 leg 决策向量单次评估的全部产物（``solve`` 内部缓存用）。"""

    tofs: npt.NDArray[np.floating]
    dv_list: list[npt.NDArray[np.floating]]
    objective: float
    obj_grad: npt.NDArray[np.floating] | None
    eq: npt.NDArray[np.floating]
    eq_jac: npt.NDArray[np.floating] | None
    ineq: npt.NDArray[np.floating]
    ineq_jac: npt.NDArray[np.floating] | None
    fwd_states: list[npt.NDArray[np.floating]]
    bwd_states: list[npt.NDArray[np.floating]]
    eq_raw: list[npt.NDArray[np.floating]]
    flybys: list[_FlybyEval]
    cap_km_s: npt.NDArray[np.floating]
    dv_norms: npt.NDArray[np.floating]
    total_dv_km_s: float
    final_mass_kg: float


class SimsFlanaganMultiLegProblem:
    """多 leg 链（rendezvous / flyby 节点）的 Sims-Flanagan 预设计问题（双档）。

    链结构：固定出发状态 → ``nodes[0]`` → … → ``nodes[-1]``，共 ``L = len(nodes)``
    个 leg。节点分两类（:class:`SimsFlanaganNode`）：

    - **rendezvous**：位置+速度匹配——leg 两 pass 锚定到固定节点状态，位置
      连续由锚定结构性满足，速度连续进入该 leg 的 6 维匹配点等式约束；
    - **flyby**：位置匹配 + V∞ 旋转——flyby 之后的 leg 前向锚定
      ``[r_B; v_B + v∞_out]``、之前的 leg 后向锚定 ``[r_B; v_B + v∞_in]``
      （v∞ 是决策变量），等模 ``|v∞_in| = |v∞_out|`` 为等式约束，转角
      ``δ ≤ δ_max(r_p_min, v_eff, μ_B)`` 为不等式约束（闭式复用
      :func:`.mga.flyby_turn_angle` / :func:`.mga.flyby_pericenter_radius`）。

    质量 ``m̄ₖ = m₀·exp(−Σ_{j<k}‖ΔVⱼ‖/c)`` 按全部 leg 段的**全局时序**递推
    （flyby 不改质量）。两档语义下节点状态都是数据，不随 TOF 决策变量变化；
    ephemeris 档 leg 的起始绝对历元取出发历元加前序 leg 的决策 TOF
    前缀和（保证 leg 时序连续），``min_time``/``weighted`` 下 TOF 决策变量
    移动 leg 时序时节点状态**不**随历元重查星历（逐 leg 重定目标归 #726 裁决）。

    Args:
        departure_state: 出发状态 ``[r, v]`` ``(6,)``，km / km/s。
        nodes: 节点序列（非空；末节点必须为 rendezvous；flyby 只用于中间节点）。
        leg_tofs_s: 逐 leg 名义飞行时间 ``(L,)``（s），正有限。
        propulsion: 推进配置（:class:`SimsFlanaganPropulsion`）。
        initial_mass_kg: 出发质量（kg），必须为正。
        mu_km3_s2: 中心天体引力常数（km³/s²），必须为正。
        backend: 保真度档位，必填关键字：``"conic"``（二体封闭解）或
            ``"ephemeris"``（星历 N 体数值传播，#727）；其余取值报错。
        epoch_et_s: ephemeris 档必填：出发历元 SPICE et 秒（TDB past
            J2000）；conic 档必须为 None。
        ephemeris_system: ephemeris 档必填：``EphemerisSystem`` 实例
            （``bodies`` 须含 ``origin``，中心引力项由 origin 提供）；conic
            档必须为 None。
    """

    def __init__(
        self,
        departure_state: npt.ArrayLike,
        nodes: Sequence[SimsFlanaganNode],
        leg_tofs_s: npt.ArrayLike,
        propulsion: SimsFlanaganPropulsion,
        initial_mass_kg: float,
        mu_km3_s2: float,
        *,
        backend: str,
        epoch_et_s: float | None = None,
        ephemeris_system: EphemerisSystem | None = None,
    ) -> None:
        departure = np.asarray(departure_state, dtype=float)
        if departure.shape != (6,):
            raise ValueError(f"departure_state 必须为 (6,) 状态向量，得到 shape {departure.shape}")
        if not np.all(np.isfinite(departure)):
            raise ValueError("departure_state 含非有限分量")
        node_list = list(nodes)
        if not node_list:
            raise ValueError("nodes 不能为空")
        for i, node in enumerate(node_list):
            if not isinstance(node, SimsFlanaganNode):
                raise ValueError(f"nodes[{i}] 必须为 SimsFlanaganNode，得到 {type(node).__name__}")
        if node_list[-1].kind != "rendezvous":
            raise ValueError("末节点 kind 必须为 'rendezvous'（不支持 flyby 型终点）")
        tofs = np.asarray(leg_tofs_s, dtype=float)
        if tofs.shape != (len(node_list),):
            raise ValueError(
                f"leg_tofs_s 必须为长度 {len(node_list)} 的正有限序列，得到 shape {tofs.shape}"
            )
        if not np.all(np.isfinite(tofs)) or np.any(tofs <= 0.0):
            raise ValueError(f"leg_tofs_s 每项必须为正的有限数，得到 {leg_tofs_s!r}")
        m0 = float(initial_mass_kg)
        if not np.isfinite(m0) or m0 <= 0.0:
            raise ValueError(f"initial_mass_kg 必须为正的有限数，得到 {initial_mass_kg!r}")
        mu = float(mu_km3_s2)
        if not np.isfinite(mu) or mu <= 0.0:
            raise ValueError(f"mu_km3_s2 必须为正的有限数，得到 {mu_km3_s2!r}")
        propulsion.validate()
        if backend not in ("conic", "ephemeris"):
            raise ValueError(f"backend 必须为 'conic' 或 'ephemeris'，得到 {backend!r}")
        self._backend = backend
        self._epoch_et_s: float | None = None
        self._eph_dyn: EphemerisDynamics | None = None
        if backend == "ephemeris":
            if epoch_et_s is None:
                raise ValueError(
                    "backend='ephemeris' 必须提供 epoch_et_s（SPICE et 秒，TDB past J2000）"
                )
            if ephemeris_system is None:
                raise ValueError(
                    "backend='ephemeris' 必须提供 ephemeris_system（EphemerisSystem 实例）"
                )
            if not isinstance(ephemeris_system, EphemerisSystem):
                raise ValueError(
                    "ephemeris_system 必须为 EphemerisSystem 实例，得到 "
                    f"{type(ephemeris_system).__name__}"
                )
            if not np.isfinite(float(epoch_et_s)):
                raise ValueError(f"epoch_et_s 必须为有限数，得到 {epoch_et_s!r}")
            if ephemeris_system.origin not in ephemeris_system.bodies:
                raise ValueError(
                    f"ephemeris_system.bodies 须包含 origin 天体 {ephemeris_system.origin!r}"
                    "（中心引力项由 origin 提供）"
                )
            self._epoch_et_s = float(epoch_et_s)
            self._eph_dyn = EphemerisDynamics(system=ephemeris_system)
            # 60 s 上限是 LEO 短弧调优值；置 inf 后经 _get_max_step 变为逐半段
            # span/10，避免日量级段被强加 60 s 步长（性能）。容差保持默认 1e-12。
            self._eph_dyn.max_step = float("inf")
        elif epoch_et_s is not None or ephemeris_system is not None:
            raise ValueError(
                "backend='conic' 不接受 epoch_et_s/ephemeris_system"
                "（星历上下文仅 ephemeris 档使用）"
            )

        self._departure = departure.copy()
        self._nodes = tuple(node_list)
        self._tof_nominal = tofs.copy()
        self._propulsion = propulsion
        self._m0_kg = m0
        self._mu_km3_s2 = mu

        # 约束归一化尺度：departure 与**全部**节点状态的幅值（退化取 1，避免除零）。
        all_states = [departure] + [np.asarray(nd.state, dtype=float) for nd in node_list]
        self._len_scale = max(max(float(np.linalg.norm(s[:3])) for s in all_states), 1.0)
        self._vel_scale = max(max(float(np.linalg.norm(s[3:])) for s in all_states), 1.0)

    # ---- 公开入口 ----

    def solve(
        self,
        n_segments: int | Sequence[int],
        *,
        cost: str = "min_fuel",
        cost_weights: Sequence[float] | None = None,
        tof_bounds_s: npt.ArrayLike | None = None,
        x0: npt.ArrayLike | None = None,
        guess: str | None = None,
        ftol: float = 1e-9,
        maxiter: int = 200,
        use_analytic_jac: bool = True,
        verbose: bool = False,
    ) -> SimsFlanaganMultiLegSolution:
        """求解多 leg NLP（软失败，不抛异常）。

        Args:
            n_segments: int 均匀段数（每 leg ``n ≥ 2``）或逐 leg 序列（长度
                = 节点数，每项 ≥ 2）。
            cost: 成本函数（定义见模块 docstring）：``"min_fuel"``（默认）、
                ``"min_time"``、``"weighted"``、``"gtoc"``。
            cost_weights: 仅 ``cost="weighted"`` 使用：``(w_dv, w_t)`` 两正数；
                其余 cost 下必须为 None。
            tof_bounds_s: None（TOF 固定为名义值；min_time/weighted 不合法）
                或 ``(lb, ub)`` 共用对或逐 leg ``(L, 2)``（``lb > 0``、
                ``lb ≤ ub``）；给定则逐 leg TOF 进入决策变量（min_time/
                weighted 必须给定）。
            x0: 显式初猜 ``(n_var,)`` 扁平决策向量（布局见下）。
            guess: ``"edelbaum"`` 逐 leg 闭式初猜；``None`` 冲量全零。
                v∞ 初猜恒为 ``1e-3 km/s × v̂_body``（in/out 同向，规避等模
                约束在 v∞=0 处梯度退化），TOF 初猜为名义值。
            ftol: SLSQP 目标容差。
            maxiter: SLSQP 最大迭代次数。
            use_analytic_jac: True（默认）解析雅可比；False 退 SLSQP 数值差分。
            verbose: 是否打印 SLSQP 迭代信息。

        决策变量布局（``x`` 的拼接次序）::

            [ΔV leg0 (3·n₀), ΔV leg1 (3·n₁), …,
             逐 flyby 节点（按节点序）: v∞_in (3), v∞_out (3), …,
             逐 leg TOF (L，仅 tof_bounds_s 给定时)]

        Returns:
            :class:`SimsFlanaganMultiLegSolution`；收敛失败以 SLSQP 结束码翻译
            与后验闸门改判的 ``(status, cause, message)`` 三元组表达。
        """
        require_rust_extension("propagate_kepler_py")
        n_legs = len(self._nodes)
        n_list = self._resolve_n_segments(n_segments, n_legs)

        if cost not in _ML_COSTS:
            raise ValueError(f"cost 必须为 {'/'.join(_ML_COSTS)} 之一，得到 {cost!r}")
        weights: tuple[float, float] | None = None
        if cost == "weighted":
            if cost_weights is None:
                raise ValueError("cost='weighted' 必须提供 cost_weights=(w_dv, w_t)")
            w_arr = np.asarray(cost_weights, dtype=float)
            if w_arr.shape != (2,):
                raise ValueError(
                    f"cost_weights 必须为 (w_dv, w_t) 两正数，得到 shape {w_arr.shape}"
                )
            if not np.all(np.isfinite(w_arr)) or np.any(w_arr <= 0.0):
                raise ValueError(f"cost_weights 两项必须为正的有限数，得到 {cost_weights!r}")
            weights = (float(w_arr[0]), float(w_arr[1]))
        elif cost_weights is not None:
            raise ValueError(f"cost_weights 仅在 cost='weighted' 时接受，cost={cost!r} 下须为 None")

        tof_pairs: npt.NDArray[np.floating] | None = None
        if tof_bounds_s is None:
            if cost in ("min_time", "weighted"):
                raise ValueError(f"cost={cost!r} 必须提供 tof_bounds_s（TOF 为决策变量）")
        else:
            b_arr = np.asarray(tof_bounds_s, dtype=float)
            if b_arr.shape == (2,):
                b_arr = np.tile(b_arr, (n_legs, 1))
            if b_arr.shape != (n_legs, 2):
                raise ValueError(
                    f"tof_bounds_s 必须为 (lb, ub) 共用对或逐 leg "
                    f"({n_legs}, 2)，得到 shape {b_arr.shape}"
                )
            if not np.all(np.isfinite(b_arr)):
                raise ValueError("tof_bounds_s 含非有限分量")
            if np.any(b_arr[:, 0] <= 0.0):
                raise ValueError("tof_bounds_s 下界必须为正")
            if np.any(b_arr[:, 1] < b_arr[:, 0]):
                raise ValueError("tof_bounds_s 上界不得小于下界")
            tof_pairs = b_arr

        layout = self._make_layout(n_list, cost, weights, tof_free=tof_pairs is not None)
        n_var = layout.n_var

        x_start = self._ml_initial_guess(layout, x0=x0, guess=guess)

        # 冲量变量盒：MVP 同款慷慨盒（按各 leg 名义 dt 算 cap_box）；v∞ 变量
        # 不设盒（±inf）；TOF 变量取调用方界。
        lb = np.full(n_var, -np.inf)
        ub = np.full(n_var, np.inf)
        for leg in range(n_legs):
            n = n_list[leg]
            off = layout.imp_offsets[leg]
            state_f = (
                self._departure if leg == 0 else np.asarray(self._nodes[leg - 1].state, dtype=float)
            )
            state_b = np.asarray(self._nodes[leg].state, dtype=float)
            v_cap = 2.0 * max(
                float(np.linalg.norm(state_f[3:])), float(np.linalg.norm(state_b[3:]))
            )
            r_ref = min(float(np.linalg.norm(state_f[:3])), float(np.linalg.norm(state_b[:3])))
            t_ref_leg = self._propulsion.max_thrust_n(r_ref)
            cap_box = (
                _CAP_BOX_FACTOR * t_ref_leg / self._m0_kg * (self._tof_nominal[leg] / n) / 1000.0
            )
            dv_cap = min(v_cap, cap_box) if cap_box > 0.0 else v_cap
            lb[off : off + 3 * n] = -dv_cap
            ub[off : off + 3 * n] = dv_cap
        if layout.tof_free:
            assert tof_pairs is not None
            lb[layout.tof_off :] = tof_pairs[:, 0]
            ub[layout.tof_off :] = tof_pairs[:, 1]
        bounds = Bounds(lb, ub)

        want_sens = bool(use_analytic_jac)
        cache: dict[bytes, _MultiLegEvaluation] = {}

        def evaluated(x_flat: npt.NDArray[np.floating]) -> _MultiLegEvaluation:
            key = np.ascontiguousarray(x_flat, dtype=float).tobytes()
            hit = cache.get(key)
            if hit is None:
                hit = self._evaluate(np.asarray(x_flat, dtype=float), layout, with_sens=want_sens)
                if len(cache) > 32:
                    cache.clear()
                cache[key] = hit
            return hit

        eq_dict: dict[str, object] = {"type": "eq", "fun": lambda x: evaluated(x).eq}
        ineq_dict: dict[str, object] = {
            "type": "ineq",
            "fun": lambda x: evaluated(x).ineq,
        }
        if want_sens:
            eq_dict["jac"] = lambda x: evaluated(x).eq_jac
            ineq_dict["jac"] = lambda x: evaluated(x).ineq_jac

        result = minimize(
            lambda x: evaluated(x).objective,
            x_start,
            method="SLSQP",
            jac=(lambda x: evaluated(x).obj_grad) if want_sens else None,
            bounds=bounds,
            constraints=[eq_dict, ineq_dict],
            options={"ftol": ftol, "maxiter": maxiter, "disp": verbose},
        )

        status, cause = scipy_slsqp_status(bool(result.success), int(result.status))
        message = str(result.message)
        final = evaluated(result.x)

        # 后验闸门（成功也要过，ADR 0020 红线）：全 leg 归一化匹配点残差、
        # 全段 |ΔV|/cap、flyby 等模与转角裕度，任一超限改判不可行。
        eq_max = float(np.max(np.abs(final.eq[: 6 * n_legs])))
        cap_safe = np.where(final.cap_km_s > 0.0, final.cap_km_s, 1.0)
        over_max = float(np.max(final.dv_norms / cap_safe))
        fly_eq_max = max(
            (abs(fe.n_in - fe.n_out) / self._vel_scale for fe in final.flybys),
            default=0.0,
        )
        fly_ineq_max = max(
            ((fe.delta - fe.delta_max) / float(np.pi) for fe in final.flybys),
            default=0.0,
        )
        if (
            eq_max > _POST_EQ_TOL
            or over_max > 1.0 + _POST_INEQ_RATIO_TOL
            or fly_eq_max > _POST_EQ_TOL
            or fly_ineq_max > _POST_INEQ_RATIO_TOL
        ):
            raw_max = max((float(np.max(np.abs(r))) for r in final.eq_raw), default=0.0)
            message = (
                f"{message}；后验闸门未过：匹配点残差 max|c| = {eq_max:.3e}"
                f"（原始 {raw_max:.3e}，阈值 {_POST_EQ_TOL}），"
                f"段界相对越界 max|ΔV|/cap = {over_max:.12f}"
                f"（阈值 1 + {_POST_INEQ_RATIO_TOL}），"
                f"flyby 等模 max‖Δ|v∞|‖/vel = {fly_eq_max:.3e}，"
                f"flyby 转角 max(δ−δ_max)/π = {fly_ineq_max:.3e}"
            )
            status = ConvergenceState.INFEASIBLE
            cause = FailureCause.CONSTRAINT_VIOLATION

        legs: list[SimsFlanaganLegSolution] = []
        for leg in range(n_legs):
            n = n_list[leg]
            dt = float(final.tofs[leg]) / n
            legs.append(
                SimsFlanaganLegSolution(
                    impulses_km_s=final.dv_list[leg].copy(),
                    impulse_times_s=(np.arange(n) + 0.5) * dt,
                    node_times_s=np.arange(n + 1) * dt,
                    forward_states=final.fwd_states[leg].copy(),
                    backward_states=final.bwd_states[leg].copy(),
                    matchpoint_residual=final.eq_raw[leg].copy(),
                )
            )
        flyby_results: list[SimsFlanaganFlybyResult] = []
        for node_idx, fe in zip(layout.flyby_nodes, final.flybys, strict=True):
            node = self._nodes[node_idx]
            assert node.mu_km3_s2 is not None
            if fe.v_eff <= 0.0:
                r_p_km = float(np.inf)
            elif fe.delta >= float(np.pi):
                r_p_km = 0.0  # δ=π 的零退化极限（反平行 V∞ ⇔ r_p=0）
            else:
                r_p_km = float(flyby_pericenter_radius(fe.delta, fe.v_eff, node.mu_km3_s2))
            flyby_results.append(
                SimsFlanaganFlybyResult(
                    v_inf_in_km_s=fe.v_in.copy(),
                    v_inf_out_km_s=fe.v_out.copy(),
                    v_inf_km_s=fe.n_in,
                    turn_angle_rad=fe.delta,
                    pericenter_radius_km=r_p_km,
                )
            )

        return SimsFlanaganMultiLegSolution(
            legs=legs,
            flybys=flyby_results,
            leg_tofs_s=final.tofs.copy(),
            delta_v_total_km_s=final.total_dv_km_s,
            final_mass_kg=final.final_mass_kg,
            fuel_kg=self._m0_kg - final.final_mass_kg,
            objective_value=final.objective,
            cost=cost,
            n_iter=int(result.nit),
            status=status,
            cause=cause,
            message=message,
        )

    def _make_layout(
        self,
        n_list: Sequence[int],
        cost: str,
        weights: tuple[float, float] | None,
        *,
        tof_free: bool,
    ) -> _MultiLegLayout:
        """由（已校验的）段数、成本与 TOF 模式构建决策变量布局。"""
        n_list_t = tuple(int(n) for n in n_list)
        imp_offsets_list = [0]
        for n in n_list_t[:-1]:
            imp_offsets_list.append(imp_offsets_list[-1] + 3 * n)
        flyby_nodes = tuple(i for i, nd in enumerate(self._nodes) if nd.kind == "flyby")
        n_imp = 3 * sum(n_list_t)
        vinf_off = n_imp
        tof_off = n_imp + 6 * len(flyby_nodes)
        n_legs = len(n_list_t)
        return _MultiLegLayout(
            n_list=n_list_t,
            imp_offsets=tuple(imp_offsets_list),
            n_imp=n_imp,
            flyby_nodes=flyby_nodes,
            vinf_off=vinf_off,
            tof_off=tof_off,
            tof_free=tof_free,
            n_var=tof_off + (n_legs if tof_free else 0),
            cost=cost,
            weights=weights,
        )

    # ---- 内部：入参归一与初猜 ----

    @staticmethod
    def _resolve_n_segments(n_segments: int | Sequence[int], n_legs: int) -> list[int]:
        """段数归一为逐 leg 列表（int 均匀；序列每个 ≥ 2）。"""
        if isinstance(n_segments, (str, bytes)):
            raise ValueError(f"n_segments 必须为 ≥ 2 的整数或其序列，得到 {n_segments!r}")
        if isinstance(n_segments, (int, np.integer)):
            # bool 会因 n < 2 被拒，无需单列分支。
            values = [float(n_segments)] * n_legs
        else:
            values = [float(v) for v in n_segments]
            if len(values) != n_legs:
                raise ValueError(f"n_segments 序列长度必须等于节点数 {n_legs}，得到 {len(values)}")
        n_list = []
        for value in values:
            n = int(value)
            if n != value or n < 2:
                raise ValueError(f"n_segments 每项必须为 ≥ 2 的整数，得到 {value!r}")
            n_list.append(n)
        return n_list

    def _default_vinf(self, node_idx: int) -> npt.NDArray[np.floating]:
        """flyby 节点 v∞ 默认初猜向量：``1e-3 km/s × v̂_body``（in/out 同向）。"""
        v_body = np.asarray(self._nodes[node_idx].state, dtype=float)[3:]
        mag = float(np.linalg.norm(v_body))
        if mag < _UHAT_EPS_KM_S:
            return np.zeros(3)
        return v_body / mag * _VINF_GUESS_KM_S

    def _ml_initial_guess(
        self, layout: _MultiLegLayout, *, x0: npt.ArrayLike | None, guess: str | None
    ) -> npt.NDArray[np.floating]:
        """初猜决策向量：显式 x0 优先；否则逐 leg Edelbaum/全零 + v∞ 默认初猜。"""
        if x0 is not None:
            arr = np.asarray(x0, dtype=float)
            if arr.shape != (layout.n_var,):
                raise ValueError(f"x0 必须为 ({layout.n_var},) 决策向量，得到 shape {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError("x0 含非有限分量")
            return arr.copy()
        if guess not in (None, "edelbaum"):
            raise ValueError(f"guess 必须为 'edelbaum' 或 None，得到 {guess!r}")
        x = np.zeros(layout.n_var)
        for leg, n in enumerate(layout.n_list):
            if guess != "edelbaum":
                continue
            # flyby 端的锚定速度含默认 v∞（与决策变量初猜一致）。
            fwd = (
                self._departure.copy()
                if leg == 0
                else np.asarray(self._nodes[leg - 1].state, dtype=float).copy()
            )
            bwd = np.asarray(self._nodes[leg].state, dtype=float).copy()
            if leg > 0 and self._nodes[leg - 1].kind == "flyby":
                fwd[3:] += self._default_vinf(leg - 1)
            if self._nodes[leg].kind == "flyby":
                bwd[3:] += self._default_vinf(leg)
            off = layout.imp_offsets[leg]
            x[off : off + 3 * n] = _edelbaum_leg_guess(fwd, bwd, n)
        for f, node_idx in enumerate(layout.flyby_nodes):
            v_def = self._default_vinf(node_idx)
            base = layout.vinf_off + 6 * f
            x[base : base + 3] = v_def
            x[base + 3 : base + 6] = v_def
        if layout.tof_free:
            x[layout.tof_off :] = self._tof_nominal
        return x

    # ---- 内部：评估 ----

    def _leg_pass(
        self,
        anchor_state: npt.NDArray[np.floating],
        anchor_sens: npt.NDArray[np.floating],
        dv: npt.NDArray[np.floating],
        dt: float,
        *,
        forward: bool,
        imp_offset: int,
        tof_col: int | None,
        n_var: int,
        with_sens: bool,
        t_start: float | None = None,
    ) -> tuple[
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
    ]:
        """单 leg 的前向/后向 pass（MVP pass 的推广：锚速度可含 v∞ 决策变量、
        段长 dt 可为 TOF 决策变量的函数）。

        ``forward=True`` 自前向锚接龙段 ``0..m−1``，否则自后向锚反向接龙段
        ``m..n−1``（内核支持负 dt）。``t_start`` 为本 leg **起始端**绝对历元
        （ephemeris 档必填：前向游标自它出发，后向游标自 ``t_start + n·dt``
        反向递减；conic 档忽略）。返回 ``(节点状态, 段中点位置, 段中点半径,
        匹配节点灵敏度 (6, n_var), 段中点位置灵敏度 (count, 3, n_var))``；后向
        的节点状态按 ``[匹配节点..锚]`` 排列、段序数组按段升序翻转（MVP 同语义）。
        锚速度灵敏度经 ``anchor_sens`` 进入链式递推；TOF 灵敏度在每次半段传播
        后累加 ``RHS(x_out)·(±1/(2n))``（后向带负号；RHS 按 backend 取二体
        闭式或星历 ``equations_of_motion``，在传播**输出**时刻/态取值）。
        """
        n = dv.shape[0]
        m = n // 2
        sign = 1.0 if forward else -1.0
        half = sign * 0.5 * dt
        x = anchor_state.copy()
        sens = anchor_sens.copy()
        seg_ids = range(m) if forward else range(n - 1, m - 1, -1)
        count = m if forward else n - m
        states = [x.copy()]
        mid_pos = np.zeros((count, 3))
        r_mid = np.zeros(count)
        spos = np.zeros((count, 3, n_var))
        t = None if t_start is None else (t_start if forward else t_start + n * dt)
        for idx, k in enumerate(seg_ids):
            t_mid = None if t is None else t + half
            x_mid, phi1 = _propagate_half(
                x, t, half, mu_km3_s2=self._mu_km3_s2, dyn=self._eph_dyn, with_stm=with_sens
            )
            mid_pos[idx] = x_mid[:3]
            r_mid[idx] = float(np.linalg.norm(x_mid[:3]))
            if with_sens:
                assert phi1 is not None  # with_stm=True 时 _propagate_half 恒返回 STM
                s_mid = phi1 @ sens
                if tof_col is not None:
                    # 半段时长 h = ±TOF/(2n)：∂x_out/∂TOF += RHS(x_out)·∂h/∂TOF。
                    s_mid[:, tof_col] += (
                        sign
                        * _tier_rhs(t_mid, x_mid, mu_km3_s2=self._mu_km3_s2, dyn=self._eph_dyn)
                        / (2.0 * n)
                    )
                spos[idx] = s_mid[:3, :]
                s_mid[:, imp_offset + 3 * k : imp_offset + 3 * k + 3] += sign * _B_IMPULSE
            x_mid = x_mid + sign * (_B_IMPULSE @ dv[k])
            t_out = None if t_mid is None else t_mid + half  # 传播输出时刻（与 x 同步）
            x, phi2 = _propagate_half(
                x_mid, t_mid, half, mu_km3_s2=self._mu_km3_s2, dyn=self._eph_dyn, with_stm=with_sens
            )
            if with_sens:
                assert phi2 is not None
                sens = phi2 @ s_mid
                if tof_col is not None:
                    sens[:, tof_col] += (
                        sign
                        * _tier_rhs(t_out, x, mu_km3_s2=self._mu_km3_s2, dyn=self._eph_dyn)
                        / (2.0 * n)
                    )
            states.append(x.copy())
            t = t_out
        if forward:
            return np.asarray(states), mid_pos, r_mid, sens, spos
        return np.asarray(states[::-1]), mid_pos[::-1], r_mid[::-1], sens, spos[::-1]

    def _flyby_constraints(
        self,
        v_in: npt.NDArray[np.floating],
        v_out: npt.NDArray[np.floating],
        node: SimsFlanaganNode,
    ) -> _FlybyEval:
        """flyby 节点的等模等式与转角不等式（值 + 相对 ``[v_in; v_out]`` 的梯度）。

        等模 ``(|v_in|² − |v_out|²)/vel_scale²``；转角 ``g = δ_max − δ``，
        ``δ = acos(clip(v̂_in·v̂_out))``，``δ_max`` 复用 :func:`.mga.flyby_turn_angle`
        在 ``v_eff = (|v_in|+|v_out|)/2`` 处闭式。退化守卫：任一模低于
        ``_UHAT_EPS_KM_S`` 时 δ 及其梯度取 0；``v_eff`` 低于阈值时 ``g ≡ 1.0``
        （常数可行、梯度 0）。``∂δ/∂v_in = −(v̂_out − cosδ·v̂_in)/(n_in·sinδ)``
        （``sinδ`` 低于阈值取 0）；``dδ_max/dv_eff = −4√k/((1+k·v²)·√(2+k·v²))``
        （``k = r_p_min/μ``，``1−w²`` 的稳定重排，``k → 0`` 时无 catastrophic
        cancellation），再链 ``∂v_eff/∂v_in = ½·v̂_in``。
        """
        eps = _UHAT_EPS_KM_S
        vel2 = self._vel_scale**2
        n_in = float(np.linalg.norm(v_in))
        n_out = float(np.linalg.norm(v_out))
        eq_val = (n_in**2 - n_out**2) / vel2
        eq_grad = np.concatenate([2.0 * v_in, -2.0 * v_out]) / vel2

        u_in: npt.NDArray[np.floating] = np.zeros(3)
        u_out: npt.NDArray[np.floating] = np.zeros(3)
        dd_in: npt.NDArray[np.floating] = np.zeros(3)
        dd_out: npt.NDArray[np.floating] = np.zeros(3)
        delta = 0.0
        if n_in >= eps and n_out >= eps:
            u_in = v_in / n_in
            u_out = v_out / n_out
            cos_d = float(np.clip(np.dot(u_in, u_out), -1.0, 1.0))
            delta = float(np.arccos(cos_d))
            sin_d = float(np.sin(delta))
            if sin_d >= eps:
                dd_in = -(u_out - cos_d * u_in) / (n_in * sin_d)
                dd_out = -(u_in - cos_d * u_out) / (n_out * sin_d)

        v_eff = 0.5 * (n_in + n_out)
        ineq_grad = np.zeros(6)
        delta_max = 0.0
        if v_eff <= eps:
            ineq_val = 1.0  # 常数可行（零退化极限），梯度 0
        else:
            assert node.r_p_min_km is not None and node.mu_km3_s2 is not None
            delta_max = flyby_turn_angle(node.r_p_min_km, v_eff, node.mu_km3_s2)
            ineq_val = delta_max - delta
            k = node.r_p_min_km / node.mu_km3_s2
            kv2 = k * v_eff**2
            dd_max = -4.0 * np.sqrt(k) / ((1.0 + kv2) * np.sqrt(2.0 + kv2))
            ineq_grad = np.concatenate(
                [0.5 * dd_max * u_in, 0.5 * dd_max * u_out]
            ) - np.concatenate([dd_in, dd_out])
        return _FlybyEval(
            v_in=np.asarray(v_in, dtype=float),
            v_out=np.asarray(v_out, dtype=float),
            n_in=n_in,
            n_out=n_out,
            delta=delta,
            delta_max=delta_max,
            v_eff=v_eff,
            eq_val=float(eq_val),
            eq_grad=eq_grad,
            ineq_val=float(ineq_val),
            ineq_grad=ineq_grad,
        )

    def _ml_objective(
        self,
        layout: _MultiLegLayout,
        dv_flat: npt.NDArray[np.floating],
        dv_norms: npt.NDArray[np.floating],
        leg_of_seg: npt.NDArray[np.integer],
        uhat: npt.NDArray[np.floating],
        tofs: npt.NDArray[np.floating],
        c_kms: float,
        *,
        with_grad: bool,
    ) -> tuple[float, npt.NDArray[np.floating] | None]:
        """成本函数值与解析梯度（定义见模块 docstring）。"""
        smooth = np.sqrt(dv_norms**2 + _EPS_KM_S**2)
        t_ref = float(np.sum(self._tof_nominal))
        grad = np.zeros(layout.n_var) if with_grad else None
        if layout.cost == "min_fuel":
            objective = float(np.sum(smooth))
            if grad is not None:
                grad[: layout.n_imp] = (dv_flat / smooth[:, None]).reshape(-1)
        elif layout.cost == "min_time":
            objective = float(np.sum(tofs) / t_ref)
            if grad is not None and layout.tof_free:
                grad[layout.tof_off :] = 1.0 / t_ref
        elif layout.cost == "weighted":
            assert layout.weights is not None
            w_dv, w_t = layout.weights
            objective = w_dv * float(np.sum(smooth)) / c_kms + w_t * float(np.sum(tofs)) / t_ref
            if grad is not None:
                grad[: layout.n_imp] = (w_dv * (dv_flat / smooth[:, None])).reshape(-1) / c_kms
                if layout.tof_free:
                    grad[layout.tof_off :] = w_t / t_ref
        else:  # gtoc：J = −Σ_{交会节点含终点} exp(−p_j/c)，p_j = 节点 j 前全段 Σ‖ΔV‖
            seg_offsets = np.concatenate([[0], np.cumsum(layout.n_list)[:-1]]).astype(int)
            leg_cum = np.cumsum(np.add.reduceat(dv_norms, seg_offsets))
            is_rdv = np.array([nd.kind == "rendezvous" for nd in self._nodes])
            objective = -float(np.sum(np.exp(-leg_cum[is_rdv] / c_kms)))
            if grad is not None:
                # coef[leg] = Σ_{j ≥ leg 且节点 j 交会} exp(−p_j/c)：leg 的每段
                # 只进入其后的交会节点质量项。
                exp_all = np.where(is_rdv, np.exp(-leg_cum / c_kms), 0.0)
                coef = np.cumsum(exp_all[::-1])[::-1]
                grad[: layout.n_imp] = (
                    coef[leg_of_seg.astype(int)][:, None] * uhat / c_kms
                ).reshape(-1)
        return objective, grad

    def _ml_ineq_jacobian(
        self,
        layout: _MultiLegLayout,
        dv_flat: npt.NDArray[np.floating],
        dv_norms: npt.NDArray[np.floating],
        cap: npt.NDArray[np.floating],
        r_mid: npt.NDArray[np.floating],
        mid_pos: npt.NDArray[np.floating],
        m_bar: npt.NDArray[np.floating],
        dt_seg: npt.NDArray[np.floating],
        leg_of_seg: npt.NDArray[np.integer],
        spos: npt.NDArray[np.floating],
        uhat: npt.NDArray[np.floating],
        c_kms: float,
        flyby_evals: list[_FlybyEval],
    ) -> npt.NDArray[np.floating]:
        """段界 + flyby 转角不等式约束的解析雅可比（原始量纲，(Σn+n_fly, n_var)）。

        段界行推广 MVP :meth:`SimsFlanaganProblem._ineq_jacobian`：质量链支集
        扩为**全局时序**（``j < s`` 跨 leg）；``T'(r)`` 链支集由 ``spos`` 决定
        （本 leg 冲量 + 本 leg 锚 v∞ 变量列 + 本 leg TOF 列，零支集自然截断）；
        另加直接项 ``∂cap_s/∂TOF_leg = T(r̄_s)/m̄_s/n_leg/1000``（段长随
        TOF 线性伸缩）::

            ∂cap_s/∂x = Δt_s/1000·[ T'(r̄_s)·(r̂_s·Spos_s)/m̄_s
                                     + T(r̄_s)·û_j/(c·m̄_s)（j < s，全局时序） ]
                        + T(r̄_s)/m̄_s/n_leg/1000·δ_{TOF_leg(s)}
            ∂g_s/∂x   = 2·cap_s·∂cap_s/∂x − 2·ΔV_s·δ_{own}
        """
        n_seg = dv_flat.shape[0]
        jac = np.zeros((n_seg + len(layout.flyby_nodes), layout.n_var))
        for s in range(n_seg):
            leg = int(leg_of_seg[s])
            scale = float(dt_seg[s]) / 1000.0
            thrust_s = self._propulsion.max_thrust_n(float(r_mid[s]))
            grad_t = self._propulsion._thrust_gradient_n_per_km(float(r_mid[s]))
            m_bar_s = float(m_bar[s])
            dcap = np.zeros(layout.n_var)
            if grad_t != 0.0:
                rhat = mid_pos[s] / float(r_mid[s])
                dcap += scale * grad_t * (rhat @ spos[s]) / m_bar_s
            # 质量链（全局时序 j < s；全局段序与冲量列序一致，前缀切片即可）。
            dcap[: 3 * s] += (scale * thrust_s / (c_kms * m_bar_s) * uhat[:s]).reshape(-1)
            if layout.tof_free:
                dcap[layout.tof_off + leg] += thrust_s / m_bar_s / layout.n_list[leg] / 1000.0
            jac[s] = 2.0 * float(cap[s]) * dcap
            jac[s, 3 * s : 3 * s + 3] -= 2.0 * dv_flat[s]
        for f, fe in enumerate(flyby_evals):
            base = layout.vinf_off + 6 * f
            jac[n_seg + f, base : base + 6] = fe.ineq_grad
        return jac

    def _ml_penalty_evaluation(
        self,
        x: npt.NDArray[np.floating],
        layout: _MultiLegLayout,
        dv_list: list[npt.NDArray[np.floating]],
        tofs: npt.NDArray[np.floating],
        dv_norms: npt.NDArray[np.floating],
        *,
        with_sens: bool,
    ) -> _MultiLegEvaluation:
        """传播失败试探点的深罚评估（同 MVP：目标罚大数、约束给深罚值）。"""
        n_legs = len(layout.n_list)
        n_fly = len(layout.flyby_nodes)
        n_seg = dv_norms.shape[0]
        c_kms = self._propulsion.isp_s * G0_MPS2 / 1000.0
        total_dv = float(np.sum(dv_norms))
        flybys = []
        for f in range(n_fly):
            base = layout.vinf_off + 6 * f
            v_in = np.asarray(x[base : base + 3], dtype=float)
            v_out = np.asarray(x[base + 3 : base + 6], dtype=float)
            flybys.append(
                _FlybyEval(
                    v_in=v_in,
                    v_out=v_out,
                    n_in=float(np.linalg.norm(v_in)),
                    n_out=float(np.linalg.norm(v_out)),
                    delta=0.0,
                    delta_max=0.0,
                    v_eff=0.0,
                    eq_val=_PENALTY_EQ,
                    eq_grad=np.zeros(6),
                    ineq_val=_PENALTY_INEQ,
                    ineq_grad=np.zeros(6),
                )
            )
        return _MultiLegEvaluation(
            tofs=tofs.copy(),
            dv_list=[d.copy() for d in dv_list],
            objective=_PENALTY_OBJECTIVE,
            obj_grad=np.zeros(layout.n_var) if with_sens else None,
            eq=np.full(6 * n_legs + n_fly, _PENALTY_EQ),
            eq_jac=np.zeros((6 * n_legs + n_fly, layout.n_var)) if with_sens else None,
            ineq=np.full(n_seg + n_fly, _PENALTY_INEQ),
            ineq_jac=np.zeros((n_seg + n_fly, layout.n_var)) if with_sens else None,
            fwd_states=[np.zeros((n // 2 + 1, 6)) for n in layout.n_list],
            bwd_states=[np.zeros((n - n // 2 + 1, 6)) for n in layout.n_list],
            eq_raw=[np.zeros(6) for _ in range(n_legs)],
            flybys=flybys,
            cap_km_s=np.zeros(n_seg),
            dv_norms=dv_norms.copy(),
            total_dv_km_s=total_dv,
            final_mass_kg=self._m0_kg * float(np.exp(-total_dv / c_kms)),
        )

    def _evaluate(
        self,
        x: npt.NDArray[np.floating],
        layout: _MultiLegLayout,
        *,
        with_sens: bool,
    ) -> _MultiLegEvaluation:
        """多 leg 决策向量 → 目标/约束/雅可比/状态剖面的统一评估。

        等式约束 = 每 leg 6 维归一化匹配点残差 + 每 flyby 1 维等模；不等式 =
        全部段（全局时序）``g_s = cap_s² − ‖ΔV_s‖²`` + 每 flyby 1 维转角裕度。
        """
        n_list = layout.n_list
        n_var = layout.n_var
        n_legs = len(n_list)
        c_kms = self._propulsion.isp_s * G0_MPS2 / 1000.0

        # -- 解包决策向量 --
        dv_list: list[npt.NDArray[np.floating]] = []
        for leg, n in enumerate(n_list):
            off = layout.imp_offsets[leg]
            dv_list.append(np.asarray(x[off : off + 3 * n], dtype=float).reshape(n, 3))
        tofs = (
            np.asarray(x[layout.tof_off : layout.tof_off + n_legs], dtype=float).copy()
            if layout.tof_free
            else self._tof_nominal.copy()
        )

        # ephemeris 档的逐 leg 起始绝对历元（出发历元 + 前序 leg 决策 TOF
        # 前缀和，保证 leg 时序连续；conic 档为 None，逐 leg 传 None 即可）。
        leg_t0 = (
            self._epoch_et_s + np.concatenate([[0.0], np.cumsum(tofs)[:-1]])
            if self._backend == "ephemeris"
            else None
        )
        vinf_cols = {
            node_idx: (layout.vinf_off + 6 * f, layout.vinf_off + 6 * f + 3)
            for f, node_idx in enumerate(layout.flyby_nodes)
        }
        vinf: dict[int, tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]] = {}
        for node_idx, (in_off, _) in vinf_cols.items():
            vinf[node_idx] = (
                np.asarray(x[in_off : in_off + 3], dtype=float),
                np.asarray(x[in_off + 3 : in_off + 6], dtype=float),
            )

        # -- 质量链（全局时序：flyby 不改质量）--
        dv_flat = np.vstack(dv_list)
        dv_norms = np.linalg.norm(dv_flat, axis=1)
        prefix = np.concatenate([[0.0], np.cumsum(dv_norms)[:-1]])
        m_bar = self._m0_kg * np.exp(-prefix / c_kms)

        # -- 逐 leg pass（锚定解耦：位置匹配由锚定结构性满足）--
        fwd_states: list[npt.NDArray[np.floating]] = []
        bwd_states: list[npt.NDArray[np.floating]] = []
        eq_raw: list[npt.NDArray[np.floating]] = []
        r_mid_parts: list[npt.NDArray[np.floating]] = []
        mid_pos_parts: list[npt.NDArray[np.floating]] = []
        sens_f: list[npt.NDArray[np.floating]] = []
        sens_b: list[npt.NDArray[np.floating]] = []
        spos_parts: list[npt.NDArray[np.floating]] = []
        try:
            for leg in range(n_legs):
                n = n_list[leg]
                dt = float(tofs[leg]) / n
                tof_col = layout.tof_off + leg if layout.tof_free else None
                # 前向锚：leg 0 固定 departure；rendezvous 固定节点状态；flyby
                # 位置固定、速度含 v∞_out（锚灵敏度 [0₃; I₃]）。
                if leg == 0:
                    anchor_f = self._departure.copy()
                    sens_f0 = np.zeros((6, n_var))
                else:
                    node_prev = self._nodes[leg - 1]
                    anchor_f = np.asarray(node_prev.state, dtype=float).copy()
                    sens_f0 = np.zeros((6, n_var))
                    if node_prev.kind == "flyby":
                        anchor_f[3:] += vinf[leg - 1][1]
                        sens_f0[:, vinf_cols[leg - 1][1] : vinf_cols[leg - 1][1] + 3] = _B_IMPULSE
                # 后向锚：末节点/rendezvous 固定状态；flyby 速度含 v∞_in。
                node_end = self._nodes[leg]
                anchor_b = np.asarray(node_end.state, dtype=float).copy()
                sens_b0 = np.zeros((6, n_var))
                if node_end.kind == "flyby":
                    anchor_b[3:] += vinf[leg][0]
                    sens_b0[:, vinf_cols[leg][0] : vinf_cols[leg][0] + 3] = _B_IMPULSE

                f_out = self._leg_pass(
                    anchor_f,
                    sens_f0,
                    dv_list[leg],
                    dt,
                    forward=True,
                    imp_offset=layout.imp_offsets[leg],
                    tof_col=tof_col,
                    n_var=n_var,
                    with_sens=with_sens,
                    t_start=None if leg_t0 is None else leg_t0[leg],
                )
                b_out = self._leg_pass(
                    anchor_b,
                    sens_b0,
                    dv_list[leg],
                    dt,
                    forward=False,
                    imp_offset=layout.imp_offsets[leg],
                    tof_col=tof_col,
                    n_var=n_var,
                    with_sens=with_sens,
                    t_start=None if leg_t0 is None else leg_t0[leg],
                )
                fwd_states.append(f_out[0])
                bwd_states.append(b_out[0])
                eq_raw.append(f_out[0][-1] - b_out[0][0])
                r_mid_parts.append(np.concatenate([f_out[2], b_out[2]]))
                mid_pos_parts.append(np.concatenate([f_out[1], b_out[1]]))
                sens_f.append(f_out[3])
                sens_b.append(b_out[3])
                spos_parts.append(np.concatenate([f_out[4], b_out[4]], axis=0))
        except ValueError:
            # 线搜索试探点落在闭式 Kepler 病态能量带（近抛物线 Newton 发散）：
            # 深罚评估让 SLSQP 的 L1 罚函数回退步长，而不是让整个求解崩溃。
            return self._ml_penalty_evaluation(
                np.asarray(x, dtype=float),
                layout,
                dv_list,
                tofs,
                dv_norms,
                with_sens=with_sens,
            )

        # -- 段可行域（全局时序质量链）--
        r_mid = np.concatenate(r_mid_parts)
        mid_pos = np.concatenate(mid_pos_parts)
        leg_of_seg = np.repeat(np.arange(n_legs), n_list)
        n_arr = np.asarray(n_list, dtype=float)
        dt_seg = tofs[leg_of_seg] / n_arr[leg_of_seg]
        thrust = np.array([self._propulsion.max_thrust_n(r) for r in r_mid])
        cap = thrust / m_bar * dt_seg / 1000.0
        ineq_seg = cap**2 - dv_norms**2

        # -- flyby 约束（值 + 局部梯度）--
        flyby_evals = [
            self._flyby_constraints(vinf[j][0], vinf[j][1], self._nodes[j])
            for j in layout.flyby_nodes
        ]

        # -- 等式/不等式约束 --
        eq = np.concatenate(
            [
                np.concatenate([raw[:3] / self._len_scale, raw[3:] / self._vel_scale])
                for raw in eq_raw
            ]
            + [np.array([fe.eq_val for fe in flyby_evals])]
        )
        ineq = np.concatenate([ineq_seg, np.array([fe.ineq_val for fe in flyby_evals])])

        # -- 目标 --
        uhat = np.zeros_like(dv_flat)
        mag_mask = dv_norms > _UHAT_EPS_KM_S
        uhat[mag_mask] = dv_flat[mag_mask] / dv_norms[mag_mask, None]
        objective, obj_grad = self._ml_objective(
            layout,
            dv_flat,
            dv_norms,
            leg_of_seg,
            uhat,
            tofs,
            c_kms,
            with_grad=with_sens,
        )

        # -- 雅可比 --
        eq_jac = None
        ineq_jac = None
        if with_sens:
            eq_rows = []
            for leg in range(n_legs):
                sf, sb = sens_f[leg], sens_b[leg]
                eq_rows.append(
                    np.vstack([sf[:3] / self._len_scale, sf[3:] / self._vel_scale])
                    - np.vstack([sb[:3] / self._len_scale, sb[3:] / self._vel_scale])
                )
            for f, fe in enumerate(flyby_evals):
                row = np.zeros(n_var)
                base = layout.vinf_off + 6 * f
                row[base : base + 6] = fe.eq_grad
                eq_rows.append(row)
            eq_jac = np.vstack(eq_rows)
            ineq_jac = self._ml_ineq_jacobian(
                layout,
                dv_flat,
                dv_norms,
                cap,
                r_mid,
                mid_pos,
                m_bar,
                dt_seg,
                leg_of_seg,
                np.concatenate(spos_parts, axis=0),
                uhat,
                c_kms,
                flyby_evals,
            )

        total_dv = float(np.sum(dv_norms))
        return _MultiLegEvaluation(
            tofs=tofs,
            dv_list=dv_list,
            objective=objective,
            obj_grad=obj_grad,
            eq=eq,
            eq_jac=eq_jac,
            ineq=ineq,
            ineq_jac=ineq_jac,
            fwd_states=fwd_states,
            bwd_states=bwd_states,
            eq_raw=eq_raw,
            flybys=flyby_evals,
            cap_km_s=cap,
            dv_norms=dv_norms,
            total_dv_km_s=total_dv,
            final_mass_kg=self._m0_kg * float(np.exp(-total_dv / c_kms)),
        )
