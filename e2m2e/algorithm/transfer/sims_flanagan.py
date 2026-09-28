"""Sims-Flanagan 单 leg rendezvous 预设计：段中冲量直接法 NLP（#740，MALTO 核心算法 MVP）。

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

``backend`` 为必填关键字参数：``"conic"``（本期实现）或 ``"ephemeris"``（未
实现，显式报错不静默回退，ADR 0050 理由 6）；其余取值 ``ValueError``。

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

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, minimize

from e2m2e.data.constants import AU_KM
from e2m2e.integrators import propagate_kepler_py, require_rust_extension

from ...data.templates import ConvergenceState, FailureCause
from ..results import ResultStatus, scipy_slsqp_status
from .sep import (
    edelbaum_delta_v,
    edelbaum_delta_v_inclined,
    sep_available_power,
    thrust_from_power,
)
from .thrust_arcs import G0_MPS2

__all__ = ["SimsFlanaganProblem", "SimsFlanaganPropulsion", "SimsFlanaganSolution"]

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
        ``−4η·P₀·AU²/(Isp·g₀·r³)``（P_bus 项导数为零）。
        """
        if self.t_max_n is not None:
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
    """单 leg 出发→到达交会的 Sims-Flanagan 预设计问题（conic 档）。

    Args:
        departure_state: 出发状态 ``[r, v]``，``(6,)``，km / km/s。
        arrival_state: 到达状态 ``[r, v]``，``(6,)``，km / km/s。
        tof_s: 飞行时间（s），必须为正。
        propulsion: 推进配置（:class:`SimsFlanaganPropulsion`）。
        initial_mass_kg: 出发质量（kg），必须为正。
        mu_km3_s2: 中心天体引力常数（km³/s²），必须为正。
        backend: 保真度档位，必填关键字：``"conic"`` 或 ``"ephemeris"``；
            ``"ephemeris"`` 本期未实现，显式报错（ADR 0050 理由 6）。
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
        if backend == "ephemeris":
            raise ValueError("ephemeris 档未实现（当前仅支持 conic 档）")

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
        """Edelbaum 闭式初猜：总 ΔV 均分 n 段，方向沿出发速度（抬升取 +）。"""
        r_dep, v_dep = self._departure[:3], self._departure[3:]
        r_arr, v_arr = self._arrival[:3], self._arrival[3:]
        v_dep_mag = float(np.linalg.norm(v_dep))
        v_arr_mag = float(np.linalg.norm(v_arr))
        inc_deg = self._plane_change_deg(r_dep, v_dep, r_arr, v_arr)
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

    @staticmethod
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

    # ---- 内部：段传播与灵敏度 ----

    def _forward_pass(
        self, dv: npt.NDArray[np.floating], n: int, m: int, dt: float, *, with_sens: bool
    ) -> tuple[
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
    ]:
        """前向 pass：出发态经段 0..m-1 接龙（段中冲量）到匹配节点。

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
        for k in range(m):
            out = propagate_kepler_py(x.tolist(), [half], self._mu_km3_s2, with_stm=with_sens)
            x_mid = np.asarray(out["states"][0], dtype=float)
            mid_pos[k] = x_mid[:3]
            r_mid[k] = float(np.linalg.norm(x_mid[:3]))
            if with_sens:
                phi1 = np.asarray(out["stm"][0], dtype=float).reshape(6, 6)
                s_mid = phi1 @ sens
                s_mid_pos[k] = s_mid[:3, :]
                # 冲量是段中点的状态跳变：∂x_mid⁺/∂ΔV_k = s_mid[:,k] + B（无 Φ1）。
                s_mid[:, 3 * k : 3 * k + 3] += _B_IMPULSE
            x_mid = x_mid + _B_IMPULSE @ dv[k]
            out = propagate_kepler_py(x_mid.tolist(), [half], self._mu_km3_s2, with_stm=with_sens)
            x = np.asarray(out["states"][0], dtype=float)
            if with_sens:
                phi2 = np.asarray(out["stm"][0], dtype=float).reshape(6, 6)
                sens = phi2 @ s_mid
            states.append(x.copy())
        return np.asarray(states), mid_pos, r_mid, sens, s_mid_pos

    def _backward_pass(
        self, dv: npt.NDArray[np.floating], n: int, m: int, dt: float, *, with_sens: bool
    ) -> tuple[
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
    ]:
        """后向 pass：到达态经段 n-1..m 反向接龙（内核支持负 dt）到匹配节点。

        返回 ``(节点状态 (n-m+1,6) [匹配节点..到达], 段中点位置 (n-m,3),
        段中点半径 (n-m,), ∂匹配节点/∂ΔV (6,3n), 段中点位置灵敏度 (n-m,3,3n))``。
        冲量在后向路径上取 ``v_pre = v_post − ΔV_k``，灵敏度带负号。
        """
        half = -0.5 * dt
        x = self._arrival
        nodes = [x.copy()]
        nb = n - m
        mid_pos = np.zeros((nb, 3))
        r_mid = np.zeros(nb)
        sens = np.zeros((6, 3 * n))
        s_mid_pos = np.zeros((nb, 3, 3 * n))
        for idx in range(nb):
            k = n - 1 - idx
            out = propagate_kepler_py(x.tolist(), [half], self._mu_km3_s2, with_stm=with_sens)
            x_mid = np.asarray(out["states"][0], dtype=float)
            mid_pos[idx] = x_mid[:3]
            r_mid[idx] = float(np.linalg.norm(x_mid[:3]))
            if with_sens:
                phi1 = np.asarray(out["stm"][0], dtype=float).reshape(6, 6)
                s_mid = phi1 @ sens
                s_mid_pos[idx] = s_mid[:3, :]
                # 后向冲量是段中点的反向状态跳变：v_pre = v_post − ΔV_k（无 Φ1）。
                s_mid[:, 3 * k : 3 * k + 3] -= _B_IMPULSE
            x_mid = x_mid - _B_IMPULSE @ dv[k]
            out = propagate_kepler_py(x_mid.tolist(), [half], self._mu_km3_s2, with_stm=with_sens)
            x = np.asarray(out["states"][0], dtype=float)
            if with_sens:
                phi2 = np.asarray(out["stm"][0], dtype=float).reshape(6, 6)
                sens = phi2 @ s_mid
            nodes.append(x.copy())
        return np.asarray(nodes[::-1]), mid_pos, r_mid, sens, s_mid_pos

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
                dv, n, m, dt, with_sens=with_sens
            )
            bwd_states, bwd_mid_pos, r_bwd, s_bwd, spos_bwd = self._backward_pass(
                dv, n, m, dt, with_sens=with_sens
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

        ``Spos_k`` 是该段**中点**位置灵敏度（段 k < m 取前向 pass、否则取后向
        pass），支集分别为 j < k / j > k；``m̄ₖ`` 对 ``ΔV_j``（j < k）的导数为
        ``−m̄ₖ·û_j/c``，两 pass 同式（后向锚定推导中 j ≥ k 项相消），j ≥ k 的
        两支全为零，故内层只需遍历 j < k。
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
            for j in range(k):
                block = slice(3 * j, 3 * j + 3)
                if grad_t != 0.0:
                    dcap[block] += scale * grad_t * (rhat_k @ spos_k[:, block]) / m_bar_k
                if dv_norm[j] > _UHAT_EPS_KM_S:
                    dcap[block] += scale * thrust_k * (dv[j] / dv_norm[j]) / (c_kms * m_bar_k)
            jac[k] = 2.0 * cap_k * dcap
            jac[k, 3 * k : 3 * k + 3] -= 2.0 * dv[k]
        return jac
