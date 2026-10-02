"""MGA 链的连续精化（multisegment 框架第二实例，#726）。

对 :func:`.mga.search_mga_chains` 的网格候选做**变量连续化**精化：决策变量
为发射历元 + 逐 leg TOF（``multisegment.VariableLayout`` 登记），逐 leg 仍
解 short-way 零圈 Lambert；每个中间天体的无动力 flyby 约束（V∞ 等模等式
+ 转角不等式）复用 :func:`.multisegment.flyby_node_constraints`，目标与
``MgaCandidate.total_delta_v_km_s`` 同口径（出发 v∞ + 到达 v∞）。求解走
:func:`.nlp_scipy.solve_slsqp`（SLSQP 数值差分，与 DRO 路径同策略）；
成功解过后验闸门（ADR 0020 红线）再经 :func:`.mga.evaluate_flyby` 重建
精化候选。节点天体态随历元由星历重查（变量即历元，与 SF“节点状态是
数据”的口径不同——这是 MGA 精化的本职）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds

from e2m2e.data.constants import SECONDS_PER_DAY
from e2m2e.status import ConvergenceState, FailureCause, ResultStatus

from ..results import scipy_slsqp_status
from .lambert import solve_lambert
from .mga import MgaCandidate, evaluate_flyby
from .multisegment import FlybyConstraintEval, VariableLayout, flyby_node_constraints
from .nlp_core import NLPSpec
from .nlp_scipy import solve_slsqp
from .qlaw import rv_to_keplerian

__all__ = ["MgaRefinementResult", "refine_mga_chain"]

#: 日心状态查询帧与观察者（与 ``mga.py`` 同约定：MGA 链一律日心 J2000 几何）。
_FRAME = "J2000"
_OBSERVER = "SUN"

#: Lambert 无解（线搜索试探点）时的惩罚值（同 SF 模式：目标罚大数、等式
#: 大残差、不等式深负值，让 SLSQP 的 L1 罚函数回退步长）。
_PENALTY_OBJECTIVE = 1e9
_PENALTY_EQ = 1e3
_PENALTY_INEQ = -1e3

#: 后验闸门阈值（ADR 0020 红线）：flyby 等模原始差（km/s）与转角越限（rad）。
_POST_EQ_TOL_KM_S = 1e-6
_POST_INEQ_TOL_RAD = 1e-6


@dataclass(frozen=True)
class MgaRefinementResult:
    """MGA 链精化结果（软失败三元组随解携带，不抛异常）。

    Attributes:
        status: 算法最终状态。
        cause: 算法最终原因码。
        message: 求解器状态消息（含后验闸门结论）。
        candidate: 精化后的候选；非收敛或闸门未过时为 ``None``。
        n_iter: SLSQP 迭代次数。
        initial_total_delta_v_km_s: 入参候选的总 ΔV（km/s，精化前的基准）。
    """

    status: ConvergenceState
    cause: FailureCause
    message: str
    candidate: MgaCandidate | None
    n_iter: int
    initial_total_delta_v_km_s: float

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)


def refine_mga_chain(
    body_sequence: list[str],
    candidate: MgaCandidate,
    *,
    min_flyby_pericenter_km: list[float],
    launch_epoch_bounds_s: tuple[float, float],
    tof_bounds_s: npt.ArrayLike,
    ephemeris: Any | None = None,
    kernel_dir: str | None = None,
    mu_sun_km3_s2: float | None = None,
    ftol: float = 1e-9,
    maxiter: int = 200,
) -> MgaRefinementResult:
    """MGA 链网格候选的连续精化（发射历元 + 逐 leg TOF 为决策变量）。

    评估：节点历元 = 发射历元 + TOF 前缀和，天体态经
    ``ephemeris.get_body_state(body, et, "J2000", "SUN")`` 重查（缺省懒加载
    ``SPICEManager``，同 :func:`.mga.search_mga_chains` 模式）；逐 leg 解
    short-way 零圈 Lambert；flyby j 取 ``v_in = vf_{j−1} − v_B``、
    ``v_out = v0_j − v_B``，等模等式与转角不等式复用
    :func:`.multisegment.flyby_node_constraints`（``vel_scale`` 取候选各
    flyby V∞ 模的最大值与 1.0 的较大者，固定归一尺度）。目标 = 出发 v∞ +
    到达 v∞（同 ``MgaCandidate.total_delta_v_km_s`` 口径）。求解：SLSQP
    数值差分；评估内 Lambert 抛 ``ValueError`` 时返回惩罚值。成功解过
    后验闸门（等模 ≤ 1e-6 km/s、转角越限 ≥ −1e-6 rad）后用
    :func:`.mga.evaluate_flyby`（等模容差 1e-6 km/s）重建精化候选。

    Args:
        body_sequence: 天体序列（SPICE 天体名），与候选同链：首项出发体、
            末项到达体、中间项为飞越体，长度 ≥ 2。
        candidate: 待精化的网格候选（历元/TOF 为初猜）。
        min_flyby_pericenter_km: 逐飞越天体最小近心点半径（km），长度 =
            天体数 − 2（两体序列为空表），元素为正的有限值。
        launch_epoch_bounds_s: 发射历元搜索区间 ``(lo, hi)``（SPICE et 秒），
            须包含候选历元。
        tof_bounds_s: 逐 leg TOF 界（秒）：``(lb, ub)`` 共用对或逐 leg
            ``(L, 2)``；下界须为正、上界不小于下界。
        ephemeris: 星历提供者（需 ``get_body_state``/``get_gm``）；缺省
            懒加载 ``SPICEManager()`` 并 ``load_design_kernels``。
        kernel_dir: 缺省星历路径的内核目录；``None`` 用仓库缺省。
        mu_sun_km3_s2: 太阳 GM（km³/s²）；``None`` 取 ``ephemeris.get_gm("SUN")``。
        ftol: SLSQP 目标容差。
        maxiter: SLSQP 最大迭代次数。

    Returns:
        :class:`MgaRefinementResult`；不可行（闸门未过或 SLSQP 判不可行）
        以 ``(INFEASIBLE, CONSTRAINT_VIOLATION)`` 三元组表达，不抛异常。

    Raises:
        ValueError: 入参长度/取值非法（bounds 形状、TOF 下界非正、界序
            颠倒、历元区间不含候选值等）。
    """
    bodies = [str(body) for body in body_sequence]
    n_bodies = len(bodies)
    if n_bodies < 2:
        raise ValueError(f"body_sequence 至少两个天体（出发体…到达体），得到 {n_bodies} 个")
    n_legs = n_bodies - 1
    if not isinstance(candidate, MgaCandidate):
        raise ValueError(f"candidate 必须为 MgaCandidate，得到 {type(candidate).__name__}")
    if len(candidate.leg_tofs_days) != n_legs:
        raise ValueError(
            f"candidate.leg_tofs_days 长度 {len(candidate.leg_tofs_days)} 与链的 leg 数 "
            f"{n_legs} 不符（天体序列与候选不同链）"
        )
    r_p_mins = [float(value) for value in min_flyby_pericenter_km]
    if len(r_p_mins) != n_bodies - 2:
        raise ValueError(
            f"min_flyby_pericenter_km 长度须为 {n_bodies - 2}（天体序列长度 − 2），"
            f"得到 {len(r_p_mins)}"
        )
    if any(not np.isfinite(value) or value <= 0.0 for value in r_p_mins):
        raise ValueError(f"min_flyby_pericenter_km 须全为正的有限值，得到 {r_p_mins}")
    epoch_lo, epoch_hi = (float(v) for v in launch_epoch_bounds_s)
    if not (np.isfinite(epoch_lo) and np.isfinite(epoch_hi)):
        raise ValueError(f"launch_epoch_bounds_s 须为有限值，得到 {(epoch_lo, epoch_hi)!r}")
    if epoch_hi < epoch_lo:
        raise ValueError(f"launch_epoch_bounds_s 上界不得小于下界，得到 {(epoch_lo, epoch_hi)!r}")
    if not epoch_lo <= candidate.launch_epoch_et <= epoch_hi:
        raise ValueError(
            f"launch_epoch_bounds_s 区间 [{epoch_lo}, {epoch_hi}] 须包含候选历元 "
            f"{candidate.launch_epoch_et}"
        )
    tof_arr = np.asarray(tof_bounds_s, dtype=float)
    if tof_arr.shape == (2,):
        tof_arr = np.tile(tof_arr, (n_legs, 1))
    if tof_arr.shape != (n_legs, 2):
        raise ValueError(
            f"tof_bounds_s 必须为 (lb, ub) 共用对或逐 leg ({n_legs}, 2)，得到 shape {tof_arr.shape}"
        )
    if not np.all(np.isfinite(tof_arr)):
        raise ValueError("tof_bounds_s 含非有限分量")
    if np.any(tof_arr[:, 0] <= 0.0):
        raise ValueError("tof_bounds_s 下界必须为正")
    if np.any(tof_arr[:, 1] < tof_arr[:, 0]):
        raise ValueError("tof_bounds_s 上界不得小于下界")

    if ephemeris is None:
        from e2m2e.data.kernels.manager import SPICEManager

        from ..design.design_orbit import load_design_kernels

        spice = SPICEManager()
        load_design_kernels(spice, kernel_dir)
        ephemeris = spice
    mu_sun = float(ephemeris.get_gm("SUN")) if mu_sun_km3_s2 is None else float(mu_sun_km3_s2)
    if not np.isfinite(mu_sun) or mu_sun <= 0.0:
        raise ValueError(f"太阳 GM 须为正的有限值，得到 {mu_sun}")

    def body_state(body: str, et: float) -> np.ndarray:
        """日心状态 (6,)；形状不符显式报错。"""
        state = np.asarray(ephemeris.get_body_state(body, et, _FRAME, _OBSERVER), dtype=float)
        if state.shape != (6,):
            raise ValueError(f"星历返回 {body} 在 et={et} 的状态形状 {state.shape}，期望 (6,)")
        return state

    # 只有飞越天体需要 GM（出发/到达体只用到位置与速度）。
    mu_body_by_idx = {j: float(ephemeris.get_gm(bodies[j])) for j in range(1, n_legs)}

    variables = VariableLayout()
    variables.register("epoch", 1)
    variables.register("tof", n_legs)
    x0 = np.concatenate(
        [[float(candidate.launch_epoch_et)], np.asarray(candidate.leg_tofs_days) * SECONDS_PER_DAY]
    )
    bounds = Bounds(
        np.concatenate([[epoch_lo], tof_arr[:, 0]]),
        np.concatenate([[epoch_hi], tof_arr[:, 1]]),
    )
    vel_scale = max(
        max((float(fe.v_inf_km_s) for fe in candidate.flybys), default=0.0),
        1.0,
    )
    n_fly = n_legs - 1

    class _ChainEval:
        """一次决策向量的全链评估（目标/等式/不等式值 + flyby 约束面）。"""

        objective: float
        eq: npt.NDArray[np.floating]
        ineq: npt.NDArray[np.floating]
        flybys: list[FlybyConstraintEval]

        def __init__(self, z: npt.NDArray[np.floating]) -> None:
            epoch = float(z[0])
            tofs = np.asarray(z[1:], dtype=float)
            t_nodes = epoch + np.concatenate([[0.0], np.cumsum(tofs)])
            try:
                v0_by_leg = []
                vf_by_leg = []
                for k in range(n_legs):
                    r0 = body_state(bodies[k], t_nodes[k])[:3]
                    rf = body_state(bodies[k + 1], t_nodes[k + 1])[:3]
                    sol = solve_lambert(r0, rf, float(tofs[k]), mu_sun, "short", 0)
                    v0_by_leg.append(sol.v0)
                    vf_by_leg.append(sol.vf)
            except ValueError:
                # Lambert 无解的线搜索试探点（几何不可达/TOF 过短）：
                # 惩罚评估让 SLSQP 的 L1 罚函数回退步长。
                self.objective = _PENALTY_OBJECTIVE
                self.eq = np.full(n_fly, _PENALTY_EQ)
                self.ineq = np.full(n_fly, _PENALTY_INEQ)
                self.flybys = []
                return
            dep_vinf = float(np.linalg.norm(v0_by_leg[0] - body_state(bodies[0], t_nodes[0])[3:]))
            arr_vinf = float(
                np.linalg.norm(vf_by_leg[-1] - body_state(bodies[-1], t_nodes[-1])[3:])
            )
            eq_vals = []
            ineq_vals = []
            flyby_evals = []
            for j in range(1, n_legs):
                v_body = body_state(bodies[j], t_nodes[j])[3:]
                flyby_evals.append(
                    flyby_node_constraints(
                        vf_by_leg[j - 1] - v_body,
                        v0_by_leg[j] - v_body,
                        r_p_min_km=r_p_mins[j - 1],
                        mu_km3_s2=mu_body_by_idx[j],
                        vel_scale=vel_scale,
                    )
                )
                eq_vals.append(flyby_evals[-1].eq_val)
                ineq_vals.append(flyby_evals[-1].ineq_val)
            self.objective = dep_vinf + arr_vinf
            self.eq = np.asarray(eq_vals, dtype=float)
            self.ineq = np.asarray(ineq_vals, dtype=float)
            self.flybys = flyby_evals

    cache: dict[bytes, _ChainEval] = {}

    def evaluated(z: npt.NDArray[np.floating]) -> _ChainEval:
        key = np.ascontiguousarray(z, dtype=float).tobytes()
        hit = cache.get(key)
        if hit is None:
            hit = _ChainEval(np.asarray(z, dtype=float))
            if len(cache) > 64:
                cache.clear()
            cache[key] = hit
        return hit

    spec = NLPSpec(
        objective=lambda z: evaluated(z).objective,
        eq=(lambda z: evaluated(z).eq) if n_fly > 0 else None,
        ineq=(lambda z: evaluated(z).ineq) if n_fly > 0 else None,
        bounds=bounds,
    )
    result = solve_slsqp(spec, x0, ftol=ftol, maxiter=maxiter)
    status, cause = scipy_slsqp_status(bool(result.success), int(result.status))
    message = str(result.message)
    final = evaluated(np.asarray(result.x, dtype=float))
    n_iter = int(result.nit)
    initial_dv = float(candidate.total_delta_v_km_s)

    if status is ConvergenceState.CONVERGED:
        # 后验闸门（ADR 0020 红线）：等模原始差与转角越限任一超限即改判
        # 不可行，不谎报 success。
        eq_max = max((abs(fe.n_in - fe.n_out) for fe in final.flybys), default=0.0)
        ineq_min = min((fe.ineq_val for fe in final.flybys), default=0.0)
        if eq_max > _POST_EQ_TOL_KM_S or ineq_min < -_POST_INEQ_TOL_RAD:
            status = ConvergenceState.INFEASIBLE
            cause = FailureCause.CONSTRAINT_VIOLATION
            message = (
                f"{message}；后验闸门未过：flyby 等模 max‖Δ|v∞|‖ = {eq_max:.3e}"
                f"（阈值 {_POST_EQ_TOL_KM_S} km/s），转角裕度 min(δ_max−δ) = "
                f"{ineq_min:.3e}（阈值 −{_POST_INEQ_TOL_RAD} rad）"
            )

    if status is not ConvergenceState.CONVERGED:
        return MgaRefinementResult(status, cause, message, None, n_iter, initial_dv)

    # -- 用解值重建精化候选（evaluate_flyby 的等模容差与闸门一致）--
    z = np.asarray(result.x, dtype=float)
    epoch = float(z[0])
    tofs = np.asarray(z[1:], dtype=float)
    t_nodes = epoch + np.concatenate([[0.0], np.cumsum(tofs)])
    states = [body_state(body, t) for body, t in zip(bodies, t_nodes, strict=True)]
    v0_by_leg = []
    vf_by_leg = []
    for k in range(n_legs):
        sol = solve_lambert(states[k][:3], states[k + 1][:3], float(tofs[k]), mu_sun, "short", 0)
        v0_by_leg.append(sol.v0)
        vf_by_leg.append(sol.vf)
    flybys = []
    for j in range(1, n_legs):
        v_body = states[j][3:]
        a_ref, *_ = rv_to_keplerian(states[j][:3], v_body, mu_sun)
        evaluation = evaluate_flyby(
            vf_by_leg[j - 1] - v_body,
            v0_by_leg[j] - v_body,
            states[j],
            bodies[j],
            mu_body_by_idx[j],
            mu_sun,
            float(a_ref),
            r_p_mins[j - 1],
            _POST_EQ_TOL_KM_S,
        )
        if evaluation is None:
            return MgaRefinementResult(
                ConvergenceState.INFEASIBLE,
                FailureCause.CONSTRAINT_VIOLATION,
                "解处 flyby 评估未通过等模/近心点筛选（闸门口径）",
                None,
                n_iter,
                initial_dv,
            )
        flybys.append(evaluation)
    dep_vinf = float(np.linalg.norm(v0_by_leg[0] - states[0][3:]))
    arr_vinf = float(np.linalg.norm(vf_by_leg[-1] - states[-1][3:]))
    refined = MgaCandidate(
        launch_epoch_et=epoch,
        leg_tofs_days=tuple(float(tof / SECONDS_PER_DAY) for tof in tofs),
        departure_v_inf_km_s=dep_vinf,
        arrival_v_inf_km_s=arr_vinf,
        total_delta_v_km_s=dep_vinf + arr_vinf,
        flybys=tuple(flybys),
    )
    message = (
        f"MGA 链精化收敛：总 ΔV {initial_dv:.6f} → {refined.total_delta_v_km_s:.6f} km/s"
        f"（{n_iter} 次迭代）"
    )
    return MgaRefinementResult(status, cause, message, refined, n_iter, initial_dv)
