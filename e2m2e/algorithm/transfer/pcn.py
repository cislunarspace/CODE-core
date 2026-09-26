"""PCN patched-conic 目标参数化打靶编排（#635）。

在 Patched-Conic 近似下把地月转移分为地球段（地心二体双曲出发）与月球段
（月心二体双曲到达）两段，用 B-plane / 双曲渐近线参数化连接：

- **到达模式**（``bplane_target``）：决策变量为地球出发渐近线
  ``(RHA, DHA, C3)``，打靶残差 ``R = (B·R − B·R*, B·T − B·T*, r_p − r_p*)``
  使地球段传播 tof 后的月心相对态命中给定 B-plane 目标（近月点高度 +
  B·T，B·R 默认 0）。残差雅可比 = :func:`bplane_jacobian_from_state`（解析）
  × ``∂月心态/∂x``（中心差分），阻尼 Newton + 线搜索。
- **出发模式**（``departure_asymptote``）：给定出发渐近线，无迭代；构造地球
  出发双曲态、传播到 tof、解月心 B-plane，近月点与 LOI 由闭式给出。

月球几何采用 CR3BP 圆型理想化（θ₀=0 约定，与 ADR 0040 的 HMN 先例一致），
``moon_state_fn`` 由编排器注入（``(t_sec) → (6,)`` 月球 GCRS 状态），本模块
不反向导入 ``__init__``（防循环导入）。

软失败路径（不可达目标、非双曲到达、不收敛）一律走
``(ConvergenceState, FailureCause, message)`` 状态三元组，不抛异常（ADR 0020）。

层级：algorithm 层，仅依赖 numpy/同包 transfer 模块/data 常量。
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ...data.constants import SECONDS_PER_DAY
from ...data.constants.bodies import MOON
from ...data.constants.datums import Datum
from ...data.templates import ConvergenceState, FailureCause
from ...exceptions import PropagationFailure
from ..results import ResultStatus
from ..spatiography.scales import soi_laplace_moon
from .bplane import (
    AsymptoteParams,
    BPlaneParams,
    bplane_from_state,
    bplane_jacobian_from_state,
    hyperbolic_time_to_periapsis,
    perilune_state_from_bplane,
    vinf_vector_from_asymptote,
)
from .hohmann import R_EARTH, TliParams
from .multi_impulse import propagate_two_body

logger = logging.getLogger(__name__)

__all__ = [
    "PcnBplaneTarget",
    "PcnSearchParams",
    "PcnSolution",
    "solve_pcn",
]

#: 月球引力参数 GM☾ (km³/s²)，DE421 基准（与编排器 CR3BP 特征尺度同源）。
_MU_MOON: float = Datum.DE421.moon_gm
#: 月球平均半径 (km)。
_R_MOON_KM: float = MOON.require_mean_radius_km()
#: 月球影响球半径 (km)：Laplace–Tisserand 代理 rho_SOI（单一来源为 spatiography 的
#: ``soi_laplace_moon``，与 atlas 的 "Moon SOI (Laplace-Tisserand)" 同口径；
#: 黄金值 66010 km）。出发模式用它判定交接点是否落在月交会可行域内（#698）。
_R_SOI_MOON_KM: float = soi_laplace_moon()
#: 地球引力参数 GM⊕ (km³/s²)。
_MU_EARTH: float = Datum.WGS84.earth_gm

#: 参考 z 轴（出发平面退化时的回退基准）。
_Z_HAT = np.array([0.0, 0.0, 1.0])

#: 平面法向退化判据。
_PLANE_DEGENERATE_EPS = 1e-12

#: 线搜索最大回退次数（阻尼 Newton：整步增大残差时步长减半）。
_LINE_SEARCH_STEPS = 10

#: 残差雅可比中 ∂月心态/∂(RHA, DHA) 的中心差分步长（deg）。
_ANGLE_FD_STEP_DEG = 1e-6
#: 残差雅可比中 ∂月心态/∂C3 的中心差分步长（km²/s²）。
_C3_FD_STEP = 1e-6


@dataclass(frozen=True)
class PcnBplaneTarget:
    """PCN 到达模式目标：月心 B-plane（Vallado 定义）。

    Attributes:
        perilune_alt_km: 目标近月点高度 (km，月面以上)。
        bdot_t_km: 目标 B·T (km)。
        bdot_r_km: 目标 B·R (km)，默认 0。
    """

    perilune_alt_km: float
    bdot_t_km: float
    bdot_r_km: float = 0.0


@dataclass(frozen=True)
class PcnSearchParams:
    """PCN 打靶搜索参数。

    Attributes:
        tof_range_days: 地球段飞行时间网格范围 (天)。
        n_tof: 飞行时间网格点数。
        rha_grid_deg: 出发渐近线赤经**相对交会时刻月球位置方向**的偏移角网格范围 (deg)。
        n_rha: 赤经网格点数。
        dha_grid_deg: 出发渐近线赤纬网格范围 (deg)。
        n_dha: 赤纬网格点数。
        c3_grid_km2_s2: 出发 C3 网格范围 (km²/s²)，仅双曲（下界须 > 0）。
        n_c3: C3 网格点数。
        tolerance_km: 打靶残差 ``max|·|`` 收敛容差 (km)。
        max_iterations: 最大 Newton 迭代次数。
        n_trajectory_samples: 地球段轨迹采样点数（编排器组装用）。
    """

    tof_range_days: tuple[float, float] = (1.0, 3.0)
    n_tof: int = 9
    rha_grid_deg: tuple[float, float] = (-45.0, 45.0)
    n_rha: int = 8
    dha_grid_deg: tuple[float, float] = (-35.0, 35.0)
    n_dha: int = 5
    c3_grid_km2_s2: tuple[float, float] = (0.1, 2.0)
    n_c3: int = 5
    tolerance_km: float = 1.0
    max_iterations: int = 20
    n_trajectory_samples: int = 200


@dataclass
class PcnSolution:
    """PCN 打靶解（成功携带几何，失败仅携带状态三元组）。

    Attributes:
        mode: ``"arrival"`` 或 ``"departure"``。
        status: 最终状态。
        cause: 原因码。
        message: 人类可读诊断。
        departure_state_gcrs: 地球出发态 (6,)，地心惯性 km / km s⁻¹。
        dv_tli_km_s: TLI 出发脉冲 (km/s)。
        departure_asymptote: 实际使用的出发渐近线（arrival 模式为收敛值）。
        encounter_state_moon: handoff 月心惯性态 (6,)，km / km s⁻¹。
        bplane: 达成的月心 B-plane。
        perilune_state_moon: 月心惯性近月点态 (6,)，km / km s⁻¹。
        dv_loi_km_s: 近月点圆化脉冲 (km/s)。
        earth_leg_tof_sec: 地球段飞行时间 (s)。
        moon_leg_tof_sec: 月心段（handoff → 近月点）飞行时间 (s)。
        n_grid_evals: 网格初猜评估次数。
        n_newton_iter: Newton 迭代次数。
    """

    mode: str
    status: ConvergenceState
    cause: FailureCause
    message: str
    departure_state_gcrs: np.ndarray | None = None
    dv_tli_km_s: float = float("inf")
    departure_asymptote: AsymptoteParams | None = None
    encounter_state_moon: np.ndarray | None = None
    bplane: BPlaneParams | None = None
    perilune_state_moon: np.ndarray | None = None
    dv_loi_km_s: float = float("inf")
    earth_leg_tof_sec: float = 0.0
    moon_leg_tof_sec: float = 0.0
    n_grid_evals: int = 0
    n_newton_iter: int = 0

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)


# ---------------------------------------------------------------------------
# 出发构造
# ---------------------------------------------------------------------------


def _canonical_rha_dha(rha_deg: float, dha_deg: float) -> tuple[float, float]:
    """把渐近线角归一到规范域 RHA ∈ [0°360)、DHA ∈ [−90°, 90°]。

    Newton 决策变量按连续分支步进，可能落到区间外的等价角；回显/入库前
    归一，避免响应模型（``ge``/``lt`` 约束）把成功解误判为非法入参。
    ``|DHA| > 90°`` 的等价方向为 ``(±180° − DHA, RHA + 180°)``。
    """
    rha = math.fmod(float(rha_deg), 360.0)
    if rha < 0.0:
        rha += 360.0
    dha = float(dha_deg)
    if dha > 90.0 or dha < -90.0:
        rha = math.fmod(rha + 180.0, 360.0)
        if rha < 0.0:
            rha += 360.0
        dha = (180.0 - dha) if dha > 90.0 else (-180.0 - dha)
    return rha, dha


def _departure_state_from_asymptote(
    asym: AsymptoteParams,
    tli_params: TliParams,
) -> tuple[np.ndarray, float]:
    """地心出发双曲态（出射分支）与 TLI 脉冲。

    停泊轨道按圆轨道理想化（γ=0），双曲轨道面法向取「含 Ŝ 且最接近参考
    z 轴的平面」——``ĥ = normalize(ẑ − (ẑ·Ŝ)Ŝ)``（Ŝ∥ẑ 时回退 x̂ 投影）。
    该构造不依赖月球相位（稳定），且严格满足出射渐近线 = Ŝ 的近心点方向
    ``ê = −(Ŝ + √(e²−1)·(ĥ×Ŝ))/e``。

    Returns:
        (地心惯性出发态 (6,)，Δv_TLI (km/s))。
    """
    c3 = asym.c3_km2_s2
    if not math.isfinite(c3) or c3 <= 0.0:
        raise ValueError(f"出发 C3 须为有限正数（双曲），得到 {c3}")
    if not math.isfinite(tli_params.parking_alt_km) or tli_params.parking_alt_km < 0.0:
        raise ValueError(f"停泊轨道高度须为有限非负数，得到 {tli_params.parking_alt_km}")
    r_park = R_EARTH + tli_params.parking_alt_km
    v_inf_e = math.sqrt(c3)
    s_hat = vinf_vector_from_asymptote(asym) / v_inf_e

    e_dep = 1.0 + r_park * c3 / _MU_EARTH
    s = math.sqrt(e_dep * e_dep - 1.0)

    h_vec = _Z_HAT - float(_Z_HAT @ s_hat) * s_hat
    if float(np.linalg.norm(h_vec)) < _PLANE_DEGENERATE_EPS:  # Ŝ∥ẑ，回退 x̂ 投影
        x_hat = np.array([1.0, 0.0, 0.0])
        h_vec = x_hat - float(x_hat @ s_hat) * s_hat
    h_norm = float(np.linalg.norm(h_vec))
    if h_norm < _PLANE_DEGENERATE_EPS:
        raise ValueError("渐近线方向退化，无法构造出发平面")
    h_hat = h_vec / h_norm

    e_hat = -(s_hat + s * np.cross(h_hat, s_hat)) / e_dep
    q_hat = np.cross(h_hat, e_hat)

    v_park = math.sqrt(_MU_EARTH / r_park)
    v_mag = math.sqrt(c3 + 2.0 * _MU_EARTH / r_park)
    r_vec = r_park * e_hat
    v_vec = v_mag * q_hat
    return np.concatenate([r_vec, v_vec]), v_mag - v_park


def _moon_encounter_state(
    departure_state: np.ndarray,
    tof_sec: float,
    moon_state_fn: Callable[[float], np.ndarray],
) -> np.ndarray:
    """地心出发态传播 tof → 月心相对态 (6,)。

    校验二体传播确实到达请求时刻且末态有限；否则抛
    :class:`PropagationFailure`（由调用方按软失败处理，避免用错时刻的
    月心态算出静默错误的几何）。
    """
    tof = float(tof_sec)
    if not math.isfinite(tof) or tof <= 0.0:
        raise PropagationFailure(f"地球段 tof 须为有限正数，得到 {tof_sec}")
    out = propagate_two_body(departure_state, np.array([0.0, tof]), _MU_EARTH)
    states = np.asarray(out["states"], dtype=float)
    times = np.asarray(out["time"], dtype=float)
    if states.shape[0] != 2 or abs(float(times[-1]) - tof) > 1e-6 * max(1.0, abs(tof)):
        raise PropagationFailure(f"二体传播未到达请求时刻 tof={tof}")
    sc = states[-1]
    moon = np.asarray(moon_state_fn(tof), dtype=float)
    if not np.all(np.isfinite(sc)) or not np.all(np.isfinite(moon)):
        raise PropagationFailure("月心相对态的传播末态或月球态非有限")
    return sc - moon


def _dv_loi(bplane: BPlaneParams) -> float:
    """近月点圆化脉冲：``√(v∞² + 2μ/r_p) − √(μ/r_p)`` (km/s)。"""
    rp = bplane.perilune_radius_km
    return math.sqrt(bplane.v_inf_km_s**2 + 2.0 * _MU_MOON / rp) - math.sqrt(_MU_MOON / rp)


def _departure_feasibility(
    closest_km: float, bplane: BPlaneParams
) -> tuple[ConvergenceState, FailureCause, str] | None:
    """出发模式的月交会可行域判定（#698）：不可行返回状态三元组，可行返回 ``None``。

    Patched-conic 口径下，出发模式只有落在月交会可行域内才是解：

    - 近月点半径 ``r_p ≤ R_moon``：可达近月点在地面以下，轨迹撞月，
      报 ``COLLISION``/``BODY_COLLISION``；
    - 网格最近月心距离超过月球影响球 ``_R_SOI_MOON_KM``：只是远距离飞越，
      月心段交接不成立，报 ``INFEASIBLE``/``CONSTRAINT_VIOLATION``。

    判据只用解自身的几何（近月点半径与最近月心距离），不依赖网格分辨率以外的
    外部状态；不可行解仍回显 B-plane / 渐近线，供调用方诊断。
    """
    if bplane.perilune_radius_km <= _R_MOON_KM:
        return (
            ConvergenceState.COLLISION,
            FailureCause.BODY_COLLISION,
            f"出发模式最近月心距离处近月点半径 {bplane.perilune_radius_km:.1f} km "
            f"≤ 月球半径 {_R_MOON_KM:.1f} km：轨迹撞月，非月交会",
        )
    if closest_km > _R_SOI_MOON_KM:
        return (
            ConvergenceState.INFEASIBLE,
            FailureCause.CONSTRAINT_VIOLATION,
            f"出发模式最近月心距离 {closest_km:.1f} km 超出月球影响球 "
            f"{_R_SOI_MOON_KM:.1f} km：远距离飞越，非月交会",
        )
    return None


def _perilune_state_from_bplane(bplane: BPlaneParams) -> np.ndarray:
    """由达成的月心 B-plane 闭式给出月心近月点态。"""
    asym = AsymptoteParams(
        rha_deg=bplane.rha_deg, dha_deg=bplane.dha_deg, c3_km2_s2=bplane.c3_km2_s2
    )
    return perilune_state_from_bplane(asym, bplane.bdot_r_km, bplane.bdot_t_km, _MU_MOON)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def solve_pcn(
    tli_params: TliParams,
    *,
    bplane_target: PcnBplaneTarget | None,
    departure_asymptote: AsymptoteParams | None,
    moon_state_fn: Callable[[float], np.ndarray],
    system: object,
    params: PcnSearchParams | None = None,
) -> PcnSolution:
    """PCN patched-conic 打靶（到达/出发两模式，XOR 二选一）。

    Args:
        tli_params: 停泊轨道参数（TLI 高度等）。
        bplane_target: 到达模式目标（与 departure_asymptote 二选一）。
        departure_asymptote: 出发模式渐近线（与 bplane_target 二选一）。
        moon_state_fn: ``(t_sec) → (6,)`` 月球 GCRS 状态（编排器注入）。
        system: CR3BP 系统（保留，供未来特征尺度使用）。
        params: 搜索参数；None 用默认。

    Returns:
        :class:`PcnSolution`；失败为状态三元组而非异常。

    Raises:
        ValueError: 两目标参数未恰给一个。
    """
    del system  # 当前几何由注入的 moon_state_fn 完全决定
    if (bplane_target is None) == (departure_asymptote is None):
        raise ValueError("bplane_target 与 departure_asymptote 必须恰给一个")
    p = params if params is not None else PcnSearchParams()

    if departure_asymptote is not None:
        return _solve_departure_mode(tli_params, departure_asymptote, moon_state_fn, p)
    assert bplane_target is not None
    return _solve_arrival_mode(tli_params, bplane_target, moon_state_fn, p)


# ---------------------------------------------------------------------------
# 出发模式
# ---------------------------------------------------------------------------


def _solve_departure_mode(
    tli_params: TliParams,
    departure_asymptote: AsymptoteParams,
    moon_state_fn: Callable[[float], np.ndarray],
    p: PcnSearchParams,
) -> PcnSolution:
    """出发模式：给定渐近线，在 tof 网格上取入射分支最近月心距离（真实交会），无迭代。

    选定网格点后过 ``_departure_feasibility`` 可行域门禁：撞月或落在月球
    影响球外的最近点不是月交会，报实际失败状态（并回显几何），不冒报 ``CONVERGED``。
    """
    dep_state, dv_tli = _departure_state_from_asymptote(departure_asymptote, tli_params)

    tof_grid = np.linspace(p.tof_range_days[0], p.tof_range_days[1], p.n_tof) * SECONDS_PER_DAY
    # 交接点须在入射分支（到近月点时间 ≥ 0）；在满足该条件的 tof 中取最近月心距离，
    # 得真实月交会（而非远距离飞越）。
    best: tuple[float, float, np.ndarray, BPlaneParams, float] | None = None
    for tof in tof_grid:
        try:
            enc = _moon_encounter_state(dep_state, float(tof), moon_state_fn)
            bplane = bplane_from_state(enc, _MU_MOON)
            moon_leg = hyperbolic_time_to_periapsis(enc, _MU_MOON)
        except (ValueError, RuntimeError, PropagationFailure, FloatingPointError):
            continue
        if moon_leg < 0.0:  # 出射分支（已过近月点），跳过
            continue
        dist = float(np.linalg.norm(enc[:3]))
        if best is None or dist < best[0]:
            best = (dist, float(tof), enc, bplane, moon_leg)

    if best is None:
        return PcnSolution(
            mode="departure",
            status=ConvergenceState.INFEASIBLE,
            cause=FailureCause.NO_INTERSECTION,
            message="出发态在 tof 网格上无入射双曲月交会（全部退化/非双曲/出射分支）",
            departure_state_gcrs=dep_state,
            dv_tli_km_s=dv_tli,
            departure_asymptote=departure_asymptote,
            n_grid_evals=len(tof_grid),
        )

    dist, tof, enc, bplane, moon_leg = best
    infeasible = _departure_feasibility(dist, bplane)
    if infeasible is not None:
        status, cause, message = infeasible
        return PcnSolution(
            mode="departure",
            status=status,
            cause=cause,
            message=message,
            departure_state_gcrs=dep_state,
            dv_tli_km_s=dv_tli,
            departure_asymptote=departure_asymptote,
            encounter_state_moon=enc,
            bplane=bplane,
            earth_leg_tof_sec=tof,
            moon_leg_tof_sec=moon_leg,
            n_grid_evals=len(tof_grid),
        )

    perilune = _perilune_state_from_bplane(bplane)
    return PcnSolution(
        mode="departure",
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="PCN 出发模式解算完成（无迭代，入射分支最近月心距离 tof）",
        departure_state_gcrs=dep_state,
        dv_tli_km_s=dv_tli,
        departure_asymptote=departure_asymptote,
        encounter_state_moon=enc,
        bplane=bplane,
        perilune_state_moon=perilune,
        dv_loi_km_s=_dv_loi(bplane),
        earth_leg_tof_sec=tof,
        moon_leg_tof_sec=moon_leg,
        n_grid_evals=len(tof_grid),
    )


# ---------------------------------------------------------------------------
# 到达模式
# ---------------------------------------------------------------------------


def _residual(bplane: BPlaneParams, target: PcnBplaneTarget, rp_target_km: float) -> np.ndarray:
    """残差 ``(B·R − B·R*, B·T − B·T*, r_p − r_p*)`` (km)。"""
    return np.array(
        [
            bplane.bdot_r_km - target.bdot_r_km,
            bplane.bdot_t_km - target.bdot_t_km,
            bplane.perilune_radius_km - rp_target_km,
        ]
    )


def _solve_arrival_mode(
    tli_params: TliParams,
    target: PcnBplaneTarget,
    moon_state_fn: Callable[[float], np.ndarray],
    p: PcnSearchParams,
) -> PcnSolution:
    """到达模式：网格初猜 + 阻尼 Newton 打靶命中月心 B-plane 目标。"""
    rp_target = _R_MOON_KM + target.perilune_alt_km

    def evaluate(x: np.ndarray, tof: float) -> tuple[np.ndarray, BPlaneParams] | None:
        """决策变量 x=(RHA, DHA, C3) → (月心相对态, B-plane)；失败/出射分支 None。"""
        try:
            asym = AsymptoteParams(rha_deg=float(x[0]), dha_deg=float(x[1]), c3_km2_s2=float(x[2]))
            dep, _ = _departure_state_from_asymptote(asym, tli_params)
            enc = _moon_encounter_state(dep, tof, moon_state_fn)
            bplane = bplane_from_state(enc, _MU_MOON)
            # 仅接受入射分支（到近月点时间 ≥ 0）：交接点须在近月点之前
            if hyperbolic_time_to_periapsis(enc, _MU_MOON) < 0.0:
                return None
            return enc, bplane
        except (ValueError, RuntimeError, PropagationFailure, FloatingPointError):
            return None

    # 1. 网格初猜：逐点最小残差 max-范数。
    # 触发交会的出发渐近线赤经应接近交会时刻月球位置方向（理想化下月球恒在 xy 面），
    # 故 rha 网格按「相对月球方向偏移」采样（绝对 rha = 月球 RHA + 偏移）。
    tof_grid = np.linspace(p.tof_range_days[0], p.tof_range_days[1], p.n_tof) * SECONDS_PER_DAY
    rha_offset_grid = np.linspace(p.rha_grid_deg[0], p.rha_grid_deg[1], p.n_rha)
    dha_grid = np.linspace(p.dha_grid_deg[0], p.dha_grid_deg[1], p.n_dha)
    c3_grid = np.linspace(p.c3_grid_km2_s2[0], p.c3_grid_km2_s2[1], p.n_c3)

    n_evals = 0
    best: tuple[float, float, np.ndarray, np.ndarray, BPlaneParams] | None = None
    for tof in tof_grid:
        moon_vec = np.asarray(moon_state_fn(float(tof))[:3], dtype=float)
        moon_rha = math.degrees(math.atan2(float(moon_vec[1]), float(moon_vec[0])))
        for rha_offset in rha_offset_grid:
            rha = (moon_rha + float(rha_offset)) % 360.0
            for dha in dha_grid:
                for c3 in c3_grid:
                    n_evals += 1
                    x = np.array([rha, dha, c3])
                    res = evaluate(x, tof)
                    if res is None:
                        continue
                    enc, bplane = res
                    norm = float(np.max(np.abs(_residual(bplane, target, rp_target))))
                    if best is None or norm < best[0]:
                        best = (norm, float(tof), x, enc, bplane)

    if best is None:
        return PcnSolution(
            mode="arrival",
            status=ConvergenceState.INFEASIBLE,
            cause=FailureCause.NO_INTERSECTION,
            message="出发渐近线网格无可行入射双曲到达（全部退化/非双曲/出射分支）",
            n_grid_evals=n_evals,
        )

    _, tof_fixed, x, _, bplane_best = best

    def ensure_feasible(x_cur: np.ndarray) -> tuple[np.ndarray | None, BPlaneParams | None]:
        """决策变量 → B-plane（供 Newton 使用）；失败 None。"""
        res = evaluate(x_cur, tof_fixed)
        if res is None:
            return None, None
        return res[0], res[1]

    # 2. 阻尼 Newton（雅可比 = 解析 B-plane 雅可比 × ∂月心态/∂x 中心差分）
    x_cur = x.copy()
    bplane_cur = bplane_best
    res_cur = _residual(bplane_cur, target, rp_target)
    n_newton = 0
    for it in range(1, p.max_iterations + 1):
        n_newton = it
        if float(np.max(np.abs(res_cur))) < p.tolerance_km:
            break
        jac = _residual_jacobian(ensure_feasible, x_cur)
        if jac is None:
            return PcnSolution(
                mode="arrival",
                status=ConvergenceState.DIVERGED,
                cause=FailureCause.DIVERGENCE_DETECTED,
                message=f"第 {it} 次迭代雅可比构造失败（传播/非双曲）",
                n_grid_evals=n_evals,
                n_newton_iter=it,
            )
        try:
            delta = np.linalg.solve(jac, -res_cur)
        except np.linalg.LinAlgError:
            return _arrival_failure(
                ConvergenceState.FAILED,
                FailureCause.SINGULAR_JACOBIAN,
                "打靶雅可比奇异",
                n_evals,
                it,
            )
        if not np.all(np.isfinite(delta)):
            return _arrival_failure(
                ConvergenceState.FAILED,
                FailureCause.SINGULAR_JACOBIAN,
                "打靶修正量非有限",
                n_evals,
                it,
            )

        alpha = 1.0
        improved = False
        for _ in range(_LINE_SEARCH_STEPS):
            x_trial = x_cur + alpha * delta
            enc_trial, bplane_trial = ensure_feasible(x_trial)
            if bplane_trial is not None:
                res_trial = _residual(bplane_trial, target, rp_target)
                if float(np.max(np.abs(res_trial))) < float(np.max(np.abs(res_cur))):
                    x_cur, bplane_cur, res_cur = x_trial, bplane_trial, res_trial
                    improved = True
                    break
            alpha *= 0.5
        if not improved:
            return _arrival_failure(
                ConvergenceState.STAGNATED,
                FailureCause.STAGNATION_DETECTED,
                "线搜索无法降低残差",
                n_evals,
                it,
                bplane_cur,
            )

    res_final = float(np.max(np.abs(res_cur)))
    if res_final >= p.tolerance_km:
        return _arrival_failure(
            ConvergenceState.MAX_ITERATIONS,
            FailureCause.MAX_ITERATIONS_REACHED,
            f"达到最大迭代次数，残差 {res_final:.3e} km",
            n_evals,
            n_newton,
            bplane_cur,
            x_cur,
            tof_fixed,
        )

    enc_final, bplane_final = ensure_feasible(x_cur)
    if enc_final is None or bplane_final is None:
        return _arrival_failure(
            ConvergenceState.FAILED,
            FailureCause.BACKEND_FAILURE,
            "收敛后重新求值失败",
            n_evals,
            n_newton,
        )
    rha_c, dha_c = _canonical_rha_dha(float(x_cur[0]), float(x_cur[1]))
    asym_final = AsymptoteParams(rha_deg=rha_c, dha_deg=dha_c, c3_km2_s2=float(x_cur[2]))
    dep_final, dv_tli = _departure_state_from_asymptote(asym_final, tli_params)
    return PcnSolution(
        mode="arrival",
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message=f"PCN 到达模式打靶收敛，残差 {res_final:.3e} km",
        departure_state_gcrs=dep_final,
        dv_tli_km_s=dv_tli,
        departure_asymptote=asym_final,
        encounter_state_moon=enc_final,
        bplane=bplane_final,
        perilune_state_moon=_perilune_state_from_bplane(bplane_final),
        dv_loi_km_s=_dv_loi(bplane_final),
        earth_leg_tof_sec=tof_fixed,
        moon_leg_tof_sec=hyperbolic_time_to_periapsis(enc_final, _MU_MOON),
        n_grid_evals=n_evals,
        n_newton_iter=n_newton,
    )


def _arrival_failure(
    status: ConvergenceState,
    cause: FailureCause,
    message: str,
    n_evals: int,
    n_newton: int,
    bplane: BPlaneParams | None = None,
    x: np.ndarray | None = None,
    tof: float = 0.0,
) -> PcnSolution:
    """构造到达模式失败解（附残差已知的达成 B-plane，便于诊断）。"""
    asym = None
    if x is not None:
        rha_c, dha_c = _canonical_rha_dha(float(x[0]), float(x[1]))
        asym = AsymptoteParams(rha_deg=rha_c, dha_deg=dha_c, c3_km2_s2=float(x[2]))
    return PcnSolution(
        mode="arrival",
        status=status,
        cause=cause,
        message=message,
        departure_asymptote=asym,
        bplane=bplane,
        earth_leg_tof_sec=tof,
        n_grid_evals=n_evals,
        n_newton_iter=n_newton,
    )


def _residual_jacobian(
    ensure_feasible: Callable[[np.ndarray], tuple[np.ndarray | None, BPlaneParams | None]],
    x: np.ndarray,
) -> np.ndarray | None:
    """残差对决策变量的雅可比 (3, 3)。

    ``∂R/∂x = ∂(B·R,B·T,r_p)/∂(月心态) @ ∂月心态/∂x``，前者为解析
    :func:`bplane_jacobian_from_state`，后者为 3 列中心差分。任一侧失败返回 None。
    """
    enc0, _ = ensure_feasible(x)
    if enc0 is None:
        return None
    try:
        jac_state = bplane_jacobian_from_state(enc0, _MU_MOON)  # (3, 6)
    except ValueError:
        return None

    steps = np.array([_ANGLE_FD_STEP_DEG, _ANGLE_FD_STEP_DEG, _C3_FD_STEP])
    denc_dx = np.zeros((6, 3))
    for j in range(3):
        xp = x.copy()
        xp[j] += steps[j]
        xm = x.copy()
        xm[j] -= steps[j]
        enc_p, _ = ensure_feasible(xp)
        enc_m, _ = ensure_feasible(xm)
        if enc_p is None or enc_m is None:
            return None
        denc_dx[:, j] = (enc_p - enc_m) / (2.0 * steps[j])
    jac = jac_state @ denc_dx
    if not np.all(np.isfinite(jac)):
        return None
    return jac
