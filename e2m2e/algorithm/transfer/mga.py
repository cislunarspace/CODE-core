"""行星际多借力（MGA）链的纯网格搜索。

三个部件都在本模块，都不依赖优化器：

- leg 转移速度由既有 Rust 批量 Lambert 给出（
  :func:`~e2m2e.algorithm.transfer.lambert.solve_lambert_batch`，每片每 leg 一次调用，
  方向 short、revs=0 写死）；
- flyby 只做**无动力**（零冲量）V∞ 旋转可行性筛选：``|v∞in| = |v∞out|`` 等模容差
  加近心点半径下限；转角与近心点半径互为闭式；
- Tisserand 参数以该飞越天体日心态的密切半长轴为参考，借力前后各报一个值，供
  调用方自筛（本模块不拿它当剔除条件）。

搜索是纯网格枚举（无 nsga2、无精化、无通用段框架）：发射历元网格 × 逐 leg TOF
网格的全部组合逐格求解，任一 leg 无解或任一 flyby 不可行即弃格，剩余候选按
``(总 ΔV, 到达 v∞)`` 升序取 top-N。

⑦ 类人工对照（ADR 0055 决策 3/5）：ARM 任务链（Strange 2013 §III）报告的行星际段
ΔV 为数 km/s 量级——非断言人工对照（ADR 0055 决策 3：⑦ 类结果性数值禁入断言）。

星历边界只要求 ``ephemeris.get_body_state(target, et, frame, observer)`` 与
``ephemeris.get_gm(body)`` 两个方法，故闭式与网格组装可由合成星历注入测试；
模块级不 import ``SPICEManager``（缺省星历路径在函数内懒加载）。
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from e2m2e.data.constants import SECONDS_PER_DAY

from ...status import ConvergenceState, FailureCause, ResultStatus
from .lambert import solve_lambert_batch
from .qlaw import rv_to_keplerian

__all__ = [
    "FlybyEvaluation",
    "MgaCandidate",
    "MgaSearchResult",
    "evaluate_flyby",
    "flyby_pericenter_radius",
    "flyby_turn_angle",
    "heliocentric_tisserand",
    "search_mga_chains",
]

#: 日心状态查询帧（J2000 惯性）与观察者（日心原点）；MGA 链一律日心几何。
_FRAME = "J2000"
_OBSERVER = "SUN"


def flyby_turn_angle(r_p_km: float, v_inf_km_s: float, mu_km3_s2: float) -> float:
    """无动力双曲飞越的 V∞ 转角闭式 ``δ = 2·asin(1/(1 + r_p·v∞²/μ))``，弧度。

    等价于 ``sin(δ/2) = 1/e``（双曲偏心率 ``e = 1 + r_p·v∞²/μ``，ADR 0055 决策 5 ①）。
    r_p 越小转角越大；``r_p·v∞²/μ → ∞`` 时 ``δ → 0``（擦过头顶的零退化极限）。

    Raises:
        ValueError: 任一入参非正的有限值。
    """
    for name, value in (
        ("r_p_km", r_p_km),
        ("v_inf_km_s", v_inf_km_s),
        ("mu_km3_s2", mu_km3_s2),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} 必须为正的有限值，得到 {value!r}")
    eccentricity = 1.0 + r_p_km * v_inf_km_s**2 / mu_km3_s2
    return 2.0 * math.asin(1.0 / eccentricity)


def flyby_pericenter_radius(delta_rad: float, v_inf_km_s: float, mu_km3_s2: float) -> float:
    """转角反闭式 ``r_p = μ/v∞²·(1/sin(δ/2) − 1)``，km。

    ``δ == 0``（V∞ 方向不变）时返回 ``math.inf``：零退化极限下任意远的近心点
    都能给出零转角，即该 flyby 无几何约束。

    Raises:
        ValueError: ``v_inf_km_s``/``mu_km3_s2`` 非正，或 ``delta_rad`` 不在 ``[0, π)``。
    """
    for name, value in (("v_inf_km_s", v_inf_km_s), ("mu_km3_s2", mu_km3_s2)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} 必须为正的有限值，得到 {value!r}")
    if not math.isfinite(delta_rad) or not 0.0 <= delta_rad < math.pi:
        raise ValueError(f"delta_rad 须在 [0, π) 内，得到 {delta_rad!r}")
    half_sine = math.sin(0.5 * delta_rad)
    if half_sine == 0.0:
        return math.inf
    return mu_km3_s2 / v_inf_km_s**2 * (1.0 / half_sine - 1.0)


def heliocentric_tisserand(
    r_km: npt.ArrayLike,
    v_km_s: npt.ArrayLike,
    mu_sun_km3_s2: float,
    a_ref_km: float,
    ref_normal: npt.ArrayLike | None = None,
) -> float:
    """以参考行星**轨道面与半长轴**归一化的日心 Tisserand 参数。

    ``T = a_ref/a + 2·(ĥ·n̂_ref)·|h|/sqrt(μ☉·a_ref)``，其中 ``r``/``v`` 为日心
    J2000 状态、``a`` 为该状态的密切半长轴、``n̂_ref`` 为**参考行星的轨道面法向**
    （``ref_normal``；``None`` 时退化为 J2000 赤道极点 ẑ）。第二项的 ``p`` 因子写成
    ``|h|/sqrt(μ·a_ref)``（对双曲日心弧仍为实值，不依赖 e）。

    参考面必须取参考行星的轨道面：经典 Tisserand 参数的 ``cos i`` 是相对该行星轨道
    面的倾角——黄道面内侧向转移若误用赤道极点，本项被 ``cos 23.44°`` 系统性压低
    （约低 0.17，T ≈ 2.9 时相对偏差 ~6%）。当 ``|r| = a_ref`` 且 ``n̂_ref`` 即该天体
    轨道法向时恒等 ``T = 3 − v∞²/v_c²``，故无动力借力前后 T 名义不变。

    ``a_ref`` 由参考行星自身的日心态密切根数给出（不建常数表）。

    Raises:
        ValueError: ``mu_sun_km3_s2``/``a_ref_km`` 非正，状态形状非 ``(3,)``，
            ``ref_normal`` 形状非 ``(3,)`` 或为零向量，或日心角动量为零
            （退化状态，T 无定义）。
    """
    for name, value in (("mu_sun_km3_s2", mu_sun_km3_s2), ("a_ref_km", a_ref_km)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} 必须为正的有限值，得到 {value!r}")
    r_vec = np.asarray(r_km, dtype=float)
    v_vec = np.asarray(v_km_s, dtype=float)
    if r_vec.shape != (3,) or v_vec.shape != (3,):
        raise ValueError(f"r_km/v_km_s 须为长度 3 的向量，得到 {r_vec.shape} 与 {v_vec.shape}")
    h_vec = np.cross(r_vec, v_vec)
    h_norm = float(np.linalg.norm(h_vec))
    if h_norm == 0.0:
        raise ValueError("日心角动量为零（退化状态）：Tisserand 参数无定义")
    if ref_normal is None:
        h_projected = float(h_vec[2])
    else:
        normal = np.asarray(ref_normal, dtype=float)
        if normal.shape != (3,):
            raise ValueError(f"ref_normal 须为长度 3 的向量，得到 {normal.shape}")
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm == 0.0:
            raise ValueError("ref_normal 为零向量：参考轨道面未定义")
        h_projected = float(np.dot(h_vec, normal)) / normal_norm
    semi_major, *_ = rv_to_keplerian(r_vec, v_vec, mu_sun_km3_s2)
    return float(a_ref_km / semi_major + 2.0 * h_projected / math.sqrt(mu_sun_km3_s2 * a_ref_km))


@dataclass(frozen=True)
class FlybyEvaluation:
    """单次无动力飞越的可行性评估（已通过等模与近心点下限筛选）。

    Attributes:
        body: 飞越天体名（SPICE 天体名）。
        v_inf_km_s: 进入/离开双曲段 V∞ 模（等模筛选后 in/out 的均值，km/s）。
        turn_angle_rad: V∞ 方向转角 δ，弧度。
        pericenter_radius_km: 达成 δ 所需的双曲近心点半径（km）；δ == 0 时为
            ``math.inf``（零退化，无几何约束）。
        tisserand_before: 借力前日心 Tisserand 参数（参考 = 该天体半长轴）。
        tisserand_after: 借力后同口径 Tisserand 参数（无动力下名义不变）。
    """

    body: str
    v_inf_km_s: float
    turn_angle_rad: float
    pericenter_radius_km: float
    tisserand_before: float
    tisserand_after: float


@dataclass(frozen=True)
class MgaCandidate:
    """一条可行的 MGA 链候选。

    Attributes:
        launch_epoch_et: 发射历元，SPICE ET 秒。
        leg_tofs_days: 逐 leg 飞行时间（天），长度 = 天体数 − 1。
        departure_v_inf_km_s: 出发双曲剩余速度模（km/s）。
        arrival_v_inf_km_s: 到达双曲剩余速度模（km/s）。
        total_delta_v_km_s: 目标函数，``出发 v∞ + 到达 v∞``（MGA 快筛惯例；MVP
            不含中途冲量）。
        flybys: 逐中间天体的飞越评估，长度 = 天体数 − 2（两体链为空元组）。
    """

    launch_epoch_et: float
    leg_tofs_days: tuple[float, ...]
    departure_v_inf_km_s: float
    arrival_v_inf_km_s: float
    total_delta_v_km_s: float
    flybys: tuple[FlybyEvaluation, ...]


@dataclass(frozen=True)
class MgaSearchResult:
    """MGA 网格搜索的最终结果（统一状态三元组 + top-N 候选）。"""

    status: ConvergenceState
    cause: FailureCause
    message: str
    candidates: tuple[MgaCandidate, ...]

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)


def evaluate_flyby(
    v_inf_in: npt.ArrayLike,
    v_inf_out: npt.ArrayLike,
    body_state: npt.ArrayLike,
    body: str,
    mu_body_km3_s2: float,
    mu_sun_km3_s2: float,
    a_ref_km: float,
    r_p_min_km: float,
    v_inf_match_tol_km_s: float,
) -> FlybyEvaluation | None:
    """评估一次无动力飞越，返回 ``None`` 表示该格被剔除。

    ``v_inf_in``/``v_inf_out`` 为进入/离开时相对飞越天体的速度矢量（km/s，
    已减去天体速度）；``body_state`` 为该天体日心 ``(6,)`` 状态（km, km/s）。
    顺序：等模筛选 ``| |v∞in| − |v∞out| | <= tol`` → 转角 → 近心点半径下限
    ``r_p >= r_p_min_km`` → 借力前后 Tisserand 值。

    剔除面：等模超差、``r_p`` 低于下限、V∞ 为零向量（与天体同速，转角与近心点
    半径无定义）。``δ == 0`` 时 ``r_p`` 为 ``inf``，恒通过下限。

    ``tisserand_before``/``tisserand_after`` 的参考为**该飞越天体日心态的轨道面与
    密切半长轴**。无动力飞越下 T 不变由等模约束承担（``|ΔT|`` 只由 ``Δ|v∞|`` 驱动，
    ``T = 3 − v∞²/v_c²``），故本函数不另设基于 T 的剔除条件——报告值供调用方自筛与
    审计（若改为独立剔除项，与等模容差等价，且在 ``|r| ≠ a_ref`` 处 T 只是名义不变，
    反而会剔掉合法格）。

    Args:
        v_inf_in: 进入速度矢量（3,），km/s。
        v_inf_out: 离开速度矢量（3,），km/s。
        body_state: 飞越天体日心状态（6,），km / km/s。
        body: 飞越天体名。
        mu_body_km3_s2: 天体 GM，km³/s²。
        mu_sun_km3_s2: 太阳 GM，km³/s²。
        a_ref_km: 参考半长轴（该天体日心态密切 a），km。
        r_p_min_km: 允许的最小近心点半径，km（体半径倍数换算由调用方负责）。
        v_inf_match_tol_km_s: 等模筛选容差，km/s（含边界，``<=`` 不剔）。
    """
    v_in = np.asarray(v_inf_in, dtype=float)
    v_out = np.asarray(v_inf_out, dtype=float)
    for name, vec in (("v_inf_in", v_in), ("v_inf_out", v_out)):
        if vec.shape != (3,):
            raise ValueError(f"{name} 须为长度 3 的向量，得到 {vec.shape}")
    norm_in = float(np.linalg.norm(v_in))
    norm_out = float(np.linalg.norm(v_out))
    if norm_in == 0.0 or norm_out == 0.0:
        return None
    if abs(norm_in - norm_out) > v_inf_match_tol_km_s:
        return None
    v_inf_eff = 0.5 * (norm_in + norm_out)
    cos_delta = float(np.dot(v_in, v_out) / (norm_in * norm_out))
    delta = math.acos(max(-1.0, min(1.0, cos_delta)))
    r_p = flyby_pericenter_radius(delta, v_inf_eff, mu_body_km3_s2)
    if r_p < r_p_min_km:
        return None
    state = np.asarray(body_state, dtype=float)
    if state.shape != (6,):
        raise ValueError(f"body_state 须为长度 6 的日心状态，得到 {state.shape}")
    position = state[:3]
    velocity = state[3:]
    # Tisserand 参考面 = 该天体自身的轨道面（经典定义相对参考行星轨道面，而非赤道面）。
    body_momentum = np.cross(position, velocity)
    body_momentum_norm = float(np.linalg.norm(body_momentum))
    if body_momentum_norm == 0.0:
        raise ValueError(f"{body} 的日心态角动量为零：Tisserand 参考轨道面未定义")
    ref_normal = body_momentum / body_momentum_norm
    tisserand_before = heliocentric_tisserand(
        position, velocity + v_in, mu_sun_km3_s2, a_ref_km, ref_normal
    )
    tisserand_after = heliocentric_tisserand(
        position, velocity + v_out, mu_sun_km3_s2, a_ref_km, ref_normal
    )
    return FlybyEvaluation(
        body=body,
        v_inf_km_s=v_inf_eff,
        turn_angle_rad=delta,
        pericenter_radius_km=r_p,
        tisserand_before=tisserand_before,
        tisserand_after=tisserand_after,
    )


def search_mga_chains(
    body_sequence: Sequence[str],
    launch_epochs_et: Sequence[float],
    leg_tof_grids_days: Sequence[Sequence[float]],
    min_flyby_pericenter_km: Sequence[float],
    *,
    ephemeris: Any | None = None,
    kernel_dir: str | None = None,
    mu_sun_km3_s2: float | None = None,
    v_inf_match_tol_km_s: float = 0.1,
    top_n: int = 5,
    progress_callback: Callable[[int], Any] | None = None,
) -> MgaSearchResult:
    """MGA 链纯网格搜索：发射历元 × 逐 leg TOF 全组合，逐格筛选后取 top-N。

    网格结构：外层发射历元（``launch_epochs_et``），内层逐 leg TOF（
    ``leg_tof_grids_days``）。每个 leg 的批量 Lambert 在**该片的全部前缀 TOF
    组合**上一次求解，再按 (前缀组合, 该 leg 的 TOF) 索引取回对角项——代价是
    每片每 leg 最多 M 倍过量求解（M = 该 leg 的 TOF 取值数），网格规模下可忽略，
    换来逐片单次 Rust 调用。

    候选定义：每个 leg 取 short-way/零圈 Lambert 解；出发 v∞ = ``|v0₀ − v_出发体|``，
    到达 v∞ = ``|vf_末 − v_到达体|``，总 ΔV = 两者之和；每个中间天体在相应历元做
    无动力 flyby 评估（:func:`evaluate_flyby`），任一 leg 无解或任一 flyby 不可行
    即弃格。借力前后日心 Tisserand 随候选同报（参考面与半长轴取该飞越天体日心态）：
    “T 不变”即无动力飞越的等模约束本身（``|ΔT|`` 只由 ``Δ|v∞|`` 驱动），故不作
    独立剔除条件，值供调用方自筛与审计。

    Args:
        body_sequence: 天体序列（SPICE 天体名）：首项出发体、末项到达体、中间项为
            飞越体，长度 ≥ 2。
        launch_epochs_et: 发射历元网格，SPICE ET 秒（非空）。
        leg_tof_grids_days: 逐 leg TOF 网格（天），长度 = 天体数 − 1，每格非空且
            元素为正的有限值。
        min_flyby_pericenter_km: 逐飞越天体最小近心点半径（km），长度 = 天体数 − 2
            （两体序列为空表），元素为正的有限值。
        ephemeris: 星历提供者；缺省 ``None`` 时懒加载 ``SPICEManager()`` 并
            ``load_design_kernels(spice, kernel_dir)``。需要
            ``get_body_state(target, et, frame, observer)`` 与 ``get_gm(body)``。
        kernel_dir: 缺省星历路径的内核目录；``None`` 用仓库缺省。
        mu_sun_km3_s2: 太阳 GM（km³/s²）；``None`` 取 ``ephemeris.get_gm("SUN")``。
        v_inf_match_tol_km_s: 无动力 flyby 等模筛选容差（km/s），含边界。
        top_n: 返回候选数上限（按总 ΔV 升序，同值按到达 v∞ 升序）。
        progress_callback: ``cb(delta: int)``，每完成一个 (发射历元, leg) 批调用
            一次 ``cb(1)``；本层不吞回调异常（适配器负责吞，见 Facade）。

    Returns:
        :class:`MgaSearchResult`：``(CONVERGED, NONE)`` 且有候选；无任何格全 leg
        有解为 ``(INFEASIBLE, NO_INTERSECTION)``；有解格全被 flyby 约束剔除为
        ``(INFEASIBLE, CONSTRAINT_VIOLATION)``。

    Raises:
        ValueError: 入参长度/取值非法，或星历返回值形状不符。
    """
    bodies = [str(body) for body in body_sequence]
    n_bodies = len(bodies)
    if n_bodies < 2:
        raise ValueError(f"body_sequence 至少两个天体（出发体…到达体），得到 {n_bodies} 个")
    n_legs = n_bodies - 1
    if len(leg_tof_grids_days) != n_legs:
        raise ValueError(
            f"leg_tof_grids_days 长度须为 {n_legs}（天体序列长度 − 1），"
            f"得到 {len(leg_tof_grids_days)}"
        )
    grids: list[tuple[float, ...]] = []
    for k, grid in enumerate(leg_tof_grids_days):
        tofs = tuple(float(tof) for tof in grid)
        if not tofs:
            raise ValueError(f"leg {k} 的 TOF 网格为空")
        if any(not math.isfinite(tof) or tof <= 0.0 for tof in tofs):
            raise ValueError(f"leg {k} 的 TOF 网格须全为正的有限天数，得到 {tofs}")
        grids.append(tofs)
    r_p_mins = [float(value) for value in min_flyby_pericenter_km]
    if len(r_p_mins) != n_bodies - 2:
        raise ValueError(
            f"min_flyby_pericenter_km 长度须为 {n_bodies - 2}（天体序列长度 − 2），"
            f"得到 {len(r_p_mins)}"
        )
    if any(not math.isfinite(value) or value <= 0.0 for value in r_p_mins):
        raise ValueError(f"min_flyby_pericenter_km 须全为正的有限值，得到 {r_p_mins}")
    epochs = [float(epoch) for epoch in launch_epochs_et]
    if not epochs:
        raise ValueError("launch_epochs_et 为空")
    if any(not math.isfinite(epoch) for epoch in epochs):
        raise ValueError("launch_epochs_et 须全为有限值")
    if top_n < 1:
        raise ValueError(f"top_n 须 ≥ 1，得到 {top_n}")
    if not math.isfinite(v_inf_match_tol_km_s) or v_inf_match_tol_km_s < 0.0:
        raise ValueError(f"v_inf_match_tol_km_s 须为非负的有限值，得到 {v_inf_match_tol_km_s}")

    if ephemeris is None:
        from e2m2e.data.kernels.manager import SPICEManager

        from ..design.design_orbit import load_design_kernels

        spice = SPICEManager()
        load_design_kernels(spice, kernel_dir)
        ephemeris = spice
    mu_sun = float(ephemeris.get_gm("SUN")) if mu_sun_km3_s2 is None else float(mu_sun_km3_s2)
    if not math.isfinite(mu_sun) or mu_sun <= 0.0:
        raise ValueError(f"太阳 GM 须为正的有限值，得到 {mu_sun}")

    state_cache: dict[tuple[str, float], np.ndarray] = {}

    def body_state(body: str, et: float) -> np.ndarray:
        """日心状态（6,），按 (body, et) 记忆化。"""
        key = (body, et)
        cached = state_cache.get(key)
        if cached is None:
            cached = np.asarray(ephemeris.get_body_state(body, et, _FRAME, _OBSERVER), dtype=float)
            if cached.shape != (6,):
                raise ValueError(f"星历返回 {body} 在 et={et} 的状态形状 {cached.shape}，期望 (6,)")
            state_cache[key] = cached
        return cached

    # 只有飞越天体需要 GM（出发/到达体只用到位置与速度），每个天体查一次。
    mu_body_by_name = {body: float(ephemeris.get_gm(body)) for body in bodies[1:-1]}

    # ---- 逐片逐 leg 批量 Lambert：对角项即该 leg 的转移速度 ----
    # per_leg[k] = (v0, vf)，两者形状均为 (prod_{i<=k} len(grids[i]), 3)；行索引是
    # 该 leg 全部前缀 TOF 组合的混合进制编码。
    solved_lists: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for launch_et in epochs:
        per_leg: list[tuple[np.ndarray, np.ndarray]] = []
        for k in range(n_legs):
            prefixes = list(itertools.product(*grids[: k + 1]))
            tof_values = sorted(set(grids[k]))
            tof_index = {tof: index for index, tof in enumerate(tof_values)}
            n_prefixes = len(prefixes)
            r0 = np.empty((n_prefixes, 3))
            rf = np.empty((n_prefixes, 3))
            diagonal = np.empty(n_prefixes, dtype=int)
            for i, combo in enumerate(prefixes):
                dep_et = launch_et + sum(combo[:k]) * SECONDS_PER_DAY
                arr_et = dep_et + combo[k] * SECONDS_PER_DAY
                r0[i] = body_state(bodies[k], dep_et)[:3]
                rf[i] = body_state(bodies[k + 1], arr_et)[:3]
                diagonal[i] = tof_index[combo[k]]
            solutions = solve_lambert_batch(
                r0,
                rf,
                [tof * SECONDS_PER_DAY for tof in tof_values],
                mu_sun,
            )
            rows = np.arange(n_prefixes)
            per_leg.append((solutions[rows, diagonal, 0, :], solutions[rows, diagonal, 1, :]))
            if progress_callback is not None:
                progress_callback(1)
        solved_lists.append(per_leg)

    # 混合进制后缀积：leg k 的前缀组合索引 = 全组合索引 // suffix[k]。
    suffix = [1] * n_legs
    accumulator = 1
    for k in range(n_legs - 1, -1, -1):
        suffix[k] = accumulator
        accumulator *= len(grids[k])
    n_cells = accumulator * len(epochs)

    candidates: list[MgaCandidate] = []
    n_solved_cells = 0
    for launch_et, per_leg in zip(epochs, solved_lists, strict=True):
        for index, combo in enumerate(itertools.product(*grids)):
            v0_by_leg: list[np.ndarray] = []
            vf_by_leg: list[np.ndarray] = []
            all_solved = True
            for k in range(n_legs):
                prefix = index // suffix[k]
                v0 = per_leg[k][0][prefix]
                vf = per_leg[k][1][prefix]
                if not (np.isfinite(v0).all() and np.isfinite(vf).all()):
                    all_solved = False
                    break
                v0_by_leg.append(v0)
                vf_by_leg.append(vf)
            if not all_solved:
                continue
            n_solved_cells += 1
            feasible = True
            epochs_et = [launch_et]
            for tof in combo:
                epochs_et.append(epochs_et[-1] + tof * SECONDS_PER_DAY)
            departure_velocity = body_state(bodies[0], epochs_et[0])[3:]
            departure_v_inf = float(np.linalg.norm(v0_by_leg[0] - departure_velocity))
            flybys: list[FlybyEvaluation] = []
            for j in range(1, n_legs):
                flyby_body = bodies[j]
                flyby_state = body_state(flyby_body, epochs_et[j])
                a_ref, *_ = rv_to_keplerian(flyby_state[:3], flyby_state[3:], mu_sun)
                evaluation = evaluate_flyby(
                    vf_by_leg[j - 1] - flyby_state[3:],
                    v0_by_leg[j] - flyby_state[3:],
                    flyby_state,
                    flyby_body,
                    mu_body_by_name[flyby_body],
                    mu_sun,
                    float(a_ref),
                    r_p_mins[j - 1],
                    v_inf_match_tol_km_s,
                )
                if evaluation is None:
                    feasible = False
                    break
                flybys.append(evaluation)
            if not feasible:
                continue
            arrival_velocity = body_state(bodies[-1], epochs_et[-1])[3:]
            arrival_v_inf = float(np.linalg.norm(vf_by_leg[-1] - arrival_velocity))
            candidates.append(
                MgaCandidate(
                    launch_epoch_et=launch_et,
                    leg_tofs_days=tuple(float(tof) for tof in combo),
                    departure_v_inf_km_s=departure_v_inf,
                    arrival_v_inf_km_s=arrival_v_inf,
                    total_delta_v_km_s=departure_v_inf + arrival_v_inf,
                    flybys=tuple(flybys),
                )
            )

    if candidates:
        ordered = sorted(
            candidates, key=lambda cand: (cand.total_delta_v_km_s, cand.arrival_v_inf_km_s)
        )[:top_n]
        return MgaSearchResult(
            status=ConvergenceState.CONVERGED,
            cause=FailureCause.NONE,
            message=f"MGA 网格搜索完成：{len(candidates)} 可行候选 / {n_cells} 格",
            candidates=tuple(ordered),
        )
    if n_solved_cells == 0:
        return MgaSearchResult(
            status=ConvergenceState.INFEASIBLE,
            cause=FailureCause.NO_INTERSECTION,
            message="网格内 Lambert 无可行 leg",
            candidates=(),
        )
    return MgaSearchResult(
        status=ConvergenceState.INFEASIBLE,
        cause=FailureCause.CONSTRAINT_VIOLATION,
        message="全部格被 flyby 约束剔除（|v∞| 匹配或 r_p,min）",
        candidates=(),
    )
