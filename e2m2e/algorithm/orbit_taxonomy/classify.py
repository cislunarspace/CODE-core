"""轨道分类学分类器（orbit taxonomy classify）。

42 标签词汇见 ``labels.py``；本模块实现"给一条 CR3BP 会合系周期轨迹
推断族标签"的解析判据。词汇采自 STK 未发布 CODE 组件，**判据为
e2m2e 自定**——无公开出处可抄，全部阈值与级联顺序的权威记录是
ADR 0042，测试以随包 baseline 数据集为回归锚点。

输入两种形态（ADR 0042 决策 2）：

- 整条周期轨迹（``states`` n≥2，覆盖一个周期、首末闭合）——按输入
  原样消费，不重传播；
- 最小形态（``states`` (1,6) + ``period``——轨道族成员的仓库存储
  形态）——内部传播一个周期物化轨迹后统一处理。

判据级联（顺序即优先序，ADR 0042 决策 3）：

1. ``periodicity="quasi-periodic"`` → unclassified（合法输出非失败）。
2. **月心分支**：平面轨道（z 恒为零）、绕月净卷绕 ≈ ±2π、月心最大
   半径 ≤ ρ_SOI = μ^(2/5)（Chebotarev 口径，Primer §5.4.2）。SOI
   展布闸把深近月 NRHO 端成员（绕月但展布超 SOI）留给平动点分支。
   逆行 → distant_retrograde；顺行按 ρ_max 分 distant_prograde /
   low_prograde（东西 = 月心会合系近月点方向半平面，+y 朝月球公转
   方向为东）。
3. **L4/L5 分支**：平面轨道绕 L4 或 L5 净卷绕 ≈ ±2π：T/T☾ > 2 →
   longperiod_l{4,5}，否则 shortperiod_l{4,5}。
4. **共线平动点分支**：绕某共线 L 净卷绕，或（深近月端成员不卷绕
   任何 L 的回退路径）三维且有 x-z 面垂直穿越、距共线 L 足够近。
   侧别（L1/L2/L3）由轨迹时间平均 x 相对月球的符号定（L3 另加
   x̄ < -0.5 闸）。族形态按穿越几何：

   - x-z 面垂直穿越（穿越处 vx≈vz≈0）每周期 2/4/6 次 → halo /
     butterfly / dragonfly（南北 = vy<0 穿越处的 z 符号）；
   - 三维、有 x-z 面穿越但非垂直（axial 族种子 vz≠0）→ axial；
   - 三维、无垂直 x-z 穿越但有垂直 z=0 面穿越 → vertical；
   - 平面且垂直穿越 → lyapunov。

5. **共振分支**：绕质心净卷绕 ≈ ±2π 且 T/T☾ 与 11 个比值的 q/p
   通约（容差 0.01）→ resonant_p_q。
6. 其余 → unclassified("no_matching_label")。

多标签：月心/L4/L5 命中且周期同时通约 → 追加 resonant 标签
（primary 取月心/L4/L5 族）。设计侧族 ↔ 分类标签的映射与入库
冲突策略见 ADR 0042 映射表，落在 catalog_ingest 模块。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ...data.catalog.terminology import TAXONOMY_BY_CANONICAL, Hemisphere, TaxonomyLabel
from ...data.constants import Datum
from ...status import ConvergenceState, FailureCause, ResultStatus
from ..dynamics import CR3BP_Dynamics
from ..dynamics.cr3bp_system import CR3BP_System

__all__ = ["TaxonomyResult", "classify_orbit"]

#: 绕天体/平动点净卷绕判据：|Δθ| ≥ 1.5π 视为环绕一周。
_WINDING_GATE = 1.5 * math.pi

#: 月心分支的展布闸指数：ρ_max ≤ μ^(2/5)（Chebotarev SOI）。
_SOI_EXPONENT = 0.4

#: 顺行月心族的 low/distant 分界（无量纲 ≈12000 km）：仓库低月轨道
#: 参数域（近月高度 ≤10000 km）上缘加月半径的量级。
_LOW_PROGRADE_RHO_MAX = 0.031

#: L4/L5 长短周期分界：T/T☾ > 2 为长周期（baseline：spo≈1.05、lpo≈3.36）。
_LONGPERIOD_T_RATIO = 2.0

#: 平面判据：全程 |z| 最大值（平面族是会合系不变流形，传播后严格为 0；
#: 三维族最小 z 振幅 ≈6.5e-4）。
_PLANAR_Z_MAX = 1e-7

#: 垂直穿越判据：穿越处横向速度 |vx|、|vz|（y=0 面）或 |vx|、|vy|
#: （z=0 面）。halo/lyapunov 穿越处理论为 0；axial 族种子 vz ≥ 1e-3。
_PERP_V_GATE = 5e-4

#: 共振通约容差：|T/T☾ − q/p| < 0.01。
_RESONANCE_TOL = 0.01

#: L1/L3 侧别闸：时间平均 x < -0.5 归 L3（L1 族 x̄ ≥ -0.15）。
_L3_MEAN_X_GATE = -0.5

#: L4/L5 分支的局域化闸：轨道到三角平动点的最小距离（spo/lpo 实测
#: ≤0.095；远距绕地圆虽把 L4 圈在内但距离 ≥0.6，不得误入）。
_TRIANGULAR_EXTENT_GATE = 0.15

#: 共线分支的轨道-共线 L 最小距离闸（卷绕与回退路径共用；深近月
#: NRHO 端 ≈0.17，远距绕地圆 ≥0.6 不得误入）。
_COLLINEAR_EXTENT_GATE = 0.25

#: 最小形态内部传播的每周期采样点数。
_N_SAMPLES = 720

#: 轨迹闭合判据（首末状态差的相对范数）。
_CLOSURE_TOL = 1e-5

#: 11 个共振比 (p, q)，p:q = 卫星:月球，T/T☾ = q/p。
_RESONANCES: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 1),
    (3, 1),
    (3, 2),
    (3, 4),
    (2, 3),
    (4, 1),
    (4, 3),
)


@dataclass(frozen=True)
class TaxonomyResult:
    """分类结果（状态契约：status/cause/message 三元组）。

    unclassified 是合法输出（labels 为空 + unclassified_reason），
    不是失败；失败（非法输入、最小形态传播失败）走 FAILED 状态。
    """

    status: ConvergenceState
    cause: FailureCause
    message: str
    labels: tuple[TaxonomyLabel, ...] = ()
    unclassified_reason: str | None = None
    diagnostics: dict[str, float | int | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)

    @property
    def primary(self) -> TaxonomyLabel | None:
        """主标签（多标签时取第一项）。"""
        return self.labels[0] if self.labels else None

    @property
    def canonical_labels(self) -> tuple[str, ...]:
        """规范字符串形式的标签序列（序列化键，MCP/catalog 用）。"""
        return tuple(label.canonical for label in self.labels)


@lru_cache(maxsize=4)
def _default_dynamics(mu: float) -> CR3BP_Dynamics:
    """按 μ 缓存的默认地月 CR3BP 动力学（最小形态传播用）。"""
    return CR3BP_Dynamics(CR3BP_System(mu=mu, primary="earth", secondary="moon"))


@lru_cache(maxsize=4)
def _libration_points(mu: float) -> dict[int, np.ndarray]:
    """五平动点位置（解析求根并按 μ 缓存）。"""
    system = CR3BP_System(mu=mu, primary="earth", secondary="moon")
    points = system.compute_libration_points()
    return {int(point.name[1]): np.asarray(pos, dtype=float) for point, pos in points.items()}


def _resonance_label(t_ratio: float) -> TaxonomyLabel | None:
    """T/T☾ 与 11 个比值通约时返回 resonant 标签（取最近者）。"""
    best: tuple[float, tuple[int, int]] | None = None
    for p, q in _RESONANCES:
        gap = abs(t_ratio - q / p)
        if gap < _RESONANCE_TOL and (best is None or gap < best[0]):
            best = (gap, (p, q))
    if best is None:
        return None
    return TAXONOMY_BY_CANONICAL[f"resonant_{best[1][0]}_{best[1][1]}"]


@dataclass
class _Features:
    """一个周期轨迹的分类特征（全部无量纲会合系量）。"""

    t_ratio: float
    winding_moon: float
    winding_earth: float
    windings_l: dict[int, float]
    rho_max: float
    z_abs_max: float
    mean_x_rel_moon: float
    moon_prograde: bool | None  # None = 近月点处 h_z≈0（不判月心顺逆）
    perilune_azimuth: float  # 月心会合系近月点方向角（弧度，(-π, π]）
    crossings_y0: list[np.ndarray]  # y=0 面穿越状态（线性内插）
    perp_y0: list[np.ndarray]  # 其中垂直穿越（|vx|、|vz| 均小于门）的子集
    perp_z0: list[np.ndarray]  # z=0 面垂直穿越（|vx|、|vy| 均小于门）
    min_collinear_dist: float
    min_triangular_dist: float


def _winding(pos: np.ndarray, center_x: float, center_y: float) -> float:
    """绕 (center_x, center_y) 的净方位角变化（解卷绕）。"""
    theta = np.unwrap(np.arctan2(pos[:, 1] - center_y, pos[:, 0] - center_x))
    return float(theta[-1] - theta[0])


def _crossings(states: np.ndarray, axis: int) -> list[np.ndarray]:
    """``axis`` 分量变号处的线性内插穿越状态。

    ``axis=1`` 为 y=0（x-z 面）穿越；``axis=2`` 为 z=0（轨道面）穿越。
    周期轨迹首末点重复落在同一穿越上（如 halo 种子 y(0)=y(T)=0），
    末段（内插分数 >0.999）按周期重复丢弃，避免重复计数。
    """
    values = states[:, axis]
    crossed: list[np.ndarray] = []
    for k in np.where(np.diff(np.sign(values)) != 0)[0]:
        frac = -values[k] / (values[k + 1] - values[k])
        if frac > 0.999:
            continue
        crossed.append(states[k] + frac * (states[k + 1] - states[k]))
    return crossed


def _extract_features(states: np.ndarray, period: float, mu: float) -> _Features:
    pos = states[:, :3]
    moon_x = 1.0 - mu
    rel = pos - np.array([moon_x, 0.0, 0.0])
    rho = np.linalg.norm(rel, axis=1)
    l_positions = _libration_points(mu)

    perilune = int(np.argmin(rho))
    h_z = float(rel[perilune, 0] * states[perilune, 4] - rel[perilune, 1] * states[perilune, 3])
    crossings_y0 = _crossings(states, 1)
    perp_y0 = [s for s in crossings_y0 if abs(s[3]) < _PERP_V_GATE and abs(s[5]) < _PERP_V_GATE]
    perp_z0 = [
        s for s in _crossings(states, 2) if abs(s[3]) < _PERP_V_GATE and abs(s[4]) < _PERP_V_GATE
    ]
    return _Features(
        t_ratio=period / (2.0 * math.pi),
        winding_moon=_winding(pos, moon_x, 0.0),
        winding_earth=_winding(pos, 0.0, 0.0),
        windings_l={point: _winding(pos, lp[0], lp[1]) for point, lp in l_positions.items()},
        rho_max=float(rho.max()),
        z_abs_max=float(np.abs(pos[:, 2]).max()),
        mean_x_rel_moon=float(np.mean(pos[:, 0]) - moon_x),
        moon_prograde=(None if abs(h_z) < 1e-12 else h_z > 0.0),
        perilune_azimuth=float(np.arctan2(rel[perilune, 1], rel[perilune, 0])),
        crossings_y0=crossings_y0,
        perp_y0=perp_y0,
        perp_z0=perp_z0,
        min_collinear_dist=min(
            float(np.min(np.linalg.norm(pos - l_positions[point], axis=1))) for point in (1, 2, 3)
        ),
        min_triangular_dist=min(
            float(np.min(np.linalg.norm(pos - l_positions[point], axis=1))) for point in (4, 5)
        ),
    )


def _diagnostics(f: _Features, mu: float, reason: str | None) -> dict[str, float | int | str]:
    diag: dict[str, float | int | str] = {
        "t_over_t_moon": round(f.t_ratio, 6),
        "winding_moon": round(f.winding_moon, 3),
        "winding_earth": round(f.winding_earth, 3),
        "rho_max": round(f.rho_max, 6),
        "soi_moon": round(mu**_SOI_EXPONENT, 6),
        "z_abs_max": float(f.z_abs_max),
        "mean_x_rel_moon": round(f.mean_x_rel_moon, 6),
        "n_crossings_y0": len(f.crossings_y0),
        "n_perp_crossings_y0": len(f.perp_y0),
        "n_perp_crossings_z0": len(f.perp_z0),
    }
    if reason is not None:
        diag["unclassified_reason"] = reason
    return diag


def _ok(labels: list[TaxonomyLabel], f: _Features, mu: float) -> TaxonomyResult:
    return TaxonomyResult(
        ConvergenceState.CONVERGED,
        FailureCause.NONE,
        "ok",
        labels=tuple(labels),
        diagnostics=_diagnostics(f, mu, None),
    )


def _unclassified(reason: str, f: _Features, mu: float) -> TaxonomyResult:
    return TaxonomyResult(
        ConvergenceState.CONVERGED,
        FailureCause.NONE,
        "ok",
        unclassified_reason=reason,
        diagnostics=_diagnostics(f, mu, reason),
    )


def _classify_features(f: _Features, mu: float) -> TaxonomyResult:
    """对已提取特征执行判据级联。"""
    planar = f.z_abs_max <= _PLANAR_Z_MAX
    soi = mu**_SOI_EXPONENT
    wind_moon = abs(f.winding_moon) >= _WINDING_GATE

    # 2. 月心分支：平面、绕月、展布在 SOI 内；顺逆不可判（近月点 h_z≈0
    #    的退化轨迹）时不强贴月心标签，落到后续分支。
    if planar and wind_moon and f.rho_max <= soi and f.moon_prograde is not None:
        labels: list[TaxonomyLabel] = []
        if f.moon_prograde is False:
            labels.append(TAXONOMY_BY_CANONICAL["distant_retrograde"])
        else:
            if f.rho_max > _LOW_PROGRADE_RHO_MAX:
                labels.append(TAXONOMY_BY_CANONICAL["distant_prograde"])
            else:
                east = 0.0 < f.perilune_azimuth < math.pi
                labels.append(
                    TAXONOMY_BY_CANONICAL[
                        "low_prograde_eastern" if east else "low_prograde_western"
                    ]
                )
        resonance = _resonance_label(f.t_ratio)
        if resonance is not None:
            labels.append(resonance)
        return _ok(labels, f, mu)

    # 3. L4/L5 分支：平面、绕三角平动点一周且局域在其邻域内。
    if planar:
        for point in (4, 5):
            if (
                abs(f.windings_l[point]) >= _WINDING_GATE
                and f.min_triangular_dist <= _TRIANGULAR_EXTENT_GATE
            ):
                family = "longperiod" if f.t_ratio > _LONGPERIOD_T_RATIO else "shortperiod"
                labels = [TAXONOMY_BY_CANONICAL[f"{family}_l{point}"]]
                resonance = _resonance_label(f.t_ratio)
                if resonance is not None:
                    labels.append(resonance)
                return _ok(labels, f, mu)

    # 4. 共线平动点分支：卷绕或（三维垂直穿越形态的）回退路径，都要求
    # 轨道局域在共线点邻域内（远距绕地圆把共线点圈在内也不得误入）。
    wound = [point for point in (1, 2, 3) if abs(f.windings_l[point]) >= _WINDING_GATE]
    spatial = not planar
    halo_like = spatial and len(f.perp_y0) >= 2
    vertical_like = spatial and len(f.perp_z0) >= 2
    near_collinear = f.min_collinear_dist <= _COLLINEAR_EXTENT_GATE
    if near_collinear and (wound or halo_like or vertical_like):
        label = _collinear_family(f, _collinear_side(f, wound), spatial)
        if label is not None:
            return _ok([label], f, mu)

    # 5. 共振分支：绕质心环绕且周期通约。
    if abs(f.winding_earth) >= _WINDING_GATE:
        resonance = _resonance_label(f.t_ratio)
        if resonance is not None:
            return _ok([resonance], f, mu)

    return _unclassified("no_matching_label", f, mu)


def _collinear_side(f: _Features, wound: list[int]) -> int:
    """共线侧别：卷绕命中者优先；未卷绕（深近月端回退路径）由时间
    平均 x 相对月球的符号定（L1 族全体 x̄<0、L2 族全体 x̄>0，零重叠）。"""
    if wound:
        if len(wound) > 1:
            return min(wound)  # 横跨多点的极端振幅取靠地侧（ADR 0042）
        return wound[0]
    if f.mean_x_rel_moon < _L3_MEAN_X_GATE:
        return 3
    return 1 if f.mean_x_rel_moon < 0.0 else 2


def _hemisphere(crossings: list[np.ndarray]) -> Hemisphere:
    """南北：vy<0 的垂直穿越处 z 符号（与设计侧 halo_class 同源的几何量）。"""
    for state in crossings:
        if state[4] < 0.0:
            return Hemisphere.NORTHERN if state[2] > 0.0 else Hemisphere.SOUTHERN
    return Hemisphere.NORTHERN if crossings[0][2] > 0.0 else Hemisphere.SOUTHERN


def _collinear_family(f: _Features, point: int, spatial: bool) -> TaxonomyLabel | None:
    """共线平动点族形态判定（穿越几何，ADR 0042 决策 3）。"""
    if spatial:
        by_count = {2: "halo", 4: "butterfly", 6: "dragonfly"}
        family = by_count.get(len(f.perp_y0))
        if family is not None:
            hemisphere = _hemisphere(f.perp_y0)
            if family == "halo":
                return TAXONOMY_BY_CANONICAL[f"halo_l{point}_{hemisphere.value}"]
            return TAXONOMY_BY_CANONICAL[f"{family}_{hemisphere.value}"]
        if f.crossings_y0:
            # 三维且有 x-z 面穿越但非垂直（axial 族种子 vz≠0）。
            return TAXONOMY_BY_CANONICAL[f"axial_l{point}"]
        if f.perp_z0:
            return TAXONOMY_BY_CANONICAL[f"vertical_l{point}"]
        return None
    if f.perp_y0:
        return TAXONOMY_BY_CANONICAL[f"lyapunov_l{point}"]
    return None


def classify_orbit(
    states: np.ndarray,
    times: np.ndarray | None = None,
    *,
    period: float | None = None,
    mu: float | None = None,
    periodicity: str = "periodic",
) -> TaxonomyResult:
    """对一条 CR3BP 会合系周期轨迹做轨道分类学判定。

    Args:
        states: 状态序列 (n, 6)（无量纲会合系）。n=1 为最小形态（须给
            ``period``，内部传播一个周期）；n≥2 按输入原样消费，须覆盖
            一个周期且首末闭合。
        times: 时间序列（无量纲），``period`` 缺省时取首末跨度并校验闭合。
        period: 周期（无量纲）。
        mu: CR3BP 质量比；缺省取 DE421 地月值。
        periodicity: 族元数据的周期性标注；非 periodic 取值（如
            quasi-periodic）直接判 unclassified。

    Returns:
        :class:`TaxonomyResult`；unclassified 是合法输出（labels 空 +
        原因），非法输入/最小形态传播失败为 FAILED 状态。
    """
    array = np.asarray(states, dtype=float)
    if array.ndim != 2 or array.shape[1] != 6 or array.shape[0] == 0:
        return TaxonomyResult(
            ConvergenceState.FAILED,
            FailureCause.INVALID_INPUT,
            f"states 须为 (n, 6)，当前 {array.shape}",
        )
    if not np.all(np.isfinite(array)):
        return TaxonomyResult(
            ConvergenceState.FAILED, FailureCause.INVALID_INPUT, "states 含非有限值"
        )
    if periodicity != "periodic":
        return TaxonomyResult(
            ConvergenceState.CONVERGED,
            FailureCause.NONE,
            "ok",
            unclassified_reason="quasi_periodic",
            diagnostics={"periodicity": periodicity},
        )

    mu_value = Datum.DE421.mu if mu is None else float(mu)
    if period is None and array.shape[0] >= 2 and times is not None:
        time_array = np.asarray(times, dtype=float)
        if time_array.shape[0] != array.shape[0]:
            return TaxonomyResult(
                ConvergenceState.FAILED,
                FailureCause.INVALID_INPUT,
                "states 与 times 长度不一致",
            )
        period = float(time_array[-1] - time_array[0])
    if period is None:
        return TaxonomyResult(
            ConvergenceState.FAILED,
            FailureCause.INVALID_INPUT,
            "最小形态（单状态）必须提供 period",
        )
    if not math.isfinite(period) or period <= 0.0:
        return TaxonomyResult(
            ConvergenceState.FAILED,
            FailureCause.INVALID_INPUT,
            f"period 须为正有限值，当前 {period}",
        )

    if array.shape[0] == 1:
        try:
            propagated = _default_dynamics(mu_value).propagate(
                array[0], (0.0, period), t_eval=np.linspace(0.0, period, _N_SAMPLES)
            )
        except Exception as exc:  # 最小形态传播失败按积分失败契约上报
            return TaxonomyResult(
                ConvergenceState.FAILED,
                FailureCause.INTEGRATION_FAILED,
                f"周期传播失败：{exc}",
            )
        trajectory = np.asarray(propagated["states"], dtype=float)
    else:
        trajectory = array

    closure = float(np.linalg.norm(trajectory[-1] - trajectory[0]))
    scale = max(1.0, float(np.linalg.norm(trajectory[0])))
    if closure > _CLOSURE_TOL * scale:
        return TaxonomyResult(
            ConvergenceState.CONVERGED,
            FailureCause.NONE,
            "ok",
            unclassified_reason="non_periodic",
            diagnostics={"closure_error": closure},
        )

    features = _extract_features(trajectory, period, mu_value)
    return _classify_features(features, mu_value)
