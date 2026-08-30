"""六域两层天图制图管线（Primer §7.3 / Table 4，ADR 0041 Phase 3c）。

把 Phase 3a（解析骨架）与 3b（MEGNO + fate）组装成 (a/a☾, e) 网格天图：

- **统一初始条件切片**（论文 line 1413，命名场景）：历元 2027-08-02
  10:06:37 UTC（日全食）、(Ω, ω, M) = (311.07°, 355.84°, 0°)、
  ω = ω☾ + 180° 反 aligned、M = 0 近点起算、i = 历元月球轨道面——与
  Gallardo atlas / Rawat 近点 Poincaré 图同切片可直接对比。
- **六制图域**（Table 4，``table4_bands()`` 已有边界）：SC / CR / CG
 （19 yr）→ IT（38 yr）→ OT / TF（57 yr），相邻区 deliberate-overlap。
- **EM/EMS 双模型**：EM = 椭圆点质量地月（星历初值后地月孤立演化，
  月球固定开普勒椭圆）；EMS = +太阳点质量（固定日心视椭圆）。
  对照结论按区输出：骨架存续 / 仅提示 / 被重组。

自由参数（承接 #579，ADR 0041 Phase 3c 登记）：

- 月/日初始相位缺省值：月球 (Ω☾, ω☾) = (311.07°, 175.84°)（论文
  反 aligned 约定反推）、M☾ = 0；太阳真黄经 = 历元月球真黄经（日全食
  合）；可用 SPICE 星历初值覆盖。
- 终端事件早停（缺省开）；积分 max_step = 6 小时（近点偶极漏检防护）。
- 各区网格分辨率（Table 4 只给 a 范围与圈数；分辨率由调用方定，
  CI 走抽查小网格，全量制图走 scripts/ 手动）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...integrators import propagate_geocentric_fate_map_py, require_rust_extension
from ...status import ConvergenceState, FailureCause, ResultStatus
from . import regions as _regions
from .constants import PRIMER_DEFAULTS, PrimerConstants
from .fate import FATE_CLASSES, FateDiagnostics, FateThresholds, classify_fate
from .scales import hill_radius_earth, hill_radius_moon

__all__ = [
    "MAP_ZONE_NAMES",
    "ECLIPSE_EPOCH_SCENARIO",
    "MapResult",
    "Scenario",
    "compare_models",
    "default_scenario",
    "dynamical_map",
    "elements_to_state",
    "zone_grid",
]

#: 六制图域名（Table 4 行序）。
MAP_ZONE_NAMES: tuple[str, ...] = ("SC", "CR", "CG", "IT", "OT", "TF")

#: 各区缺省积分窗（Table 4 Span 列，年）。
_ZONE_SPAN_YEARS: dict[str, float] = {
    "SC": 19.0,
    "CR": 19.0,
    "CG": 19.0,
    "IT": 38.0,
    "OT": 57.0,
    "TF": 57.0,
}

#: 各区诊断聚焦（Table 4 Description 列的口径，随区升级）。
_ZONE_DIAGNOSTIC_FOCUS: dict[str, str] = {
    "SC": "纯 MEGNO（低倾切片近全 Ȳ ≈ 2，vZLK/Kozai 被抑制）",
    "CR": "MEGNO + 命运（共振岛/粘滞/碰撞带交织）",
    "CG": "两层全开（gateway 混合输运层：逃逸海 + 撞月域 + 再入带）",
    "IT": "两层 + 首次逃逸时刻（外月共振输运网）",
    "OT": "生存/逃逸为主（混合日月共振边域）",
    "TF": "生存/逃逸为主（Hill 界内侧终端 shelf）",
}


@dataclass(frozen=True)
class Scenario:
    """命名制图场景（论文 line 1413 统一切片 + 天体初值）。

    Attributes:
        epoch_utc: 历元（2027-08-02 日全食）。
        raan_deg / argp_deg / mean_anom_deg: 卫星初始角根数（反 aligned
            约定：Ω = Ω☾、ω = ω☾ + 180°、M = 0 近点起算）。
        inclination_is_moon_plane: True 表示 i 取历元月球轨道面（黄道
            参考的 Simon 1994 平均值 ≈ 5.157°，论文 "ecliptic-referenced
            ≈ 5°"）。
        moon_elements: 月球地心开普勒根数 [a_km, e, i_rad, Ω_rad, ω_rad,
            M0_rad]（缺省 Simon 1994 + 论文反推相位）。
        sun_elements: 太阳地心视轨道根数（同形；EMS 模型用）。
        provenance: 各初值的出处注记（SPICE 或论文推定）。
    """

    epoch_utc: str
    raan_deg: float
    argp_deg: float
    mean_anom_deg: float
    inclination_is_moon_plane: bool
    moon_elements: tuple[float, ...]
    sun_elements: tuple[float, ...]
    provenance: str


def default_scenario(constants: PrimerConstants = PRIMER_DEFAULTS) -> Scenario:
    """缺省场景：论文固定角 + Simon 1994 月根数 + 日全食合相位。

       月球 (Ω☾, ω☾) = (311.07°, 175.84°) 由反 aligned 约定反推
    （ω = ω☾ + 180° = 355.84°、Ω = Ω☾）；M☾ = 0 与太阳真黄经 = 历元
       月球真黄经（日全食合）为登记的缺省自由参数（可用 SPICE 覆盖）。
    """
    inc_rad = math.radians(constants.moon_inc_deg)
    raan_rad = math.radians(311.07)
    argp_rad = math.radians(175.84)
    moon_elements = (
        constants.moon_a_km,
        constants.moon_ecc,
        inc_rad,
        raan_rad,
        argp_rad,
        0.0,
    )
    # 太阳：视轨道黄道共面（i = 0），真黄经与历元月球真黄经重合（日全食合）。
    sun_argp = math.radians(103.0)  # 近日点黄经（一月初）量级，登记缺省
    moon_true_long = raan_rad + argp_rad  # M = 0 → 真黄经 = ϖ☾
    sun_m0 = moon_true_long - sun_argp
    sun_elements = (constants.sun_a_km, constants.sun_ecc, 0.0, 0.0, sun_argp, sun_m0)
    return Scenario(
        epoch_utc="2027-08-02T10:06:37",
        raan_deg=311.07,
        argp_deg=355.84,
        mean_anom_deg=0.0,
        inclination_is_moon_plane=True,
        moon_elements=moon_elements,
        sun_elements=sun_elements,
        provenance=(
            "卫星角根数 = 论文 line 1413 固定值；月根数 = Simon 1994 + 论文"
            "反 aligned 约定反推 (Ω☾, ω☾)；M☾ = 0 与太阳相位 = 日全食合为"
            "登记缺省自由参数（ADR 0041 Phase 3c）"
        ),
    )


#: 论文统一切片的命名常量（MCP details 透传）。
ECLIPSE_EPOCH_SCENARIO = default_scenario()


def elements_to_state(
    a_km: float,
    ecc: float,
    inc_rad: float,
    raan_rad: float,
    argp_rad: float,
    mean_anom_rad: float,
    gm_km3_s2: float,
) -> np.ndarray:
    """经典根数 → 地心情性系笛卡尔态（Curtis Alg. 4.2 的平面化重排）。"""
    ecc_anom = _solve_kepler(mean_anom_rad, ecc)
    r_km = a_km * (1.0 - ecc * math.cos(ecc_anom))
    cos_f = (math.cos(ecc_anom) - ecc) / (1.0 - ecc * math.cos(ecc_anom))
    sin_f = math.sqrt(1.0 - ecc * ecc) * math.sin(ecc_anom) / (1.0 - ecc * math.cos(ecc_anom))
    x_p, y_p = r_km * cos_f, r_km * sin_f
    p = a_km * (1.0 - ecc * ecc)
    v_factor = math.sqrt(gm_km3_s2 / p)
    vx_p = -v_factor * sin_f
    vy_p = v_factor * (ecc + cos_f)
    (so, co), (si, ci), (sw, cw) = (
        (math.sin(raan_rad), math.cos(raan_rad)),
        (math.sin(inc_rad), math.cos(inc_rad)),
        (math.sin(argp_rad), math.cos(argp_rad)),
    )
    r11, r12 = co * cw - so * sw * ci, -co * sw - so * cw * ci
    r21, r22 = so * cw + co * sw * ci, -so * sw + co * cw * ci
    r31, r32 = sw * si, cw * si
    return np.array(
        [
            r11 * x_p + r12 * y_p,
            r21 * x_p + r22 * y_p,
            r31 * x_p + r32 * y_p,
            r11 * vx_p + r12 * vy_p,
            r21 * vx_p + r22 * vy_p,
            r31 * vx_p + r32 * vy_p,
        ]
    )


def _solve_kepler(mean_anom_rad: float, ecc: float) -> float:
    ecc_anom = mean_anom_rad
    for _ in range(60):
        residual = ecc_anom - ecc * math.sin(ecc_anom) - mean_anom_rad
        ecc_anom -= residual / (1.0 - ecc * math.cos(ecc_anom))
        if abs(residual) < 1e-14:
            break
    return ecc_anom


def zone_grid(
    zone: str,
    n_a: int,
    n_e: int,
    *,
    e_min: float = 0.0,
    e_max: float = 0.9,
    constants: PrimerConstants = PRIMER_DEFAULTS,
) -> tuple[np.ndarray, np.ndarray]:
    """制图域网格：a/a☾ 在 Table 4 区带内线性、e 线性。

    Args:
        zone: :data:`MAP_ZONE_NAMES` 之一。
        n_a / n_e: 两轴格点数。
        e_min / e_max: 偏心率范围（缺省 0–0.9）。

    Returns:
        (a_over_a_moon, e) 两轴数组。
    """
    if zone not in MAP_ZONE_NAMES:
        raise ValueError(f"未知制图域 zone={zone!r}，支持 {'/'.join(MAP_ZONE_NAMES)}")
    if n_a < 2 or n_e < 2:
        raise ValueError(f"网格至少 2×2，得到 {n_a}×{n_e}")
    idx = MAP_ZONE_NAMES.index(zone)
    bands = _regions.table4_bands(constants)
    lo, hi = bands.lower[idx], bands.upper[idx]
    return (
        np.linspace(lo, hi, n_a),
        np.linspace(e_min, e_max, n_e),
    )


@dataclass(frozen=True)
class MapResult:
    """一张两层天图。

    Attributes:
        zone / model: 制图域与模型（"em" / "ems"）。
        a_over_a_moon / e_grid: 两轴。
        ybar_field: Ȳ 场 (n_a, n_e)。
        fate_ids: 命运类 id 场 (n_a, n_e)，索引 :data:`FATE_CLASSES`。
        t_escape_years_field: 首次逃逸时刻场 (n_a, n_e)，未逃逸 NaN。
        min_r_sel_km_field / min_r_geo_km_field: 最小距离场。
        span_years / thresholds / scenario / diagnostic_focus: 回显。
    """

    zone: str
    model: str
    a_over_a_moon: np.ndarray
    e_grid: np.ndarray
    ybar_field: np.ndarray
    fate_ids: np.ndarray
    t_escape_years_field: np.ndarray
    min_r_sel_km_field: np.ndarray
    min_r_geo_km_field: np.ndarray
    span_years: float
    thresholds: FateThresholds
    scenario: Scenario
    diagnostic_focus: str
    status: ConvergenceState
    cause: FailureCause
    message: str

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)

    @property
    def cells(self) -> int:
        return int(self.ybar_field.size)

    def fate_fractions(self) -> dict[str, float]:
        """各类命运占比（0–1；按格点计）。"""
        total = self.fate_ids.size
        counts = np.bincount(self.fate_ids.ravel(), minlength=len(FATE_CLASSES))
        return {FATE_CLASSES[i]: float(counts[i]) / total for i in range(len(FATE_CLASSES))}


def dynamical_map(
    zone: str,
    *,
    model: str = "em",
    n_a: int = 12,
    n_e: int = 8,
    e_min: float = 0.0,
    e_max: float = 0.9,
    span_years: float | None = None,
    thresholds: FateThresholds | None = None,
    rtol: float = 1e-9,
    max_step_hours: float = 6.0,
    stop_on_terminal: bool = True,
    scenario: Scenario | None = None,
    constants: PrimerConstants = PRIMER_DEFAULTS,
) -> MapResult:
    """逐格传播两层天图（EM/EMS；Rust 内核，纯物理单位）。

    Args:
        zone: :data:`MAP_ZONE_NAMES` 之一。
        model: ``"em"``（椭圆点质量地月）或 ``"ems"``（+太阳点质量）。
        n_a / n_e: 网格分辨率（CI 用小网格；全量制图走 scripts/）。
        e_min / e_max: 偏心率范围。
        span_years: 积分窗；None = Table 4 区带缺省（SC/CR/CG 19、IT 38、
            OT/TF 57 年）。
        thresholds: 命运分类阈值（None = Phase 3b 登记缺省）。
        rtol / max_step_hours: 积分器配置（max_step 为近点漏检防护，
            登记自由参数）。
        stop_on_terminal: 终端事件早停。
        scenario: 命名场景（None = :data:`ECLIPSE_EPOCH_SCENARIO`）。
        constants: Primer 常数集。

    Returns:
        :class:`MapResult`。
    """
    if model not in ("em", "ems"):
        raise ValueError(f"model 须为 em/ems，得到 {model!r}")
    require_rust_extension("propagate_geocentric_fate_map_py")
    thresholds = thresholds or FateThresholds()
    scen = scenario or ECLIPSE_EPOCH_SCENARIO
    span = float(span_years if span_years is not None else _ZONE_SPAN_YEARS[zone])
    a_axis, e_axis = zone_grid(zone, n_a, n_e, e_min=e_min, e_max=e_max, constants=constants)

    c = constants
    hill_e = hill_radius_earth(c)
    hill_m = hill_radius_moon(c)
    span_s = span * 365.25 * 86400.0
    inc_rad = math.radians(c.moon_inc_deg) if scen.inclination_is_moon_plane else 0.0
    sun_gm = c.sun_gm if model == "ems" else 0.0
    sun_elements = list(scen.sun_elements) if model == "ems" else None

    ybar_field = np.full((n_a, n_e), np.nan)
    fate_ids = np.full((n_a, n_e), -1, dtype=int)
    t_esc = np.full((n_a, n_e), np.nan)
    min_sel = np.full((n_a, n_e), np.nan)
    min_geo = np.full((n_a, n_e), np.nan)

    for i, a_ratio in enumerate(a_axis):
        a_km = float(a_ratio) * c.moon_a_km
        for j, ecc in enumerate(e_axis):
            ecc_f = float(ecc)
            # 初值即终端短路（与 Dynamics._immediate_collision 同语义）：
            # 近点在地表内直接判再入；初值在月面内判撞月。不进入传播。
            if a_km * (1.0 - ecc_f) <= c.earth_ref_radius_km:
                ybar_field[i, j] = math.nan
                fate_ids[i, j] = FATE_CLASSES.index("earth_reentry")
                min_geo[i, j] = a_km * (1.0 - ecc_f)
                min_sel[i, j] = math.nan
                continue
            state = elements_to_state(
                a_km,
                ecc_f,
                inc_rad,
                math.radians(scen.raan_deg),
                math.radians(scen.argp_deg),
                math.radians(scen.mean_anom_deg),
                c.earth_gm,
            )
            cell = propagate_geocentric_fate_map_py(
                c.earth_gm,
                c.moon_gm,
                list(scen.moon_elements),
                sun_gm,
                sun_elements,
                c.earth_ref_radius_km,
                c.moon_radius_km,
                hill_e,
                hill_m,
                span_s,
                stop_on_terminal,
                2,
                [float(v) for v in state],
                rtol,
                max_step_hours * 3600.0,
                None,
            )
            ybar_field[i, j] = float(cell["ybar"])
            min_sel[i, j] = float(cell["min_r_sel_km"])
            min_geo[i, j] = float(cell["min_r_geo_km"])
            if cell["t_escape_s"] is not None:
                t_esc[i, j] = float(cell["t_escape_s"]) / (365.25 * 86400.0)
            diagnostics = FateDiagnostics(
                escaped=bool(cell["escaped"]),
                t_escape_days=(
                    None if cell["t_escape_s"] is None else float(cell["t_escape_s"]) / 86400.0
                ),
                earth_reentry=bool(cell["reentry"]),
                t_reentry_days=(
                    None if cell["t_reentry_s"] is None else float(cell["t_reentry_s"]) / 86400.0
                ),
                moon_impact=bool(cell["impact"]),
                t_impact_days=(
                    None if cell["t_impact_s"] is None else float(cell["t_impact_s"]) / 86400.0
                ),
                min_geocentric_km=float(cell["min_r_geo_km"]),
                min_selenocentric_km=float(cell["min_r_sel_km"]),
                moon_hill_entries=int(cell["moon_hill_entries"]),
                span_days=span * 365.25,
            )
            label, _, _, _ = classify_fate(diagnostics, float(cell["ybar"]), thresholds)
            fate_ids[i, j] = FATE_CLASSES.index(label)

    return MapResult(
        zone=zone,
        model=model,
        a_over_a_moon=a_axis,
        e_grid=e_axis,
        ybar_field=ybar_field,
        fate_ids=fate_ids,
        t_escape_years_field=t_esc,
        min_r_sel_km_field=min_sel,
        min_r_geo_km_field=min_geo,
        span_years=span,
        thresholds=thresholds,
        scenario=scen,
        diagnostic_focus=_ZONE_DIAGNOSTIC_FOCUS[zone],
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="ok",
    )


def compare_models(em: MapResult, ems: MapResult) -> dict[str, Any]:
    """EM/EMS 双模型对照结论（架构持续性检验，论文 §7.3 协议）。

    对同一 zone 的两张图逐格对比命运标签，输出：

    - ``persisted``：两模型同类（骨架存续）；
    - ``prompted``：EM 未逃逸 → EMS 逃逸（太阳把 EM 架构排干）；
    - ``redirected``：终端结局互换（如撞月→逃逸）；
    - ``reorganized``：其余换类（含反向）。

    按 :data:`FATE_CLASSES` 给出逐类占比差（EMS − EM）。
    """
    if em.zone != ems.zone or em.a_over_a_moon.shape != ems.a_over_a_moon.shape:
        raise ValueError("compare_models 需要同域同网格的两张图")
    f_em, f_ems = em.fate_ids, ems.fate_ids
    same = f_em == f_ems
    escape = {
        "stable_quasiperiodic",
        "sticky_resident",
        "bounded_unclassified",
    }
    escaped = {"orderly_escape", "chaotic_escape", "escape_unclassified"}
    em_bounded = np.isin(f_em, list(escape))
    ems_escaped = np.isin(f_ems, list(escaped))
    prompted = (~same) & em_bounded & ems_escaped
    terminal = {"earth_reentry", "moon_impact"}
    em_term = np.isin(f_em, list(terminal))
    ems_esc = np.isin(f_ems, list(escaped))
    redirected = (~same) & em_term & ems_esc
    reorganized = (~same) & ~prompted & ~redirected
    total = f_em.size
    delta = {}
    for idx, name in enumerate(FATE_CLASSES):
        delta[name] = float((f_ems == idx).sum() - (f_em == idx).sum()) / total
    return {
        "zone": em.zone,
        "persisted_fraction": float(same.sum()) / total,
        "solar_drained_fraction": float(prompted.sum()) / total,
        "terminal_redirected_fraction": float(redirected.sum()) / total,
        "reorganized_fraction": float(reorganized.sum()) / total,
        "fate_fraction_delta_ems_minus_em": delta,
    }
