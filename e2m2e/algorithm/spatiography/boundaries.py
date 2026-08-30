"""分区边界几何（spatiography boundaries）。

为可视化（tod 画布 / 论文式图件）生成边界的离散几何：
- :func:`synodic_planar_elements`：地月会合旋转系（质心原点、z=0 平面）
  中的边界圆族 / Battin 非对称闭合曲线 / L1–L5 平动点，物理单位 km。
- :func:`ae_curves`：地心 osculating (a, e) 根数平面的走廊曲线族
  （论文 Fig. 5/8/11 的 overlay：掠地线、Hill 远点线、月 Hill 相遇走廊、
  GEO 穿越线、共振竖线、Tisserand 等值线）。这些是 **元素空间 crossing
  diagnostics 而非物理面**（论文 Fig. 11 caption 原话），输出 docstring
  与响应 schema 均须传达这一定性。

"界面不碰算法"：前端只做 km→DU 归一与绘制，全部几何在此离散化。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...data.templates.enums import LibrationPoint
from ...status import ConvergenceState, FailureCause, ResultStatus
from .constants import PRIMER_DEFAULTS, PrimerConstants
from .regions import primer_cr3bp_system
from .resonances import resonance_centers
from .scales import (
    battin_soi_moon,
    chebotarev_radius_moon,
    geo_radius_km,
    hill_radius_earth,
    hill_radius_moon,
    laplace_radius_geolunar,
    soi_laplace_earth,
    soi_laplace_moon,
    tidal_parity_radius,
)

__all__ = [
    "AE_CURVE_NAMES",
    "BoundaryElement",
    "BoundarySetResult",
    "SYNODIC_ELEMENT_NAMES",
    "ae_curves",
    "synodic_planar_elements",
]

#: synodic_planar 输出支持的边界元素名。
SYNODIC_ELEMENT_NAMES: tuple[str, ...] = (
    "laplace_radius",
    "tidal_parity",
    "earth_soi",
    "earth_hill",
    "moon_hill",
    "moon_soi",
    "moon_battin",
    "moon_chebotarev",
    "libration_points",
)

#: ae_curves 输出支持的曲线族名。
AE_CURVE_NAMES: tuple[str, ...] = (
    "graze",
    "hill_apocenter",
    "moon_hill_encounter",
    "geo_crossing",
    "resonance_verticals",
    "tisserand_contours",
)

#: Tisserand 等值线族缺省 T 值（论文 Fig. 3 走廊量级；T☾=3 为 a=a☾ 圆轨
#: 参考值，非 gateway 阈值）。
_DEFAULT_TISSERAND_VALUES: tuple[float, ...] = (2.4, 2.7, 3.0)


@dataclass(frozen=True)
class BoundaryElement:
    """单个边界几何元素。

    Attributes:
        kind: ``"circle"``（圆，用 center_km+radius_km+points_km）/
            ``"polyline"``（闭合或开放折线，points_km）/ ``"point"``（点标记）/
            ``"curve_ae"``（(a,e) 根数平面曲线，points_ae）/
            ``"vertical_ae"``（(a,e) 平面竖直线，a_km + points_ae）。
        label: 显示名（如 ``"Laplace radius r_L"``、``"5:1☾"``）。
        formula_id: 论文式号或表出处（如 ``"Eq.98"``、``"Table1"``）。
        center_km: 圆心（会合系质心原点 km，z=0）；非空间元素为 None。
        radius_km: 圆半径，km；非圆元素为 None。
        points_km: 会合系平面离散点 (n, 3)，km；空间曲线用。
        points_ae: (a, e) 离散点 (n, 2)，a 单位 km；根数曲线用。
        a_km: 竖直线位置，km；非竖线为 None。
        note: 语义注记（如 crossing-diagnostic 定性、角度约定）。
    """

    kind: str
    label: str
    formula_id: str
    center_km: np.ndarray | None = None
    radius_km: float | None = None
    points_km: np.ndarray | None = None
    points_ae: np.ndarray | None = None
    a_km: float | None = None
    note: str = ""


@dataclass(frozen=True)
class BoundarySetResult:
    """边界几何查询结果（状态契约三元组齐备）。"""

    status: ConvergenceState
    cause: FailureCause
    message: str
    elements: tuple[BoundaryElement, ...]

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)

    def __len__(self) -> int:
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements)


def _circle_points(center_km: np.ndarray, radius_km: float, resolution: int) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=True)
    pts = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)
    return center_km + radius_km * pts


def _earth_center_km(constants: PrimerConstants) -> np.ndarray:
    return np.array([-constants.moon_mass_parameter * constants.moon_a_km, 0.0, 0.0])


def _moon_center_km(constants: PrimerConstants) -> np.ndarray:
    return np.array([(1.0 - constants.moon_mass_parameter) * constants.moon_a_km, 0.0, 0.0])


def synodic_planar_elements(
    resolution: int = 720,
    boundary_set: list[str] | tuple[str, ...] | None = None,
    *,
    constants: PrimerConstants = PRIMER_DEFAULTS,
) -> BoundarySetResult:
    """会合系（质心原点，z=0）边界几何：圆族 + Battin 非对称曲线 + L1–L5。

    Args:
        resolution: 圆与闭合曲线的离散点数（闭合，首尾相接）。
        boundary_set: :data:`SYNODIC_ELEMENT_NAMES` 的子集；None = 全部。
        constants: Primer 常数集。

    Returns:
        :class:`BoundarySetResult`。Battin 曲线 psi 从反地方向量起
        （psi=0 在 +x，即背地点 = 曲线最远点），见 scales 模块角度约定。

    Raises:
        ValueError: boundary_set 含未知元素名。
    """
    if resolution < 8:
        raise ValueError(f"resolution 至少为 8，得到 {resolution}")
    names = SYNODIC_ELEMENT_NAMES if boundary_set is None else tuple(boundary_set)
    unknown = [n for n in names if n not in SYNODIC_ELEMENT_NAMES]
    if unknown:
        raise ValueError(f"未知的边界元素：{unknown}；支持 {SYNODIC_ELEMENT_NAMES}")

    c = constants
    earth = _earth_center_km(c)
    moon = _moon_center_km(c)
    elements: list[BoundaryElement] = []

    def _circle(name: str, label: str, formula: str, center: np.ndarray, radius: float) -> None:
        if name in names:
            elements.append(
                BoundaryElement(
                    kind="circle",
                    label=label,
                    formula_id=formula,
                    center_km=center.copy(),
                    radius_km=radius,
                    points_km=_circle_points(center, radius, resolution),
                )
            )

    _circle(
        "laplace_radius",
        "Laplace radius r_L (geolunar)",
        "Eq.98",
        earth,
        laplace_radius_geolunar(c),
    )
    _circle("tidal_parity", "Lunisolar tidal parity a_TP", "Eq.127", earth, tidal_parity_radius(c))
    _circle("earth_soi", "Earth SOI (Laplace-Tisserand)", "Eq.120", earth, soi_laplace_earth(c))
    _circle("earth_hill", "Earth Hill sphere r_H", "Eq.111", earth, hill_radius_earth(c))
    _circle("moon_hill", "Moon Hill sphere rho_H", "Eq.110", moon, hill_radius_moon(c))
    _circle("moon_soi", "Moon SOI (Laplace-Tisserand)", "Eq.116", moon, soi_laplace_moon(c))
    _circle(
        "moon_chebotarev",
        "Moon Chebotarev sphere rho_Ch",
        "Eq.117",
        moon,
        chebotarev_radius_moon(c),
    )

    if "moon_battin" in names:
        psi = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=True)
        radii = np.array([battin_soi_moon(float(p), c) for p in psi])
        pts = moon + np.stack(
            [radii * np.cos(psi), radii * np.sin(psi), np.zeros_like(psi)], axis=1
        )
        elements.append(
            BoundaryElement(
                kind="polyline",
                label="Moon Battin SOI rho_B(psi)",
                formula_id="Eq.118",
                center_km=moon.copy(),
                points_km=pts,
                note="psi 从反地方向量起（psi=0 为背地最远点）；朝地 52009 km /"
                " 背地 64201 km（论文数值口径）",
            )
        )

    if "libration_points" in names:
        system = primer_cr3bp_system(c)
        system.compute_libration_points()
        for name, point in (
            ("L1", "cislunar 侧内颈"),
            ("L2", "translunar 侧外颈"),
            ("L3", "system equilibrium"),
            ("L4", "system equilibrium"),
            ("L5", "system equilibrium"),
        ):
            key = getattr(LibrationPoint, name)
            pos_nd = system.get_libration_point(key)
            elements.append(
                BoundaryElement(
                    kind="point",
                    label=name,
                    formula_id="Sec.5.4",
                    center_km=pos_nd * c.moon_a_km,
                    note=f"精确求根（scipy fsolve）；{name} {point}",
                )
            )

    return BoundarySetResult(
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="ok",
        elements=tuple(elements),
    )


def _ae_polyline(
    label: str, formula: str, a_grid: np.ndarray, e_values: np.ndarray, note: str
) -> BoundaryElement:
    pts = np.stack([a_grid, e_values], axis=1)
    return BoundaryElement(
        kind="curve_ae", label=label, formula_id=formula, points_ae=pts, note=note
    )


def ae_curves(
    n_points: int = 400,
    boundary_set: list[str] | tuple[str, ...] | None = None,
    *,
    tisserand_values: tuple[float, ...] = _DEFAULT_TISSERAND_VALUES,
    constants: PrimerConstants = PRIMER_DEFAULTS,
) -> BoundarySetResult:
    """(a, e) 根数平面走廊曲线族（论文 Fig. 5/8/11 overlay 的数据层）。

    Args:
        n_points: 每条曲线的采样点数。
        boundary_set: :data:`AE_CURVE_NAMES` 的子集；None = 全部。
        tisserand_values: Tisserand 等值线族的 T 值。
        constants: Primer 常数集。

    Returns:
        :class:`BoundarySetResult`。全部曲线为元素空间 crossing diagnostics
        而非物理面；a 横轴单位 km（换算 a/a☾ 除以 ``constants.moon_a_km``）。

    Raises:
        ValueError: boundary_set 含未知曲线族名。
    """
    if n_points < 16:
        raise ValueError(f"n_points 至少为 16，得到 {n_points}")
    names = AE_CURVE_NAMES if boundary_set is None else tuple(boundary_set)
    unknown = [n for n in names if n not in AE_CURVE_NAMES]
    if unknown:
        raise ValueError(f"未知的曲线族：{unknown}；支持 {AE_CURVE_NAMES}")

    c = constants
    r_l = laplace_radius_geolunar(c)
    r_h = hill_radius_earth(c)
    rho_h = hill_radius_moon(c)
    r_geo = geo_radius_km(c)
    a_lo = 0.02 * c.moon_a_km
    a_hi = 1.02 * r_h
    grid = np.geomspace(a_lo, a_hi, n_points)
    elements: list[BoundaryElement] = []

    diag = "元素空间 crossing diagnostic，非物理面（论文 Fig.8/11 caption 口径）"

    if "graze" in names:
        e_vals = 1.0 - c.earth_ref_radius_km / grid
        mask = (e_vals >= 0.0) & (e_vals <= 1.0)
        elements.append(
            _ae_polyline(
                "Earth grazing a(1-e)=R+",
                "Fig.8/11",
                grid[mask],
                e_vals[mask],
                "近地点触地（再入包络）。" + diag,
            )
        )
    if "hill_apocenter" in names:
        e_vals = r_h / grid - 1.0
        mask = (e_vals >= 0.0) & (e_vals <= 1.0)
        elements.append(
            _ae_polyline(
                "Earth Hill apocenter a(1+e)=r_H",
                "Fig.11",
                grid[mask],
                e_vals[mask],
                "远点达地球 Hill 界（逃逸判据外边界）。" + diag,
            )
        )
    if "moon_hill_encounter" in names:
        peri_grid = grid[grid >= c.moon_a_km + rho_h]
        e_peri = 1.0 - (c.moon_a_km + rho_h) / peri_grid
        elements.append(
            _ae_polyline(
                "Moon-Hill encounter (periapsis) a(1-e)=a_M+rho_H",
                "Fig.11",
                peri_grid,
                e_peri,
                "近点达月球 Hill 遭遇区（近月走廊）。" + diag,
            )
        )
        apo_grid = grid[grid >= c.moon_a_km - rho_h]
        e_apo = (c.moon_a_km - rho_h) / apo_grid - 1.0
        mask = (e_apo >= 0.0) & (e_apo <= 1.0)
        elements.append(
            _ae_polyline(
                "Moon-Hill encounter (apoapsis) a(1+e)=a_M-rho_H",
                "Fig.8",
                apo_grid[mask],
                e_apo[mask],
                "远点达月球 Hill 内界（内月遭遇包络）。" + diag,
            )
        )
    if "geo_crossing" in names:
        e_vals = 1.0 - r_geo / grid
        mask = (e_vals >= 0.0) & (e_vals <= 1.0)
        elements.append(
            _ae_polyline(
                "GEO crossing a(1-e)=r_GEO",
                "Fig.8/11",
                grid[mask],
                e_vals[mask],
                "近点返回地球业务域；r_GEO 按恒星日派生"
                f"（{r_geo:.1f} km）。GEO 非分区判据。" + diag,
            )
        )
    if "resonance_verticals" in names:
        ladder = resonance_centers("all", c)
        for center in ladder.centers:
            if center.kind == "exterior_terrestrial_selenocentric":
                continue  # 月心系共振不在地心 (a,e) 平面
            elements.append(
                BoundaryElement(
                    kind="vertical_ae",
                    label=center.label,
                    formula_id="Table1/2",
                    a_km=center.a_km,
                    points_ae=np.array([[center.a_km, 0.0], [center.a_km, 1.0]]),
                    note=f"名义中心 {center.label}；周期 {center.period_days:.2f} d",
                )
            )
        # Laplace 半径竖直参考线（内带起点）。
        elements.append(
            BoundaryElement(
                kind="vertical_ae",
                label="r_L",
                formula_id="Eq.98",
                a_km=r_l,
                points_ae=np.array([[r_l, 0.0], [r_l, 1.0]]),
                note=f"Laplace 半径 {r_l:.1f} km = {r_l / c.moon_a_km:.4f} a☾",
            )
        )
    if "tisserand_contours" in names:
        ratio = grid / c.moon_a_km
        for t_val in tisserand_values:
            inner = 1.0 - ((t_val - 1.0 / ratio) ** 2) / (4.0 * ratio)
            mask = inner >= 0.0
            e_vals = np.sqrt(np.where(mask, inner, 0.0))
            mask &= ratio > 0
            elements.append(
                _ae_polyline(
                    f"Tisserand T_M={t_val:g}",
                    "Eq.140",
                    grid[mask],
                    e_vals[mask],
                    "共面 Tisserand 等值线（Jacobi 的根数类比；可达性导引）。" + diag,
                )
            )

    return BoundarySetResult(
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="ok",
        elements=tuple(elements),
    )
