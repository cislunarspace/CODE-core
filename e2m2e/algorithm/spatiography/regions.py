"""区域分类器（spatiography regions）。

五省分区体系与判定函数（论文 §5 Table 1、附录 B Table 4、§3.2 五拓扑）。

命名铁律（论文 §2.6，ADR 0041）：``Cislunar`` 只用于两个带级区域、绝不作
伞式值；整个地月耦合环境的总称是 geolunar space / Earth-Moon system space；
不得以 GEO 作任何判据；L1 属 cislunar 侧、L2 属 translunar 侧、L3/L4/L5
属 system equilibria。

多标签设计（论文校验结论）：分区边界存在 deliberate overlap——Table 1 自身
中 5:4ζ（0.86）落在 circumlunar 包络 [L1, L2] 内、L2 与 4:5ζ 同值、a_TP 与
地球 SOI 落在 translunar 带内部，circumlunar "cuts across the sequence"。
因此分类器返回**有序标签列表**而非单值；``include_overlaps=False`` 时按
优先序（terrestrial > 内带 > circumlunar > 外带 > translunar > heliocentric）
取主标签。
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

import numpy as np

from ...data.templates.enums import LibrationPoint
from ...status import ConvergenceState, FailureCause, ResultStatus
from ..dynamics.cr3bp_system import CR3BP_System
from .constants import PRIMER_DEFAULTS, PrimerConstants
from .scales import hill_radius_earth, laplace_radius_geolunar

__all__ = [
    "REGION_LEGEND",
    "RegionId",
    "StateDiagnostics",
    "Table4Bands",
    "classify_by_semi_major_axis",
    "classify_state",
    "jacobi_critical_values",
    "jacobi_topology_case",
    "primer_cr3bp_system",
    "table4_bands",
]


class RegionId(enum.IntEnum):
    """五省分区枚举（论文 §2.6/§5 命名纪律，见模块 docstring）。"""

    TERRESTRIAL = 0
    CISLUNAR_INNER_SECULAR = 1
    CISLUNAR_OUTER_RESONANT = 2
    CIRCUMLUNAR = 3
    TRANSLUNAR = 4
    HELIOCENTRIC = 5


#: 区域 id → 名称（MCP 响应 legend 用；snake_case，语义与论文一致）。
REGION_LEGEND: dict[int, str] = {
    int(RegionId.TERRESTRIAL): "terrestrial",
    int(RegionId.CISLUNAR_INNER_SECULAR): "cislunar_inner_secular",
    int(RegionId.CISLUNAR_OUTER_RESONANT): "cislunar_outer_resonant",
    int(RegionId.CIRCUMLUNAR): "circumlunar",
    int(RegionId.TRANSLUNAR): "translunar",
    int(RegionId.HELIOCENTRIC): "heliocentric",
}

#: Table 4 制图分区带（a/a☾，附录 B；区限为 Gallardo separatrix-edge 包络，
#: 相邻区端部有意重叠）。SC 上缘 0.35 与 Table 1 的 5:1ζ=0.34 差 0.01 属两
#: 表口径差，分类器按所选 reference 各自忠实复现。
_TABLE4_BANDS: tuple[tuple[float, float], ...] = (
    (0.13, 0.35),  # SC → CISLUNAR_INNER_SECULAR
    (0.33, 0.89),  # CR → CISLUNAR_OUTER_RESONANT
    (0.84, 1.16),  # CG → CIRCUMLUNAR
    (1.08, 2.03),  # IT → TRANSLUNAR
    (1.91, 3.34),  # OT → TRANSLUNAR
    (3.03, 3.90),  # TF → TRANSLUNAR
)
_TABLE4_TO_REGION: tuple[RegionId, ...] = (
    RegionId.CISLUNAR_INNER_SECULAR,
    RegionId.CISLUNAR_OUTER_RESONANT,
    RegionId.CIRCUMLUNAR,
    RegionId.TRANSLUNAR,
    RegionId.TRANSLUNAR,
    RegionId.TRANSLUNAR,
)

# Table 1 模式下的主标签优先序（include_overlaps=False 时取第一个命中）。
_PRIMARY_PRECEDENCE: tuple[RegionId, ...] = (
    RegionId.TERRESTRIAL,
    RegionId.CISLUNAR_INNER_SECULAR,
    RegionId.CIRCUMLUNAR,
    RegionId.CISLUNAR_OUTER_RESONANT,
    RegionId.TRANSLUNAR,
    RegionId.HELIOCENTRIC,
)


@dataclass(frozen=True)
class Table4Bands:
    """Table 4 六制图分区的实际边界（a/a☾），由 Primer 常数解析派生。"""

    lower: tuple[float, ...]
    upper: tuple[float, ...]


def table4_bands(constants: PrimerConstants = PRIMER_DEFAULTS) -> Table4Bands:
    """把 Table 4 的静态表值换算为随常数集自洽的边界（r_L、r_H 解析派生）。"""
    r_l = laplace_radius_geolunar(constants) / constants.moon_a_km
    r_h = hill_radius_earth(constants) / constants.moon_a_km
    lower = (
        r_l,
        _TABLE4_BANDS[1][0],
        _TABLE4_BANDS[2][0],
        _TABLE4_BANDS[3][0],
        _TABLE4_BANDS[4][0],
        _TABLE4_BANDS[5][0],
    )
    upper = (
        _TABLE4_BANDS[0][1],
        _TABLE4_BANDS[1][1],
        _TABLE4_BANDS[2][1],
        _TABLE4_BANDS[3][1],
        _TABLE4_BANDS[4][1],
        r_h,
    )
    return Table4Bands(lower=lower, upper=upper)


def primer_cr3bp_system(constants: PrimerConstants = PRIMER_DEFAULTS) -> CR3BP_System:
    """构造 Primer 口径的地月 CR3BP 系统（mu_bar、a☾ 自洽特征尺度）。

    与 ``tests/conftest.py`` 的 ``earth_moon_system``（DE421 基准）口径不同：
    此处 mu_bar = GM☾/(GM⊕+GM☾)、特征长度 a☾ = 383397.7725 km、特征周期
    2π/n（n = sqrt((GM⊕+GM☾)/a☾³)）。Jacobi 常数约定为 Parker 式
    C = 2U − v²（无常数项），与论文 §6.1 一致。
    """
    mu_bar = constants.moon_mass_parameter
    period_s = 2.0 * math.pi / constants.cr3bp_mean_motion_rad_s
    system = CR3BP_System(mu=mu_bar, primary="Earth", secondary="Moon")
    system.set_characteristic_scales(distance=constants.moon_a_km, period=period_s)
    return system


def jacobi_critical_values(
    system: CR3BP_System | None = None, constants: PrimerConstants = PRIMER_DEFAULTS
) -> dict[str, float]:
    """五个平动点处的临界 Jacobi 值 C1..C5（Case I–V 分级用）。

    平动点用现有精确求根（``compute_libration_points``，scipy fsolve）。
    论文 §5 标称值 57868/64347 km 为级数近似口径（且质量参数取 GM☾/GM⊕），
    与精确求根差 1.2–2.3%；本库以精确值为准，论文值作文档注记（ADR 0041）。
    """
    sys = system if system is not None else primer_cr3bp_system(constants)
    sys.compute_libration_points()
    out: dict[str, float] = {}
    for name, point in (
        ("C1", LibrationPoint.L1),
        ("C2", LibrationPoint.L2),
        ("C3", LibrationPoint.L3),
        ("C4", LibrationPoint.L4),
        ("C5", LibrationPoint.L5),
    ):
        pos = sys.get_libration_point(point)
        out[name] = float(sys.get_jacobi_constant([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0]))
    return out


def jacobi_topology_case(
    jacobi_constant: float, critical_values: dict[str, float]
) -> tuple[int, tuple[str, ...]]:
    """Hill 区域五拓扑分级（论文 §3.2 Case I–V）。

    Returns:
        (case, open_necks)：case ∈ 1..5；open_necks 为已开启颈口列表
        （元素取 ``"L1"``/``"L2"``，Case IV 起外加 ``"L3"`` 语义上的外域
        连通，论文按 C4=C5 处理，这里只报告 L1/L2 颈）。
    """
    cj = jacobi_constant
    c1, c2 = critical_values["C1"], critical_values["C2"]
    c3, c4 = critical_values["C3"], critical_values["C4"]
    if cj > c1:
        return 1, ()
    if cj > c2:
        return 2, ("L1",)
    if cj > c3:
        return 3, ("L1", "L2")
    if cj > c4:
        return 4, ("L1", "L2", "L3")
    return 5, ("L1", "L2", "L3")


def classify_by_semi_major_axis(
    a_over_a_moon: float,
    *,
    reference: str = "table1",
    include_overlaps: bool = True,
    constants: PrimerConstants = PRIMER_DEFAULTS,
    system: CR3BP_System | None = None,
) -> list[int]:
    """按 osculating 半长轴（以 a☾ 归一）判定地心轨道所处分区。

    Args:
        a_over_a_moon: a/a☾（地心 osculating 半长轴，月球平均半长轴归一）。
        reference: ``"table1"``（分区语义，论文 Table 1 口径）或
            ``"table4"``（附录 B 六制图带口径，含 deliberate-overlap）。
        include_overlaps: False 时只返回主标签（优先序见模块 docstring）。
        constants: Primer 常数集（r_L、a_TP、r_H 由其解析派生）。
        system: 复用已构造的 Primer CR3BP 系统（缺省自建，用于 L1/L2）。

    Returns:
        区域 id 列表（:data:`REGION_LEGEND` 的键），升序；重叠带多值。

    Raises:
        ValueError: reference 不受支持。
    """
    if reference not in ("table1", "table4"):
        raise ValueError(f"未知的 reference={reference!r}，支持 table1/table4")

    labels: set[RegionId] = set()
    if reference == "table4":
        bands = table4_bands(constants)
        for lo, hi, region in zip(bands.lower, bands.upper, _TABLE4_TO_REGION, strict=True):
            if lo <= a_over_a_moon <= hi:
                labels.add(region)
        if a_over_a_moon > hill_radius_earth(constants) / constants.moon_a_km:
            labels.add(RegionId.HELIOCENTRIC)
    else:
        x = a_over_a_moon
        r_l = laplace_radius_geolunar(constants) / constants.moon_a_km
        r_h = hill_radius_earth(constants) / constants.moon_a_km
        # 内月球 MMR 阶梯端点：5:1（0.3420，内带起点）与 5:4（0.8618，
        # 最内低阶共振末端）由共振条件解析派生（与 resonances 同式）。
        five_to_one = (1.0 / 5.0) ** (2.0 / 3.0)
        five_to_four = (4.0 / 5.0) ** (2.0 / 3.0)
        sys = system if system is not None else primer_cr3bp_system(constants)
        sys.compute_libration_points()
        l1 = float(
            np.linalg.norm(
                sys.get_libration_point(LibrationPoint.L1) + sys.mu * np.array([1.0, 0.0, 0.0])
            )
        )
        l2 = float(
            np.linalg.norm(
                sys.get_libration_point(LibrationPoint.L2) + sys.mu * np.array([1.0, 0.0, 0.0])
            )
        )
        if x < r_l:
            labels.add(RegionId.TERRESTRIAL)
        if r_l <= x < five_to_one:
            labels.add(RegionId.CISLUNAR_INNER_SECULAR)
        if five_to_one <= x <= five_to_four:
            labels.add(RegionId.CISLUNAR_OUTER_RESONANT)
        if l1 <= x <= l2:
            labels.add(RegionId.CIRCUMLUNAR)
        if x > l2:
            labels.add(RegionId.TRANSLUNAR)
        if x > r_h:
            labels.add(RegionId.HELIOCENTRIC)
        if not labels:
            labels.add(RegionId.CISLUNAR_OUTER_RESONANT)

    ordered = sorted(labels, key=lambda r: r.value)
    if include_overlaps:
        return [int(r) for r in ordered]
    for primary in _PRIMARY_PRECEDENCE:
        if primary in labels:
            return [int(primary)]
    return []


@dataclass(frozen=True)
class StateDiagnostics:
    """单状态分区诊断（classify_state 的返回值，状态契约三元组齐备）。"""

    status: ConvergenceState
    cause: FailureCause
    message: str
    r_geocentric_km: float
    rho_selenocentric_km: float
    a_geocentric_km: float
    a_over_a_moon: float
    jacobi_constant: float
    topology_case: int
    open_necks: tuple[str, ...] = field(default_factory=tuple)
    zone_ids: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)


def classify_state(
    state: list[float] | np.ndarray,
    *,
    frame: str = "synodic_barycentric_km",
    reference: str = "table1",
    include_overlaps: bool = True,
    constants: PrimerConstants = PRIMER_DEFAULTS,
    system: CR3BP_System | None = None,
) -> StateDiagnostics:
    """对单个会合系状态做完整分区诊断。

    Args:
        state: 6 维状态 [x, y, z, vx, vy, vz]。
            ``frame="synodic_barycentric_km"``：地月会合旋转系、质心原点、
            物理单位 km / km/s（ADR 0040 同款措辞）。
            ``frame="synodic_barycentric_nd"``：同系无量纲（长度 a☾、速度
            a☾·n，n = sqrt((GM⊕+GM☾)/a☾³)）。
        reference / include_overlaps / constants: 同
            :func:`classify_by_semi_major_axis`。
        system: 复用 Primer CR3BP 系统（缺省自建）。

    Returns:
        :class:`StateDiagnostics`：地心距/月心距、osculating a（绕 GM⊕ 二体
        闭合式 a = 1/(2/r − v²/GM⊕)）、Jacobi 值与 Case、分区多标签。

    Raises:
        ValueError: frame 不受支持或状态维数不对。
    """
    if frame not in ("synodic_barycentric_km", "synodic_barycentric_nd"):
        raise ValueError(
            f"不支持的 frame={frame!r}；当前支持 synodic_barycentric_km/"
            "synodic_barycentric_nd（gcrs_km 待星历批次接入，见 ADR 0041）"
        )
    arr = np.asarray(state, dtype=float)
    if arr.shape != (6,):
        raise ValueError(f"状态须为 6 维 [x,y,z,vx,vy,vz]，得到 shape={arr.shape}")

    c = constants
    sys = system if system is not None else primer_cr3bp_system(constants)
    n_rad_s = c.cr3bp_mean_motion_rad_s
    if frame == "synodic_barycentric_km":
        pos_nd = arr[:3] / c.moon_a_km
        vel_nd = arr[3:] / (c.moon_a_km * n_rad_s)
        v_km_s = float(np.linalg.norm(arr[3:]))
    else:
        pos_nd = arr[:3]
        vel_nd = arr[3:]
        v_km_s = float(np.linalg.norm(vel_nd)) * c.moon_a_km * n_rad_s

    state_nd = np.concatenate([pos_nd, vel_nd])
    # 地心距（地球位于 (-mu, 0, 0)）与月心距（月球位于 (1-mu, 0, 0)）。
    earth_pos = np.array([-sys.mu, 0.0, 0.0])
    moon_pos = np.array([1.0 - sys.mu, 0.0, 0.0])
    r_geo_km = float(np.linalg.norm(pos_nd - earth_pos)) * c.moon_a_km
    rho_km = float(np.linalg.norm(pos_nd - moon_pos)) * c.moon_a_km

    denom = 2.0 / r_geo_km - v_km_s**2 / c.earth_gm
    a_km = float("inf") if denom <= 0.0 else 1.0 / denom

    cj = float(sys.get_jacobi_constant(state_nd))
    crits = jacobi_critical_values(sys, c)
    case, necks = jacobi_topology_case(cj, crits)

    zones: tuple[int, ...] = ()
    if math.isfinite(a_km):
        zones = tuple(
            classify_by_semi_major_axis(
                a_km / c.moon_a_km,
                reference=reference,
                include_overlaps=include_overlaps,
                constants=c,
                system=sys,
            )
        )

    return StateDiagnostics(
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="ok",
        r_geocentric_km=r_geo_km,
        rho_selenocentric_km=rho_km,
        a_geocentric_km=a_km,
        a_over_a_moon=(a_km / c.moon_a_km if math.isfinite(a_km) else float("inf")),
        jacobi_constant=cj,
        topology_case=case,
        open_necks=necks,
        zone_ids=zones,
    )
