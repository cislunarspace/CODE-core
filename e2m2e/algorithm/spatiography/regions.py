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
    "IntervalStateDiagnostics",
    "REGION_LEGEND",
    "RegionId",
    "StateDiagnostics",
    "Table4Bands",
    "classify_by_semi_major_axis",
    "classify_by_semi_major_axis_interval",
    "classify_state",
    "classify_state_interval",
    "jacobi_critical_values",
    "jacobi_topology_case",
    "jacobi_topology_case_interval",
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


@dataclass(frozen=True)
class IntervalStateDiagnostics:
    """状态盒分区诊断（classify_state_interval 的返回值，区间界输出）。

    各诊断量为 ``(lo, hi)`` 区间：判据链（地心距、月心距、osculating 半长
    轴、Jacobi 常数）在截断阶 Taylor 多项式下的保守包围。``zone_ids_possible``
    为区间可能触及的全部分区（含跨界歧义），``zone_ids_certain`` 为区间整体
    落入的分区（possible 子集）；``topology_case_min``/``topology_case_max``
    为 Hill 拓扑 Case 区间界，``ambiguous_critical_values`` 显式列出被区间
    严格跨越的临界 Jacobi 值。
    """

    status: ConvergenceState
    cause: FailureCause
    message: str
    r_geocentric_km: tuple[float, float]
    rho_selenocentric_km: tuple[float, float]
    a_geocentric_km: tuple[float, float]
    a_over_a_moon: tuple[float, float]
    jacobi_constant: tuple[float, float]
    topology_case_min: int
    topology_case_max: int
    ambiguous_critical_values: tuple[str, ...] = field(default_factory=tuple)
    open_necks: tuple[str, ...] = field(default_factory=tuple)
    zone_ids_possible: tuple[int, ...] = field(default_factory=tuple)
    zone_ids_certain: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)


def classify_by_semi_major_axis_interval(
    a_bounds: tuple[float, float],
    *,
    reference: str = "table1",
    constants: PrimerConstants = PRIMER_DEFAULTS,
    system: CR3BP_System | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """按 osculating 半长轴区间（a/a☾ 口径）判定分区的 possible/certain 双标签。

    点版判据（:func:`classify_by_semi_major_axis`）逐条区间化：possible 为
    区间与带相交，certain 为区间整体含于带；阈值与点版同式重算（L1/L2 由
    平动点求根派生，不做硬编码）。

    Args:
        a_bounds: ``(lo, hi)`` 半长轴区间；hi 可为 inf（盒内含逃逸态）。
        reference: 同 :func:`classify_by_semi_major_axis`。
        constants: Primer 常数集（r_L、r_H 由其解析派生）。
        system: 复用已构造的 Primer CR3BP 系统（缺省自建，用于 L1/L2）。

    Returns:
        ``(possible, certain)``：各为升序区域 id 元组（:data:`REGION_LEGEND`
        的键）；certain 必为 possible 子集。

    Raises:
        ValueError: reference 不受支持。
    """
    if reference not in ("table1", "table4"):
        raise ValueError(f"未知的 reference={reference!r}，支持 table1/table4")

    x_lo, x_hi = a_bounds
    possible: set[RegionId] = set()
    certain: set[RegionId] = set()

    def _add(region: RegionId, hits: bool, inside: bool) -> None:
        if hits:
            possible.add(region)
        if inside:
            certain.add(region)

    if reference == "table4":
        bands = table4_bands(constants)
        for lo_k, hi_k, region in zip(bands.lower, bands.upper, _TABLE4_TO_REGION, strict=True):
            _add(region, x_lo <= hi_k and x_hi >= lo_k, x_lo >= lo_k and x_hi <= hi_k)
        r_h = hill_radius_earth(constants) / constants.moon_a_km
        _add(RegionId.HELIOCENTRIC, x_hi > r_h, x_lo > r_h)
    else:
        r_l = laplace_radius_geolunar(constants) / constants.moon_a_km
        r_h = hill_radius_earth(constants) / constants.moon_a_km
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
        # 点版各判据的区间化：hits = 区间与带相交（possible），inside =
        # 区间整体含于带（certain）；含等号口径与点版边界一致。
        _add(RegionId.TERRESTRIAL, x_lo < r_l, x_hi < r_l)
        _add(
            RegionId.CISLUNAR_INNER_SECULAR,
            x_lo < five_to_one and x_hi >= r_l,
            x_lo >= r_l and x_hi < five_to_one,
        )
        _add(
            RegionId.CISLUNAR_OUTER_RESONANT,
            x_lo <= five_to_four and x_hi >= five_to_one,
            x_lo >= five_to_one and x_hi <= five_to_four,
        )
        _add(
            RegionId.CIRCUMLUNAR,
            x_lo <= l2 and x_hi >= l1,
            x_lo >= l1 and x_hi <= l2,
        )
        _add(RegionId.TRANSLUNAR, x_hi > l2, x_lo > l2)
        _add(RegionId.HELIOCENTRIC, x_hi > r_h, x_lo > r_h)
        if not possible:
            possible.add(RegionId.CISLUNAR_OUTER_RESONANT)

    return (
        tuple(int(r) for r in sorted(possible, key=lambda r: r.value)),
        tuple(int(r) for r in sorted(certain, key=lambda r: r.value)),
    )


def jacobi_topology_case_interval(
    c_bounds: tuple[float, float], critical_values: dict[str, float]
) -> tuple[int, int, tuple[str, ...]]:
    """Jacobi 常数区间上的 Hill 拓扑 Case 界与跨界歧义（论文 §3.2）。

    Args:
        c_bounds: ``(lo, hi)`` Jacobi 常数区间。
        critical_values: :func:`jacobi_critical_values` 的输出。

    Returns:
        ``(case_min, case_max, ambiguous)``：C 上界给最小 Case、下界给最大
        Case（C 越大零速面越闭合）；ambiguous 为被区间严格内部跨越的临界值
        名（只查 C1..C4，与点版 :func:`jacobi_topology_case` 口径一致）。
    """
    c_lo, c_hi = c_bounds
    case_min, _ = jacobi_topology_case(c_hi, critical_values)
    case_max, _ = jacobi_topology_case(c_lo, critical_values)
    ambiguous = tuple(k for k in ("C1", "C2", "C3", "C4") if c_lo < critical_values[k] < c_hi)
    return case_min, case_max, ambiguous


def classify_state_interval(
    state: list[float] | np.ndarray,
    half_widths: list[float] | np.ndarray,
    *,
    frame: str = "synodic_barycentric_km",
    reference: str = "table1",
    truncation_order: int = 3,
    constants: PrimerConstants = PRIMER_DEFAULTS,
    system: CR3BP_System | None = None,
) -> IntervalStateDiagnostics:
    """对标称会合系状态加对角不确定度盒做区间化分区诊断（issue #785）。

    判据链（瞬时地心/月心距、osculating 半长轴、Jacobi 常数）在标称点按
    ``truncation_order`` 阶截断 Taylor 多项式展开（#784 DA 原语），经保守
    包围得区间后与分带阈值、C1–C5 临界值比较，输出带界的 possible/certain
    多标签与显式跨界歧义。

    口径注意：区间包围的是判据的 **k 阶截断 Taylor 多项式**（``Da.bound()``
    对自变量域 [-1,1]^6 的保守包围），不是解析函数的验证性包围；截断余项
    O(|h|^{k+1})，半宽越小包围越紧。``half_widths`` 全零时退化为点判定，
    逐位复用 :func:`classify_state`（DA 路径的浮点求和序不保证与点版逐位
    相同，零宽锚点由委托实现保证一致）。

    Args:
        state: 标称 6 维状态 [x, y, z, vx, vy, vz]，frame 语义同
            :func:`classify_state`（synodic_barycentric_km / _nd）。
        half_widths: 对角盒各分量半宽 [hx, hy, hz, hvx, hvy, hvz]，与
            state 同 frame 同单位，各分量 ≥ 0。
        reference: 同 :func:`classify_by_semi_major_axis`。
        truncation_order: DA 截断阶，≥ 1；每次调用重建进程级 DA 上下文
            （当前 Python 侧无跨调用持有 Da 对象的其他路径）。
        constants: Primer 常数集。
        system: 复用 Primer CR3BP 系统（缺省自建）。

    Returns:
        :class:`IntervalStateDiagnostics`：各诊断量的区间界、Case 区间、
        跨越的临界值与 possible/certain 分区多标签。

    Raises:
        ValueError: frame 不受支持、维数不对、半宽为负、截断阶 < 1，或
            标称状态位于主天体奇点（地心/月心距 < 1e-12，判据不可微）。
        RustExtensionUnavailableError: Rust 扩展缺失或 DA 符号不可用。
    """
    if frame not in ("synodic_barycentric_km", "synodic_barycentric_nd"):
        raise ValueError(
            f"不支持的 frame={frame!r}；当前支持 synodic_barycentric_km/"
            "synodic_barycentric_nd（gcrs_km 待星历批次接入，见 ADR 0041）"
        )
    arr = np.asarray(state, dtype=float)
    if arr.shape != (6,):
        raise ValueError(f"状态须为 6 维 [x,y,z,vx,vy,vz]，得到 shape={arr.shape}")
    widths = np.asarray(half_widths, dtype=float)
    if widths.shape != (6,):
        raise ValueError(f"half_widths 须为 6 维各分量半宽，得到 shape={widths.shape}")
    if bool(np.any(widths < 0.0)):
        raise ValueError("half_widths 各分量须 ≥ 0（对角盒半宽）")
    if truncation_order < 1:
        raise ValueError(f"truncation_order 须 ≥ 1，得到 {truncation_order}")

    c = constants
    sys = system if system is not None else primer_cr3bp_system(constants)

    # 零宽锚：半宽全零退化为点判定（include_overlaps=True 口径），各标量
    # 复制为退化区间，状态三元组照抄。
    if not bool(np.any(widths > 0.0)):
        diag = classify_state(
            arr,
            frame=frame,
            reference=reference,
            include_overlaps=True,
            constants=constants,
            system=sys,
        )
        return IntervalStateDiagnostics(
            status=diag.status,
            cause=diag.cause,
            message=diag.message,
            r_geocentric_km=(diag.r_geocentric_km, diag.r_geocentric_km),
            rho_selenocentric_km=(diag.rho_selenocentric_km, diag.rho_selenocentric_km),
            a_geocentric_km=(diag.a_geocentric_km, diag.a_geocentric_km),
            a_over_a_moon=(diag.a_over_a_moon, diag.a_over_a_moon),
            jacobi_constant=(diag.jacobi_constant, diag.jacobi_constant),
            topology_case_min=diag.topology_case,
            topology_case_max=diag.topology_case,
            ambiguous_critical_values=(),
            open_necks=diag.open_necks,
            zone_ids_possible=diag.zone_ids,
            zone_ids_certain=diag.zone_ids,
        )

    from e2m2e.integrators import Da, da_init_py, require_rust_extension

    require_rust_extension("da_init_py", "da_initialized_py", "da_truncation_order_py", "Da")

    n_rad_s = c.cr3bp_mean_motion_rad_s
    if frame == "synodic_barycentric_km":
        state_nd = np.concatenate([arr[:3] / c.moon_a_km, arr[3:] / (c.moon_a_km * n_rad_s)])
        h_nd = np.concatenate([widths[:3] / c.moon_a_km, widths[3:] / (c.moon_a_km * n_rad_s)])
    else:
        state_nd = arr.copy()
        h_nd = widths.copy()

    # 进程级 DA 上下文按需重建（阶 truncation_order、6 变量）。
    da_init_py(truncation_order, 6)

    mu = float(sys.mu)
    gamma = 1.0 - mu
    q = [Da.constant(float(state_nd[i])) + float(h_nd[i]) * Da.variable(i + 1) for i in range(6)]
    x, y, z, vx, vy, vz = q

    # 地心距 r1（地球位于 (-mu,0,0)）与月心距 r2（月球位于 (1-mu,0,0)）。
    r1_sq = (x + mu) * (x + mu) + y * y + z * z
    r2_sq = (x - 1.0 + mu) * (x - 1.0 + mu) + y * y + z * z
    if r1_sq.cons() < 1.0e-24 or r2_sq.cons() < 1.0e-24:
        raise ValueError(
            "标称状态位于主天体奇点（地心距或月心距 < 1e-12 无量纲），"
            "判据在此不可微，区间判定不可用"
        )
    r1 = r1_sq.sqrt()
    r2 = r2_sq.sqrt()
    v2 = vx * vx + vy * vy + vz * vz
    # Jacobi（Parker 约定，同 sys.get_jacobi_constant）与无量纲半长轴分母
    # （a/a☾ = 1/(2/r1 − v²/γ)，γ = 1−μ）。
    cj_da = x * x + y * y + 2.0 * (1.0 - mu) / r1 + 2.0 * mu / r2 - v2
    denom = 2.0 / r1 - v2 / gamma

    r1_lo, r1_hi = r1.bound()
    r2_lo, r2_hi = r2.bound()
    cj_lo, cj_hi = cj_da.bound()
    d_lo, d_hi = denom.bound()

    # a(D) = 1/D 在 D > 0 单调递减：整盒非正 → 全逃逸（a = inf，镜像点版对
    # 非有限 a 跳过分带的语义，zones 置空）；区间跨 0 → 上界无界；整盒为正
    # → 双侧有界（端点取对侧倒数）。
    possible_zones: tuple[int, ...] = ()
    certain_zones: tuple[int, ...] = ()
    if d_hi <= 0.0:
        a_nd: tuple[float, float] = (float("inf"), float("inf"))
    else:
        a_hi = float("inf") if d_lo <= 0.0 else 1.0 / d_lo
        a_nd = (1.0 / d_hi, a_hi)
        possible_zones, certain_zones = classify_by_semi_major_axis_interval(
            a_nd, reference=reference, constants=constants, system=sys
        )

    crits = jacobi_critical_values(sys, c)
    case_min, case_max, ambiguous = jacobi_topology_case_interval((cj_lo, cj_hi), crits)
    open_necks = jacobi_topology_case(cj_lo, crits)[1]

    return IntervalStateDiagnostics(
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="ok",
        r_geocentric_km=(r1_lo * c.moon_a_km, r1_hi * c.moon_a_km),
        rho_selenocentric_km=(r2_lo * c.moon_a_km, r2_hi * c.moon_a_km),
        a_geocentric_km=(a_nd[0] * c.moon_a_km, a_nd[1] * c.moon_a_km),
        a_over_a_moon=a_nd,
        jacobi_constant=(cj_lo, cj_hi),
        topology_case_min=case_min,
        topology_case_max=case_max,
        ambiguous_critical_values=ambiguous,
        open_necks=open_necks,
        zone_ids_possible=possible_zones,
        zone_ids_certain=certain_zones,
    )
