"""地月空间分区（spatiography）。

论文式分区体系的解析内核：Rosengren et al. 2026《The Astrodynamics Primer
on Cislunar and Translunar Space》§5 的全部边界尺度、共振梯、五省分类器与
可视化边界几何。属二档查询能力（ADR 0014）；常数与复现陷阱见 ADR 0041。

核心概念：

- **五省层级（five provinces）**：terrestrial → cislunar（内带长期主导 /
  外带共振结构化）→ circumlunar（L1/L2 gateway 飞地）→ translunar →
  heliocentric。``Cislunar`` 只用于两个带级区域名，绝不作伞式值；系统
  总称用 geolunar space（论文 §2.6 命名纪律）。
- **Laplace 半径**：地心 r_L（terrestrial/cislunar 界，式 98）与月心
  rho_L（式 124）——日月力矩与中心天体扁率长期效应的交点。
- **影响球族**：Hill（gateway 稳定性）、Laplace–Tisserand（patched-conic
  开关）、Chebotarev（直接力对等）、Battin（一阶非对称）——物理含义互
  不相同、不可互换（论文 §5.4.2）。
- **gateway 拓扑**：Jacobi 五临界值 Case I–V 与 L1/L2 颈口开启
  （精确求根口径，论文表值为级数近似注记）。
- **deliberate overlap**：分区边界有意重叠（Table 4 口径），分类器返回
  有序多标签而非互斥单值。

对外入口（MCP 工具经 ``api/facade.py`` 暴露）：``spatiography_scales`` /
``spatiography_classify`` / ``spatiography_boundaries``。
"""

from __future__ import annotations

from .boundaries import (
    AE_CURVE_NAMES,
    SYNODIC_ELEMENT_NAMES,
    BoundaryElement,
    BoundarySetResult,
    ae_curves,
    synodic_planar_elements,
)
from .cartography import (
    ECLIPSE_EPOCH_SCENARIO,
    MAP_ZONE_NAMES,
    MapResult,
    Scenario,
    compare_models,
    default_scenario,
    dynamical_map,
    elements_to_state,
    zone_grid,
)
from .constants import PRIMER_CITATION, PRIMER_DEFAULTS, PrimerConstants
from .fate import (
    FATE_CLASSES,
    FATE_THRESHOLDS_DEFAULT,
    FateDiagnostics,
    FateThresholds,
    build_cr3bp_fate_events,
    classify_fate,
    extract_cr3bp_fate,
)
from .megno import megno_reference, propagate_bcr4bp_megno, propagate_cr3bp_megno
from .regions import (
    REGION_LEGEND,
    RegionId,
    StateDiagnostics,
    Table4Bands,
    classify_by_semi_major_axis,
    classify_state,
    jacobi_critical_values,
    jacobi_topology_case,
    primer_cr3bp_system,
    table4_bands,
)
from .resonances import (
    PRIMER_RESONANCE_KINDS,
    ResonanceCenter,
    ResonanceLadderResult,
    ResonanceWidthEnvelope,
    ResonanceWidthProfile,
    ResonanceWidthResult,
    gallardo_resonance_width,
    gallardo_width_envelopes,
    resonance_centers,
)
from .scales import (
    activity_surface_moon,
    battin_soi_earth,
    battin_soi_moon,
    characteristic_rate_j2,
    characteristic_rate_lunar_exterior,
    characteristic_rate_solar_exterior,
    chebotarev_radius_earth,
    chebotarev_radius_moon,
    geo_radius_km,
    hill_radius_earth,
    hill_radius_moon,
    laplace_radius_geolunar,
    laplace_radius_selenocentric,
    soi_laplace_earth,
    soi_laplace_moon,
    tidal_parity_radius,
    tisserand_parameter,
)
from .secular import (
    VZLK_CRITICAL_INCLINATION_DEG,
    SecularLocusCurve,
    VzlkPortrait,
    VzlkValidity,
    apsidal_rate_cislunar,
    apsidal_rate_translunar,
    apsidal_stationary_inclination_translunar,
    nodal_rate_ext_moon,
    nodal_rate_int_moon,
    secular_loci_curves,
    secular_prefactor_ext_moon,
    secular_prefactor_ext_sun,
    secular_prefactor_int_moon,
    vzlk_frequency_rad_s,
    vzlk_phase_portrait,
    vzlk_tidal_sum,
    vzlk_timescale_days,
    vzlk_validity,
)

__all__ = [
    "AE_CURVE_NAMES",
    "BoundaryElement",
    "BoundarySetResult",
    "ECLIPSE_EPOCH_SCENARIO",
    "MAP_ZONE_NAMES",
    "MapResult",
    "PRIMER_CITATION",
    "PRIMER_DEFAULTS",
    "PRIMER_RESONANCE_KINDS",
    "REGION_LEGEND",
    "ResonanceCenter",
    "ResonanceLadderResult",
    "ResonanceWidthEnvelope",
    "ResonanceWidthProfile",
    "ResonanceWidthResult",
    "RegionId",
    "Scenario",
    "SYNODIC_ELEMENT_NAMES",
    "FATE_CLASSES",
    "FATE_THRESHOLDS_DEFAULT",
    "FateDiagnostics",
    "FateThresholds",
    "SecularLocusCurve",
    "PrimerConstants",
    "StateDiagnostics",
    "Table4Bands",
    "VZLK_CRITICAL_INCLINATION_DEG",
    "VzlkPortrait",
    "VzlkValidity",
    "activity_surface_moon",
    "ae_curves",
    "apsidal_rate_cislunar",
    "apsidal_rate_translunar",
    "apsidal_stationary_inclination_translunar",
    "battin_soi_earth",
    "battin_soi_moon",
    "characteristic_rate_j2",
    "characteristic_rate_lunar_exterior",
    "characteristic_rate_solar_exterior",
    "chebotarev_radius_earth",
    "chebotarev_radius_moon",
    "classify_by_semi_major_axis",
    "classify_state",
    "compare_models",
    "default_scenario",
    "dynamical_map",
    "elements_to_state",
    "build_cr3bp_fate_events",
    "classify_fate",
    "extract_cr3bp_fate",
    "gallardo_resonance_width",
    "gallardo_width_envelopes",
    "geo_radius_km",
    "hill_radius_earth",
    "hill_radius_moon",
    "jacobi_critical_values",
    "jacobi_topology_case",
    "laplace_radius_geolunar",
    "laplace_radius_selenocentric",
    "megno_reference",
    "nodal_rate_ext_moon",
    "nodal_rate_int_moon",
    "primer_cr3bp_system",
    "propagate_bcr4bp_megno",
    "propagate_cr3bp_megno",
    "resonance_centers",
    "secular_loci_curves",
    "secular_prefactor_ext_moon",
    "secular_prefactor_ext_sun",
    "secular_prefactor_int_moon",
    "soi_laplace_earth",
    "soi_laplace_moon",
    "synodic_planar_elements",
    "table4_bands",
    "tidal_parity_radius",
    "tisserand_parameter",
    "vzlk_frequency_rad_s",
    "vzlk_phase_portrait",
    "vzlk_tidal_sum",
    "vzlk_timescale_days",
    "vzlk_validity",
    "zone_grid",
]
