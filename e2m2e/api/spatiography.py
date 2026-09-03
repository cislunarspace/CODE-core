"""空间分区接口类：spatiography 五省分区分析（ADR 0043 决策 3）。


``Spatiography`` 是接口层的空间分区类：五个分区分析工具（scales /
classify / boundaries / resonance atlas / dynamical map，ADR 0041）。
MCP/CLI/sidecar 经单一工具清单到达这些方法；进程内调用方从
``Facade().spatiography`` 取实例或直接构造（ADR 0043，ADR 0014 决策 2
的类分家）。
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

import numpy as np

from e2m2e.data.templates import ConvergenceState, FailureCause

from .facade import _exception_triplet, _serialize_value, mcp_exposed
from .models import (
    OrbitError,
    SpatiographyAtlasRequest,
    SpatiographyAtlasResponse,
    SpatiographyBoundariesRequest,
    SpatiographyBoundariesResponse,
    SpatiographyClassifyRequest,
    SpatiographyClassifyResponse,
    SpatiographyMapRequest,
    SpatiographyMapResponse,
    SpatiographyScalesRequest,
    SpatiographyScalesResponse,
)

__all__ = ["Spatiography"]


class Spatiography:
    """空间分区接口类（ADR 0043 决策 3）：无状态，纯分析入口。"""

    @mcp_exposed(request_model=SpatiographyScalesRequest)
    def spatiography_scales(self, **params) -> SpatiographyScalesResponse:
        """分区解析尺度计算（spatiography，二档）。/ Spatiography analytic scales (tier 2).

        计算地月空间分区（Rosengren et al. 2026 Primer §5）的全部闭式边界
        尺度：Laplace 半径（地心/月心）、影响球族（Hill / Laplace-Tisserand /
        Chebotarev / Battin）、tidal parity、共振梯（Table 1/2）、平动点精确
        解与 Jacobi 临界值。Primer 常数口径（SPICE GM + Simon 1994 月根数）。
        """
        try:
            request = SpatiographyScalesRequest(**params)
            from e2m2e.algorithm import spatiography as sp

            c = sp.PRIMER_DEFAULTS
            system = sp.primer_cr3bp_system(c)
            system.compute_libration_points()
            import math

            scales: dict[str, float] = {
                "laplace_radius_geolunar_km": sp.laplace_radius_geolunar(c),
                "laplace_radius_selenocentric_km": sp.laplace_radius_selenocentric(c),
                "hill_radius_moon_km": sp.hill_radius_moon(c),
                "hill_radius_earth_km": sp.hill_radius_earth(c),
                "soi_laplace_moon_km": sp.soi_laplace_moon(c),
                "soi_chebotarev_moon_km": sp.chebotarev_radius_moon(c),
                "battin_moon_anti_earthward_km": sp.battin_soi_moon(0.0, c),
                "battin_moon_earthward_km": sp.battin_soi_moon(math.pi, c),
                "soi_laplace_earth_km": sp.soi_laplace_earth(c),
                "soi_chebotarev_earth_km": sp.chebotarev_radius_earth(c),
                "tidal_parity_radius_km": sp.tidal_parity_radius(c),
                "geo_radius_km": sp.geo_radius_km(c),
                "moon_period_days": c.moon_period_days,
            }
            unknown = set(request.elements) - set(scales)
            if unknown:
                raise ValueError(f"未知的尺度名：{sorted(unknown)}")
            if request.elements:
                scales = {k: v for k, v in scales.items() if k in set(request.elements)}
            for key in (
                "laplace_radius_geolunar_km",
                "tidal_parity_radius_km",
                "hill_radius_earth_km",
                "soi_laplace_earth_km",
            ):
                if key in scales:
                    scales[key.replace("_km", "_over_a_moon")] = scales[key] / c.moon_a_km

            libration_points_km = {
                name: (getattr(system, name) * c.moon_a_km).tolist()
                for name in ("L1", "L2", "L3", "L4", "L5")
            }
            ladder = sp.resonance_centers("all", c)
            return SpatiographyScalesResponse(
                status=ConvergenceState.CONVERGED,
                cause=FailureCause.NONE,
                message="ok",
                scales=scales,
                libration_points_km=libration_points_km,
                jacobi_criticals=sp.jacobi_critical_values(system, c),
                resonance_ladder=[dataclasses.asdict(center) for center in ladder],
                constants_used=dataclasses.asdict(c),
                citation=sp.PRIMER_CITATION,
                details={"moon_a_km": c.moon_a_km},
            )
        except OrbitError:
            raise
        except (ValueError, TypeError) as exc:
            raise OrbitError("INVALID_PARAMS", str(exc)) from exc
        except Exception as exc:
            status, cause, message = _exception_triplet(exc)
            raise OrbitError("SPATIOGRAPHY_FAILED", message, status=status, cause=cause) from exc

    @mcp_exposed(request_model=SpatiographyClassifyRequest)
    def spatiography_classify(self, **params) -> SpatiographyClassifyResponse:
        """分区区域分类（spatiography，二档）。/ Spatiography region classification (tier 2).

        对会合系状态逐点判定五省分区（terrestrial / cislunar 内带 / cislunar
        外带 / circumlunar / translunar / heliocentric，论文 Table 1 或附录 B
        Table 4 口径），重叠带返回多标签；附 osculating a、Jacobi 值与 Hill
        五拓扑 Case 诊断。
        """
        try:
            request = SpatiographyClassifyRequest(**params)
            from e2m2e.algorithm import spatiography as sp

            system = sp.primer_cr3bp_system()
            zone_ids: list[list[int]] = []
            diagnostics: list[dict[str, Any]] = []
            for state in request.states:
                diag = sp.classify_state(
                    state,
                    frame=request.frame,
                    reference=request.reference,
                    include_overlaps=request.include_overlaps,
                    system=system,
                )
                zone_ids.append(list(diag.zone_ids))
                diagnostics.append(_serialize_value(dataclasses.asdict(diag)))
            return SpatiographyClassifyResponse(
                status=ConvergenceState.CONVERGED,
                cause=FailureCause.NONE,
                message="ok",
                zone_ids=zone_ids,
                legend={str(k): v for k, v in sp.REGION_LEGEND.items()},
                diagnostics=diagnostics,
                details={"n_states": len(request.states), "reference": request.reference},
            )
        except OrbitError:
            raise
        except (ValueError, TypeError) as exc:
            raise OrbitError("INVALID_PARAMS", str(exc)) from exc
        except Exception as exc:
            status, cause, message = _exception_triplet(exc)
            raise OrbitError("SPATIOGRAPHY_FAILED", message, status=status, cause=cause) from exc

    @mcp_exposed(request_model=SpatiographyBoundariesRequest)
    def spatiography_boundaries(self, **params) -> SpatiographyBoundariesResponse:
        """分区边界几何（spatiography，二档）。/ Spatiography boundary geometry (tier 2).

        输出可视化用边界几何数据：会合系（质心原点，z=0）的 r_L / tidal
        parity / 双系 Hill 与 SOI 圆族、Battin 非对称闭合曲线、L1–L5，或
        (a,e) 根数平面走廊曲线族。前端只做单位归一与绘制，不在界面重算。
        """
        try:
            request = SpatiographyBoundariesRequest(**params)
            from e2m2e.algorithm import spatiography as sp

            state_frame: Literal["synodic_barycentric_km", "element_space_ae"]
            if request.kind == "synodic_planar":
                result = sp.synodic_planar_elements(
                    resolution=request.resolution, boundary_set=request.boundary_set
                )
                state_frame = "synodic_barycentric_km"
            else:
                result = sp.ae_curves(
                    n_points=request.resolution, boundary_set=request.boundary_set
                )
                state_frame = "element_space_ae"
            elements = [_serialize_value(dataclasses.asdict(e)) for e in result.elements]
            return SpatiographyBoundariesResponse(
                status=ConvergenceState.CONVERGED,
                cause=FailureCause.NONE,
                message="ok",
                elements=elements,
                state_frame=state_frame,
                details={"count": len(elements), "kind": request.kind},
            )
        except OrbitError:
            raise
        except (ValueError, TypeError) as exc:
            raise OrbitError("INVALID_PARAMS", str(exc)) from exc
        except Exception as exc:
            status, cause, message = _exception_triplet(exc)
            raise OrbitError("SPATIOGRAPHY_FAILED", message, status=status, cause=cause) from exc

    @mcp_exposed(request_model=SpatiographyAtlasRequest)
    def spatiography_resonance_atlas(self, **params) -> SpatiographyAtlasResponse:
        """共振图集（spatiography，二档）。/ Resonance atlas (tier 2).

        Primer §4.2–§4.4 / §5.3 的共振与长期解析骨架：Gallardo 半解析
        共振半宽包络（式 100–104，计算设置对齐 Fig. 8：共面切片、
        Simon 1994 月根数、2ρ_H 近遇截断）、拱线驻定 loci（式 75–78）、
        vZLK 相图与时间尺度（式 64–71）。1:1 共振带宽系统性高估
        （论文 §5.3 声明），不得当 gateway 边界用。
        """
        try:
            request = SpatiographyAtlasRequest(**params)
            from e2m2e.algorithm import spatiography as sp

            products = set(request.products)
            unknown = products - {"gallardo_widths", "secular_loci", "vzlk_portrait"}
            if unknown:
                raise ValueError(f"未知产品名：{sorted(unknown)}")
            if request.e_max <= request.e_min:
                raise ValueError("e_max 必须大于 e_min")
            elements: list[dict[str, Any]] = []

            if "gallardo_widths" in products:
                pairs = (
                    None
                    if request.resonance_pairs is None
                    else [(int(p[0]), int(p[1])) for p in request.resonance_pairs]
                )
                e_grid = [
                    request.e_min + (request.e_max - request.e_min) * i / (request.n_e - 1)
                    for i in range(request.n_e)
                ]
                result = sp.gallardo_width_envelopes(
                    pairs=pairs,
                    e_grid=e_grid,
                    varpi_offset_deg=request.varpi_offset_deg,
                    n_sigma=request.n_sigma,
                    n_lambda=request.n_lambda,
                )
                caveat = (
                    "Gallardo 半解析包络（式 100–104），共面切片 + 2ρ_H 近遇截断；"
                    "crossing diagnostic 而非物理边界。1:1 带系统性高估（论文 §5.3）"
                )
                for env in result.envelopes:
                    note_extra = "；1:1 高估 caveat 适用" if env.k == env.k_body else ""
                    elements.append(
                        {
                            "kind": "envelope_ae",
                            "label": f"{env.label} lower",
                            "formula_id": "Eq.100-104",
                            "points": _serialize_value(
                                np.stack([env.lower_a_km, env.eccentricities], axis=1)
                            ),
                            "note": caveat + note_extra,
                        }
                    )
                    elements.append(
                        {
                            "kind": "envelope_ae",
                            "label": f"{env.label} upper",
                            "formula_id": "Eq.100-104",
                            "points": _serialize_value(
                                np.stack([env.upper_a_km, env.eccentricities], axis=1)
                            ),
                            "note": caveat + note_extra,
                        }
                    )
                    elements.append(
                        {
                            "kind": "vertical_ae",
                            "label": env.label,
                            "formula_id": "Eq.87",
                            "a_km": env.a_center_km,
                            "note": f"名义中心（式 87）；k={env.k}, k_body={env.k_body}",
                        }
                    )

            if "secular_loci" in products:
                c = sp.PRIMER_DEFAULTS
                grid = np.linspace(
                    request.a_over_a_moon_min * c.moon_a_km,
                    request.a_over_a_moon_max * c.moon_a_km,
                    request.n_locus,
                )
                curves = sp.secular_loci_curves(
                    a_grid_km=grid,
                    e_slices=request.locus_e_slices,
                    branches=("cislunar", "translunar"),
                )
                for curve in curves:
                    elements.append(
                        {
                            "kind": "locus_ai",
                            "label": f"apsidal-stationary {curve.branch} e={curve.eccentricity:g}",
                            "formula_id": curve.formula_id,
                            "points": _serialize_value(
                                np.stack([curve.a_km, np.degrees(curve.inclination_rad)], axis=1)
                            ),
                            "note": "最低阶 spatiographic 骨架（式 75–78），非月距附近"
                            "精确局部共振位置",
                        }
                    )

            vzlk_scalars: dict[str, float] = {}
            if "vzlk_portrait" in products:
                portrait = sp.vzlk_phase_portrait(request.vzlk_c1)
                for level, pts in portrait.curves:
                    elements.append(
                        {
                            "kind": "portrait_curve",
                            "label": f"c2={level:g}",
                            "formula_id": "Eq.68",
                            "c2": float(level),
                            "points": _serialize_value(pts),
                            "note": "vZLK 相图 c2 等值线（式 65–68）；c1 < 0.6 时 c2=0 为分离线",
                        }
                    )
                c = sp.PRIMER_DEFAULTS
                vzlk_scalars = {
                    "critical_inclination_deg": sp.VZLK_CRITICAL_INCLINATION_DEG,
                    "critical_inclination_retro_deg": 180.0 - sp.VZLK_CRITICAL_INCLINATION_DEG,
                    "c1": request.vzlk_c1,
                    "e_max_separatrix": (
                        float("nan") if not portrait.has_separatrix else portrait.e_max
                    ),
                    "nu_vzlk_rad_s_at_a_moon": sp.vzlk_frequency_rad_s(c.moon_a_km),
                    "t_vzlk_days_at_a_moon": sp.vzlk_timescale_days(c.moon_a_km),
                    "tidal_sum_inv_s2": sp.vzlk_tidal_sum(c),
                }
                validity = sp.vzlk_validity(c.moon_a_km)
                vzlk_scalars.update(
                    {
                        "validity_j2_suppressed_at_a_moon": float(validity.j2_suppressed),
                        "validity_double_averaging_warning_at_a_moon": float(
                            validity.double_averaging_warning
                        ),
                    }
                )

            return SpatiographyAtlasResponse(
                status=ConvergenceState.CONVERGED,
                cause=FailureCause.NONE,
                message="ok",
                elements=elements,
                state_frames={
                    "envelope_ae": "element_space_ae",
                    "vertical_ae": "element_space_ae",
                    "locus_ai": "element_space_ai",
                    "portrait_curve": "vzlk_phase_plane",
                },
                vzlk=vzlk_scalars,
                details={
                    "count": len(elements),
                    "products": sorted(products),
                    "varpi_offset_deg": request.varpi_offset_deg,
                },
            )
        except OrbitError:
            raise
        except (ValueError, TypeError) as exc:
            raise OrbitError("INVALID_PARAMS", str(exc)) from exc
        except Exception as exc:
            status, cause, message = _exception_triplet(exc)
            raise OrbitError("SPATIOGRAPHY_FAILED", message, status=status, cause=cause) from exc

    @mcp_exposed(request_model=SpatiographyMapRequest)
    def spatiography_dynamical_map(self, **params) -> SpatiographyMapResponse:
        """六域两层天图（spatiography，二档）。/ Spatiographic dynamical map (tier 2).

        Primer §7.3 (a, e) 天图管线：Table 4 六制图域网格 × 命名场景
        （2027-08-02 日全食历元统一初值切片），逐格传播 EM/EMS 点质量
        模型，输出 MEGNO Ȳ 场与八类命运场 + 诊断量。大数组建议走
        sidecar 二进制帧（E2M2 帧，ADR 0035）。
        """
        try:
            request = SpatiographyMapRequest(**params)
            from e2m2e.algorithm.spatiography import cartography as cart
            from e2m2e.algorithm.spatiography.fate import FATE_CLASSES, FateThresholds

            result = cart.dynamical_map(
                request.zone,
                model=request.model,
                n_a=request.n_a,
                n_e=request.n_e,
                e_min=request.e_min,
                e_max=request.e_max,
                span_years=request.span_years,
                thresholds=FateThresholds(
                    ybar_ordered_band=request.ybar_ordered_band,
                    ybar_chaotic_excess=request.ybar_chaotic_excess,
                ),
                rtol=request.rtol,
                max_step_hours=request.max_step_hours,
                stop_on_terminal=request.stop_on_terminal,
            )

            def _field(array):
                return [[None if not np.isfinite(v) else float(v) for v in row] for row in array]

            c = cart.PRIMER_DEFAULTS
            from e2m2e.algorithm.spatiography import regions as sp_regions
            from e2m2e.algorithm.spatiography import scales as sp_scales

            system = sp_regions.primer_cr3bp_system(c)
            system.compute_libration_points()
            crit = sp_regions.jacobi_critical_values(system, c)
            t_moon = sp_scales.tisserand_parameter(c.moon_a_km, 0.0, 0.0, c)
            return SpatiographyMapResponse(
                status=result.status,
                cause=result.cause,
                message=result.message,
                zone=result.zone,
                model=result.model,
                span_years=result.span_years,
                a_over_a_moon=[float(v) for v in result.a_over_a_moon],
                e_grid=[float(v) for v in result.e_grid],
                ybar_field=_field(result.ybar_field),
                fate_ids=result.fate_ids.astype(int).tolist(),
                t_escape_years_field=_field(result.t_escape_years_field),
                min_r_sel_km_field=_field(result.min_r_sel_km_field),
                min_r_geo_km_field=_field(result.min_r_geo_km_field),
                fate_legend={str(i): name for i, name in enumerate(FATE_CLASSES)},
                thresholds={
                    "ybar_ordered_band": result.thresholds.ybar_ordered_band,
                    "ybar_chaotic_excess": result.thresholds.ybar_chaotic_excess,
                },
                scenario={
                    "epoch_utc": result.scenario.epoch_utc,
                    "raan_deg": result.scenario.raan_deg,
                    "argp_deg": result.scenario.argp_deg,
                    "mean_anom_deg": result.scenario.mean_anom_deg,
                    "inclination_is_moon_plane": result.scenario.inclination_is_moon_plane,
                    "provenance": result.scenario.provenance,
                },
                diagnostic_focus=result.diagnostic_focus,
                details={
                    "cells": result.cells,
                    "zone_bands_a_over_a_moon": [
                        [lo, hi]
                        for lo, hi in zip(
                            sp_regions.table4_bands(c).lower,
                            sp_regions.table4_bands(c).upper,
                            strict=True,
                        )
                    ],
                    "gateway_tisserand_note": (
                        f"共面 T☾(a=a☾, e=0) = {t_moon:.3f}，低于第一颈口阈值 "
                        f"C1 = {crit['C1']:.3f}（精确求根口径）：CG 域为开放"
                        " gateway 拓扑（论文 §7.3）"
                    ),
                },
            )
        except OrbitError:
            raise
        except (ValueError, TypeError) as exc:
            raise OrbitError("INVALID_PARAMS", str(exc)) from exc
        except Exception as exc:
            status, cause, message = _exception_triplet(exc)
            raise OrbitError("SPATIOGRAPHY_FAILED", message, status=status, cause=cause) from exc
