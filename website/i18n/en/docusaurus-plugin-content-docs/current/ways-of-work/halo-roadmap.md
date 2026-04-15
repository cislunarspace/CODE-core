---
format: md
title: Halo Orbit Features — Follow-Up Development Roadmap
---

# Halo Orbit Features — Follow-Up Development Roadmap

This document tracks the evolution of **Halo single-orbit / orbit family / PAL continuation** separately from the [overall quality improvement plan](test-coverage-project-plan.md), to stay aligned with Continuation, differential correction, testing, and documentation work.

---

## Current Deliverables (Baseline)

- Richardson initial guess, `setup_halo_orbit_fixed_z0` / `fixed_x0`, single-orbit and seed generation scripts.
- `pseudo_arclength_continuation` (XZ-symmetric PAL + multiple `dc_scheme` options).
- `halo_pseudo_arclength_continuation` (bidirectional branches, step-size and direction parameters can be aligned with MATLAB example scripts).
- PAL inner-layer numerical safeguards: convergence check ordering, Newton step clamping, non-physical branch rollback, DC strategy fallback when needed.
- Plotting scripts: `plot_halo_orbit.py`, `plot_halo_family.py`.
- Documentation: `docs/algorithms/halo.md` (and English summary `halo_en.md`).

---

## Short Term (1–2 Iterations)

| Priority | Item | Description |
|----------|------|-------------|
| P0 | **PAL vs MATLAB line-by-line alignment switch** | Optional implementation of an "inner layer uses fixed previous curve point \(\mathbf{X}\) to compute \(F\)" branch, for binary-level comparison with `continuation_PAL_CR3BP.m`; default still uses \(\mathbf{X}_{new}\) to compute \(F\). |
| P0 | **Continuation regression tests** | Add small-scale PAL steps in `tests/algorithms/` (fixed random seed or analytic initial values), asserting \(\mathbf{X}\) falls within a reasonable interval and period \(T\in[0.5,5]\) (dimensionless). |
| P1 | **`dc_scheme=matlab_halo_type1` robustness** | Investigate differences between STM Newton and MATLAB `newton_symPeriodicXZ_fixedX`; or standardize on running `adaptive` first after PAL with optional refinement. |
| P1 | **Performance** | `compute_F_and_dF_symmetric_xz_plane` and dynamics integration: evaluate reducing `t_eval` point count, reusing STM, or coarse-grid prediction + fine correction. |

---

## Medium Term

| Item | Description |
|------|-------------|
| **L2 Halo and southern Halo** | Script parameters and documentation partially support `libration_point` / `halo_class` already; add example JSON files and plotting default ranges. |
| **Natural-parameter family `generate_halo_family`** | Clarify applicable use cases vs PAL families (small-amplitude sweep vs tracking family curves); document limitations. |
| **Turning points and bifurcations** | Behavior of true vs pseudo-arclength near turning points; optionally integrate a more general `pseudo_arclength` augmented equation (unified interface with existing Lyapunov/DRO continuation). |

---

## Long Term

| Item | Description |
|------|-------------|
| **Multi-body / ephemeris interface** | Map dimensionless CR3BP results to dimensional quantities and mission timelines. |
| **GUI / Notebook** | Wrap existing scripts into interactive family generation and stability browsing. |

---

## Relationship to Quality Improvement Plan

- **Algorithm Testing Epic**: Incorporate "Continuation PAL + Halo" into the coverage targets for `continuation.py` and `differential_correction.py` (see the master plan **Feature: Algorithm Module Tests**).
- **Documentation**: This roadmap is kept in sync with `docs/algorithms/halo.md`; significant API changes also update `docs/reference/api-reference.md` and `docs/guides/orbit-generation.md`.

---

## Revision History

| Date | Change |
|------|--------|
| 2026-03 | Initial version: first roadmap after Halo PAL, scripts, and documentation baseline |
