# Issue #62 Plan: Coordinate System Core and ITRF Contract

**Date**: 2026-06-12  
**Parent**: #62  
**Related slices**: #80-#93

## Goal

Deliver the coordinate-system vertical slice for `Axes + Origin + CoordinateSystem`, precise SPICE-backed ITRF defaults, explicit first-phase GMAT-compatible native ITRF, and thin `System` delegation.

## Confirmed Contract

- Public/default precise ITRF is `ITRFSpiceAxes(frame="ITRF93")`.
- `ITRFAxes` is a compatibility alias/thin subclass of the SPICE-backed default.
- `IAU_EARTH` is not a fallback for precise ITRF.
- `GMATITRFAxes` is explicit opt-in and first-phase native compatible, not exact GMAT table parity.
- Public coordinate APIs accept SPICE ET seconds.
- A1MJD behavior is low-level/test-only for GMAT parity checks.
- EOP out-of-range raises by default; GMAT clamp behavior is explicit opt-in.
- `System.transform()` is a thin delegate to `CoordinateSystem.transform_state()`.

## Vertical Slices

1. #80 records the ADR contract.
2. #81 establishes static coordinate transforms.
3. #82 adds `rotation_and_rate(et)` state semantics.
4. #83 adds committed fixture discovery and optional `GMAT_DATA_DIR` policy.
5. #84 makes SPICE-backed `ITRF93` the public default.
6. #85 parses GMAT leap-second and EOP fixtures.
7. #86 keeps public ET while adding GMAT-compatible time conversion.
8. #87 adds the pyerfa-backed XYS provider interface.
9. #88 composes deterministic GMAT-compatible reduction stages.
10. #89 exposes `GMATITRFAxes` as explicit opt-in.
11. #90 adds thin `System` delegation.
12. #91 finalizes public exports and documentation signals.
13. #92 adds focused E2E coordinate smoke tests.
14. #93 is the review gate for closing parent #62.

## Test Plan

- `uv run pytest tests/core/coordinate tests/core/system/test_system_coordinate_transform.py -q`
- `uv run pytest -q`

Optional high-precision checks that require full GMAT/SPICE Earth BPC data skip clearly when `GMAT_DATA_DIR` is unset or when `ITRF93` coverage is unavailable.

## Follow-Up Candidates

- Add a GMAT table-backed `XysProvider` for exact GMAT XYS interpolation parity.
- Tighten native-vs-SPICE tolerances only after confirming identical XYS, EOP interpolation, and time-scale conventions.
- Promote full GMAT data/BPC checks in CI only if the project adopts a distributable kernel fixture policy.
