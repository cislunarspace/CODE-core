# ADR 0003: Axes, ITRF93 defaults, GMAT-compatible Earth orientation

**Status**: Adopted
**Date**: 2026-06-12
**Related Issues**: #62, #80

## Context

Issue #62 introduces GMAT-style coordinate systems, composed from independent
axes and origins. The original request mixes a simplified IAU 2006
implementation with SPICE validation at extremely tight tolerances. That is
not a stable contract for the Earth-fixed frame: a simplified native model can
neither independently match SPICE's high-precision Earth orientation at
`1e-12` nor account for EOP data.

GMAT R2026a uses `ITRF93` as its high-precision Earth SPICE frame and treats
`IAU_EARTH` as low precision. GMAT's native ITRF path also has several
compatibility-specific behaviors: A1MJD inputs, C04 EOP parsing, linear
interpolation of `UT1-UTC` and polar motion, non-interpolated `LOD`, and
optional clamping outside EOP coverage.

## Decision

1. **The default high-precision ITRF is SPICE-backed `ITRF93`.**
   - Public/default ITRF factories and compatibility `ITRFAxes` point to
     `ITRFSpiceAxes(frame="ITRF93")`.
   - `IAU_EARTH` never acts as a silent fallback for high-precision ITRF.
   - `ITRFApproxAxes` remains low-precision/educational only.

2. **GMAT-compatible native ITRF is opt-in.**
   - `GMATITRFAxes` is an independent axes implementation.
   - First-phase native reduction uses a pyerfa/SOFA-based `ErfaXysProvider`
     (an implementation of `XysProvider`) supplying IAU `X, Y, s`.
   - If exact table-level agreement with GMAT is later required, a
     `GMATXysProvider` can replace that source.

3. **State transformation is rotation-rate based.**
   - `Axes.rotation_matrix(et)` returns `R`, with the convention
     `r_icrf = R @ r_axes`.
   - `Axes.rotation_and_rate(et)` returns `(R, Rdot)`, with
     `v_icrf = R @ v_axes + Rdot @ r_axes`.
   - `Axes.state_transform_matrix(et)` derives from `(R, Rdot)`.
   - `CoordinateSystem.transform_state()` prefers `(R, Rdot)` before falling
     back to any angular-velocity compatible path.

4. **Public coordinate epoch inputs remain ET seconds.**
   - Public axes and coordinate-system APIs take SPICE ET seconds.
   - GMAT A1MJD support is lower-level and test-facing, used only for
     consistency checks.

5. **Verification is two-tiered.**
   - `ITRFSpiceAxes("ITRF93")` verifies against `spiceypy.pxform/sxform` to
     `<1e-12` when high-precision kernels are available.
   - `GMATITRFAxes` initially sanity-checks against SPICE `ITRF93` at about
     `1e-7`; main native-chain verification is carried by parser/time/EOP/
     per-stage tests.

6. **Test fixtures and data strategy are explicit.**
   - Committed slim text fixtures cover J2000, the 2017 leap-second boundary,
     and a 2026-06-12 window.
   - Full GMAT data is opt-in via `GMAT_DATA_DIR`.
   - Optional high-precision Earth BPC checks skip explicitly when
     unavailable.
   - Missing required committed fixtures or kernels raise clear errors.

7. **Errors are explicit; precision never silently degrades.**
   - A missing `ITRF93` kernel raises a coordinate error with actionable hints.
   - Missing GMAT native data raises a data error at construction or query
     time.
   - Out-of-range EOP raises by default.
   - GMAT-style clamping exists only behind an explicit compatibility option.

8. **System integration offers no frame-conversion shortcuts.**
   - `System.coordinate_system` holds an optional coordinate system so force
     models can query which frame input states are in.
   - Conversion entry points live at the `CoordinateSystem` layer: callers use
     `system.coordinate_system.transform_state()` / `transform_vector()`
     directly or construct their own `CoordinateSystem`.
   - `System` does **not** provide a `transform()` shortcut.

   > **Revision note (2026-06-15, issue #79)**: original decision 8 specified
   > `System.transform()` as a thin delegate over
   > `CoordinateSystem.transform_state()`. After landing, real production code
   > (drag, gravity, thrust, SRP models) all bypassed `System.transform()` and
   > used `system.coordinate_system.transform_*` directly. Re-discussion
   > concluded the thin delegation was needless indirection; the shortcut is
   > unnecessary. #79 removed `System.transform()` and this clause was revised
   > accordingly. The original `System.transform()` landed in #90 on an
   > under-considered design call now reversed.

## Consequences

### Added

- A stable contract covering SPICE-backed defaults, GMAT-compatible native
  behaviors, fixtures, and tolerances.
- A rotation-rate-based axes abstraction serving both SPICE `sxform` and GMAT
  native `Rdot`.

### Unchanged

- Public coordinate APIs keep taking ET seconds.
- Approximate ITRF remains explicitly labeled low-precision.

### Follow-up work

- If exact agreement with GMAT XYS interpolation is required later, implement
  a GMAT-table-backed `XysProvider`.
- Tighten native-vs-SPICE tolerances only after equivalence of data sources,
  interpolation, and model versions is proven.
