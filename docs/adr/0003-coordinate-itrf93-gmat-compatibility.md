# ADR 0003: Coordinate Axes, ITRF93 Defaults, and GMAT-Compatible Earth Orientation

**Status**: Accepted  
**Date**: 2026-06-12  
**Issue**: #62, #80

## Context

Issue #62 introduces GMAT-style coordinate systems built from independent axes and origins. The original request combined a simplified IAU 2006 implementation with SPICE validation at very tight tolerances. That is not a stable contract for Earth-fixed frames: a simplified native model cannot independently match SPICE high-precision Earth orientation at `1e-12` while also omitting EOP data.

GMAT R2026a uses `ITRF93` as the high-fidelity Earth SPICE frame and treats `IAU_EARTH` as low fidelity. GMAT's native ITRF path also has compatibility-specific behavior: A1MJD input, C04 EOP parsing, linear `UT1-UTC` and polar-motion interpolation, non-interpolated `LOD`, and optional clamping outside EOP coverage.

## Decision

1. **Default precise ITRF is SPICE-backed `ITRF93`.**
   - Public/default ITRF factories and compatibility `ITRFAxes` point to `ITRFSpiceAxes(frame="ITRF93")`.
   - `IAU_EARTH` is never a silent fallback for precise ITRF.
   - `ITRFApproxAxes` remains low-fidelity/educational only.

2. **GMAT-compatible native ITRF is explicit opt-in.**
   - `GMATITRFAxes` is a separate axes implementation.
   - First-phase native reduction uses a pyerfa/SOFA-backed `XysProvider` for IAU `X,Y,s`.
   - A future `GMATXysProvider` may replace that source if exact GMAT table parity is required.

3. **State transforms are based on rotation rates.**
   - `Axes.rotation_matrix(et)` returns `R` with `r_icrf = R @ r_axes`.
   - `Axes.rotation_and_rate(et)` returns `(R, Rdot)` with `v_icrf = R @ v_axes + Rdot @ r_axes`.
   - `Axes.state_transform_matrix(et)` is derived from `(R, Rdot)`.
   - `CoordinateSystem.transform_state()` uses `(R, Rdot)` before falling back to any angular-velocity compatibility path.

4. **Public coordinate epoch input remains ET seconds.**
   - Public axes and coordinate-system APIs accept SPICE ET seconds.
   - GMAT A1MJD support is low-level and test-oriented for parity checks only.

5. **Validation is split into two tiers.**
   - `ITRFSpiceAxes("ITRF93")` validates against `spiceypy.pxform/sxform` at `<1e-12` when high-precision kernels are available.
   - `GMATITRFAxes` validates against SPICE `ITRF93` initially at about `1e-7` as a sanity check, with parser/time/EOP/stage-level tests carrying the primary native-chain validation.

6. **Fixture and data policy is explicit.**
   - Commit trimmed text fixtures for J2000, the 2017 leap-second boundary, and the 2026-06-12 window.
   - Full GMAT data is opt-in through `GMAT_DATA_DIR`.
   - Optional high-precision Earth BPC checks skip clearly when unavailable.
   - Missing required committed fixtures or kernels raise explicit errors.

7. **Errors are explicit and no precision downgrades are automatic.**
   - Missing `ITRF93` kernels raise coordinate errors with actionable messages.
   - Missing GMAT native data raises data errors during construction or lookup.
   - EOP out-of-range raises by default.
   - GMAT-style clamping is available only through explicit compatibility options.

8. **System integration is a thin delegate.**
   - `System.coordinate_system` stores an optional coordinate system.
   - `System.transform()` delegates to `CoordinateSystem.transform_state()` and does not duplicate coordinate math.

## Consequences

### Added

- A stable contract for SPICE-backed defaults, GMAT-compatible native behavior, fixtures, and tolerances.
- A rotation-rate-based axes abstraction suitable for both SPICE `sxform` and GMAT native `Rdot`.

### Unchanged

- Public coordinate APIs continue to accept ET seconds.
- Approximate ITRF remains explicitly low fidelity.

### Future Work

- Implement a GMAT table-backed `XysProvider` if exact GMAT XYS interpolation parity becomes required.
- Tighten native-vs-SPICE tolerances only after the data source, interpolation, and model versions are proven equivalent.
