# ADR 0019: Drag Rust port uses ITRF93 pxform frame rotation

**Status**: Adopted
**Date**: 2026-08-07
**Related**: ADR 0003 (ITRF93 defaults & GMAT compatibility), ADR 0018
(Jacobian ∂a/∂v), issue #315, issue #317
**Related code**: `crates/e2m2e-forces/src/forces/drag.rs`,
`e2m2e/algorithm/forces/drag.py`

## Context

Drag physics is computed in the **Earth-fixed frame** (ITRF): the atmosphere
co-rotates with Earth, so relative velocity = spacecraft ITRF velocity. The
drag pipeline is thus a round trip of frame rotations: J2000 propagation frame
→ ITRF → drag formula → J2000.

Python's ``DragModel.compute_acceleration`` (`drag.py`) uses
``ITRFApproxAxes`` for this rotation. ADR 0003 decision 1 already labels
``ITRFApproxAxes`` as low-precision/educational with SPICE-backed `ITRF93` as
the high-precision default. Python drag keeping approximate axes is legacy:
drag was long a teaching/LEO-scale example where ``ITRFApproxAxes`` sufficed.

The drag Rust port (issue #315 series) brings `DragModel` into compiled force
models sharing the same SPICE FFI as `GravityField` (``pxform`` / kernel pool).
Drag's frame rotation then had two options:

- **A. Copy Python**: reimplement the ``ITRFApproxAxes`` approximate reduction
  in Rust (simplified IAU 2006 + linear LOD etc.), solely for bit-level parity
  with Python.
- **B. Align with ADR 0003 defaults**: use `ITRF93`'s ``pxform`` directly
  (same rotation path as `GravityField`).

`drag.rs`'s header comment calls this "decision 1b", but it had no documented
home (issue #317 item 1.2 noted this). This ADR provides one.

## Decision

**Option B**: the drag Rust path uses `ITRF93` ``pxform``; no reimplementation
of ``ITRFApproxAxes``.

1. **`drag_accel` rotation via ITRF93.**
   - `lookup_frame_matrix("ITRF93", propagation_frame, et)` prefers the
     ephemeris pre-sampling cache (ADR 0016), falling back to
     ``pxform("ITRF93", propagation_frame, et)``.
   - Same pattern as `GravityField`'s `pxform` usage
     (``r_itrf = Rᵀ·r_j2000``, ``a_j2000 = R·a_itrf``).

2. **Python path unchanged.**
   - `DragModel.compute_acceleration` keeps `ITRFApproxAxes`. Under current
     reality — drag needing SPICE always goes Rust — the two paths don't
     coexist: `DragModel.to_rust_spec` returns None when `system.spice` is
     missing, `ForceModel` falls back to the Python path, and without SPICE
     there is no `ITRF93`; `ITRFApproxAxes` is a reasonable degradation.

3. **`∂a/∂v` is orthogonal to frames.**
   - Drag's velocity dependence (``a ∝ |v|·v``) is physics, independent of the
     rotation choice. The triple-extension interface lives in ADR 0018, out of
     scope here.

## Rationale

1. **Avoid re-implementing an explicitly low-precision frame inside Rust.**
   ADR 0003 confines `ITRFApproxAxes` to low-precision/education. The Rust
   compiled path is the high-precision production path; porting a reduction our
   own ADR classifies as low precision is self-contradictory, and maintaining
   two Earth orientations (ITRF93 + approximation) doubles the burden.

2. **Shares the rotation path with `GravityField`.** Both are body-fixed
   forces (density vs spherical harmonics), both need J2000↔ITRF rotations.
   Using the same `pxform` path shares cache hits (ADR 0016) and reduces SPICE
   FFI calls. Forking would waste landed caching infrastructure.

3. **Bit-level parity with Python was never the goal.** The Rust compiled path
   aims to be faster and no worse than Python (ADR 0002 revision 2), not a bit-
   for-bit clone. `ITRF93` beats `ITRFApproxAxes` in accuracy; the divergence
   direction is Rust closer to truth — consistent with ADR 0003's accuracy
   ladder.

4. **Degradation semantics self-consistent.** The Python path only engages
   without SPICE; without SPICE, `ITRF93` is unavailable anyway (ADR 0003 item
   7 raises on missing kernels, never silently degrading precision).
   `ITRFApproxAxes` is then the only available option — not violating ADR 0003;
   it exists precisely as the sans-SPICE low-precision fallback.

## Boundaries

- **Drag only.** Other body-fixed forces (`GravityField`, `EarthTide`) follow
  ADR 0003 on their own; unaffected here.
- **No change to Python `DragModel`.** Its `ITRFApproxAxes` stays as sans-SPICE
  degradation. If Python drag later switches to `ITRF93`, evaluate separately
  (at that point both paths match in accuracy; this ADR's dual-path notes could
  be removed).
- **No GMAT-native ITRF.** ADR 0003 item 2's `GMATITRFAxes` is an independent
  GMAT-compatible frame; drag doesn't use it.

## Consequences

### Added

- Drag Rust path uses `ITRF93` `pxform`, aligned with ADR 0003 defaults.
- Decision 1b has a documented home (this ADR); `drag.rs` comments no longer
  dangle.

### Unchanged

- Python `DragModel.compute_acceleration`'s `ITRFApproxAxes` path (sans-SPICE
  degradation).
- Drag physics formulas, density interface, `∂a/∂v` Jacobian (ADR 0018).
- All ADR 0003 decisions (this ADR specializes them for drag without revising
  the parent).

### Trade-offs

- **Dual-path accuracy divergence.** With SPICE present, Rust (`ITRF93`) vs
  Python (`ITRFApproxAxes`) drag accelerations differ slightly in theory.
  Negligible at LEO scale (the approximate axes' rotation error is
  higher-order small for drag magnitudes); moreover the paths never coexist
  now (SPICE ⇒ Rust). If Python-path ITRF93 accuracy is ever needed, a separate
  change follows.

## Revision (2026-08-12, ADR 0020 decision 4)

**Missing SPICE no longer degrades to ITRFApproxAxes**: when `system.spice`
is absent, `DragModel.to_rust_spec` returns None and `ForceModel` explicitly
raises a capability error (issue #378) instead of silently falling back to the
Python slow path (decision 2's claim that ITRFApproxAxes was a reasonable
sans-SPICE degradation is void). The Python reference implementation was
removed wholesale by issue #378 — `DragModel` no longer carries
`compute_acceleration` (decision 2 and the keep-Python-path clauses lapse);
`ITRFApproxAxes` remains only at the coordinate layer
(`e2m2e/algorithm/coordinate/standard_axes.py`) for low-precision/teaching per
ADR 0003, absent from every drag propagation path.
