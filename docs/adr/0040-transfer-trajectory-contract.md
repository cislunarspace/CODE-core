# ADR 0040: transfer_design converged trajectory — unified synodic-frame contract

**Status**: Adopted (implemented)
**Date**: 2026-08-29
**Related Issue**: #568
**Related**: ADR 0015 (nominal-orbit coordinate contract), ADR 0031 (orbit
catalog — transfer products deliberately out of scope), transfer-orbit-design
ADR 0013 (four-slot visualization contract)

## Context

`transfer_design` converged for HMN/LGA/WSB yet returned `trajectory=None`
(only low_thrust filled it). Consumers (tod GUI canvas, MCP sidebar) got Δv
numbers but no geometry, so a designed transfer could not be drawn. Meanwhile
the internals already propagated the geometry and threw it away:

- HMN's trajectory-bearing path sits behind a `dynamics` parameter the Facade
  never passes; the geometric two-body path returned nothing.
- LGA/WSB refinement (`_refine_*_candidate`) runs `ThreeBodyLambert.solve()`,
  which re-propagates the full arrival leg (200 points, synodic rotating frame,
  physical km via `dimensionless_to_physical`), then keeps only the terminal
  velocity for the Δv update.

The `trajectory` field existed in the response model all along — the contract
was there, the producers were not wired.

## Decision

### 1. Unified trajectory contract

`TransferDesignResult.trajectory` / `TransferDesignResponse.trajectory` is
(n, 6) states in the **Earth-Moon synodic rotating frame, barycenter origin,
physical units km / km/s**; new sibling `trajectory_times` is (n,) seconds
since TLI (t=0 is the departure pulse), row-aligned. This matches the
`target_ephemeris` input contract (synodic physical, e2m2e#516) on the output
side, and is the frame tod's canvas natively renders (after ÷DU).

### 2. Producers per transfer type

- **HMN**: sample the two-body arc with `multi_impulse.propagate_two_body`
  (new public wrapper) from the post-TLI state — Lambert solution when
  `tof_range` is given, else the Hohmann tangential scaling of the parking
  velocity — then convert via `hohmann.eci_to_synodic_display`.
- **LGA**: departure leg (LEO→perilune) re-propagated in CR3BP from the
  candidate's departure state; arrival leg is the refinement's
  `ThreeBodyLambert` arc, now returned alongside the refined candidate.
  Refinement fallback (shooting failed or no improvement): the grid candidate
  is a single free-flight solution, re-propagated whole.
- **WSB**: same assembly, but the departure leg is re-propagated in BCR4BP
  with the candidate's `sun_phase0` (solar perturbation is the point of WSB);
  the arrival leg remains the CR3BP-refined arc.
- **low_thrust**: unchanged (its `trajectory` is the force-model state
  system — a known inconsistency, tracked separately).

Trajectory assembly failures (PropagationFailure) degrade to
`trajectory=None` with a warning; numeric results (Δv, details) still return.

### 3. HMN display convention (phase alignment)

`eci_to_synodic_display` rotates the ECI two-body scene by Rz(−θ) so the
arrival direction lands in the +x half-plane (z preserved), then translates
positions by −[μ·DU, 0, 0] (geocentric → barycentric). Velocities rotate but
get **no ω×r transport term**. This is a *display convention*, not an
ephemeris-consistent transformation: HMN's geometric path has no real epoch
semantics (epoch is record-keeping only). A truly consistent frame conversion
requires `spacetime_transform` with real epochs — out of scope by design.

### 4. Seam notes

- Join at perilune: positions are continuous (guarded by test, <10 km
  tolerance from integrator/interpolation error); the **velocity kink** is the
  refinement's implicit correction impulse, which the Δv accounting does not
  include — recorded, not asserted.
- WSB legs use their own systems' scales (BCR4BP DU=384405 vs CR3BP
  DU=384400, a pre-existing 1.3e-5 DU split); each converts with its own
  characteristic quantities, the canvas normalizes by 384400 — invisible, not
  fixed.
- No catalog ingestion: ADR 0031 scoped auto-ingestion to orbit products;
  transfer records (departure/arrival, Δv events, no period/family) need their
  own record type — deliberately deferred again, tracked in tod's discovery
  notes.
- `_refine_lga_candidate` / `_refine_wsb_candidate` now return
  `(candidate, arrival_arc | None)`; tests calling them unpack accordingly.

## Amendment (2026-08-30): `state_frame` data-frame label

### Context

Consumers drawing trajectories had to *guess* which frame the numbers live
in — the root cause of tod #421 (GCRS-km propagation output drawn as
dimensionless). No response self-describes its frame. tod's ADR 0013 already
uses "frame" for the *view* frame (the user's synodic/inertial display
choice), so the data-side label needs its own name.

### Decision

`TransferDesignResult` / `TransferDesignResponse` gains a top-level
`state_frame` field (string enum), emitted on every converged response:

- `synodic_barycentric_km` — the §1 contract (HMN/LGA/WSB trajectories).
- `force_model_state` — low_thrust trajectories (the §2 known inconsistency,
  named honestly rather than papered over).

The vocabulary is the canonical data-frame enum going forward; later batches
add `gcrs_km` and `synodic_barycentric_nd` when other responses (propagation,
design) adopt the field. `state_frame` is data-side and travels with the
numbers; tod's `view_frame` (ADR 0013) stays user-side — the two must not be
conflated. This branch is unreleased, so the field is mandatory (no `None`
default).

### Consequences

- tod renders by label instead of per-tool hardcoding; the interim hardcoded
  fix for propagation stays until that response adopts the field.
- MCP consumers (LLM agents) get frame semantics machine-readably.
- low_thrust's inconsistency becomes machine-readable: a future unification
  changes the enum value, not the contract shape.

## Consequences

- MCP and sidecar consumers get the trajectory inline as JSON (transfer tools
  are not in the sidecar `_BINARY_TOOLS` frame map; 200-point arrays are small
  enough — the frame map stays demand-driven per ADR 0035).
- tod's GUI can draw the arc after a run by scaling positions by 1/DU_KM and
  feeding `trajectory_times` to the timeline.
- HMN visuals are textbook-style demos (arrival on the +x/Moon side); anyone
  needing ephemeris-grade transfers should use the LGA/WSB paths or wire
  `dynamics`.
