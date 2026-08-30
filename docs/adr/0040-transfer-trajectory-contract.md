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
default). (`gcrs_km` landed in the #584 amendment below — as the label of the
parallel inertial segment, not a new value of this field.)

### Consequences

- tod renders by label instead of per-tool hardcoding; the interim hardcoded
  fix for propagation stays until that response adopts the field.
- MCP consumers (LLM agents) get frame semantics machine-readably.
- low_thrust's inconsistency becomes machine-readable: a future unification
  changes the enum value, not the contract shape.

## Amendment (2026-08-30): maneuver events + catalog transfer record (#575, #574)

### Context

Two follow-ups to §4's deliberately-deferred items. (a) Consumers needed the
Δv events *structured* — tod rendered "2 maneuvers" as prose because the
response carried only a scalar total; the perilune time was buried in
backend-specific `details`. (b) Transfer products still bypassed the catalog
(ADR 0031 scoped auto-ingestion to orbit products), so designed transfers were
not queryable/lineage-tracked.

### Decisions

**Maneuver events (#575).** `TransferDesignResult` /
`TransferDesignResponse` gain `maneuver_events` — a list of
`{kind, t_sec, dv_km_s, note?}` with `t_sec` in seconds since TLI (the §1
time basis) and `kind` an open enum (`departure` / `perilune` / `arrival`).
Per backend:

- **HMN**: `departure` (t=0) + `arrival` (t=tof); arrival *is* perilune, no
  separate event.
- **LGA/WSB**: departure + perilune (t = refinement's `perilune_time_dim` ×
  that leg's characteristic time — the same quantity the trajectory join uses,
  so event time and geometry stay consistent by construction) + arrival.
  Perilune `dv_km_s=0.0`: the velocity kink there is refinement's implicit
  correction (§4), not part of the Δv accounting — recorded honestly as a
  zero-Δv waypoint rather than papered over.
- **low_thrust**: empty list. Its burn schedule lives on SPICE ET, a different
  time basis than TLI-seconds; converting without a shared epoch would
  fabricate precision. Revisit when low_thrust adopts the contract.

**Catalog transfer record (#574).** Converged `transfer_design` responses with
a trajectory auto-ingest (the ADR 0031 `_auto_catalog` seam, now type-aware):
a record with `source_tool="transfer_design"`, the six classification keys
`None`/`False` (no orbit family semantics — do not force transfer metadata
into orbit-shaped fields), transfer metadata in `scalars`
(`transfer_type`, `delta_v_km_s`, `tli_epoch` raw as given, `tof_sec`,
`state_frame`, `n_points`), the backend `details` block stored as-is under a
`details` meta key with the #575 `maneuver_events` appended (the shared
contract), arrays under a `transfer/` segment prefix
(`transfer/states`, `transfer/times`). `SCHEMA_VERSION` stays 1: record shape
is JSON-schema-compatible, only the derived SQLite index grows columns
(`transfer_type`, `delta_v`, `tli_epoch`) plus five `CatalogFilter` fields.
Diverged runs and low_thrust-style responses without a trajectory ingest
nothing. Legacy index tables missing the new columns are detected on open and
rebuilt from the record files (ADR 0031 decision 5: the index is derived).
`tli_epoch` range filtering applies only to numeric epochs; string epochs
index as NULL (exact value still readable from the record).

### Consequences

- tod renders maneuvers structurally (timeline markers at `t_sec`) instead of
  parsing prose; total Δv remains the scalar sum of nonzero events.
- `catalog_query(transfer_type="WSB", delta_v_min_km_s=…)` works alongside
  orbit filters; summaries expose `transfer_type`/`delta_v_km_s`/`tli_epoch`.
- Zero breaking change: new response fields default empty/None; old catalogs
  self-heal their index on first open after upgrade.

## Consequences

- MCP and sidecar consumers get the trajectory inline as JSON (transfer tools
  are not in the sidecar `_BINARY_TOOLS` frame map; 200-point arrays are small
  enough — the frame map stays demand-driven per ADR 0035).
- tod's GUI can draw the arc after a run by scaling positions by 1/DU_KM and
  feeding `trajectory_times` to the timeline.
- HMN visuals are textbook-style demos (arrival on the +x/Moon side); anyone
  needing ephemeris-grade transfers should use the LGA/WSB paths or wire
  `dynamics`.

## Amendment (2026-08-30): `gcrs_km` inertial segment (#584)

### Context

tod #428 (inertial view — ADR 0013 `view_frame` through the double-layer
canvas) needs the transfer arc's GCRS inertial geometry. The contract carried
only synodic-frame geometry (`synodic_barycentric_km`), so the inertial view
had nothing to draw, and every consumer would have had to ship its own
synodic→inertial transform.

### Decision

**Vocabulary.** The `state_frame` vocabulary gains `gcrs_km`: geocentric
origin, non-rotating axes (GCRS convention), physical km / km/s.

**Parallel segment, direct representation.** `TransferDesignResult` /
`TransferDesignResponse` gain `trajectory_gcrs_km` — an (n, 6) inertial
trajectory row-aligned with `trajectory` and **sharing `trajectory_times`**
(no second time array). The top-level `state_frame` keeps labeling the
primary synodic `trajectory`; the inertial segment's data frame is the
vocabulary value embedded in its own name (`gcrs_km`). The rejected
alternative — shipping synodic→GCRS transform parameters instead — saves
storage but scatters transform code to every consumer; the accepted cost is
double geometry in the response and record.

**Producers per transfer type.**

- **HMN**: the geocentric two-body arc as constructed (the ECI states before
  the §3 display rotation). Its construction axes are an idealized inertial
  convention with no ephemeris orientation semantics — same idealization as
  §3, named honestly as GCRS-convention km.
- **LGA/WSB**: the joined synodic arc rotated back to inertial — translate
  barycentric → geocentric (`+[μ·DU, 0, 0]`), rotate Rz(θ(t)) with
  θ(t) = ω·t (ω = synodic mean motion), and **add the ω×r transport term to
  velocities**. Unlike §3's display rotation (which deliberately omits the
  transport term), this is a real frame change: without it the states are not
  inertial and velocity-bearing consumers would be wrong.
- **WSB scale note**: the joined arc already mixes BCR4BP (departure) and
  CR3BP (arrival) characteristic quantities (§4's 1.3e-5 DU split); the
  conversion applies the canonical CR3BP scales uniformly rather than a
  two-system scheme — the residual is invisible next to the θ₀ idealization.
- **low_thrust / zero-result runs**: `trajectory_gcrs_km=None` (no synodic
  geometry to mirror; the force-model states stay out of the contract).

**Orientation convention (θ₀ = 0).** At TLI the synodic +x axis (Earth→Moon
line) coincides with the GCRS x axis. This is the only non-arbitrary choice:
LGA/WSB's synodic geometry has no ephemeris orientation semantics (the
constructed ECI departure state is *interpreted* as synodic at ingestion, not
transformed by epoch), so attaching a real Moon angle would fabricate
precision. Ephemeris-consistent orientation requires `spacetime_transform`
with real epochs — out of scope, as in §3. Time basis: TLI seconds, shared.

**Catalog.** `build_transfer_record` stores the inertial geometry as a
`transfer/states_gcrs_km` array segment (present only when the segment
exists); `transfer/times` is shared by both geometries. The segment key embeds
the vocabulary value — the key *is* the frame annotation, mirroring how
`scalars.state_frame` annotates `transfer/states`. `SCHEMA_VERSION` stays 1;
no new index columns or filters.

### Consequences

- tod's inertial view draws `trajectory_gcrs_km` directly — zero transform
  code in the consumer; the synodic view is untouched (zero breakage for
  existing consumers).
- MCP/sidecar: transfer tools remain outside the sidecar `_BINARY_TOOLS` frame
  map; the inertial array serializes inline JSON like `trajectory` (payload
  roughly doubles for converged impulse transfers — the accepted cost).
- Round-trip guarantee (test-guarded): applying the inverse rotation
  (translate −[μ·DU, 0, 0], Rz(−θ), subtract ω×r) to `trajectory_gcrs_km`
  recovers the synodic `trajectory` row-for-row.
