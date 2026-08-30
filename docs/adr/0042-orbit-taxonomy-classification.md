# ADR 0042: orbit taxonomy — 42-label classification of CR3BP periodic orbits

**Status**: Adopted (implemented)
**Date**: 2026-08-30
**Related**: ADR 0012 (layering), ADR 0014 (facade tiers), ADR 0031 (catalog
records), ADR 0036 (baseline dataset), ADR 0040 (state_frame), ADR 0041
(spatiography — the sibling single-state partition axis); issue #581.

## Context

e2m2e labels an orbit's family only at design time: the differential corrector
echoes its correction setup, and catalog ingest copies the design-side family
name into `classification.orbit_family`. Nothing can answer the reverse
question — "given a trajectory, which family is it?". The spatiography
classifier (ADR 0041) answers a different question (single state → spatial
province).

Issue #581 adopts a three-tier orbit taxonomy of 42 labels. The vocabulary
comes verbatim from the STK **CODE** (Cislunar Orbit Designer) component,
which is unreleased: there is no citable source for the labels' *semantics*
and no published classification algorithm. As with ADR 0041's MEGNO
thresholds (Primer §7 left them unspecified), the criteria below are
e2m2e's own analytic definitions, calibrated against the packaged baseline
dataset (ADR 0036) and recorded here in full.

## Decision

### 1. Taxonomy and label vocabulary

42 labels = 27 libration-point + 4 moon-centered + 11 resonant (the issue
list, verbatim). Each label carries a structured form — category
(`libration_point` / `moon_centered` / `resonant`), family, libration point
(1–5 or null), hemisphere (`northern` / `southern` / `eastern` / `western`
or null), resonance (p:q or null) — plus a snake_case canonical string
(`halo_l2_northern`, `low_prograde_eastern`, `resonant_2_1`). The canonical
string is the serialization key everywhere (MCP responses, catalog records);
the structured form is restored by parsing (labels legend).

### 2. Classifier contract

`classify_orbit(states, times=None, *, period=None, mu=None,
periodicity="periodic") -> TaxonomyResult` in
`e2m2e/algorithm/orbit_taxonomy/`. The result carries the status triplet,
ordered labels (primary first), an `unclassified_reason`, and per-criterion
diagnostics. Two input forms share one criteria path:

- a full one-period closed trajectory (consumed as given), or
- the minimal form — a single crossing state + period (the repository's
  family-member storage form) — propagated one period internally.

`unclassified` is a legal converged outcome (empty labels + reason:
`non_periodic`, `quasi_periodic`, `no_matching_label`), not a failure;
invalid input and minimal-form propagation failure are FAILED states.

### 3. Criteria cascade (order is precedence; all quantities nondimensional
synodic)

1. `periodicity` gate: quasi-periodic → unclassified.
2. **Moon-centered**: planar (max |z| ≤ 1e-7), net winding about the Moon
   ≥ 1.5π, and ρ_max ≤ μ^(2/5) (Chebotarev SOI, Primer §5.4.2 — the extent
   gate that keeps deep-perilune NRHO members in the libration branch).
   Retrograde (moon-centered h_z < 0) → `distant_retrograde`; prograde
   splits at ρ_max = 0.031 (≈12 000 km, the repository's low-lunar-orbit
   parameter ceiling) into `distant_prograde` and
   `low_prograde_{eastern,western}` (east = perilune direction in the +y
   half-plane, the direction of lunar orbital motion).
3. **L4/L5**: planar, winding about L4 or L5, localization ≤ 0.15;
   T/T☾ > 2 → `longperiod_l{4,5}`, else `shortperiod_l{4,5}`.
4. **Collinear points**: winding about an L, or (the no-winding fallback for
   deep-perilune members) 3D with perpendicular x-z-plane crossings and
   orbit-to-point distance ≤ 0.25. Side: time-mean x relative to the Moon —
   < −0.5 → L3, < 0 → L1, > 0 → L2 (the four baseline halo/NRHO families
   separate with zero overlap). Family by crossing geometry: 2 / 4 / 6
   perpendicular y = 0 crossings per period → halo / butterfly / dragonfly
   (hemisphere = sign of z at the vy < 0 crossing); 3D with non-perpendicular
   y = 0 crossings → axial (the axial seed carries vz ≠ 0); 3D with
   perpendicular z = 0 crossings only → vertical; planar with perpendicular
   crossings → lyapunov.
5. **Resonant**: net winding about the barycenter ≥ 1.5π and
   |T/T☾ − q/p| < 0.01 → `resonant_p_q` (p:q = satellite:moon,
   T/T☾ = q/p — same orientation as the ADR 0041 resonance ladders).
6. Otherwise `no_matching_label`.

Multi-label: a moon-centered or L4/L5 hit whose period is also commensurate
appends the resonant label (primary stays the moon/libration family).

### 4. Verification status per label

Baseline-verified with real trajectories (all members): `halo_l1/l2`
northern+southern, `axial_l1/l2`, `distant_retrograde`, `distant_prograde`,
`low_prograde_*`, `longperiod_l4`, `shortperiod_l4`. Criteria-level synthetic
coverage only (no design capability yet): `lyapunov_l1–l3`, `axial_l3–l5`,
`vertical_l1–l5`, `butterfly_*`, `dragonfly_*`, all `resonant_*`,
`longperiod_l5` / `shortperiod_l5`, `halo_l3_*`. The synthetic curves verify
the crossing/winding geometry criteria, not dynamical solutions.

### 5. Wiring (no new MCP tool)

The maintainer's principle: expose only the highest-level Facade; the tool
count stays 22.

- **Ingest stamps measured labels**: `classification.taxonomy_labels`
  (record-level deduplicated set) and `members[].taxonomy_label` (member
  primary); the design-side family label remains as provenance. Conflict
  against the design-side expectation map logs a warning and never fails.
  Families outside the taxonomy (quasi-periodic lissajous, horseshoe, elfo)
  are stamped empty without running the classifier.
- **Index**: a `taxonomy_labels` column (schema-reset rebuild per ADR 0031
  decision 5); `promote_member` inherits the member's measured label — the
  data layer never imports the classifier.
- **Response enrichment**: `design_orbit`, `orbit_family_generation`,
  `catalog_query` / `catalog_get` responses carry `taxonomy_labels`.
  `orbit_propagation` is deliberately NOT enriched: it is the force-model
  ephemeris tool and would carry a permanent unclassified field — the same
  noise argument that excluded the transfer tools (a deviation from the
  issue brief, recorded here).
- **Backfill**: `scripts/backfill_baseline_taxonomy.py` stamps the packaged
  baseline through the same ingest path.

### 6. Conventions

p:q counts satellite revolutions per lunar revolution (T/T☾ = q/p; 2:1 is
interior). Northern/southern = sign of z at the vy < 0 crossing (the same
geometry the design side encodes as `halo_class`). Eastern/western =
perilune half-plane in the moon-centered synodic frame. NRHO folds into
halo (same family, high-amplitude near-rectilinear arc).

## Reproduction notes

- **dpo baseline members 0–3 are retrograde.** The design-side family walk
  started on the retrograde branch; geometrically they are small retrograde
  lunar orbits → `distant_retrograde`, and the backfill logs four conflicts
  against the prograde expectation. Measured labels win; design labels stay
  as provenance.
- **nrho-l2's record class used to disagree with its geometry (fixed, #586).**
  The packaged baseline recorded `halo_class=1` (south) while every seed had
  z0 > 0 at the vy < 0 crossing → the taxonomy said `halo_l2_northern`. The
  cause was not a mirror-phase storage issue: the Rust L2 NRHO kernel's
  hardcoded seed was itself a southern orbit, so its north/south mirroring
  inverted both requested families. The seed was re-based to the northern
  fold member and the baseline regenerated; geometry, `halo_class`, and
  measured labels now agree (anchored by the baseline hemisphere test).
- **The horseshoe family overlaps lpo.** Its endpoint members are identical
  to lpo-l4's largest (large-amplitude tadpoles, libration ≲ 50°). Per the
  mapping they ingest as empty labels, but a bare classifier call on such a
  trajectory returns `longperiod_l4` — the taxonomy simply has no horseshoe
  label.
- **Deep-perilune NRHO members wind the Moon** (ρ_min down to 2 700 km).
  Winding alone would misroute them to moon-centered; the ρ_SOI extent gate
  is what keeps them in the halo branch.

## Consequences

Classification is deterministic, explainable (full criteria diagnostics in
every result), and anchored to real trajectories for every family the
repository can generate. Labels whose families lack design capability rest
on self-defined geometry; when those pipelines land, the baseline-anchored
tests should be extended to replace the synthetic ones. The expectation map
in `catalog_ingest` is the single place design-side vocabulary and taxonomy
meet; a future vocabulary change (e.g. STK CODE publishing its semantics)
is a one-map edit plus an ADR amendment.
