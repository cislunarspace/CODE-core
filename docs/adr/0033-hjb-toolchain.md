# ADR 0033: HJB low-thrust toolchain — value-function product contract and online query interface

**Status**: Adopted
**Date**: 2026-08-21
**Related Issues**: #497, #498, #499, #501
**Related**: ADR 0011 (five-layer architecture), ADR 0012 (dependency
direction), ADR 0013 (verification by definition), ADR 0031 (catalog record
format), ADR 0032 (HJB dynamics crate & generic binding entry); geo-nrho
`docs/hjb-dependency-architecture.md` (downstream dependency architecture)

## Context

geo-nrho follows Bellman optimality via a two-level HJB route: offline,
e2m2e-levelset solves the HJ equation on structured grids for a value function;
online, closed-loop control derives from its gradient. The downstream dependency
architecture doc (geo-nrho `docs/hjb-dependency-architecture.md` §2.3) listed four
capabilities proposed for upstreaming: #497 (CR3BP synodic Hamiltonian), #498
(ephemeris force-model Hamiltonian), #499 (gradient interface + discrete operating-
point mapping).

Three seams needed fixing before work or later deepening reworks:

1. **Crate ownership** (fixed meanwhile by ADR 0032 during this entry's drafting):
   original HJB dynamics live in new crate e2m2e-hjb-dynamics — levelset wholesale
   inherits ToolboxLS's ACM non-commercial license; merging originals would sweep
   them under it. Decision 1 aligns without re-deciding.
2. **Value-function product contract.** Putting gradient queries at Python means
   interface↔solver decoupling — contracts shift onto data formats: double-
   integrator products use nondimensional time, synodic coordinates; ephemeris
   products will be ET seconds, possibly different frames, possibly mass-carrying.
   Without fixed format semantics, every Hamiltonian sprouts its own reader. geo-nrho
   has a `ProductMeta` prototype (`produced_by`/`frame`/`units`/`maturity`) but its key
   naming diverges from ADR 0031.
3. **Status of the time dimension.** The double integrator is autonomous — geo-nrho's
   nearest-time-snapshot sufficed by luck; ephemeris models are non-stationary, where
   no time interpolation is simply wrong. Interface design must pin this early.

## Decision

### 1. levelset stays a pure-math leaf; dynamics ownership defers to ADR 0032

e2m2e-levelset keeps only ToolboxLS counterparts and math examples (`Advection`
etc.); dynamics inject via the `Hamiltonian` trait. Original dynamics (#497's CR3BP
synodic Hamiltonian, #498's ephemeris force-model Hamiltonian) live in
**e2m2e-hjb-dynamics** (Apache-2.0); rationale in ADR 0032 (license boundary).
#498's ephemeris adapter depends inside that crate on e2m2e-forces' `CompiledForce`
and e2m2e-spice's ephemeris cache, injected at construction — pure table reads
during solving, zero CSPICE contact, never into Python callbacks. No new
`e2m2e-forces → e2m2e-levelset` dependency edge.

### 2. Python exposure solely through e2m2e-integrators, via ADR 0032's generic entry

Levelset solving capability exposes uniformly through e2m2e-integrators under the
ABI-stamp process: one generic entry (dynamics identifier + parameter table + grid
definition + terminal conditions — ADR 0032 decision 3's `solve_hjb_py`), no
per-dynamics dedicated signatures. Bindings do array shuffling, parameter
validation, error translation only; Hamiltonian assembly and control minimization
live entirely inside Rust crates, testable Rust-side.

### 3. Value-function product contract aligned to ADR 0031

Value-function products = JSON metadata + NPZ arrays, reusing catalog record
machinery rather than inventing formats:

- `schema_version` starts at 1 with no cross-version compatible reads; required keys
  follow ADR 0031's `_META_REQUIRED_KEYS` system (`source_tool`, status triple,
  `request` snapshot, `source_record_id`, etc.).
- Numeric conventions go explicitly into metadata: state-dim order, per-dim physical
  meaning, nondimensionalization (characteristic length/time/mass or none),
  `times` semantics (ET seconds vs synodic nondimensional). Products of differing
  conventions are distinguished by metadata fields, not reader guesswork — same
  spirit as ADR 0031 distinguishing dynamical models by segment presence.
- Value-function products enter catalog as a new record type, `source_record_id`
  pointing at the target-orbit record serving as terminal constraint. Ingestion is
  solver-side (#497/#498); consumers (#499's gradient interface) need only npz
  reading — no catalog dependency.
- geo-nrho's `ProductMeta` is the downstream prototype whose `frame`/`units`/
  `force_model` semantics this contract absorbs; geo-nrho migrates keys toward ADR
  0031 naming (`produced_by` → `source_tool` etc.) on its side.

### 4. Gradient query interface at the Python algorithm layer; time interpolation mandatory

Value functions are numpy arrays in consumers' hands already; online query rates
are low (control-period scale currently) — no Rust justification:

- Location: Python algorithm layer (`e2m2e/algorithm/`), pure numpy/SciPy per
  ADR 0012 dependency direction.
- Dimension-agnostic: interface takes only `axes`/`values`/`times` + query points;
  assumes neither state dimensionality nor physical meaning.
- Spatial interpolation uses tensor-product splines (e.g., cubic
  `RegularGridInterpolator`); gradients are **analytic derivatives of the
  interpolant**; center-difference-then-interpolate on grids (geo-nrho's current
  `_grid_gradient`) is forbidden — that is precisely this issue's targeted error
  source.
- Time interpolation mandatory, at least linear; autonomous systems are merely the
  degenerate-but-still-correct special case.
- Performance isn't this interface's goal. If future closed-loop sims make queries a
  bottleneck, moving to Rust is an independent decision not touching this contract.

### 5. Discrete operating-point mapping: data models migrate, constants parameterized, algorithm carries minimum-arc constraint

- `ThrustLevel` (0/60/100%), `ThrustArc`, `ThrustArcSequence` migrate verbatim from
  geo-nrho's `thrust_arcs.py` into e2m2e's Python low-thrust layer, sharing
  operating-point definitions with `LowThrustCollocation`; geo-nrho deletes its local
  copy and imports instead.
- Mission constants become constructor parameters: `MIN_ARC_DURATION_S` (geo-nrho
  currently 3600 s), `MAX_THRUST_N`, `ISP_S` — no module-level constants remain.
- Mapping must handle minimum-arc constraints (merging/splitting), not geo-nrho's
  current per-segment nearest-level approach which hard-errors when collocation
  segments run denser than minimum arcs — unusable.
- Acceptance self-contained within e2m2e: CR3BP continuous-throttle solution
  (generated by `LowThrustShooting`/`LowThrustCollocation`) → mapping → re-propagation
  → terminal residuals under L1 thresholds (384 km / 1 m/s order). No geo-nrho case
  dependencies, honoring ADR 0013 and `.out-of-scope/`'s verification-independence
  principle.

## Rationale

1. **Crate ownership defers to ADR 0032**: levelset's ACM license would cover
   merged originals — harder constraint than saving one scaffold. Centralizing
   dynamics in e2m2e-hjb-dynamics also preserves levelset's faithful-porting
   stance. Mixing domain logic into bindings (integrators) breaks their thin-FFI
   positioning — excluded.
2. **Contract aligned to 0031 rather than a new format**: catalog already solved the
   identical problem (multi-model artifacts, convention ambiguity, lineage) and gifts
   lineage mechanics: `source_record_id` pointing at terminal-constraint orbits
   directly mitigates immutable-boundary-condition pain — changing targets equals
   changing lineage pointers + re-solving; product relationships stay traceable.
3. **Gradient interface at Python**: consumers, data shape, query frequency all live
   Python-side; Rustification's benefit fails. Dimension-agnostic + mandatory-time
   design avoids rework upon ephemerization.
4. **Data models migrate verbatim**: geo-nrho already validated the contract;
   redesigning manufactures two equivalent but different concepts.

## Consequences

**Added**: catalog value-function record type (#498); Python gradient query
interface (#499) + discrete operating-point mapping module (#501).

**Changed**: none among prior decisions. #497 landed per ADR 0032
(e2m2e-hjb-dynamics + generic binding entry); decisions 1–2 align. geo-nrho
deletes `thrust_arcs.py` after #501 lands; `ProductMeta` keys align to ADR 0031
later (both geo-nrho-side actions).

**Unchanged**: e2m2e-levelset kernel & pure-math position; e2m2e-integrators thin
binding position; existing catalog record format (value-function records add a type,
not modify existing schema).

**Costs**: catalog needs segment conventions + validation for value-function
records.

**Out of scope**: three research-grade difficulties (adaptive grids, quantitative
error evolution, re-solve strategies under mutable boundary conditions) aren't solved
here; this entry only guarantees engineering architecture doesn't block them.

## Revision (2026-08-21, implementation feedback)

1. **Renumbered to 0033; decisions 1–2 aligned to landed ADR 0032**: drafted as
   0032, colliding with the ADR 0032 merged via #497 (HJB dynamics crate + binding
   generic entry) and disagreeing on decision 1; renumbered at merge, decisions
   1–2 realigned (dynamics in e2m2e-hjb-dynamics; bindings via generic
   `solve_hjb_py`).
2. **#501 added to related issues**: triage split #499 in two — #499 keeps only the
   gradient interface (decision 4); discrete mapping became #501 (decision 5).
   Decision 3's ingestion ownership unchanged.
3. **Decision 4 implementation form**: tensor-product splines assembled via
   per-axis not-a-knot cubic solves into `NdBSpline`; gradients via analytic `nu`
   derivatives, C² continuous (excluding local sliding stencils: their gradients jump
   at cell boundaries while closed-loop control consumes gradients directly for
   direction & switching functions). Cost: stateless function contract rebuilds
   splines per touched time snapshot per call (~0.5 s/snapshot at 41⁴-scale grids);
   acceptable at control-period query rates. If dense closed-loop sims bottleneck,
   introducing a coefficient-cached interpolator object is independent — contract
   unchanged.
4. **Decision 5's end-to-end case fixed as Earth-ephemeris two-body LEO two revs**
   (`GravityField` degree-0, matching existing low-thrust test fixtures), not CR3BP:
   self-contained, fast, geo-nrho-independent intent unchanged. 384 km / 1 m/s was an
   L1 mission-scale gauge — too loose for LEO; tests tightened ~10× against measured
   residuals (~0.35 km / 0.0004 m/s) as regression assertions. CR3BP end-to-end
   strengthened after #497.
5. **Decision 5 gains three interface refinements**: `validate` adds optional
   `levels` checking level legality (brief acceptance); `sequence_from_controls`
   input is segment boundary times `(N+1,)`, uniform and non-uniform both accepted;
   `ThrustLevel` enum didn't migrate — level sets parameterize as a `levels` tuple
   per issue text or user-defined conventions (default 0/60/100%).
