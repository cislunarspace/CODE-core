# ADR 0034: Ephemeris force-model Hamiltonian scope — planar full-ephemeris, non-autonomous, harmonics/SRP deferred

**Status**: Adopted
**Date**: 2026-08-21
**Numbering note**: numbers 0032/0033 were taken by ADRs merged with #497/#499;
this entry's actual decision date is 2026-08-21, numbered in sequence.
**Related Issue**: #498 (ephemeris force-model Hamiltonian)
**Related**: ADR 0013 (definition-level verification), ADR 0016 (ephemeris
cache), ADR 0020 (failure policy), ADR 0027 (model ladder), ADR 0032 (HJB
dynamics crate & binding entry), ADR 0033 (value-function product contract &
query interface); subsystem architecture at
`docs/architecture/hjb-subsystem.md`, solve-chain dataflow at
`docs/architecture/hjb-hamiltonian-dataflow.md`

## Context

Issue #498's body demands plugging the ephemeris N-body full-fidelity force
model (Earth-Moon 10×10 harmonics, solar third body, variable-mass SRP) into
the levelset grid solver to solve HJB directly. `hjb-subsystem.md` §3, merged
the same day (effective with PR #500), reached the opposite conclusion:
high-order gravity and SRP are intrinsically three-dimensional forces; the grid
layer's (dimension cap 5, no z-axis) fidelity ceiling is forces planarly
expressible in the synodic frame, and 3D high fidelity belongs to tier-two
neural networks. Two documents hours apart with opposite conclusions — an
unconverged decision left by parallel sessions.

After triage verified the physics, the conflict's full picture:

- All three mission segments (GEO, Halo, NRHO) travel far off-plane in the
  Earth-Moon synodic frame: GEO sits in the equatorial plane inclined 18°–29°
  to lunar orbit plane, out-of-plane amplitudes above ten thousand km; Halo/NRHO
  out-of-plane motion is definitional. No near-planar slice approximates the
  whole mission.
- Section projection (taking in-plane components of 3D forces at z=0) builds
  tables fine, but product control laws' domain won't cover real trajectories:
  value tables have no z axis — off-plane states can't be queried; closed-loop
  replay (verification tier 4) then can't run at all, projected errors barely
  even measurable.
- The grid layer's job in two-tier architecture is baseline + training data. As
  a baseline, model fidelity contributes nothing (cross-checks test solvers not
  models); as training data, CR3BP lacks **time dependence** (stationary model),
  not spatial fidelity — real ephemeris dynamics is time-varying, and tier-two
  networks must learn to handle that.

The maintainer ruled accordingly: #498 takes the planar full-ephemeris route
(decision 1 below), neither implementing the issue body's full-fidelity section
projection nor closing.

## Decision

1. **#498 scope: planar full-ephemeris Hamiltonian.** Real lunar ephemeris
   (position & velocity via `EphemCache` tables) defines a time-varying pulsating
   synodic frame; force models are two-primary point masses + solar third body
   (BCR4BP convention, Sun's in-plane projection of ephemeris position);
   non-autonomous with time explicit in dynamics. State is 5-dim
   `(x, y, vx, vy, m)` nondimensional synodic coordinates (4-dim dropping mass =
   constant-thrust-acceleration variant, distinguished by parameters). Fully
   compatible with `hjb-subsystem.md` §3 and its family-table row "ephemeris
   force model (4 or 5 dims, non-autonomous)" — **no revision of master's
   architecture docs**.
2. **Harmonics & SRP demoted to deferred experiments, out of #498 acceptance.**
   Adding section-projected forces onto decision 1's skeleton is small work,
   reserved as follow-up experiments: decide retention using tier-3 verification
   data (value function magnitude/iso-surface structure vs CR3BP solutions).
   Experiments get their own issue — not elaborated here.
3. **Time-varying synodic frame construction**: the frame derives from lunar
   instantaneous ephemeris (rotation + pulsation); `ω(t)`, `ω̇(t)`, pulsation all
   derive from cached lunar position/velocity — no second ephemeris query path.
   Solver t ↔ SPICE et epoch mapping, cache coverage of the whole solve window
   (out-of-range hard failure per ADR 0020 semantics), backward-in-time reversal
   land per `hjb-hamiltonian-dataflow.md`'s time-semantics section; value-function
   products' `times` semantics, state order, nondimensionalization enter metadata
   per ADR 0033 decision 3.
4. **Force models & cache path**: the ephemeris Hamiltonian lives in
   `e2m2e-hjb-dynamics` (ADR 0032 decision 1), receiving a `CompiledForce` list +
   cache time range at construction — ADR 0033's construction-time injection
   realized thus: injection sits at the **configuration** level (force list, time
   range, epoch mapping as constructor params), while query paths inside
   `CompiledForce` still go through `EphemCache`'s process singleton (ADR 0016),
   constructors responsible for enabling cache coverage of the solve window
   beforehand. No injection-style refactor changes.
5. **Variable-mass SRP contract (`SRPVariableMass`/
   `acceleration_with_mass`) lands independently of #498.** It exists only in an
   uncommitted local workspace while geo-nrho's `lowthrust_rs` already calls it —
   a load-bearing dependency of existing solvers requiring prompt independent PR.
   Under decision 2 it stays off #498's critical path.
6. **#498 repo-internal acceptance in three tiers**:
   (a) Degradation cross-check: replacing ephemeris inputs with circularized
       stationary values must degrade dynamics term-by-term to `Cr3bpSynodic`
       (#497 implementation);
   (b) Force consistency: identical (t, state) — Hamiltonian-internal post-frame-
       transform forces match direct `compute_total_acceleration` calls pointwise;
   (c) Coarse-fine regression (verification tier 3): small grids solve both
       ephemeris & CR3BP value functions; magnitude & iso-surface structure agree;
       exceeding tolerance = implementation error.
   Closed-loop replay (tier 4) runs manually on geo-nrho side, never entering
   e2m2e pytest (acceptance independent of external repo code).
7. **Crate ownership, binding entry, product contract, time interpolation**:
   from ADR 0032 (decisions 1, 3) and ADR 0033 (decisions 2, 3, 4); no re-decision
   here.

## Rationale

1. **Ruling (c) over (a)**: section projection's control-law domain can't cover
   real trajectories (no z axis to query); verification tier 4 becomes
   unexecutable; products resist even error measurement; missions are off-plane
   throughout, so projection isn't locally damaging only. (a)'s output — exact
   solution of an ad-hoc projected model — sits awkwardly research-wise and would
   overturn that day's merged §3.
2. **Ruling (c) over (b)**: closing #498 freezes the grid layer on stationary
   models forever. Tier-two networks must handle time-varying dynamics; if tier
   one provides no non-autonomous baseline, time-varying signals get neither
   training nor spot-checks. Planar full-ephemeris precisely patches this gap
   within the dimension ceiling: **upgrading the grid layer along the temporal-
   fidelity axis rather than crashing into the spatial-fidelity wall**.
3. **Decision 3's pulsating frame**: distance pulsation makes primaries drift in
   fixed-scale frames, destabilizing grid axes' physical meaning; pulsating
   coordinates pin primaries at fixed nondimensional positions — standard for
   elliptical restricted three-body and full-ephemeris models, costing explicit
   time-varying terms ω̇/pulsation corrections which is exactly this issue's wanted
   non-autonomy, not extra burden.
4. **Decision 4's config injection**: `CompiledForce::acceleration` signatures carry
   no cache parameter; query-path injection means touching whole call chains +
   Python bindings for a benefit (multiple caches per process) nobody needs today;
   constructor params satisfy ADR 0033's pure-table-reads intent.
5. **Decision 6(a)'s degradation check** is the strongest assertion: the ephemeris
   version in circularized-stationary limit must reproduce term-by-term the already-
   cross-checked #497 CR3BP dynamics, anchoring new implementation correctness onto
   verified code — matching ADR 0013's verification-by-definition strategy.

## Consequences

- This entry precedes #498's specification; issue #498 carries triage record +
  agent brief, labeled `enhancement` + `ready-for-agent`.
- The family-table's ephemeris-force-model row lands concretely via this entry;
  the doc itself un-revised.
- Once the variable-mass SRP contract PR (decision 5) lands, geo-nrho's
  `lowthrust_rs` sheds its uncommitted-workspace dependency. (2026-08-23 addendum:
  PR landed; `SRPVariableMass` + `acceleration_with_mass` now in
  `crates/e2m2e-forces`.)
- Harmonics/SRP section-projection experiments (decision 2), if started, track via
  new issue; their conclusions may revise decision 2.
