# ADR 0027: System/Dynamics separation retained — dynamics directory unsplit, two classes unmerged

**Status**: Adopted
**Date**: 2026-08-16
**Related Issues**: #430 (dynamics split evaluation), #438 (LibrationPoint
layering)
**Related**: ADR 0011 (five-layer architecture), ADR 0012 (dependency
direction), ADR 0026 (decision 1 & its follow-ups — this entry grew from that
audit)

## Context

While auditing the test suite, ADR 0026 raised a question: two kinds of things
mix inside `e2m2e/algorithm/dynamics/` — the System side
(`CR3BP_System`/`EphemerisSystem`, physical system definitions: μ, bodies,
characteristic scales) looks close to the data layer, while the Dynamics side
(`CR3BP_Dynamics`/`EphemerisDynamics`, constructing and integrating equations
of motion) is quintessential algorithm layer. #430 filed it: should System move
to data?

During triage, maintainers further questioned separation itself: System and
Dynamics were manually separated; if they're always constructed together, is
separation superfluous? Four candidate paths emerged: System to data, merge the
two, move only the `LibrationPoint` enum, or status quo.

Before adjudicating, a per-module verified data-flow investigation produced
`docs/architecture/system-dynamics-dataflow.md` (the "dataflow doc"),
verifying both class hierarchies' construction, held state, consumers, and one
propagation's data path module by module. This entry records the ruling;
structural-fact details defer to the dataflow doc.

## Decision

1. **System and Dynamics stay separated, both under
   `e2m2e/algorithm/dynamics/`.** System doesn't move to data; the classes
   don't merge.
2. **The dataflow doc enters the repo as this decision's structural
   explanation** (`docs/architecture/system-dynamics-dataflow.md`; registered
   in Sphinx toctree).
3. `LibrationPoint` enum's layer ownership is an independent pure-data-symbol
   question, deferred to #438 — not adjudicated here.

## Rationale

### Domain layer: the model ladder

Earth-Moon orbit design's basic working path is a dynamics-model ladder:
design periodic orbits in CR3BP (idealized) for initial guesses, then BCR4BP
(solar perturbation added), finally ephemeris N-body (real force environment)
where low-accuracy results seed multiple-shooting corrections yielding
high-accuracy quasi-periodic orbits. The `design_orbit` docstring's three-stage
main chain (CR3BP design → ephemeris multiple shooting → nominal ephemeris) is
this path in code form.

Each ladder rung is a System+Dynamics pair: System is the model's context
(CR3BP's μ/characteristic scales/libration points, BCR4BP's solar parameters,
ephemeris's body list/SPICE/frames); Dynamics is equations and integration
under that model. The System/Dynamics split isn't directory tidiness — it
mirrors domain structure in code.

The ladder is unfinished: Hill three-body, elliptical restricted three-body,
quasi-bicircular (QBCP) await implementation. The seam formed by the two class
hierarchies plus the `System` base is their future extension slot: each new rung
arrives as a new pair, while frame conversion, shooting, family generation and
other consumers keep their connections unchanged.

### Structural layer: three empirical facts

Verification process and sources for all below are in the dataflow doc.

1. **System has a dozen-plus independent consumers constructing no Dynamics**:
   force models (`ForceModel` holds system and mandates `coordinate_system`),
   the low-thrust trio, coordinate conversion (`SynodicJ2000System`,
   `rho_bridge`), station keeping & prediction, normal_form, and even the data
   layer's `Orbit` (duck-typed system reference for unit conversion).
   Separation lets context-only code avoid depending on the propagation
   machinery.
2. **One System instance serves multiple Dynamics and consumer paths.** In
   `design_orbit`, a single CR3BP instance simultaneously feeds dynamics
   construction, Jacobi computation, time conversion, and frame conversion —
   four paths; stability analysis and invariant manifolds rebuild their own
   Dynamics from `orbit.system` on demand.
3. **Lifecycles differ by an order of magnitude.** Systems live with the data
   (`Orbit` holds references surviving serialization round-trips); Dynamics
   live with the task: constructed, integrator-config overridden per mission,
   propagated, cache read, discarded. Merging would let consumers sharing a
   system stomp each other's integrator configs.

### Clarification: nominal polymorphism is currently thin

The `System` base promises three members (`frame`/`unit_system`/
`gravitational_parameter`) but only `gravitational_parameter` is genuinely
polymorphic across both CR3BP and ephemeris implementations; other accesses
are structural duck-typing onto implementation members
(`origin`/`coordinate_system`/`spice`/`mu` via `getattr`/`hasattr`; low-thrust
tests pass plain `SimpleNamespace`). This entry keeps that seam as-is without
exaggeration: separation's load-bearing reasons are the three empirical facts,
not the nominal abstraction. If duck-typing proves insufficient when new models
arrive, a new ADR widens the contract then.

### Why alternatives were rejected

**System to data layer**: System is a computational object, not data: libration
points solved via `fsolve` on nonlinear equations; stability analysis does
eigendecomposition; plus unit conversions and info printing. No data-layer
subdirectory (constants/frames/kernels/templates/types/catalog…) has precedent
for hosting computational objects. Also `compute_stability_index` shares
`pseudo_potential_hessian` with Dynamics' `compute_jacobian_A` — moving System
drags potential along, creating new ownership questions. Same root as ADR 0026
decision 1's coordinate ruling: "system definitions feel like data" is
functional-class intuition, and functional class ≠ code layer.

**Merging the two**: each structural fact inverts into cost: context-only
consumers forced onto propagation machinery; multi-consumer shared objects
stomping configs; data-living objects saddled with task-living caches.
Inheritance also degrades: `BCR4BPSystem` extends `CR3BP_System`, while
`BCR4BP_Dynamics` extends `Dynamics` directly (time-dependent Jacobian, no
Jacobi integral, four extra sun parameters at the Rust entry). Two non-mirrored
inheritance trees welded into one: BCR4BP would inherit CR3BP's system data
while swapping out nearly all dynamical behavior.

## Consequences

### Added

- This ADR.
- `docs/architecture/system-dynamics-dataflow.md`: structural explanation of
  System/Dynamics flows; source material for this decision's structural
  rationale.

### Unchanged

- `e2m2e/algorithm/dynamics/` layout and every line of code; System/Dynamics
  interfaces, implementations, tests stay as-is.

### Costs

- The thin nominal contract of the `System` base gets documented-and-retained:
  functions taking abstract type signatures still probe via
  `getattr`/`hasattr`. Deliberate trade-off: widening's benefit doesn't repay
  present risk; evaluated together when new models arrive.
