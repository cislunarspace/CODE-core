# ADR 0029: Orbit family generation via a unified Rust deep module

**Status**: Adopted (implemented)
**Date**: 2026-08-16
**Related Issue**: #428
**Related**: ADR 0011 (numerics/orchestration division), ADR 0014 (Facade
responses), ADR 0024 (status triple), ADR 0028 (planar full-period PAL —
partially revised by this entry's seam)

## Context

`FamilyGenerationRequest` already registers Halo, NRHO, Axial, Lissajous, SPO,
LPO, and Horseshoe, but the Facade offers only Halo family generation. Existing
numerical capabilities scatter across Python family walking, Rust differential
correction, Halo PAL, and planar full-period PAL; if the Facade loops single-
orbit entries item by item, step sizes, filtering, and failure semantics leak
into the interface layer.

Lissajous poses a second problem: it is a two-frequency quasi-periodic bounded
trajectory failing periodic closure. The single-orbit `design_lissajous` uses
high-order normal-form center manifolds; family generation also requires the
numeric core fully in Rust — it cannot call that entry from Python loops.

## Decision

### 1. Seven families share one Rust generation interface

`e2m2e-integrators::family_generation` provides a pure-Rust module of labeled
specs exposed via `generate_cr3bp_family_py`. One call performs seed
construction, propagation, STM, differential correction, PAL, step control,
member filtering, geometric metrics, and structured termination. Python only
validates requests, selects family specs, and rewraps raw members into domain
objects; there is no Python numeric fallback and no per-member FFI crossing.

Internal specs type the seven families individually. Fixed sampling rules and
continuation directions are not numeric configuration: NRHO L1 uses a single
Rust Halo PAL, while L2 walks fixed-x0 from DE421 Earth-Moon folded calibrated
members; Axial walks fixed-vz0 from DE421 vertical critical orbit calibrated
seeds; Horseshoe reuses the LPO chain.

### 2. Facade returns dedicated Pydantic responses while staying OrbitFamily-compatible

`FamilyGenerationResponse`, a Pydantic model, directly carries
`status/cause/message`, request/generated member counts, and family members,
while inheriting `OrbitFamily`'s reading interface. Success and soft failure use
the same response; soft failures retain completed members. The algorithm layer
keeps using `FamilyGenerationResult`; data-layer `OrbitFamily` carries no
algorithm status.

This satisfies ADR 0014/0024 interface status contracts and preserves #428's
requirement that successful results iterate, index, and read period semantics as
an `OrbitFamily`.

### 3. Lissajous families use Rust nonlinear center-reduced flow

Family sampling parameterizes state within the collinear point's four-
dimensional center subspace. Rust computes the reduced RHS with the full CR3BP
nonlinear potential gradient, advancing in-plane and out-of-plane central
degrees of freedom with RK4; state reconstruction always excludes linear
hyperbolic directions, so results are bounded by construction while retaining
nonlinear frequency/amplitude coupling. Results mark
`periodicity="quasi-periodic"` with `period` being only the nominal in-plane one.

The reduced flow claims no equivalence to the single-orbit entry's high-order
normal-form expansion. `design_lissajous` keeps its implementation and accuracy;
the two interfaces share amplitude/phase/bounded/quasi-periodic semantics but
serve different purposes: high-order single-orbit design vs family parameter
scanning.

### 4. Distinguish PAL trace from public amplitude window

ADR 0028's `PlanarPalRustResult` still always contains seed + full completed PAL
trace with unchanged soft-failure diagnostics. #428's SPO/LPO/Horseshoe results
apply a public-amplitude-window filter over that trace: a 1000-km numeric seed
outside the requested window doesn't enter the final `OrbitFamily`, which would
otherwise violate request scope.

Every filtered member keeps its Newton count, tangent/augmented-system
effective ranks and condition numbers, actual step size, closure error, and
Jacobi drift. The public family thus neither masquerades as raw PAL trace nor
loses numerical diagnostics of returned members.

## Rationale

A single entry concentrates the seven families' shared status/failure/FFI
contracts in one place while hiding family differences behind a Rust enum —
deeper than seven flat numeric functions. Numeric dependencies are all
in-process Rust computation: no ports or callbacks needed.

Were the Facade to return algorithm dataclasses directly, it would break the
interface layer's Pydantic response convention; stuffing status fields into
`OrbitFamily` would pollute data-layer responsibility. A Pydantic response
inheriting existing reading interfaces is the minimal compatible way to satisfy
both constraints.

Propagating full CR3BP directly for Lissajous amplifies residual hyperbolic
components; purely linear trajectories lack nonlinear coupling. The center-
reduced RHS retains nonlinearity without importing Python normal-form code, and
excludes divergent directions from state space by construction.

## Consequences

PyO3 ABI bumps to v15. All seven families share the status triple for success
and soft failures; `n_orbits` remains an upper bound on final member count.
Periodic-family members verify full-period closure, Jacobi conservation, and
applicable symmetries; Lissajous verifies multi-point, boundedness,
finiteness, strictly increasing time.

NRHO/Axial calibrated seeds currently bind to DE421 Earth-Moon context; Rust
explicitly rejects other mass parameters rather than silently applying them.
General-CR3BP calibration extension needs fresh numerical evidence and a new
decision.
