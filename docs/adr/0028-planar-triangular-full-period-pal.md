# ADR 0028: Planar triangular-libration-point families via full-period pseudo-arclength continuation

**Status**: Adopted (#428's unified Rust seam revised by ADR 0029 and
implemented)
**Date**: 2026-08-16
**Related Issues**: #436, #428, #435, #451
**Related**: ADR 0024 (unified result status contract), ADR 0013
(verification by definition)

## Context

SPO, LPO, and Horseshoe are planar periodic orbits near L4/L5. They lack Halo's
x-axis or XZ mirror symmetry, so they must close over the full period. The
existing `DifferentialCorrection` already handles three-residual full-period
correction at fixed `x0`; both `Continuation.natural_continuation()` and the
existing PAL rely on half-period symmetry constraints and cannot carry these
problems.

Natural-parameter continuation at fixed `x0` works locally but cannot serve as
LPO's global parameterization: amplitude and period turn around, and Horseshoe —
a large-amplitude LPO member — shouldn't get yet another separate continuation.
The goal is deciding the formal algorithm shape; Facade/public request models
aren't modified in this ADR.

## Numerical evidence

The experiment script was
`scripts/research_issue_436_full_period_pal.py` (removed after mission
completion; retrievable from git history at the pinned commit below). Pinned
commit `14025914956faa90a1d3e24019db5ca6c33647af`, standard Earth-Moon CR3BP
mass ratio `0.01215058560962404`, L4, small-amplitude linearized seed,
normalized arclength step `0.01`. Run after `make setup && make dev` to set up
CSPICE, kernels, and the Rust extension.

- LPO walked 60 steps toward decreasing `x0`. L4's geometric-amplitude metric
  ranged 586 km → 238,833 km; max full planar closure infinity norm `3.30e-10`,
  max Jacobi drift `3.11e-15`. Steps 9–13 saw period dip then rise, but the
  chain stayed continuous with no branch jumps or correction failures.
- L5's LPO walked 20 steps same direction. Its geometric-amplitude crossed
  110,000 km up to 138,526 km; max closure norm `8.17e-10`, max Jacobi drift
  `1.78e-15`.
- SPO walked 5 steps same direction: max closure norm `1.58e-11`, max Jacobi
  drift `1.33e-15`.
- LPO's full-closure+phase condition had effective rank 4 at relative singular
  threshold `1e-8`; adding pseudo-arclength gave effective rank 5 for the
  augmented system. Integration and STM errors lift the autonomous system's
  theoretical null space — rank can't be judged at machine-precision
  thresholds.

These results prove the formulation continuously tracks L4's long branch and
L5's corresponding extension past the current 110,000-km search claim; they
prove neither a physical amplitude ceiling nor grounds to widen #435's public
scope. Reverse branches, collision boundaries, and other normalizations/steps
need re-testing during implementation acceptance.

## Decision

### 1. Adopt planar full-period PAL

Let the planar initial state be `s=(x0,y0,vx0,vy0)`, unknowns `q=(s,T)`. The
formal algorithm uses:

```text
R(q) = Pi(phi_T(s)) - s = 0
h(q; qk) = (s - sk) dot f(sk) / ||f(sk)|| = 0
g(q; qk, tk, ds) = ((q - qk) / scales) dot tk - ds = 0
```

where `Pi` selects the four planar components, `qk` is the previous converged
orbit, `tk` its tangent vector. `R` checks all four planar closure components;
the phase condition kills the autonomous system's phase freedom; the
pseudo-arclength condition picks the neighboring family member. Tangents come
from `[dR/dq; dh/dq]`'s null space, oriented along the previous tangent. After
prediction, least-squares Newton corrects the 6-row × 5-column augmented system
`[R; h; g]`.

First version fixes `scales=(1,1,1,1,10)` matching experiments.
Implementations must record normalization and shrink steps when conditioning
degrades; neither `x0`, amplitude, period, nor Jacobi constant is forced
monotone. The initial `x0` direction only orients the first tangent's sign;
after passing a fold, motion continues along arclength.

### 2. A dedicated deep module, not generalizing Halo's PAL

Add a planar full-period PAL numeric kernel inside `crates/e2m2e-integrators`;
Python's caller-facing surface stays one family-generation entry:

```python
generate_planar_periodic_family(
    dynamics,
    seed_orbit,
    *,
    family_type,
    libration_point,
    n_orbits,
    step_size,
    initial_direction,
)
```

Rust internally encapsulates STM closure Jacobian, phase gauge, SVD tangents,
fixed-damping SVD least-squares Newton, line search, step-size contraction, and
effective-rank checks. Python reads mass parameters and integration config from
`CR3BP_Dynamics`, calls the Rust kernel, then interprets raw members into an
`OrbitFamily`; returns ADR 0024's `ContinuationResult` where `family` always
contains seed plus completed partial family. Callers never pass residual
indices, phase functions, or Jacobian shapes.

SPO, LPO, Horseshoe share this module; family names, L4/L5 labels, and
amplitude measurement stay in `algorithm.family` orchestration. Horseshoe is a
member classification of LPO — it gets no second solver.

The existing `Continuation.pseudo_arclength_continuation()` stays Halo-only. Its
XZ-symmetric free variables and physical ranges can't stretch to this problem
via parameter switches. With only one asymmetric planar adapter today, no
generic arbitrary-period adapter abstraction is introduced; extract a shared
PAL numeric kernel when a second real 3D full-period adapter appears.

Reusing multiple shooting's normal-equations Gaussian elimination fails this
problem: PAL must resolve rank-deficient null spaces of closure+phase matrices
and solve ill-conditioned augmented least squares. The Rust kernel uses
`nalgebra`'s SVD and rank decisions rather than copying that numerical duty into
Python.

### 3. Explicit failure & accuracy contract

Successful members must satisfy full six-dimensional closure infinity norm ≤
`1e-8`, planar constraints `z=vz=0`, and pass the CR3BP Jacobi drift check.
When the augmented system can't hold effective rank 5 → return
`FAILED/SINGULAR_JACOBIAN`; when steps shrink to the floor without a converged
member → `STAGNATED/STAGNATION_DETECTED`. Any soft failure retains the
generated `OrbitFamily` — never bare `None` or implicit fallback. Each member's
effective rank, condition number, Newton iteration count, and actual arclength
step persist in Rust-boundary results and numerical-diagnostic entries; data-
layer `OrbitFamily` records family semantics only, never becoming a diagnostic
container.

## Trade-offs

Stuffing `iterate_full_period_correction()` into natural continuation changes
little but stays fixed-`x0`, blind to folds. Keeping grid search finds single
orbits by amplitude but yields no continuous family and can't say where failure
occurred. Both may remain as single-orbit design baselines, but neither is
#428's family-generation method.

Going straight to generic full-6D-state-plus-period PAL would simultaneously
decide 3D residual independence, phase gauges, and new interfaces with no second
consumer validating the abstraction's value. Limiting v1 to planar buys smaller
surface and concentrated numerical verification at the cost of one more design
decision when 3D asymmetric periodic families arrive.

Experiments exceeded 110,000 km but covered one L4 one-way branch and one step
size. Immediately widening public amplitude ranges would misstate method
feasibility as physical reachability — rejected.

## Consequences

When #428 implements SPO/LPO/Horseshoe family dispatch, this ADR's Rust kernel
and Python family-generation seam are premises. Implementation acceptance
covers at least: L4/L5, both initial directions, local SPO/LPO chains, LPO's
period-turning region, structured failures, and existing Halo PAL regression;
large-amplitude scans continue via rerunnable scripts rather than long numerics
stuffed into regular pytest.

## Revision (2026-08-16, #451)

Decision 2's first draft placed the planar full-period PAL numeric module in
the Python `solver` package. Checked against the five-layer architecture and
revised: numerical iteration lives in the Rust numerical layer;
`algorithm.family` keeps the family-generation seam for problem construction
and result interpretation; formal equations, status contract, and don't-
generalize-Halo-PAL stand unchanged. The implementation introduced `nalgebra`
because the existing shooting's normal-equation Gaussian elimination can't
provide rank-deficient null spaces, effective ranks, and ill-conditioned
least-squares diagnostics that PAL requires.

#451 completed the first vertical slice on the current branch: Rust kernel,
PyO3 ABI, Python family-generation seam, L4/L5 SPO/LPO seam tests, LPO period-
turning cases, partial-family retention, and existing per-family continuation
regressions all passing. Reverse initial directions and longer-branch scans
continue per this ADR's acceptance boundaries.

## Revision (2026-08-16, ADR 0029, #428)

#428 folded decision 2's Python family-generation seam into the unified Rust
family-generation module. The underlying `PlanarPalRustResult` still always
retains seed + fully converged trace per decision 3; the Facade returns the
domain family filtered by requested amplitude window — seeds outside the window
don't enter public members. Filtered members keep effective rank, condition
number, Newton count, actual step size, closure error, and Jacobi drift.
