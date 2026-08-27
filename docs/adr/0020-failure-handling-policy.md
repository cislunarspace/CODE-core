# ADR 0020: Failure handling policy — deterministic failures raise, infeasible searches return flags, no implicit degradation

**Status**: Adopted (decision 3 revised by ADR 0024)
**Date**: 2026-08-09
**Related**: ADR 0002 (multiple scipy fallbacks revised by this ADR), ADR 0003
(the frame layer's never-auto-degrade principle — direct precursor), ADR 0009,
ADR 0014 (decision 4 error translation), ADR 0016, ADR 0017, ADR 0019

## Context

A robustness audit across `algorithm/` + `data/` + `api/` (four parallel
scans) found ~139 sites doing something other than raising or flagging on
failure: silently returning approximations, auto-switching backends, loosening
tolerances while reporting success, hiding failures inside success statistics.
They spread across layers but share one root: **treating failure as an
acceptable alternative result rather than an event to raise or explicitly
flag.**

The most typical:

- **Step-size collapse silently swallowed**: `dynamics.py:611-633` catches
  Rust's "step size collapsed" error (via string matching at `dynamics.py:54`)
  and returns empty states; `propagate_orbit_state_at_time`
  (`dynamics.py:688-699`) then interpolates from the orbit's own data on empty
  states and returns it as a successful result.
- **Lying about convergence**: `differential_correction.py:730-744` marks
  `converged=True` when Newton stalls (corrections < 1e-14) with residuals
  still at 1e-8, while configured tolerance is 1e-12 — a silent four-order-of-
  magnitude relaxation.
- **Failures hidden inside success statistics**: `monte_carlo.py:484-511`, when
  the control law returns None (no convergence / no crossing found), records
  `failed_k = False`; station-keeping Δv statistics skew systematically low.
- **Six incompatible failure-flag dialects**:
  `MultipleShootingResult.converged` /
  `TransferSolution.converged` /
  `TransferOptimizationResult.success` /
  `Orbit.correction_success` (bool|None tri-state) / DC's implicit None /
  grid search's `success:bool + free-string status`. Consequence:
  `search_parallel.py:189-198` hardcodes collision cells' `success=True`,
  and collision cells get plotted as valid solutions on the Δv-Time chart
  (`tools/viz/transfer.py:78`).
- **Resource degradation chains**: the `spice_optional=True` three-tier chain
  (`normal_form/pipeline` → `dynamical_substitution` → `quasi_floquet`)
  silently swaps physics from full ephemeris to pure CR3BP when SPICE is
  missing; `nlp_copt.py` auto-falls-back SLSQP when COPT is unavailable;
  `_HAS_RUST_*` import gates fall back to scipy when Rust is unavailable.

Astrodynamics is deterministic: same initial values + force models → unique
results. These robustness snippets cost: **computed results may have been
quietly altered with no signal to callers.**

## Unifying principle

**Behavior is decided by explicit inputs; no implicit degradation.**
Failure either raises (deterministic processes) or returns with unified flags
(search processes); no intermediate state exists where one thing failed,
another was returned, and callers cannot tell.

## Decision

### Decision 1: three failure classes, three treatments

| Class | Meaning | Treatment |
|---|---|---|
| Deterministic propagation failure | integration divergence, step collapse to machine floor, unavailable Jacobian | raise `PropagationFailure` (decision 2) |
| Search/optimization infeasibility | grid cell diverges, NLP candidate infeasible, single DC step fails | return with unified flags (decision 3) |
| Red lines (forbidden for all classes) | lying about success, hiding failures in success stats, silently swapping physics | fix always, no exceptions |

The discriminator isn't which layer code sits in but **whether callers can
distinguish "got what I wanted" from "didn't"**. Flagged returns that preserve
that distinction are compliant; those destroying it (`success=True` lies,
approximate values without flags, implicit None losing causes) are red lines.

### Decision 2: contextualized semantics of deterministic propagation failures

Raising on any step collapse is too coarse; literally applied it would ban
adaptive integrators' standard behavior and kill legitimate gravity assists.
Refined into three tiers:

1. **Step rejection** (error > tol but h still above machine floor):
   standard adaptive-controller behavior — reject, shrink h, retry
   (`cr3bp.rs:260-277`, `solve_ivp.rs:251-303`, `force_model.py` RK loop).
   **Not failure; not reported; not counted as fallback.**
2. **Collapse to machine floor / unavailable Jacobian / true divergence**:
   propagators raise **`PropagationFailure`** (new typed exception, see
   Consequences). Replaces the fragile catch matching `"step size collapsed"`
   via string at `dynamics.py:54` (one wording change breaks it).
3. **Context decides reporting semantics**: the same `PropagationFailure`
   raised to users calling `propagate` directly; search/optimization wrappers
   (grid search, NLP, multiple-shooting segments) catch and convert to
   `status=INFEASIBLE/DIVERGED` flagged returns. **Propagators themselves don't
   assume context**; callers decide whether to catch or re-raise.

Machine-precision floors (`MIN_STEP = 1e-12·span`,
`10·EPSILON·(1+|t|)`) must remain, explicitly acknowledged: they are loop
guards (preventing h→0 infinite loops on true divergence), not concealment.
Floors must be observable and enter result objects. What's forbidden are
**physical-magnitude floors**, e.g. `qlaw.py:379-380` resetting rejected steps
back to original step length (triggering ~2 million idle steps barely advancing
t before assembling a control law from unaccepted intermediate states), or
`force_model.py:804` forcing acceptance via `h=max(h,min_step)`. Removing
physical floors aligns the Python path with Rust's pattern
(`solve_ivp.rs:244-248`: min_step used solely for failure detection, never
lifts).

> Catching step-collapse in search contexts and converting to infeasible
> (`dynamics.py:610-633`, `transfer_grid_search.rs:157-187`) is compliant;
> what changes isn't the catching but replacing empty-states-with-len==0
> sniffing with structured results carrying failure flags.

### Decision 3: unified failure flags for infeasible searches

Returning converged=False isn't enough: the flag's shape must be pinned, or
each module grows its own dialect and callers get bitten by None/False/missing
flags. The six dialects already let collision cells lie success=True.

Anchored on the existing `ConvergenceState` enum
(`e2m2e/data/templates/enums.py`: ITERATING/CONVERGED/DIVERGED/STAGNATED/
MAX_ITERATIONS), **extend with `INFEASIBLE` and `COLLISION`**, and require:

- All search/correction results expose identically named
  `status: ConvergenceState` field + `cause: str`.
- **Result objects must be returned even on failure** (carrying status +
  cause). Asymmetric signatures returning objects on success and None on
  failure are banned: `DifferentialCorrection.iterate_correction`'s
  `Orbit | None` becomes always-a-result-carrying-status (Orbit as its field);
  `termination_reason` travels with the object instead of lingering on the
  solver.
- Collision is its own status enum value (COLLISION), not
  `success=True + status string`. The collision cells at
  `search_parallel.py:189-198` must move to the failure side.
- Abolish `bool | None` tri-states, free-string statuses, implicit None.
  `converged`/`success`/`correction_success` may survive compatibly as derived
  properties of `status == CONVERGED`, but must not be the only signal.

Precedents: `design_orbit.py` (raises `DesignNotConvergedError` on all
non-convergent paths), `multiple_shooting.py`
(`MultipleShootingResult(converged=False, status=...)`) are already correct
in-repo patterns; this decision generalizes them.

> **Revision (2026-08-11, ADR 0024)**: boolean compatibility projections are
> dropped: `success`/`converged`/`correction_success` are removed outright
> without a runtime compatibility layer; `ConvergenceState` gains `FAILED`
> beyond this decision's `INFEASIBLE`/`COLLISION`. The "may survive as derived
> properties" clause above is void; all other clauses stand.

### Decision 4: no implicit resource degradation; two kinds of unavailability distinguished

Auto-switching backends when resources (SPICE/Rust/COPT) are unavailable is
this ADR's core elimination target. But two kinds exist, treated differently:

- **Resource missing** (not installed/not built): **error out**. Spice is now a
  default feature, standardized by `make dev`, shipped in release wheels —
  these resources are constant in normal operation; absence means environment
  misconfiguration, never a reason to quietly switch to slow paths. Revised:
  ADR 0002's Dynamics-base scipy fallback on missing Rust, COPT→SLSQP
  fallback, silent sans-spice degradation; ADR 0009's release try/except;
  ADR 0016's cache-miss cspice fallback; ADR 0017's rust-unavailable
  processes fallback; ADR 0019's SPICE-missing ITRFApproxAxes degradation —
  all become errors.
- **Capability missing** (backend present, feature unimplemented/semantics
  unaligned): **explicit `backend="scipy"` / `backend="rust"` parameter**,
  one or the other; omitting raises (deprecation warning acceptable during
  migration, removed next major). **No `backend="auto"`** — auto still lets
  code decide backends for users, i.e., implicit. Typical case: CR3BP/BCR4BP
  event detection's Rust semantics not yet aligned with scipy (ADR 0002 event
  clause) — capability missing, explicit backend required.

**Test-injection seam exemption**: ADR 0017's monkeypatch fallback (fall back
to Python when tests inject synthetic trajectories, so injections take effect)
is test infrastructure, not production degradation — not banned, but confined
to test paths (`_geometry_methods_monkeypatched` detection); production paths
never trigger it.

### Decision 5: separate singularity regularization from collision termination

Naively removing distance clamps would delete two unrelated things together,
blowing up Hessians. Refined:

- **Machine-precision regularization kept**:
  `MIN_DISTANCE ≈ 1e-10 LU` (≈3.8 cm, far inside any body radius) prevents
  divide-by-zero NaNs at gravity's 1/rⁿ singularity — present in
  `potential.py:11`, `dynamics.py:76`, `cr3bp.rs:19`, `bcr4bp.rs:22`
  (all 1e-10 nondimensional), and `nbody_stm.rs:27` at 1e-6 km for the same
  purpose. Hessians contain 1/r⁵ terms (`potential.py:42-50`); deleting these
  yields inf/NaN near primaries. Numerical guards, not physical falsehoods.
  **All retained.**
- **Physical-magnitude clamps become collision termination**: intersecting a
  body radius (Earth ≈6378 km, Moon ≈1737 km) → event detection
  `g = |r| - R_body`, `terminal=True`, or raise. `transfer_geometry.rs:211`'s
  `check_collision` already post-hoc scans; core propagation needs the
  event-based version.
- CR3BP is a point-mass model with no intrinsic body radii — collision
  termination needs **external body-radius configuration injection**. New
  feature, not removal of old behavior.
- Wording shifts from "no distance clamping" to "**no clamping within body
  radii**".

## Revisions to existing ADRs

| ADR | Original decision | Changed to | Class |
|---|---|---|---|
| 0002 | Dynamics-base scipy fallback on missing Rust | error on missing Rust | resource missing |
| 0002 | COPT→SLSQP fallback | error; NLP backend explicit | resource missing |
| 0002 | Silent sans-spice slow-path degradation | error | resource missing |
| 0002 | CR3BP/BCR4BP events passed to scipy fallback | explicit `backend="scipy"/"rust"`, no auto | capability missing |
| 0009 | Release without spice; try/except silent degradation | error (releases ship spice; mechanism removed) | resource missing |
| 0016 | Cache miss silently falls back to cspice FFI | error or explicit selection (Strict mode generalized from parallel-only to default) | resource missing |
| 0017 | Explicitly chosen rust falls back to processes when Rust missing | error (test monkeypatch seam exempt) | resource missing |
| 0019 | SPICE missing degrades drag rotation to ITRFApproxAxes | error or explicit low-precision backend choice | resource missing |

> Note: ADR 0002 line 96 originally stated BCR4BP events raise
> NotImplementedError (#333); actual code (`bcr4bp_dynamics.py:204-212`) had
> changed to warn + scipy fallback — doc/code mismatch resolved here as
> capability missing with explicit backend.

## Rationale

1. **Direction has precedent, not invented wholesale.** ADR 0003 item 7
   (explicit errors, never auto precision degradation; clamping behind explicit
   options) established error-on-missing at the coordinate layer long ago;
   this ADR generalizes it repo-wide. ADR 0004's loud non-serializable errors
   and ADR 0018's mandatory triples turning silent corruption into compile
   failures are precedents for decision 4 (no lying); ADR 0014 decision 4's
   exception translation at api/ into `OrbitError(code/message/details)` is the
   downstream exit for decision 2 (raising); ADR 0016 Strict mode's hard-fail
   misses precede decision 4 (resource-missing ⇒ error).
2. **Coarse wording kills legitimate behavior — three counterexamples excluded
   via adversarial verification.**
   - Raise-on-any-collapse would ban adaptive reject-shrink-retry (standard RK)
     and, combined with de-clamping, legitimate low-altitude lunar flybys
     (r₂≈1e-3, 384 km from lunar center, missing the surface — a legal gravity
     assist) would be reported as integration failure. Decision 2's tiering
     excludes it.
   - Deleting 1e-10 LU regularization makes Hessians (1/r⁵ terms) inf/NaN near
     bodies. Decision 5's split excludes it.
   - Flagless converged=False returns already let grid-search collision cells
     lie success=True (`tools/viz/transfer.py:78` plotting collisions as valid
     solutions). Decision 3's unified enum excludes it.
3. **Determinism is a domain requirement.** Astrodynamical propagation is
   deterministic — same initial state and model, unique outcome. Results being
   quietly altered with no caller signal violates it. Implicit degradation's
   worst consequence isn't slowness but **wrongness without awareness**:
   `spice_optional` chains swapping physics, `ITRFApproxAxes` dropping accuracy
   tiers, DC stalling loosening tolerances — all change numbers while callers
   assume nothing changed.
4. **Cost is controlled.** Decision 5's refinement reclassifies most of the
   audit's ~30 MIN_DISTANCE clamps as machine-precision regularization
   (retained); migration scope shrinks drastically. Real deletions concentrate
   in decision 4's resource degradations (8 ADR revisions) and decision 1's
   red lines (~36 lying/hiding sites).

## Consequences

### Added

- `PropagationFailure(E2M2EError)` typed exception (`e2m2e/exceptions.py`),
  replacing the string-matching catch at `dynamics.py:54`.
- `ConvergenceState` extended with `INFEASIBLE`, `COLLISION`; unified
  `status: ConvergenceState` + `cause: str` norm for search/correction results.
- Collision termination: CR3BP/BCR4BP body-radius config injection +
  event-based termination in propagation (`g=|r|-R_body, terminal=True`).
- Explicit `backend="scipy"/"rust"` parameters for capability-missing scenarios
  (event detection etc.), no `auto`.

### Changes (migration order)

1. Add `PropagationFailure` typed exception (zero test breakage; foundation).
2. Decision 3: unify `ConvergenceState` status norms across search results
  (most tests assert happy paths; small breakage).
3. Decision 1 red lines: fix lying/hiding (DC stall shortcut, MC controller
  None treated as success, `propagate_orbit_state_at_time` empty-states
  interpolation retreat, grid-search collision success=True, qlaw idle spin,
  qlaw `_resolve_mu` silent Earth μ).
4. Decision 2: `_propagate_state_only` empty states → flagged failure; sync
  `transfer_optimization.py`'s len==0 sniffing and NLP's dv=1e10 double
  penalty (drop objective penalty, keep constraint-conflict flags).
5. Decision 4: remove resource degradations (8 ADR revisions); event detection
  gains explicit backend, no auto.
6. Decision 5: collision termination + body-radius injection (highest risk —
  touches force evaluation/STM; ensure collision-event termination before
  touching any physical-magnitude clamps).

### Unchanged

- Machine-precision regularization (MIN_DISTANCE ≈ 1e-10 LU, NaN guards).
- Adaptive integration's reject-shrink-retry standard behavior.
- Machine-precision step floors (loop guards).
- Test injection seam (ADR 0017 monkeypatch fallback, test paths only).
- `design_orbit.py` / `multiple_shooting.py` / `homotopy.py`'s raise + flag
  paradigm (already-compliant precedents).
- IEEE 754 domain protection (e.g., clip to [-1,1] before arccos).
