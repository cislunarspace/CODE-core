# ADR 0002: Rust integrator core with Python-controlled dynamics

**Status**: Adopted
**Date**: 2026-06-11
**Related Issue**: #61

## Context

Issue #60 plans to migrate propagation and force-model capabilities from GMAT
to e2m2e, using a Rust integrator core + Python force models + full
coordinate support. Issue #61 is the first vertical slice of that migration.

The existing `Dynamics` class in `e2m2e/algorithm/dynamics/dynamics.py`
already provides a stable template-method API: `propagate()` orchestrates
integration of the whole trajectory while subclasses override
`_get_eom_func()` and `_get_max_step()`. It currently delegates all integration
to `scipy.integrate.solve_ivp`.

The goal of #61 is to introduce a Rust-based single-step Runge-Kutta engine
(starting from Prince-Dormand 5(4), i.e. PD45), expose it to Python, and do so
without breaking the existing `Dynamics` API or forcing an immediate rewrite
of trajectory-level control logic.

## Decision

1. **Introduce a Rust workspace under `crates/` with maturin as the sole
   build backend.**
   - Root `Cargo.toml` defines a workspace.
   - The first crate is `crates/e2m2e-integrators/`, built with PyO3 +
     maturin.
   - `pyproject.toml`'s `build-system` switches from `hatchling` to `maturin`.

2. **The Rust crate does single-step integration only.**
   - Exposes `rk_step(method, t, y, h, tol, f)` where `f` is a Python callback.
   - Returns a `StepResult` containing `y_new`, `error`, `h_next`.
   - Does **not** do event detection, dense output, or full propagation
     control.

3. **Python-side `Dynamics` keeps trajectory-level control.**
   - In this first slice, `Dynamics.propagate()` continues using
     `scipy.integrate.solve_ivp`.
   - Rust's `rk_step` is used only by dedicated tests and serves as a base
     building block for later slices.

4. **Provide a public thin wrapper at `e2m2e.integrators`.**
   - The Rust extension module installs as `e2m2e._integrators`.
   - A thin Python module `e2m2e.integrators` re-exports `rk_step` and
     `RkMethod` so callers never import the underscore-prefixed internal
     module.

## Rationale

1. **Migration risk must be controlled incrementally.** Replacing
   `solve_ivp` inside `Dynamics.propagate()` in one step would require
   reimplementing adaptive step acceptance, dense output, event detection,
   and stiffness handling in Python. Doing these correctly far exceeds one
   slice and would introduce failure modes the current `solve_ivp` path does
   not have.

2. **Validate algorithms before wiring into default paths.** We want the Rust
   PD45 implementation proven correct before it becomes the default path for
   all propagation. Isolating it behind tests allows comparison against
   `scipy` and analytic solutions without touching production callers.

3. **Build-backend switches are hard to reverse.** Moving from `hatchling` to
   `maturin` changes how wheels are produced and how contributors set up the
   project. This is a weighty, surprising trade-off that should be documented,
   not hidden in `pyproject.toml`.

4. **A workspace reserves room for future crates without imposing them now.**
   Issue #60 mentions Rust force models and coordinate transformations. A
   workspace lets future crates live under `crates/` without later repository
   reorganization.

## Consequences

### Added

- Repository-root `Cargo.toml` defining a Cargo workspace.
- `crates/e2m2e-integrators/` with `Cargo.toml`, PyO3 bindings, PD45
  coefficients, and inline Rust unit tests.
- `[tool.maturin]` configuration in `pyproject.toml`.
- Public thin wrapper `e2m2e/integrators.py` re-exporting `rk_step` and
  `RkMethod`.
- `tests/numerical/integrators/methods/test_rk_step.py`, Python-side
  correctness and consistency tests.

### Changed

- `pyproject.toml`'s `build-system` from `hatchling` to `maturin`.
- CI workflow installs the Rust toolchain and runs `maturin develop` before
  Python tests.

### Unchanged

- Implementation and behavior of `Dynamics.propagate()` in this slice.
- Public `Dynamics`, `CR3BP_Dynamics`, `EphemerisDynamics` APIs.

### Follow-up work

- Later slices may rewrite `Dynamics.propagate()` to orchestrate Rust's
  `rk_step` from Python, adding dense output and event detection as needed.
- More RK methods (e.g. DOP853) can be added to
  `crates/e2m2e-integrators/src/` without changing the build system.

## Revision (2026-06-14, issue #67)

Decision 2 (Rust crate does single-step integration only) described the scope
of the **first slice**, not a permanent constraint. The integrator-family epic
(#67) extends the Rust crate to three method families:

- **Single-step RK** (`rk_step`): `Pd45`, `Pd78`, `Rk89`, as in the original
  slice.
- **Multistep predictor-corrector** (`multistep_step`):
  Adams-Bashforth-Moulton (`Abm`), fixed step, carrying a **history buffer**
  of derivative samples.
- **Second-order double integration** (`cowell_step`): Störmer-Cowell, fixed
  step, integrating `x'' = a(t, x)` directly from a position+acceleration mixed
  history buffer; outputs positions only.

Decision 1 (workspace + maturin) and decision 4 (public `e2m2e.integrators`
thin wrapper) still hold. The new multistep/second-order families respect the
same boundary: advance one step, return error estimate and suggested step; no
event detection, dense output, or full propagation control.

Note on decision 3: `CR3BP_Dynamics` and `EphemerisDynamics` (system classes)
still use `scipy.solve_ivp`; only `ForceModel` (force-decomposition class)
drives Rust steppers from Python. The original follow-up items are now done:
`ForceModel` orchestrates `rk_step` from Python (adaptive steps + simple event
detection), and the crate gained the multistep and second-order families.

## Revision (2026-07: crate split and spice build conventions)

The single crate splits into four: `e2m2e-integrators` (pyo3 bindings and
build entry, the only maturin packaging target), `e2m2e-propagation` (pure-math
integrators), `e2m2e-forces` (N-body STM, gravity fields), `e2m2e-spice`
(CSPICE FFI). Decision 3 partially lapses: propagation has moved into Rust
(`propagate_compiled`, `propagate_with_stm_py`) because the cspice kernel pool
is a process-level singleton — SPICE-related propagation and STM must compile
into the same extension as force models and cannot stay in the Python
orchestration layer.

Spice-feature build conventions: `cspice-sys` downloads CSPICE sources from
NAIF at build time via `downloadcspice`, no manual install needed (or point
`CSPICE_DIR` at a local installation). `maturin develop` defaults to no spice;
`maturin develop --features spice` includes STM propagation, shooting,
third-body and other Rust fast paths; without spice, Python silently degrades
to slow paths and corresponding tests skip via `importorskip`. **Release
wheels ship without spice for now**: including it would embed CSPICE in wheels
and tie builds to NAIF reachability; licensing and release stability need a
separate evaluation first. CI covers spice-gated code compilation via
`cargo clippy --workspace --features spice`.

> Revision note (2026-08, ADR 0020 decision 4): after spice became a default
> feature this section went stale: `maturin develop` defaults to spice (below),
> silent degradation to slow paths without spice became hard errors (issue
> #378), and corresponding tests' `importorskip` semantics were adjusted.

## Revision (2026-08: spice promoted to default feature)

Spice is now a default feature: crates `default = ["spice"]` plus pyproject
`features=["spice"]` as double insurance; `maturin develop` defaults to spice,
producing no no-spice subset; release wheels carry spice (ADR 0009 delivered).

**Dynamics integration core paths unified on Rust.** Integration
(`rk_step`/`multistep_step`/`cowell_step`/`solve_ivp`), propagation
(`propagate_compiled`/`propagate_with_stm_py`), force models
(PointMass/ThirdBody/GravityField/Drag/SRP/Relativistic), multiple shooting,
and transfer grid search have all moved into Rust. With spice enabled by
default, all core computation paths avoid Python scipy in normal operation.

**Scipy paths retained in the following scenarios** (deliberate design
choices, not oversights):

- **Event detection** (`CR3BP_Dynamics._propagate_with_stm(events=...)`): Rust
  `solve_ivp_events_py` exists but its event semantics don't fully align with
  scipy; event paths choose explicitly `backend="scipy"/"rust"` (ADR 0020
  decision 4) — omitting it errors out, `auto` disallowed; `"scipy"` uses
  scipy `solve_ivp`, `"rust"` uses Rust `solve_ivp_events` (accepting semantic
  differences). BCR4BP likewise (#333's NotImplementedError divergence
  resolved).
- **Defensive fallbacks** (`Dynamics` base and `EphemerisDynamics`
  `_propagate_with_stm`/`_propagate_state_only`): when the Rust extension is
  unavailable there is no fallback to scipy anymore;
  `require_rust_extension` raises `RustExtensionUnavailableError` instead
  (issue #378, ADR 0020 decision 4: missing resources raise).
- **NLP optimization** (`transfer/nlp_copt.py`): when COPT is unavailable the
  default raises (`fallback_to_scipy` defaults `False`); passing `True`
  explicitly falls back to SciPy SLSQP. ADR 0017 keeps NLP at the Python layer.
- **Normal form propagation** (`normal_form/multiple_shooting.py`,
  `dynamical_substitution.py`, `propagation.py`, `quasi_floquet.py`): moved to
  Rust `solve_ivp_py` (#336). QF↔CM high-order Lie flows
  (`coord_trans/qf_cm.py`) sunk as `qf_to_cm_py` / `cm_to_qf_py` (#465,
  12-real-dim split-complex integration); `backend="python"` remains for
  explicit comparison only. `scipy.linalg.expm` (matrix exponential) and
  `scipy.optimize.fsolve` keep awaiting Rust replacements.
- **Libration point solving and initial-value generation**
  (`scipy.optimize.fsolve`/`brentq`): for L1/L2 position solving, Halo orbit
  initial guesses etc. Single calls, low migration value.
