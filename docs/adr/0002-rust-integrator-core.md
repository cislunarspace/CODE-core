# ADR 0002: Rust Integrator Core with Python Dynamics Control

**Status**: Accepted  
**Date**: 2026-06-11  
**Issue**: #61

## Context

Issue #60 plans to migrate propagation and force-model capabilities from GMAT to e2m2e using a Rust integrator core with Python force models and full coordinate support. Issue #61 is the first vertical slice of that migration.

The existing `Dynamics` class in `e2m2e/core/dynamics.py` already provides a stable Template Method API: `propagate()` orchestrates the full trajectory integration, while subclasses override `_get_eom_func()` and `_get_max_step()`. It currently delegates the entire integration to `scipy.integrate.solve_ivp`.

The goal of #61 is to introduce a Rust-based single-step Runge-Kutta engine (starting with Prince-Dormand 5(4), "PD45") and expose it to Python, without breaking the existing `Dynamics` API or forcing an immediate rewrite of trajectory-level control logic.

## Decision

1. **Introduce a Rust workspace under `crates/` with maturin as the sole build backend.**
   - The root `Cargo.toml` defines a workspace.
   - The first crate is `crates/e2m2e-integrators/`, built with PyO3 + maturin.
   - `pyproject.toml` switches its `build-system` from `hatchling` to `maturin`.

2. **Rust crate owns only single-step integration.**
   - It exposes `rk_step(method, t, y, h, tol, f)` where `f` is a Python callback.
   - It returns a `StepResult` with `y_new`, `error`, and `h_next`.
   - It does **not** perform event detection, dense output, or full propagation control.

3. **Python `Dynamics` keeps responsibility for trajectory-level control.**
   - In this first slice, `Dynamics.propagate()` continues to use `scipy.integrate.solve_ivp`.
   - The Rust `rk_step` is used only by dedicated tests and as a low-level building block for future slices.

4. **Public API shim at `e2m2e.integrators`.**
   - Rust extension module installs as `e2m2e._integrators`.
   - A thin Python module `e2m2e.integrators` re-exports `rk_step` and `RkMethod` so callers do not need to import an underscore-prefixed internal module.

## Rationale

1. **Incremental migration risk.** Replacing `solve_ivp` inside `Dynamics.propagate()` in one step would require re-implementing adaptive step acceptance, dense output, event detection, and stiffness handling in Python. Doing that correctly is larger than one slice and adds failure modes that the current `solve_ivp` path does not have.

2. **Algorithm validation before integration.** We want to verify that the Rust PD45 implementation is correct before it becomes the default path for all propagation. Isolating it behind tests lets us compare against `scipy` and analytic solutions without affecting production callers.

3. **Build-backend switch is hard to reverse.** Moving from `hatchling` to `maturin` changes how wheels are produced and how contributors set up the project. This is a meaningful, surprising trade-off that should be documented rather than hidden in `pyproject.toml` alone.

4. **Workspace prepares future crates without forcing them now.** Issue #60 mentions Rust force models and coordinate transformations. A workspace lets future crates live under `crates/` without reorganizing the repository later.

## Consequences

### Added

- `Cargo.toml` at repository root defining a Cargo workspace.
- `crates/e2m2e-integrators/` with `Cargo.toml`, PyO3 bindings, PD45 coefficients, and inline Rust unit tests.
- `[tool.maturin]` configuration in `pyproject.toml`.
- `e2m2e/integrators.py` public shim re-exporting `rk_step` and `RkMethod`.
- `tests/integrators/test_rk_step.py` for Python-level correctness and consistency tests.

### Modified

- `pyproject.toml` `build-system` changes from `hatchling` to `maturin`.
- CI workflow installs Rust toolchain and runs `maturin develop` before Python tests.

### Unchanged

- `Dynamics.propagate()` implementation and behavior in this slice.
- Public `Dynamics`, `CR3BP_Dynamics`, and `EphemerisDynamics` APIs.

### Future work

- A later slice may re-implement `Dynamics.propagate()` to orchestrate Rust `rk_step` calls from Python, adding dense output and event detection as needed.
- Additional RK methods (e.g., DOP853) can be added to `crates/e2m2e-integrators/src/` without changing the build system.
