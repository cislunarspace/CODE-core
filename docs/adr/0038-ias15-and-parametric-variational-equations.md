# ADR 0038: Adopt two ASSIST algorithms (IAS15 integrator, parametric variational equations); MERCURIUS not adopted

**Status**: Adopted
**Date**: 2026-08-26
**Related**: ADR 0002 (Rust integrator core), ADR 0018 (force-Jacobian
triple contract), ADR 0020 (explicit failure)

## Context

Two sets of published algorithms were evaluated for applicability to this
repo: ASSIST (Holman et al. 2023, an ephemeris-level test-particle
integrator) and MERCURIUS (Rein et al. 2019, a hybrid symplectic
integrator). Conclusion: MERCURIUS targets hundred-million-year-scale
evolution of giant-planet systems (the Wisdom-Holman split requires a
dominant Kepler term plus fully interacting massive bodies) and does not
match this repo's problem domain - massless spacecraft flying
mission-length timescales in the Earth-Moon space. The disease it cures
(close-encounter divergence of symplectic integrators) is already covered
by the existing adaptive RK, so it is **not adopted**. ASSIST's technology
stack is isomorphic to this repo's (SPICE ephemerides + N-body +
high-fidelity force models + variational equations), and two of its pieces
fill real gaps here, so they are **adopted**:

1. The existing integrator family (PD45/PD78/RK89, ABM, Cowell) has no
   high-order variable-order predictor-corrector method and no compensated
   summation; high-accuracy long-arc extrapolation (e.g. orbit lifetime
   analysis) lacks an option whose round-off grows per Brouwer's law
   (n^1/2).
2. STM propagation only covers the 6x6 partials of the state with respect
   to the initial state (ADR 0018); first-order partials with respect to
   force-model parameters (Cr, Cd) - needed by orbit determination and
   covariance propagation - are missing. Low-thrust sensitivity already
   has the A*S+B structure (`augmented_state.rs`) but was only wired to
   control parameters.

License constraint: ASSIST/REBOUND are GPL v3 while this repo is
Apache-2.0, so their code cannot be referenced; both algorithms are
implemented from the mathematical descriptions published in the papers.

## Decision

1. **The IAS15 integrator goes into `e2m2e-propagation` (`ias15.rs`).**
   Implemented per the published algorithms of Rein & Spiegel (2015) and
   Everhart (1985): 8 left Radau nodes (roots of P7+P8, verified
   numerically to machine precision), Newton-basis interpolation
   polynomial with per-node Gauss-Seidel correction (at most 12 sweeps),
   second-order double integration (positions) plus single integration
   (an extra first-order component beyond velocities, reused directly for
   STM/sensitivity columns), and Neumaier-Kahan compensated summation.
   This differs from the papers' b-coefficient basis only by a change of
   basis; the convergence fixed point is the same.
2. **Parametric sensitivity columns join the STM augmented system.**
   `CompiledForce` gains `SensParam` (currently only `Cr`/`Cd`, both
   strictly linear in the acceleration so `d(a)/d(p) = a/p` analytically);
   the augmented dimension in `compiled_stm.rs` grows to `42 + 6*n_params`,
   each column satisfying `dS/dt = A*S + [0; da/dp]` with zero initial
   value. With an empty `sens` the result is bit-identical to the old
   42-dimensional path (the original `propagate_compiled_stm` signature is
   kept and delegates to the new implementation).
3. **IAS15's error estimate is calibrated on the state update, with a
   noise-floor treatment.** The highest-order term G7 is converted into an
   update error via the integration weight integral-of-P7 (Gauss-Radau
   cancellation), not from nodal values - otherwise the interpolation
   noise floor (which does not shrink with h) is misread as truncation
   error and the engine rejects steps in a loop. Under ephemeris force
   models (SPICE evaluation, third-body direct/indirect cancellation) the
   measured effective smoothness of the acceleration is about 1e-11
   relative; 7th-order divided differences amplify it into an error
   estimate that stops decreasing with step size. After consecutive
   rejections whose eps shrinks by less than 2x, the floor is deemed
   reached, the effective tolerance is raised above it, and the step is
   accepted. Purely analytic forces (point mass) have no such floor and
   follow tol strictly.
4. **External seams.** Rust binding `propagate_compiled_ias15_py` (with
   optional with_stm and sens_params); `propagate_compiled_stm_py` gains
   an optional `sens_params` (ABI bumped to v22); on the Python side,
   `ForceModel.propagate` gains `integrator="rk"|"ias15"` and
   `sens_params=["srp_cr"|"drag_cd"]` (requires `with_stm=True`; labels
   are resolved to force indices at the Python layer, with ambiguity or
   absence reported explicitly per ADR 0020).

## Rationale

- **IAS15 rather than another Butcher tableau**: adding an RK method needs
  only a coefficient table, but a variable-order predictor-corrector's
  efficiency on smooth solutions and compensated summation's long-arc
  round-off behavior are things RK cannot provide; IAS15's automatic step
  shrinkage at close encounters is validated by ASSIST's Apophis example.
- **Self-implemented Newton basis rather than copied coefficient tables**:
  Everhart's c/d/r recursion coefficients and REBOUND's hard-coded
  constants can both be generated exactly from the nodes via
  Gauss-Legendre quadrature (integrand degree at most 8, exact to rounding
  with 16 GL points), which proves correctness by construction while never
  touching GPL code.
- **Sensitivity columns only for linear parameters**: Cr/Cd have analytic
  and exact `da/dp` (a/p), covering the two dominant stochastic
  force-model error sources of orbit determination; nonlinear parameters
  (e.g. atmospheric density model parameters) would need numerical
  difference columns - add them when a real need appears (extension
  points: the `SensParam` enum plus `param_accel_derivative`).
- **Not following ASSIST's full PPN / Marsden A1A2A3 / solar J2**: the
  central-body 1PN terms (Schwarzschild + LT + de Sitter) already exist in
  this repo; full PPN matters for asteroid mas-level ephemerides, the
  Marsden model serves comets, and solar J2 is negligible in Earth-Moon
  space - all out of scope.

## Consequences

### Added

- `crates/e2m2e-propagation/src/ias15.rs`: the IAS15 engine (with Rust
  unit tests for the analytic Kepler solution, closed e=0.9 orbits,
  out-and-back reversibility, long-arc energy behavior).
- `crates/e2m2e-forces/src/forces/compiled_ias15.rs`: IAS15 driver for
  compiled force models (state / +STM / +sensitivity columns, with
  burn start/stop boundary truncation for low thrust).
- `SensParam` and `param_accel_derivative` (`compiled.rs`);
  `propagate_compiled_stm_sens` (`compiled_stm.rs`).
- Binding `propagate_compiled_ias15_py`; `propagate_compiled_stm_py` gains
  optional `sens_params`; ABI v22.
- `ForceModel.propagate(integrator=..., sens_params=...)`.
- Tests: `tests/numerical/integrators/methods/test_ias15.py` (analytic
  solutions / conserved quantities / backend comparison) and
  `tests/numerical/forces/container/test_force_model_sensitivity.py`
  (cross-integrator sensitivity agreement at ~1e-12, shadow-particle FD
  comparison, contract errors).

### Unchanged

- Behavior of the RK family, ABM, Cowell and existing propagation paths
  (bit-identical when `sens` is empty).
- `ForceModel.propagate` defaults (`integrator="rk"`, no sensitivity
  columns).

### Trade-offs

- Under ephemeris force models IAS15's effective accuracy is capped at
  ~1e-11 relative by ephemeris sampling smoothness (decision 3); purely
  analytic force models are not capped. Higher accuracy requires fixing
  the smoothness of the ephemeris sampling chain first, to be evaluated
  separately.
- Each sensitivity parameter adds 6 to the augmented dimension, so the STM
  path's cost grows linearly; Cr and Cd together give 54 dimensions,
  acceptable.
