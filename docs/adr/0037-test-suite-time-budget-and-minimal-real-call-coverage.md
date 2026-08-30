# ADR 0037: Test suite time budget, minimal real-call coverage, and e2e test boundaries

**Status**: Adopted
**Date**: 2026-08-23
**Related Issues**: #534, #536
**Related**: ADR 0013 (verification strategy), ADR 0020 (failure policy),
ADR 0021 (functional categories & time bounds), ADR 0025 (suite convergence)

## Context

While investigating full-suite runtime and failures (#534), audit found multiple
time-budget violations (single tests >30 s, files >4 min killed by timeout):

1. **Nine deleted orbit-family e2e test files** (2038 lines, ~35 min): directly
   calling `design_*` to generate multi-family real orbits on top of correctors
   at 1e-12 research tolerance + scipy STM propagation (#536's root cause);
2. **`tests/api/test_facade.py` smoke cases**: genuinely generating long-arc
   families one by one — horseshoe (76 s), nrho (46 s), lpo (38 s);
3. **`tests/algorithm/transfer/test_lga.py`**: pure-Python grid search
   (360 departure angles × 5 TOFs) exceeding 4 minutes;
4. **`tests/algorithm/design/` ephemeris-correction tests** (5+ files): direct
   SPICE + differential corrector calls, minutes per file.

The core contradiction producing these slow tests: **ADR 0013 opposes mocks and
ADR 0021 decision 4 requires orchestrators' "one real call", but no executable
standard existed for that call's scale ceiling or a per-test time budget** — so
test authors wrote production-scale parameter sweeps and long-arc family
generation straight into pytest.

## Decision

1. **Test suite time budget**:
   - **Per-test wall-clock ceiling: 10 seconds**;
   - **Per-file wall-clock ceiling: 60 seconds**;
   - Tests exceeding budget stay out of default pytest; shrink problem scale
     (small amplitudes/short arcs/coarse grids/screening tolerances) into budget;
     irreducible ones move to `scripts/` manual diagnostics or benchmarks.

2. **Minimal real-call coverage standard** (interpreting & refining ADR 0021
   rationale 4):
   - `orchestration`/`interface` orchestrator entries (`design_orbit`,
     `transfer_orbit`, `Facade`) **must keep exactly one minimal-scale real-call
     smoke test** proving chain connectivity and return-type contracts;
   - Minimal calls must pick the **cheapest parameter combinations** (small-
     amplitude Halo, planar Lyapunov, loose-perilune NRHO); horseshoe (months-long
     periods), long-arc LPO (T≈21) etc. are strictly forbidden as smoke samples;
   - Exhaustive multi-family generation and grid-density-sensitive physical
     searches aren't smoke material and never enter default suites.

3. **Test tolerance orthogonal to production tolerance**:
   - Default suites uniformly use **screening tolerances**
     (`rtol/atol ≈ 1e-9–1e-10`), never dynamics-benchmark integration's
     `DEFAULT_TOLERANCE = 1e-12`;
   - Correctors & search entries should support overriding propagation tolerance
     via parameters (#536's landing scope).

4. **Computation-sharing invariant**:
   - When multiple assertions in one file depend on the same generated numbers,
     share via `pytest.fixture(scope="module")` or `@functools.cache`; repeatedly
     invoking expensive generation inside test bodies is forbidden.

## Rationale

1. **Speed isn't a correctness category, but time bounds decide whether the
   regression gate is usable** (ADR 0021 #420 basis): half-hour tests dying to
   timeouts provide no deterministic pre-merge protection.
2. **Smoke verifies chain glue, not physical feasibility exhaustion**: horseshoe's
   physical closure and Halo's chain glue are isomorphic at the interface layer;
   smoke needs only the fastest walkable path.
3. **Eliminate ambiguity**: concrete 10s/60s numbers prevent future accumulation
   of e2e recomputation debt.

## Migration plan

1. **Phase 1 (now)**: delete `tests/algorithm/family/`'s nine heavy-compute
   files keeping registry contracts; WSB tests lower tolerance + cache
   (commits `92b798e`, `acb2037`); establish ADR 0037 & CONTEXT.md.
2. **Phase 2 (#536)**: once corrector tolerance configurability lands, restore
   one **<3-second minimal real-call coverage** per family under
   `tests/algorithm/family/`.
3. **Phase 3 (targeted optimization)**: shrink `tests/api/test_facade.py` smoke
   parameters (drop horseshoe/nrho long arcs for small-amplitude samples);
   evaluate sinking `test_lga.py` onto Rust search kernels (WSB pattern).

## Amendment (2026-08-30): end-to-end purge to the API + math-derivation core

The suite had re-accumulated end-to-end debt past the Phase 1–3 scope: real
family generation, ephemeris-correction, force-model propagation, and
production-scale grid searches had crept into the default gate. This amendment
records a purge wave that returns the default suite to the API-contract +
math-derivation core this ADR intends.

**Outcome**: `2164 passed / 29 min` → `1981 passed / 31 skipped / ~37 s`
(xdist `-n auto --dist loadscope`); 49 files changed (19 deleted, 30
surgically shrunk), net −5304 lines. After the wave, **no default-suite test
exceeds the 10 s per-test ceiling** (slowest: NSGA-II parallel-consistency at
~9.6 s, dominated by Windows `ProcessPoolExecutor` spawn — irreducible without
dropping the parallel path the test exists to exercise).

**What was deleted outright** (end-to-end / heavy real computation with the
contract already covered elsewhere): the remaining `design/` ephemeris-
correction and frozen-orbit integration files, whole-force-model propagation
files under `numerical/forces/physics/` and `numerical/forces/container/`
(STM/config), GIL/cache integration-binding probes, and the `tools/` GMAT
full-force-model comparison. Each deletion kept the registry/API contract
tests in the same directories.

**Shrink techniques applied** (reusable for future over-budget tests):

- **Pick the cheap point on a production walk, not a smaller problem.** A
  periodic-orbit design via `_walk_family` converges in one `correct_at` step
  near the family seed and walks many steps far from it — `design_dpo` is
  ~0.3 s at 25000 km (the seed neighborhood) but ~10 s at 20000 km and ~37 s
  at 8000 km. The REQ-003 Jacobi-conservation test moved to the seed-side
  amplitude (the conservation semantics are amplitude-independent) rather than
  shrinking the orbit.
- **Hoist loop-invariant reference computations.** The value-function gradient
  accuracy test recomputed `np.gradient` over the full 4-D grid inside its
  sampling loop; hoisting it out (it does not depend on the sample point) plus
  a coarser grid — which *widens* the spline-vs-central-difference gap the
  test asserts — cut 15.5 s to 0.2 s.
- **Stub the batch boundary, keep the glue assertion.** The Jacobi-window
  grouping test only needs to prove "same-(family, point, params) windows
  share one batch generation call"; it now stubs `generate_rust_family_windows`
  with synthetic results instead of paying a real trace, while the window
  *membership* behavior tests keep real generation.

**Production-side floors found (flagged, deliberately NOT changed — out of
test-surgery scope)**: `MIN_TRACE_MEMBERS = 200` and the 25000 km / 1e-12
defaults in the Rust family-generation path force ≥200-member traces, giving
every catalog window/sweep test a ~5–6 s floor regardless of how few members
the test requests. If the per-test budget tightens further, these knobs (not
the tests) are the lever — they need a maintainer decision on whether tests
may request smaller traces.

**Edge items kept pending adjudication**: `test_dual_instance_sync` (sole
double-CSPICE-instance guard), `test_kernel_future_coverage` pxform regression
(#556), and the seconds-scale minimal real solves (three-body Lambert,
low-thrust shooting, multi-impulse, WSB Rust backend) retained as the ADR 0021
rationale-4 minimal real-call anchors.

**Post-review corrections** (the pre-merge review caught two over-deletions,
restored): (1) the `design_orbit` orchestrator's decision-2 minimal real call
had been removed with the heavy ephemeris-correction files — restored as a
~1.6 s ELFO smoke (`tests/algorithm/design/test_design_orbit_smoke.py`; ELFO is
the cheapest real path — no ephemeris correction, `correction=None`); (2)
`test_low_thrust_variable_mass.py` had lost its only fast analytic comparisons
(mass-consumption / semi-major-axis-rate / Rust-7D-path / zero-thrust guard,
all <2 s) along with the two over-budget SRP cases — the four fast math-derivation
tests are restored, the two SRP cases (17 s / 10 s) stay removed. The
transfer-side anchors listed above never covered `design_orbit`; each
orchestrator entry needs its own minimal real call, which this restores.
