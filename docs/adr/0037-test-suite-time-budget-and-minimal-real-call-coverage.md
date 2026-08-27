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
