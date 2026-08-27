# ADR 0017: Transfer grid search — pure-numerics kernel sunk to Rayon

**Status**: Adopted
**Date**: 2026-08-07
**Related**: ADR 0002 (Rust integrator core), ADR 0011 (five-layer
architecture), ADR 0012 (dependency direction), ADR 0013 (verification
strategy), ADR 0016 (EphemCache architecture)

## Context

The DRO→RO transfer search+optimize two-step method (Cui et al. 2025): step
one forward-integrates point-by-point over an α (tangential velocity ratio) ×
n_departure (departure epoch) grid with geometric feasibility screening,
producing a candidate set. Each grid point in the search phase is fully
independent with no cross-point state — a textbook pure-numerics kernel that
iterates on fed numbers.

Search previously parallelized with Python `ProcessPoolExecutor` /
`ThreadPoolExecutor`. The bottleneck wasn't integration itself (already Rust's
`propagate_cr3bp`) but two Python-side costs:

1. **Cross-process pickling**: the `processes` backend pickles config tuples +
   `departure_state` + `arrival_states` arrays into every subprocess, growing
   linearly with departure count.
2. **Per-α Python loop overhead**: inside `search_single_departure`'s inner α
   loop, `detect_local_minimum`'s Python for-loops, `dict` assembly,
   `np.concatenate`/`np.linalg.norm` per-point overhead are the real burden of
   the `threads` backend: they hold the GIL, so threads never truly
   parallelize.

Benchmarks confirmed it: at large grids (320 evaluations) `threads` was twice
as slow as `processes` (GIL serializing Python loop overhead plus thread-
switching cost); see benchmark data at the end.

## Decision

Sink the search phase's **6-step evaluation unit** (once per α, purely
numerical, no domain knowledge) plus **grid distribution** wholesale into
Rust, replacing Python process/thread pools with Rayon `par_iter`.

6-step evaluation unit (sinking list):

| Step | Content | Sinking notes |
|---|---|---|
| 1 Departure velocity composition | `v_mag`, tangential/normal unit vectors, `alpha·v_mag·t_hat` | nalgebra; precompute/cache α-independent quantities |
| 2 Forward propagation | `propagate_cr3bp` | **direct pure-Rust call** (not via GIL-holding `propagate_cr3bp_py`) |
| 3 Collision detection | distance to earth/moon centers | vectorized |
| 4 Distance series | n_traj×n_orbit broadcast + argmin | ndarray parallel |
| 5 Intersection/local minima | `detect_intersection` + `detect_local_minimum` | local minimum vectorized (compare neighbors) |
| 6 Result assembly | scalar/array/state aggregation | `TransferPointResult` pyclass |

**Parallel architecture** copies the order-preserving pattern from
`multiple_shooting.rs:465-531`:

```rust
#[pyfunction]
fn transfer_grid_search_py(...) -> Vec<TransferPointResult> {
    py.allow_threads(move || {
        (0..n_dep * n_alpha).into_par_iter()
            .map(|idx| evaluate_point(idx, ...))  // directly call pure-Rust propagate_cr3bp
            .collect::<Vec<_>>()
    })
}
```

- **GIL release**: `py.allow_threads` wraps the outer layer
  (`multiple_shooting.rs:810-828` template).
- **Direct pure-Rust call**: workers call
  `e2m2e_forces::cr3bp::propagate_cr3bp`, **never routing through
  `propagate_cr3bp_py`**: it holds the GIL and would make Rayon effectively
  serial (the easiest trap in this design).
- **Order-preserving bit-level identity**: `par_iter` + `collect` preserve
  order; `E2M2E_SEARCH_PARALLEL=0` forces serial mode for comparison
  (reusing `E2M2E_MS_PARALLEL`'s verification pattern).
- **CR3BP has no SPICE FFI**: pure math needs neither `multiple_shooting`'s
  `StrictGuard` nor `ephem_cache`; rayon safety preconditions are simpler than
  multiple shooting's.

## Precedents

- **`multiple_shooting.rs:528`** `(0..n_seg).into_par_iter()`: this repo's
  landed numerical-layer `par_iter` + `allow_threads` + env-var toggle pattern.
  This ADR's search sinking reuses that pattern, extending the env var from
  `E2M2E_MS_PARALLEL` symmetrically to `E2M2E_SEARCH_PARALLEL`.
- **ADR 0002 revision 2 (`propagate_compiled`)**: propagation entering Rust
  was a **special case** forced by the cspice kernel-pool singleton constraint
  (SPICE-related propagation must compile with force models into one
  extension). This ADR is the **other kind of precedent**: pure-numerics
  atomic sinking; the evaluation unit has no SPICE and sinks for performance
  (eliminating Python scheduling/loop overhead), not concurrency safety.

Together they extend ADR 0002's Rust-kernel boundary from single-step
integration / single-segment shooting to grid evaluation units.

## Boundaries (fixed)

Sinking touches only the search phase's pure-numerical evaluation units. The
following explicitly stay in Python:

- **Orchestration stays in Python**: `TransferSearch` (parameter management,
  `search`/`optimize` entries, feasibility filtering), `dispatch_grid_search`
  (backend dispatch), `set_parallel_backend` (backend validation/routing).
- **NLP optimization stays in Python**: SLSQP / COPT serial iteration is
  Python's strength (early architecture consensus); multi-candidate parallelism
  uses `ProcessPoolExecutor` — outside this ADR.
- **Geometry thin-wrappers retained**: the six on `TransferSearch`
  (`_forward_integrate` / `_check_collision` / `_compute_distance_series` /
  `_detect_intersection` / `_detect_local_minimum` /
  `_compute_min_distance`) are Python-side's **only dispatch seam**, kept for
  monkeypatch compatibility + numpy reference benchmarks.
- **CR3BP/BCR4BP pure-math paths need no ephem_cache**: `transfer_grid_search`
  and WSB's BCR4BP search call Rust pure-math propagators and can safely use
  Rayon; ephemeris paths (`EphemerisDynamics`) still need `ephem_cache` +
  `StrictGuard` (ADR 0016), handled under their own concurrency boundary.
- **low-thrust / porkchop / nsga2**: may reuse this infrastructure (same
  multiprocessing→rayon pattern) in later separate migrations; WSB was already
  sunk by #447 as an independent BCR4BP Rust/Rayon numeric kernel.

## Architecture compliance

- **ADR 0011 (five layers)**: the search evaluation unit is a pure-numerics
  atom within numerical-layer responsibility (integration + geometry), not
  algorithm orchestration sinking. Orchestration (`TransferSearch`) remains at
  the algorithm layer.
- **ADR 0002 (Rust boundary)**: what sinks is feed-numbers-and-iterate
  numerical evaluation, not orchestration. Citing the `multiple_shooting` +
  `propagate_compiled` precedents extends the Rust kernel boundary to grid
  evaluation units.
- **ADR 0012 (dependency direction)**: Python algorithm layer
  (`algorithm/transfer/`) calling Rust numerical layer (`crates/`) is the
  legal direction.
- **ADR 0013 (verification strategy)**: dual backends coexist; Python
  algorithm unit tests remain (thin-wrapper dispatch seam keeps monkeypatch
  tests unchanged); Rust gains equivalence comparisons
  (`test_rust_backend_equivalence` per-candidate per-field;
  `test_geometry_rust_vs_numpy` per geometry function) — no external software.

## Dual-backend coexistence and the monkeypatch seam

`search(parallel_backend='rust')` requires both: Rust extension built, and
geometry methods not monkeypatched. Either failing falls back to the Python
path automatically (correct results, just slower):

- **Monkeypatch seam**: tests inject synthetic trajectories via
  `monkeypatch.setattr(TransferSearch, "_forward_integrate", ...)`. The Rust
  kernel bypasses Python method dispatch so patches wouldn't apply;
  `_geometry_methods_monkeypatched` detects `__qualname__` deviation and falls
  back to Python, preserving test semantics.
- **Missing Rust extension**: `grid_search_rust` raises `RuntimeError`
  (`transfer_grid_search_py` is None); fall back to `processes`.

All 12 existing search tests (including 4 monkeypatches) pass unchanged.

## Benchmark data

Three scales uniformly `n_workers=4`; each scale/backend runs 3 times taking
median wall-time (48-core machine, CR3BP Earth-Moon, DOP853 rtol=atol=1e-9).
Benchmark script: `scripts/benchmark_transfer_search.py`, reproducible.

| Scale (dep×α) | Evals | processes(s) | threads(s) | rust(s) | rust vs processes |
|---|---|---|---|---|---|
| Small (2×3)   | 6   | 0.028 | 0.029 | 0.005 | 5.50× |
| Medium (8×10)  | 80  | 0.112 | 0.202 | 0.015 | 7.52× |
| Large (16×20) | 320 | 0.438 | 0.913 | 0.042 | 10.35× |

**Readings**:

- Rust speedup grows with grid size (5.5× → 7.5× → 10.4×). Python's per-α
  loop overhead grows linearly with points while rayon scheduling is near-zero
  overhead and loop-free — more points, more absolute time saved.
- `threads` slower than even `processes` at medium/large grids (0.55× / 0.48×):
  per-α Python loops hold the GIL, so threads never truly parallelize while
  adding switching cost. Confirms the bottleneck is Python loops, not
  integration.

## Consequences

### Added

- Search evaluation unit's Rust implementation under `crates/e2m2e-forces/`
  (`transfer_geometry` geometric kernel + `transfer_grid_search` grid
  distribution + Rayon parallelism; pyfunction wrapper in `e2m2e-integrators`).
- Python thin wrappers `grid_search_rust` / `grid_search_rust_serial`
  (re-exported via `e2m2e.integrators`).
- `dispatch_grid_search` third branch `grid_search_rust_dispatch` (POD input
  flattening + monkeypatch fallback detection).
- `'rust'` added to `set_parallel_backend`'s validated set.
- Equivalence tests (`test_rust_backend_equivalence`,
  `test_geometry_rust_vs_numpy`, `test_rust_backend_via_search`).

### Unchanged

- `TransferSearch` public API, `search()`/`optimize()` signatures & behavior.
- `processes` / `threads` / sequential Python execution paths (kept for
  testing and fallback).
- All 12 existing search tests (incl. 4 monkeypatches) unchanged.
- NLP optimization phase (SLSQP / COPT, staying in Python).

### Trade-offs

- **New Rust maintenance surface**: `transfer_geometry` +
  `transfer_grid_search` are new Rust code to maintain; geometry functions must
  stay equivalent to numpy references (guarded by
  `test_geometry_rust_vs_numpy`).
- **Progress granularity regression**: the rust backend callbacks at departure
  granularity (per completed departure, not per α), unlike processes/threads'
  per-α tqdm: per-α FFI crossings would negate throughput.

## Revision (2026-08-12, ADR 0020 decision 4)

- **Explicitly chosen but unavailable rust now errors**: previously missing
  Rust extension → `grid_search_rust` raised `RuntimeError` and fell back to
  `processes`; now it errors outright (issue #378). Default
  `_default_parallel_backend` stays `rust` regardless of extension presence —
  the parallel model doesn't silently change; `processes`/`threads` only used
  when callers choose them explicitly.
- **Monkeypatch-seam exemption retained**: falling back to the Python path when
  geometry methods are monkeypatched (test `setattr` injection of synthetic
  trajectories) remains, restricted to test paths (`_geometry_methods_
  monkeypatched` detection); production paths never trigger it (ADR 0020
  decision 4's test-injection seam exemption).
