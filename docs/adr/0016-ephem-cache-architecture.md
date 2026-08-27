# ADR 0016: EphemCache ephemeris cache architecture

**Status**: Adopted
**Date**: 2026-08-02
**Related**: ADR 0011 (five-layer architecture), ADR 0013 (verification
strategy)

## Context

e2m2e's force models (`GravityField`, `SRP`, SPK acceleration) query SPICE
ephemerides at every integration step: body positions (`spkpos`) and frame
rotations (`pxform`/`sxform`). The original implementation called cspice FFI
directly, with two problems:

1. **Performance**: multi-year multi-rev shooting (e.g. 2 years × 50 revs ×
   8 nodes × 50 iterations) crossing the FFI boundary for every SPICE lookup
   at every step of every force — call volumes unacceptable.
2. **Concurrency safety**: cspice maintains global state internally (the
   kernel pool); concurrent multithreaded calls trigger `DAFFRNOTFOUND` or
   panics, blocking parallel shooting (rayon `par_iter` segment integration)
   outright.

EphemCache pre-samples ephemerides into in-memory cubic-spline tables before
shooting; the whole integration then reads tables instead of calling cspice FFI.

## Cache coverage

EphemCache caches two kinds of SPICE queries:

| Kind | API | Cache key | Interpolation |
|------|-----|--------|----------|
| Body position | `spkpos` → `lookup_body_position` | `(target, observer, et)` | Cubic spline (`[f64; 3]`) |
| Frame rotation | `pxform`/`sxform` → `lookup_frame_matrix` / `lookup_sxform` | `(from, to, et)` | Per-element cubic spline (`[[f64; 3]; 3]` or `[[f64; 6]; 6]`) |

**Pre-sampling flow**: `EphemCache::build(bodies, frames, sxform_pairs,
et_start, et_end, dt)` samples via cspice over `[et_start, et_end]` at step
`dt` (default 3600 s), building cubic-spline tables. Queries outside the range
are misses.

## Known limitations

**Force models that don't use the cache** — these never query ephemerides:
- `PointMassGravity`: central-body two-body acceleration is analytic, no
  SPICE.
- Thrust models (`FiniteBurn` / `VariableMassFiniteBurn`): no SPICE
  dependency.

**Miss conditions**:
- Without `enable_ephem_cache`, the global cache is `None`; `lookup_*`
  returns `Ok(None)` and callers fall back to cspice (non-strict mode).
- Query epoch `et` outside the sampled `[et_start, et_end]` → miss.
- Requested target/observer or frame pair not in the sampling list → miss.
- Oversized `dt` degrading spline accuracy (current default 3600 s suffices
  for orbital mechanics).

**cspice fallback**: non-strict mode returns `Ok(None)` on miss and force
models fall back to cspice FFI per existing patterns; strict mode returns
`Err(CacheMissError)` propagated upward by callers' `?`.

## Relationship to parallel shooting

Parallel shooting (`multiple_shooting.rs`) is EphemCache's core consumer:

```
Python call chain:
  design_orbit.py
    → spice.enable_ephem_cache(bodies, frames, et0, et_end, dt=3600)
    → shooting_multiple(states, ..., parallel=True)
      → Rust: StrictGuard::new()        # strict on
      → Rust: rayon par_iter segments   # each segment reads cache independently
      → Python: spice.disable_ephem_cache()  # try/finally cleanup
```

**Strict mode** (`StrictGuard` RAII): on throughout shooting. Within scope,
any `lookup_*` miss returns `Err` (hard failure), eliminating silent cspice
fallbacks from force models — cspice is the source of kernel-pool corruption
in parallel regions. `StrictGuard` saves the prior value and restores it on
Drop; nestable.

**E2M2E_MS_PARALLEL environment variable**: `E2M2E_MS_PARALLEL=0` forces
serial segment integration to verify parallel/serial bit-level identity
(`par_iter` order-preserving + deterministic per-segment integration → same
results). Parallel by default.

**Concurrency safety mechanisms**:
- The global cache sits behind `RwLock<Option<EphemCache>>`: read locks run
  concurrently without blocking each other (segments read splines in
  parallel); write locks (enable/disable) exclude readers.
- Cubic-spline interpolation is pure numerics with no cspice FFI — thread-safe.

## Registration flow

```
Python side                                  Rust side
──────────────────────────────────────────────────────────────
spice.enable_ephem_cache(                     enable_ephem_cache()
  bodies=["EARTH","MOON","SUN"],                → EphemCache::build()
  et_start, et_end,                               sample spkpos/pxform/sxform at dt
  dt=3600,                                      → build cubic-spline tables
  observer="EARTH",                             → ephem_cache::enable(cache)
  frame_pairs=[                                   write global RwLock
    ("ITRF93","J2000"),
    ("MOON_PA","J2000"),
  ],
)

# During shooting:
#   force models call lookup_body_position / lookup_frame_matrix
#   → read lock + spline lookup → hit: interpolation / miss: Err(strict)

spice.disable_ephem_cache()                   disable_ephem_cache()
                                                → ephem_cache::disable()
                                                → write lock clears global cache
```

`enable_ephem_cache` is a PyO3-exported function
(`e2m2e-integrators/src/lib.rs`) forwarding parameters to `EphemCache::build`.
Python-side design callers pair `enable/disable` within `try/finally` to avoid
leaking caches into later calls.

## Relationship to ADR 0013 (verification strategy)

ADR 0013 requires completing tasks by definition: test assertions come from
analytic solutions and physical invariants, not external software output.
EphemCache's verification aligns:

1. **Cache-consistency tests**: the same shooting problem run with and without
   cache (`E2M2E_MS_PARALLEL=0` + serial); assert bit-identical final states.
   Verifies the cache changes no results, referencing nothing external.
2. **Spline accuracy tests**: against known analytic trajectories (two-body
   circular orbit), assert interpolation error within tolerance. By definition:
   judged by mathematics.
3. **Strict-mode behavior tests**: miss returns `Err` under strict,
   `Ok(None)` otherwise. Interface-contract testing.
4. **Parallel/serial consistency**: `E2M2E_MS_PARALLEL=0` verifies bit-level
   identity during development — regression means, no external software.

## Revision (2026-08-12, ADR 0020 decision 4)

**Cache-miss semantics: hard failure once enabled.** After explicit
`enable_ephem_cache`, misses (query outside sampled range / pair not in list)
always return `Err`; no more strict/non-strict distinction. Enablement signals
the user asked for caching, so a miss afterward is an error — never a silent
cspice FFI fallback (kernel-pool corruption risk in parallel regions).

**Not-enabled is not a miss**: when the global cache is `None`, `lookup_*`
returns `Ok(None)` and callers fall back to cspice (user didn't ask for
caching; legitimate path). `StrictGuard` (RAII, active in shooting parallel
regions) remains as extra insurance there: hard failures even if caching wasn't
enabled, guaranteeing zero cspice in parallel regions. The old non-strict
miss→`Ok(None)` behavior now applies only to the not-enabled case.
