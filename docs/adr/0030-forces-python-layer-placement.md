# ADR 0030: algorithm/forces stays at the algorithm layer — Python config/orchestration surface, numerics in crates

**Status**: Adopted
**Date**: 2026-08-17
**Related Issue**: #429 (forces source-layer ownership)
**Related**: ADR 0011 (five-layer architecture), ADR 0012 (dependency
direction), ADR 0004 (ForceModel config-driven), ADR 0021 (test functional
categories), ADR 0026 (follow-ups — forces ownership; this entry grew from that
audit), ADR 0027 (same-type precedent: evaluated and retained + ADR)

## Context

ADR 0026 consolidated forces tests into `tests/numerical/forces` but left
source-layer ownership to #429: the `e2m2e/algorithm/forces` Python package
self-describes as parameter validation + to_rust_spec serialization,
corresponding to Rust `e2m2e-forces`'s `CompiledForce`; the migration ledger
already records forces numerics as sunk. An audit could still read tests at
numerical, sources at algorithm, and self-description resembling integrators as
misplaced — then ask: should it move out of algorithm? If so, should numerical-
layer Python take the shape of a new `e2m2e/numerical/`, or sit beside
`integrators.py`?

#429's constraints state: the five-layer architecture's numerical layer is
`crates/`; Python currently has only thin binding wrappers, no numerics
directory. If status quo holds, explain why and close. This entry records the
ruling.

## Decision

1. **`e2m2e/algorithm/forces` stays put, in the algorithm layer.** The Python
   side is force-model configuration & orchestration surface, not a numeric
   core.
2. **No new `e2m2e/numerical/` (or any Python numerical-layer directory).**
   The five-layer architecture's numerical layer remains `crates/`; no sixth
   layer gets invented for this.
3. **Forces doesn't move beside `integrators.py`, nor get flattened into a
   top-level single file/thin module.** forces is a config-driven domain
   package (ADR 0004), different in kind from binding re-exports.

## Rationale

### Responsibility: computation vs orchestration already separated

Force-model numerics (spherical harmonics, tides, SRP, third-body, drag, STM…)
live in the `e2m2e-forces` crate. Python's `PhysicalModel` / `ForceModel` do only
parameter validation, `to_rust_spec` serialization, config round-trips, and post-
construction calls into `integrators`' compiled propagation entries; accelerations
and Jacobians keep no Python reference implementations.

This matches architecture docs and README's division: crates compute; Python
constructs problems, calls Rust, interprets results.
`numerics-migration-status` already marks `algorithm/forces` (numerics) as sunk.
This decision doesn't change responsibility division — only **directory
ownership**: numerics already sunk and where the Python config surface lives are
two separate matters.

### Structure: why not out of algorithm

1. **Consumers are in algorithm.** design, station_keeping, transfer,
   propagation all construct force models inside algorithm chains and propagate.
   Config surface and domain orchestration share a layer, avoiding cross-layer
   moves done purely for directory symmetry.
2. **Real runtime dependencies on algorithm exist.** The config surface depends
   on System context and coordinate (frames, axes, origins); containers call
   integrators. Moving wholesale out of algorithm would either create
   numerical-side → algorithm reverse dependencies (violating ADR 0012) or
   force System/coordinate extraction too — a #430-scale migration, with
   #430/ADR 0027 having already ruled System stays in algorithm.
3. **Five layers have no Python-numerics directory slot.** ADR 0011's numerical
   layer is `crates/`; Python's numerics entry is only the parallel thin binding
   wrapper (`integrators.py` → `_integrators`); there is no `e2m2e/numerical/`.
   forces is a multi-type domain package carrying config schema and containers —
   not a single-file re-export; flattening beside `integrators.py` would crush
   the package structure. A sixth layer changes looks only, zero behavior gain.
4. **Test directories can't back-infer source layers.** ADR 0021/0026:
   functional-class markers and code layering are two axes.
   `tests/numerical/forces` verifies Rust force-model numeric contracts — not a
   demand that Python sources enter numerical. Same root as coordinate tests
   marked `data` while source stays algorithm.

### Why alternatives were rejected

**Moving out of algorithm (beside integrators, or new numerical/).** Would first
require defining the numerical-layer Python shape that isn't written into the
five layers; untangling System/coordinate dependencies or accepting violating
ones; touching many algorithm consumers and many test imports. Behavior
unchanged; the sole benefit is directory symmetry.

**Flattening to top-level single file/thin module because it resembles
integrators.** forces is a config-driven domain package (ADR 0004), not a
binding-layer re-export; similar duty ≠ comparable form.

## Consequences

### Added

- This ADR.

### Changed

- forces entry in `docs/architecture/numerics-migration-status.md`: from "to be
  independently assessed by #429" to citing this ADR's adjudicated statement.

### Unchanged

- `e2m2e/algorithm/forces` directory, interfaces, implementation, test paths —
  untouched line by line.
- Numerics in `e2m2e-forces`, Python config/orchestration only — unchanged.
- ADR 0011 five-layer architecture & ADR 0012 dependency-direction texts.

### Costs

- Test tree stays `tests/numerical/forces`; source tree stays
  `e2m2e/algorithm/forces`. The two axes (functional class vs layer) coexist
  and invite misreading; this ADR shares explanatory duty with ADR 0026 to
  prevent repeat misjudgments.
