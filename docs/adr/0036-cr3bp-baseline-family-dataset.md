# ADR 0036: CR3BP baseline orbit-family dataset — precomputed full-family data shipped with the package

**Status**: Adopted
**Date**: 2026-08-23
**Related**: ADR 0031 (orbit catalog), ADR 0029 (unified Rust family
generation), ADR 0014 (interface layer)

## Context

The repo previously shipped no precomputed orbit-family data: all families
computed at runtime (differential correction + continuation, Rust kernels).
Teaching scenarios demand out-of-the-box availability: after installing,
students should browse complete data for nine CR3BP orbit families immediately —
not first learn family-generation APIs.

Measured (2026-08-23, DE421 Earth-Moon, n_orbits=100 cap): single-family
generation takes ~1 s (halo L2 full family 1.5 s, DRO 0.7 s); nine families
total <15 s. `Orbit` members never carry whole trajectories anyway (`states` is
the (1,6) initial state); serializing all nine families by initial-state +
period + scalar diagnostics costs only ~120 KB. Generation is fast, data small —
but fast ≠ offline-available and version-consistent, and small makes packaging
free.

## Decision

### 1. Package-distributed; data (JSON + NPZ) enters repo git

Baseline data (~3.3 MB total: JSON metadata + NPZ segments) commits to
`e2m2e/data/catalog_baseline/` as package data distributed with pip installs.
No Release downloads (too small to justify a distribution step).

> Revision (2026-08-23): the initial decision was no-git + regenerate before
> release; implementation switched to committing data directly. Why: JSON+NPZ
> must pair in-repo for fresh clones to build complete packages; size is small,
> change frequency low (regeneration only on algorithm changes) — git history
> cost acceptable.

### 2. Content: initial states + periods + scalar diagnostics; no trajectories

Each member stores only x0, period, Jacobi, amplitude and other scalar
diagnostics (per ADR 0031 record format's cr3bp_segment). Full trajectory =
initial state + propagator's deterministic derivation; a 3601-point sample
computes on demand.

### 3. Full families = each spec's maximal default coverage

Coverage doesn't grow via configuration: continuation stops naturally at ADR
0029 spec-builtin amplitude-window boundaries / folds / termination conditions;
actual member counts enter honestly (halo 79, DRO 42 are spec-determined, not
defects). Each record's metadata documents actual coverage: amplitude range,
member count, termination reason — fully auditable rather than verbally claimed.

### 4. Scope: nine families × Earth-Moon DE421

HALO, NRHO, AXIAL, LISSAJOUS, DRO, DPO, SPO, HORSESHOE, LPO. Lyapunov has no
standalone family interface (ADR 0029 unregistered) — explicitly excluded;
NRHO/Axial calibrated seeds bind DE421 Earth-Moon, consistent with defaults of
other families. Other μ values await ADR 0029's calibration extension — out of
scope here.

### 5. Shape: standard catalog records; first-use import into user library

The baseline is a batch of ADR 0031 catalog records (one per family), uniformly
tagged `tags: ["baseline"]`. Integration is first-use import: when user catalogs
lack the baseline or versions mismatch, copy from package into the user library
directory and rebuild index; storage engine untouched; everything thereafter uses
existing `catalog_query`. No read-only multi-source mounting: that changes the
engine, and package directories aren't writable/annotatable.

### 6. Freshness: no CI gate; user reports + issues

Post-algorithm-change baseline regeneration relies on manual release-checklist
steps (`make catalog-baseline` rerunning generation & committing new data); no CI
recompute-compare gate. Data doubts go through issue reports → fix → re-release.
Known cost: windows where packaged data lags algorithm output due to forgotten
regeneration — accepted.

### 7. Validation inside the generation flow

Generation script embeds assertions: per-member period-closure error, Jacobi
drift within verification tolerances, member-count floors, coverage metadata
completeness. Failed assertions write nothing. No separate test layer for file
validation.

### 8. Teaching curation deferred

This round delivers full baselines + `baseline` tag only; curated cases &
annotations wait for real teaching consumption scenarios — no designing for
never-consumed requirements.

## Rationale

1. **Into git, shipped with package**: JSON+NPZ ~3.3 MB invisible inside
   packages; paired in-repo ensures any fresh clone builds complete package
   data; download links add distribution+verification steps for zero benefit.
   Algorithm-change regeneration goes `make catalog-baseline` + commit (see
   decision 6).
2. **No trajectories**: size grows ~120 KB → ~180 MB (float64 full sampling) to
   save 7 ms/member on-demand compute — disproportionate.
3. **First-use import over multi-source query**: catalog's value is unified
   querying; adding read-only second sources deep-changes the engine while import
   only copies files + rebuilds index (ADR 0031 guarantees full index
   rebuildability).
4. **No CI gate**: full generation <15 s makes gating technically cheap but CI
   tasks have maintenance costs; the project trusts release checklists +
   user-report (issue) loops.

## Consequences

### Added

- `e2m2e/data/catalog_baseline/`: baseline records (JSON + NPZ, in git,
  package-distributed).
- `scripts/`: baseline generation script (embedded validation assertions;
  writes records + coverage metadata).
- Catalog first-use import logic: detect missing/mismatched baseline → import
  from package.
- Makefile target `catalog-baseline`.

### Changed

- No interface changes; `catalog_query` consumption unchanged.

### Unchanged

- ADR 0031 storage layout, record format, query interfaces — all untouched; the
  baseline is merely a batch of pre-generated records.
- ADR 0029 family-generation specs unchanged; baseline data snapshots their
  outputs.

## Trade-offs

- First-use import puts non-user-created records into user libraries; marked via
  `baseline` tag + baseline version — identifiable, re-importable.
- No CI gate leaves staleness windows (decision 6), mitigated by manual process.
