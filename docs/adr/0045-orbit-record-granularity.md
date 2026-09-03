# ADR 0045: Orbit record granularity — one record per trajectory, family as label

**Status**: Adopted (implemented — schema v2 with per-member records,
family_id/member_index labels, family_id filtering, promote removal, and the
bundle-expanding first-use baseline import; measured: 592 members expand in
~7 s once, idempotent re-opens skip)
**Date**: 2026-09-01
**Related Issue**: #611
**Related**: ADR 0024 (result status contract), ADR 0031 (decision 4 overturned
here; decisions 1, 2, 5 revised), ADR 0036 (baseline dataset), ADR 0037 (test
budget), ADR 0040 (transfer trajectory contract), ADR 0042 (taxonomy stamping),
ADR 0043 (interface class split), ADR 0044 (terminology list).

## Context

ADR 0031 decision 4 made an entire orbit family one record, with member
parameters and arrays inside. The catalog therefore holds two container shapes —
a bundle and a single trajectory — and nothing states how to tell them apart.

Four symptoms follow. `member_count` carries three meanings (0 for
transfer and station-keeping products, 1 for a single orbit, ≥2 for a family),
written down only in a field description. Callers reconstruct the container rule
themselves: transfer-orbit-design keeps it in TypeScript and in Python, kept in
step by shared JSON cases. `catalog_promote` exists only to lift a member out of
its bundle so downstream tools can consume it. And transfer records report
`has_cr3bp=False, has_ephemeris=False` while their trajectory sits in a
`transfer/` segment, so a generic reader that trusts the flags concludes there is
no trajectory data.

The bundle was chosen to keep query results small and to give family-level
quantities a home. Both needs have since been met by other means: the SQLite
index filters records, and a generation run can stamp its identity on every
member it writes.

## Decision

### 1. One record carries exactly one trajectory

A catalog record is an **orbit record**: one mission orbit or one transfer
orbit (CONTEXT.md, catalog data model). No record contains a collection of
trajectories.

### 2. A family is a label, not a container

Family membership is stamped on each orbit record as three fields:
`orbit_family` (the family name, already present), `family_id` (identity of the
generation run) and `member_index` (position in the continuation walk).
Retrieving a family is a query by `family_id`; grouping for display is the
caller's presentation step. Run-level provenance — the request snapshot,
requested and generated counts, run scalars — is replicated on every member
record of that run: one writer per run, so the copies cannot drift.

`orbit_family` alone is insufficient, since two independent runs of the same
family would be indistinguishable; `family_id` carries run identity.

### 3. `member_count` and `members[]` are removed; schema version 2

Both fields disappear from records, summaries and the index. `SCHEMA_VERSION`
becomes 2. Legacy records are not migrated and not read — deleted wholesale,
following ADR 0031 decision 9.

### 4. Content discrimination reduces to one rule

`transfer_type` non-null means the record is a transfer orbit; otherwise it is a
mission orbit. This is the only derivation a caller needs, and it is specified
here rather than left to be reconstructed.

### 5. `catalog_promote` is removed

With every member already a record, member promotion has no meaning. Its
removal lands with this ADR, not with ADR 0043, so that in-process callers keep
working until the granularity change ships. Annotation needs are served by
`catalog_tag`.

### 6. Field and segment specification

The record specification states, for every summary field, its meaning, unit and
null semantics; for every closed-value field, the terminology module of ADR 0044
as its source; and for every array segment, its key prefix and when it is
present:

- `cr3bp/` — nondimensional state and period of a mission orbit;
- `eph/` — an EphemerisTable (GCRS, synodic, times_et);
- `transfer/` — a transfer trajectory (states, times, and optionally
  `states_gcrs_km`), per ADR 0040;
- `result/` — station-keeping maneuver sequences and statistics.

`has_cr3bp` and `has_ephemeris` describe the `cr3bp/` and `eph/` segments only.
They must not be read as "this record carries no trajectory data": a transfer
orbit has both flags false and a populated `transfer/` segment.

### 7. Write policy

Each orbit record is written atomically (temporary file, then rename). There is
no cross-record transaction: a continuation walk that fails midway leaves its
converged members as records, which is the honest partial result. The SQLite
index is rebuilt by scanning `records/`, so a partial write leaves no
inconsistency that a rebuild cannot repair.

### 8. The baseline dataset ships as a distribution bundle

The packaged baseline (13 bundled files, 592 members, 3.5 MB today) stays
bundled for distribution and is expanded into 592 orbit records on first use.
The bundle is a transport and compression unit, not an orbit record; the library
still holds exactly one record shape. Shipping 592 expanded records — about
1 184 files — inside the wheel is rejected.

## Rationale

1. **The two reasons behind decision 4 now have better answers.** Flooded query
   results are a display concern, answered by index filters and `family_id`
   grouping. Family-level quantities are answered by stamping run identity and
   provenance on each member.
2. **Uniformity removes three debts at once**: the three meanings of
   `member_count`, the container rule copied into two languages downstream, and
   the existence of `catalog_promote`.
3. **Station-keeping consumption gets simpler**: a controlled run points at a
   member record directly, with no promotion step in between.
4. **Distribution and storage are different problems.** Keeping them separate
   preserves both a small wheel and a single in-library record shape.
5. **Partial results are preferable to all-or-nothing.** A 79-member walk that
   fails at member 40 currently yields one record describing a partial family;
   it will yield 40 records, each individually valid.

## Consequences

### Added

- `family_id` and `member_index` on records, summaries and the index; a baseline
  expansion step; a granularity contract test asserting that a record carries
  one trajectory.

### Changed

- ADR 0031 decision 4 is overturned; decision 1 (record contents), decision 2
  (classification fields) and decision 5 (index columns) are revised
  accordingly.
- Ingest builders write one record per converged member instead of one bundle.
- `catalog_query` results grow by roughly the family member count; callers
  filter by `family_id`.
- transfer-orbit-design rewrites its catalog data model handling: container
  discrimination, member-count display and the promote path.
- Baseline-related tests move to small subsets to stay inside the ADR 0037
  budget of 10 s per case and 60 s per file.

### Unchanged

- Record files as the source of truth, SQLite as a derived and rebuildable
  index, storage layout, auto-ingest (ADR 0031 decisions 5, 8), the result
  status contract (ADR 0024), and per-record taxonomy stamping (ADR 0042).

### Cost

- File count in a working catalog grows by roughly the average family size; a
  single `catalog_sweep` may write several hundred files.
- The packaged baseline must be regenerated, and the expansion step is new code
  on the read path.

## Implementation note (2026-09-01, #611)

Decision 3's "legacy records deleted wholesale" lands as fail-on-open, not
auto-deletion: a v1 record encountered during index rebuild raises with the
standing instruction ("旧产物不兼容读取，请删除后重算"), and the user deletes
their catalog directory manually. Silently deleting user data on upgrade was
rejected. The bundled v1 family files themselves are untouched — they are the
frozen transport format of decision 8, read structurally (no v2 validation)
and expanded by `data/catalog/bundle.py`.
