# ADR 0031: Orbit catalog — record format, storage layout, query interface

**Status**: Adopted (decision 4 overturned by ADR 0045; decisions 1, 2, 5
revised by ADR 0045; decision 7 revised by ADR 0043)
**Date**: 2026-08-19
**Related**: `docs/architecture/architecture.md` (§5 data management), ADR 0014
(interface Facade/MCP/CLI), ADR 0024 (unified result status contract),
ADR 0029 (unified Rust family generation); transfer-orbit-design ADR 0008
(output/ persistence — its scope revised by this entry); ADR 0013 (ephemeris
visualization's four-slot contract)

## Context

Orbits keep accumulating while artifacts pile up chaotically. Status quo:
results dumped under output/ with timestamp filenames; classification guessed
after the fact from subdirectory names plus filename regexes; a single orbit_type
field for all dimensions; lineage like "this orbit fed station keeping" lives
only in GUI memory and dies at process exit. Such file heaps can't serve two use
cases: data analysis (filter/statistics by family, energy, geometry) and teaching
(curated cases, annotations, packaged subset distribution).

Architecture doc §5 filed this long ago: the data management module will later
extend toward intelligent game-theoretic research as big-data foundation, noting
current npz artifacts have no schema, no versioning, needing detailed design.
This entry delivers that design. Timing is also right: ADR 0024 is executing a
breaking migration of persistence formats — legacy artifact conventions deleted
wholesale with no migration — so the new format carries no compatibility burden.

The classification system references Hu Jiaxin et al., "Intelligent Simulation
System Architecture and Implementation for Cislunar Space Awareness Mission
Design and Analysis" (Journal of Image and Graphics, 2025)'s orbital dynamics
model library: it organizes 4 classes / 27 kinds of three-body orbits along
dynamics-model × orbit-type dimensions. Its core idea — one orbit simultaneously
belonging to multiple independently filterable dimensions — is this entry's
classification prototype.

## Decision

### 1. Record = JSON metadata + NPZ arrays, dual segments

A catalog record = one JSON (full metadata) + same-named NPZ (array segments).
Orbit data keeps both:

- `cr3bp_segment`: Orbit-native types (nondimensional states/times);
- `ephemeris_segment`: EphemerisTable (GCRS, synodic, times_et).

Either segment may be null: station-keeping products carry only ephemeris;
periodic family members only CR3BP (initial state + period). Records carry
`schema_version`, starting at 1.

Implementation note (2026-08-19): beyond dual segments, station-keeping records
additionally carry a small `result/` array segment (maneuver sequences,
sk_rows statistics) — part of the station-keeping product without which records
are incomplete; tiny, not a third orbit-data segment.

### 2. Classification is a multi-dimensional field set, not a single-valued type

Each record's dimensions, filled by the algorithm layer at generation time,
each independently filterable:

| Field | Meaning |
|---|---|
| `orbit_family` | dro / halo / nrho / lyapunov / lissajous / dpo / axial / ro / spo / lpo / horseshoe etc. |
| `libration_point` | 1–5 |
| `jacobi` | CR3BP-segment Jacobi constant |
| `amplitude` | principal amplitude (km) |
| `has_cr3bp` / `has_ephemeris` | segment presence |

No single-valued dynamics-model field: dual segments make every record naturally
cross-model; filtering semantics express via segment-presence boolean combos.
Result status follows ADR 0024's triple (status/cause/message). Raw request models
are saved as JSON snapshots with records (`request`) — traceable, recomputable;
mission scalars (epoch, duration, mu, iteration counts…) save alongside result
segments. Classification fields come from the algorithm layer, never inferred
afterwards: during shooting/continuation, family type, libration point, energy are
all known quantities.

Implementation note (2026-08-19): the record builder (`api/catalog_ingest.py`)
actually lands at the interface-layer Facade seam: request snapshots depend on api
request models whose ownership (ADR 0012) forbids algorithm-layer holding;
"filled at generation time" means filled synchronously when computation completes,
distinct from after-the-fact inference from filenames/paths — decision intent
unchanged. Also: amplitude is defined differently per family (Halo's z0, SPO's
radial distance, NRHO doesn't use amplitude directly); the classification field
`amplitude` unifies as **geometric principal amplitude** (max half-range of CR3BP-
segment position components × characteristic length, km), measured from known
geometry at generation; request-level parameter amplitudes stay in the `request`
snapshot.

### 3. Lineage written at generation time

`source_record_id` (nullable) written by the algorithm layer when producing
results: station-keeping products point at controlled orbits; lifted members point
at their families. Lineage no longer depends on caller memory.

### 4. Family granularity: one record per family

An entire OrbitFamily is one record; member parameters & arrays inside. Members
may lift into standalone records (`source_record_id` → family) for downstream
consumption such as station keeping.

*(Revision note 2026-09-01, #611: overturned by ADR 0045 decisions 1 and 2 —
one record carries exactly one trajectory, and a family becomes a label
(`orbit_family` + `family_id` + `member_index`) queried by filter. `member_count`
and `members[]` are removed at schema version 2, and `catalog_promote` — the
member-lifting method this decision motivated — is removed with it. The two
reasons recorded here, flooded query results and homeless family-level
quantities, are answered there by index filters and by per-member run
provenance.)*

### 5. Storage layout: flat record files + SQLite derived index

```
catalog/
├── records/<record_id>.json + <record_id>.npz   # source of truth
└── catalog.db                                    # SQLite index, derived
```

Record files are the source of truth; catalog.db stores only filter dimensions +
file pointers, deletable and fully rebuilt by scanning records/. Directories bear
no classification duty: no family subdirectories; filename = record_id. Storage dir
injects via Config (ADR 0014 decision 7).

### 6. Teaching annotations travel with JSON

`tags` (string list) + `note` (free text) live in the JSON record. Subset exports
carry annotations in files, independent of local db.

### 7. Query & batch generation enter Facade

New Facade methods: `catalog_query` (multi-dimensional filters → summary list),
`catalog_get`, `catalog_delete`, `catalog_tag`, `catalog_export` (subset
packaging), `catalog_sweep` (parameter-space scan batch generation + ingestion;
orchestration reuses ADR 0029's Rust family generation). CLI and MCP derive
automatically per ADR 0014 pure derivation.

*(Revision note 2026-09-01, #610: these methods move off `Facade` onto the
catalog interface class of ADR 0043 decision 2, joined there by
`orbit_family_generation` and by `catalog_terminology` (ADR 0044). Tool names,
schemas and pure derivation are unchanged; only the holding class changes.)*

### 8. Automatic ingestion

Facade's artifact-producing methods (design_orbit, orbit_family_generation,
control_orbit etc.) auto-ingest on success — callers never explicitly request.
Config can disable (testing).

### 9. No migration of legacy artifacts

All old-format products under output/ get deleted — no migration tool, no
compatibility reads.

## Rationale

1. **Dual segments, not forced EphemerisTable unification**: CR3BP initial guesses
   vs ephemerides are products of different dynamical models; one-way coercion
   loses guesses' correctability (differential correction consumes nondimensional
   states). Coexistence also matches transfer-orbit-design ADR 0013's four-slot
   visualization contract — GUI consumption unchanged.
2. **Multi-dimensional fields, not single-valued orbit_type**: classification's
   value lies in filter combinations (has-ephemeris AND L2 AND Jacobi 3.0–3.1
   NRHO-type conditions) that single values can't express. Fields from the
   algorithm layer rather than retro-guesses: generation time has maximal
   information; after-the-fact inference is precisely today's chaos source.
3. **SQLite as derived index, not source of truth**: files-as-source keep manual
   copy/backup/sharing and CLI+GUI sharing; index is query acceleration only —
   corrupted or format-upgraded, delete and rebuild; query-dimension changes never
   touch record files.
4. **Flat directories**: path-encoded classification is today's chronic disease:
   renames lose types; regex misses lose products. With classification moved into
   the index, directory organization goes free.
5. **One record per family**: families hold tens-to-hundreds of members —
   individual records would flood query results; family semantics (continuation
   parameter, amplitude sequence) would otherwise have nowhere to live.
6. **Automatic ingestion**: catalog value = everything computed is there;
   explicit ingestion is the step callers forget.
7. **Annotations with JSON**: teaching distribution units are file subsets;
   annotations must travel with data.
8. **No migration**: few legacy products, low retention value; tool cost exceeds
   product value. ADR 0024 already established fail-on-read + hint-migration for
   old formats; this entry goes further — not even migration hints.

## Consequences

### Added

- `data/catalog/`: record types, storage engine (write/read/index rebuild),
  SQLite index.
- `algorithm/`: catalog_sweep parameter-scan orchestration.
- `api/`: Facade catalog methods & Pydantic models; record builder
  (`catalog_ingest.py`, see decision 2 implementation note); CLI/MCP auto-derived.

### Changed

- Architecture doc §5's planned design lands; artifact formats enter schema'd,
  versioned state.
- Facade artifact methods gain auto-ingest side effects.

### Unchanged

- Orbit/OrbitFamily/EphemerisTable domain-data duties (ADR 0024); catalog records
  reference them, never replace them.
- `algorithm/normal_form/catalog.py` (libration-point parameter-catalog
  transformer) and this orbit catalog are different concepts; original name kept;
  docs reserve "orbit catalog" for this entry's catalog.

### Handover

- transfer-orbit-design's output/ scanning (discovery.py) deprecated; its ADR 0008
  revised: artifact lists now come via Facade catalog queries — files-as-source-of-
  truth unchanged.

## Trade-offs

- Dual segments roughly double each design-product record's size (two
  representations of one orbit) for lossless information and zero-conversion
  consumers; station-keeping products already had ephemeris-only segments —
  unaffected.
- Auto-ingest gives pure computation calls file-I/O side effects; Config-injected
  directory + disable switch keep testing unaffected.
- SQLite adds a binary file — but derived, never format lock-in.
