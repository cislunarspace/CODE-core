# ADR 0021: Test suite organized by functional categories

**Status**: Adopted
**Date**: 2026-08-09
**Related**: ADR 0013 (verification strategy), ADR 0011 (five-layer
architecture)

## Context

ADR 0013 established correctness by physical definition, appending one clause
of test tiering: Rust units → Python units → integration → physical
invariants. The tiering never landed:

- Of ~203 test files repo-wide, only 43 carry explicit tier markers; `l2`
  appears just twice — "default L1+L2" is empty talk.
- `e2e`/`l3`/`slow` are three markers for the same concept (slow/integration),
  used sporadically.
- CI runs no tests (lint+mypy+inter-layer import checks only); tiering serves
  only local intuition.
- `tests/core/` and `tests/algorithms/` (plural) mirror source packages
  deleted during five-layer migration; tests organize around dead structure.

## Decision

1. **The classification axis changes from speed/integration depth to "what is
   verified"**: closed set of 7: `theory` (math/physics theory), `integrator`,
   `force`, `data` (data layer: kernels/frames/types/IO/templates + coordinate
   conversion), `orchestration` (layer-3 algorithm orchestration),
   `interface` (layer-4 facade), `aux` (tools/helpers). Exactly 1 primary class
   per test.
2. **Directories mirror source structure** (navigation), **markers state
   functional classes** (what's verified), unfinished features use independent
   markers to gate tests (e.g. `low_thrust`); no speed tiering is rebuilt.
3. **Abolish `l1`/`l2`/`l3`/`l4`/`e2e` and `slow` speed tiers**; `addopts`
   excludes only the not-yet-complete `low_thrust` feature.
4. **CI keeps static gates** (format/style/types/inter-layer imports);
   **full test suite runs before release**.
5. **`tests/` reorganized by the five layers** (`data/`, `numerical/`,
   `algorithm/`, `api/`, `tools/`, `mbse/`, `_meta/`), removing dead structure.

## Rationale

1. Speed is not a correctness category — how slowly something runs doesn't
   change what it proves.
2. Tiering existed to skip slow tests per PR; CI runs no tests, so tiering lost
   its reason to exist.
3. Directory-mirrors-source → predictable location of module X's tests;
   functional classes via markers → no scattering of a single module's tests.
4. Orchestrator APIs (`transfer_orbit`/`design_orbit`) naturally require one
   real call for API correctness; classify by assertion into
   `orchestration`/`interface`; `e2e` as a category dissolves (ADR 0013's anti-
   mock stance leaves no middle ground).
5. Data layer (containers/IO/templates) verifies data structures and defaults,
   independent of physics and orchestration — its own `data` class.

## Consequences

- `pyproject.toml` markers become 7 classes + `spice`; `addopts` excludes only
  `low_thrust`; no speed tiering; full run before release.
- Migration in three PRs: ① pure `git mv` moves (history kept, logic unchanged,
  markers untouched); ② per-file functional markers replacing l1–l4/e2e;
  ③ structural debt cleanup (private-symbol tests, golden/gmat/dfh terminology,
  #358 classification).

## Migration checklist (guarding against dead references)

Once directories under `tests/` are deleted, strings in sources and tests
pointing at them become dead references: `linked_tests` tracking, comments,
docstrings may all hide them. CI gates via
`scripts/check_deleted_dir_refs.py` (deleted-directory list maintained at that
script's top, `DELETED_DIRS`). Before/after deleting:

- Add the directory to `DELETED_DIRS`, run
  `uv run python scripts/check_deleted_dir_refs.py`, plus
  `grep -rn "tests/<old-dir>" e2m2e/ tests/` as double-check (the script gates
  code; documents need human review).
- Clean up or remap all old paths in source strings/comments/docstrings.
  Migration completion = zero old-path references.
- Requirement↔test tracking like `linked_tests` leaks most easily (#373 missed
  `tests/core`, #372 missed `tests/algorithms`); remap item-by-item against the
  migration table.
- Rewrite historical annotations ("ported from …") so they don't reference old
  paths (e.g., "located in the former core package pre-migration"), or delete.

## Revision (2026-08-14, #420)

`FiniteBurn` constant-mass Rust propagation, `VariableMassFiniteBurn`
variable-mass propagation, low-thrust shooting, collocation, and Q-law are all
implemented and in the default test set — `low_thrust` is no longer excluded by
default. The marker remains an orthogonal functional-classification marker for
explicitly selecting low-thrust regression sets.

`Facade.low_thrust_design` stays a separate interface-layer to-do and no longer
justifies gating low-level computation or algorithm tests. Wall-clock
comparisons belong to benchmark scripts outside pytest correctness judgments;
default-suite time bounds come from shrinking real problem sizes and removing
duplicated computation, not from restoring a `slow` category.

## Revision (2026-08-14, ADR 0026)

Decision 1 parenthesized the `data` class as kernels/frames/types/IO/templates
+ coordinate conversion. That "coordinate conversion" describes the
**functional class (what is verified)**, not code layering. Conversion
**algorithms** live at `algorithm/coordinate` (ADR 0011: algorithms belong to
the algorithm layer); conversion **data** (EOP/leap seconds/ephemerides) lives
in `data/frames`. Directories mirror source, markers follow functional classes
— two independent axes; don't infer from data-class membership that conversion
code belongs at the data layer. See ADR 0026.
