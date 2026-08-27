# ADR 0025: Test suite convergence — external references removed, primary marker invariant, explicit backend selection

**Status**: Adopted
**Date**: 2026-08-14
**Related Issue**: #425
**Related**: ADR 0011 (five-layer architecture), ADR 0012 (dependency
direction), ADR 0013 (verification strategy), ADR 0020 (failure policy),
ADR 0021 (test functional categories)

## Context

ADR 0021 switched the test classification axis from speed to "what is
verified," with two rules: directories mirror source structure; exactly one
primary class per test. Post-landing audit (`tests/algorithm`, 1249 cases)
showed the rules were words without gatekeepers; five structural debts
accumulated:

1. **Unclean verification sources**. qiao/DFH outputs entered in three forms:
   five skip-guarded empty regression placeholders (verifying nothing — task
   lists disguised as tests); one fixture cross-check depending on a personal
   workstation's absolute path (`/home/.../qiao/L1_EM_Hamilton.mat`);
   hardcoded external-software output values (e.g., Hamilton constant term
   `-862.50648692`). All violate ADR 0013 decisions 2/3: correctness judged by
   physical definition, never by other software's output.
2. **Primary markers not conserved**. Three cases in
   `test_differential_correction_stagnation.py` are collected into the default
   set without primary markers — no `-m <functional-class>` selects them.
   File-level `orchestration` markers cover cases actually verifying
   `interface`/`data` (`test_wsb_contract.py`; the `DesignOrbitRequest`
   validation sections in family files). "Exactly one primary class per test"
   lacked an executable constraint; missed markers were inevitable.
3. **Directories not mirroring**. `tests/algorithm/correction/` corresponds to
   source `e2m2e/algorithm/solver/`; `e2m2e/algorithm/nominal_orbit/` has no
   test-side counterpart; `tests/algorithm` mixes pure interface validation
   constructing only `e2m2e.api.models.DesignOrbitRequest`. Test-side layering
   diverges from ADR 0012.
4. **Environment drift**. SPICE availability checks have at least three
   implementations across test files; absolute-path fixtures tie results to
   specific machines.
5. **Production coupling**. `normal_form` frequency analysis's `prefer="auto"`
   automatic backend switching got cemented by test assertions (NAFF missing →
   FFT fallback; selected NAFF failing → FFT fallback). ADR 0020 decision 4
   forbids auto — tests were guarding a violating design.

## Decision

1. **Three-tier handling of external references**. Method-provenance comments
   (pointing at qiao Code10) stay — literature citation. qiao/DFH fixture
   cross-checks and hardcoded output values leave pytest, moving to `scripts/`
   for manual diagnostics (ADR 0013 decision 4's designated place). The five
   skip-placeholder regression tests are deleted; unfinished numeric regressions
   move to issue tracking. Deleted numeric assertions get definition-level
   replacements in the same commit: ephemeris-computed reference values,
   Hamilton equation structure, symplecticity (`BᵀJB=J`), round-trip identity.
2. **Primary-marker conservation gatekeeper**. Meta-tests in `tests/_meta/`:
   collect all cases, assert each has exactly one primary-class marker,
   listing violators' paths. ADR 0021 decision 1 turns from slogan into
   executable constraint.
3. **Directory-mirror convergence**. Rename `tests/algorithm/correction/` to
   `solver/`; add `nominal_orbit/`; pure `DesignOrbitRequest` validation moves
   to `tests/api/`; `tests/algorithm` no longer imports `e2m2e.api`.
4. **SPICE detection single-pointed**. Availability checks converge onto a
   single fixture in `tests/conftest.py` + unified `spice` marker skips;
   per-file implementations all deleted.
5. **Explicit backend selection (with production change)**. `normal_form`'s
   `prefer` drops `auto`, accepting only `naff`/`fft`; selected-NAFF failure
   raises. This implements ADR 0020 decision 4 rather than revising it.

**Six migration steps** (order-dependent, each independently verifiable):
① gatekeeper lands (currently red on the stagnation file — baseline evidence)
→ ② pure moves (syncing `linked_tests`, traceability matrix, `DELETED_DIRS`)
→ ③ marker corrections (gatekeeper green) → ④ external-reference removal
(three tiers, separate commits) → ⑤ explicit backend selection (independent
PR) → ⑥ SPICE detection convergence.

## Rationale

1. **Gatekeeper first**. ADR 0021's rules had no gatekeeper; the stagnation
   missed markers and file-level mislabeled categories are evidence. Make rules
   executable before migrating; the first red run doubles as a quantitative
   baseline of current state.
2. **External references sorted by form, not one-size-fits-all**. Method provenance is
   citation, not cross-checking — deleting loses traceability. Fixture checks
   and hardcoded outputs treating other software as standard must leave pytest.
   Skip placeholders verify nothing while inflating collection counts; their
   information (which numeric regressions await) belongs to issues, not the
   suite.
3. **Definition-level replacements leave no coverage vacuum**. Deleted numeric
   assertions (e.g., Hamilton constant) swap same-commit for references computed
   on-site from ephemerides and definitional formulas — correctness still
   judged by definition, oracle no longer tied to personal workstations.
4. **Kill `auto` in production rather than amend the ADR**. The alternative —
   adding an algorithm-equivalence exemption clause to ADR 0020 — was rejected:
   exemptions get abused, and research scenarios shouldn't tolerate silent
   backend swaps anyway. `auto` violates determinism (ADR 0020 rationale 3).
5. **Directory mirroring exists for navigational predictability**
   (ADR 0021 rationale 3). `correction/` vs `solver/` is name-reality mismatch;
   MBSE's traceability matrix already registers `e2m2e.algorithm.solver`. After
   alignment, "where are module X's tests" returns to a one-sentence answer.

## Consequences

### Added

- `tests/_meta/` primary-marker conservation meta-tests.
- Directories `tests/algorithm/solver/`, `tests/algorithm/nominal_orbit/`.
- Manual qiao/DFH diagnostic scripts under `scripts/` (migrated from pytest).

### Changed

- `tests/algorithm/correction/` deleted (merged into `solver/`); `DELETED_DIRS`,
  `linked_tests`, traceability matrix synced.
- Pure API-validation cases moved to `tests/api/`; `tests/algorithm` no longer
  imports `e2m2e.api`.
- `e2m2e/algorithm/normal_form/`: `prefer` drops `auto`; NAFF failure raises
  (production behavior change, independent PR).
- SPICE availability probing converged to `tests/conftest.py`.

### Unchanged

- Seven primary classes + `spice`/`low_thrust` orthogonal markers; no speed
  tiering restored.
- ADR 0013 verification, ADR 0020 failure handling, ADR 0021 functional
  category decisions unchanged in text. This entry strengthens without
  revising.

### Costs

- With hardcoded external values gone, regression protection of corresponding
  coefficients rests on construction quality of definition-level assertions;
  risk of wrongly built definitions falls to same-commit review.
- Moving ~20 API validation cases and updating the traceability matrix —
  one-time cost.
- Dropping `auto` is public behavior change: callers relying on default
  NAFF→FFT silent fallback must pass `fft` explicitly.

## Revision (2026-08-14, #425 implementation)

Decision 4's landing refined in two spots:

1. The universal probe lives in `tests/kernel_helpers.py`
   (`spice_kernels_available()` + `requires_spice` marker), not
   `tests/conftest.py`. Why: `SPICE_KERNEL_DIR` and kernel-loading helpers
   already live there — probing shares origin; and the five consolidated probes
   were module-level skipifs (for pytestmark lists), which fixtures can't serve
   during collection. Rule added: any case carrying `requires_spice` must also
   carry the orthogonal `spice` marker, or `-m spice` selection undercounts.
2. `test_dynamical_substitution.py`'s full-window acceptance kept runtime
   probing (calling `eval_params` to check kernel-pool load) — different
   semantics from file-existence probing; that arrangement was superseded by
   the #426 revision below.

## Revision (2026-08-16, #426 scope decision)

This revision replaces decision 1's requirement to migrate qiao/DFH cross-checks
into `scripts/`, without changing definition-level acceptance relying solely on
e2m2e and its supported SPICE kernels.

The qiao normal-form pipeline and its `.mat`/`.npz` intermediates are standalone
research tools outside e2m2e's operation contract, release contract, or
development maintenance scope. The repo maintains no qiao cross-check scripts
and treats no qiao intermediate results as oracles for pytest or other project
checks. SPICE kernels remain the project's supported standard runtime dependency.

`test_dynamical_substitution.py`'s full-window frequency-suppression acceptance
calls only e2m2e itself plus SPICE, reading no qiao data, so its runtime
kernel-pool probing stays. It is definition-level behavioral checking of e2m2e's
own ephemeris dynamics — not external-software cross-checking.
