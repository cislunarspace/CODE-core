# ADR 0026: Test suite layer clarification — coordinate ownership, forces test merge, dead-reference cleanup

**Status**: Adopted
**Date**: 2026-08-14
**Related Issues**: #429 (forces layer ownership), #430 (dynamics split),
#431 (Python numerics migration labeling)
**Related**: ADR 0011 (five-layer architecture), ADR 0012 (dependency
direction), ADR 0021 (test functional categories), ADR 0025 (suite
convergence)

## Context

Auditing `tests/algorithm` (14 subdirectories) surfaced three structural debts:

1. **coordinate markers vs directory mismatch**. All 15 files under
   `tests/algorithm/coordinate` carry the `data` functional class, yet sit at
   the `algorithm` layer. Initial reading of architecture.md (frame conversion
   belongs to the spacetime/constants module) plus ADR 0021 (`data` class
   includes coordinate conversion) proposed moving coordinate source and tests
   wholesale to the data layer.
2. **forces tests split**. Tests for source `e2m2e/algorithm/forces` (20 files)
   scatter between `tests/numerical/forces` (~45 force-marked tests across
   container/contract/physics tiers) and `tests/algorithm/forces` (only
   `test_atmosphere.py`).
3. **normal_form dead references**. `corrector.py` has a live import
   `from e2m2e.core.dynamics import CR3BP_System`, but `e2m2e.core` and
   `e2m2e.algorithms` were deleted in ADR 0011's radical renaming; executing
   that branch raises `ImportError`.

Deeper digging overturned item 1's initial reading (see decision 1 rationale);
this ADR records the corrected conclusions.

## Decision

1. **coordinate stays at the algorithm layer, not data.** Respects ADR 0011's
   landed decisions: `e2m2e/algorithm/coordinate/__init__.py` states conversion
   algorithms belong here, with `data/frames` keeping only data. Functional
   class markers (`data`) and code layering (`algorithm`) are two independent
   axes per ADR 0021 decision 2: directories mirror source, markers state what
   is verified — coexisting without contradiction.
2. **forces tests merged.** `tests/algorithm/forces/test_atmosphere.py` joins
   `tests/numerical/forces/contract/` (alongside other single-model contract
   tests); empty `tests/algorithm/forces/` deleted. The forces source-layer
   ownership question (numerical-layer config surface vs algorithm-layer model
   definitions) awaits its own ADR — not adjudicated here.
3. **Dead references cleaned in normal_form and neighbors.** Live imports
   remapped to current paths; docstring cross-references
   (`e2m2e.algorithms.*`, `e2m2e.core.*`) remapped to new paths;
   shim-history compatibility notes in `data/__init__.py`,
   `data/templates/enums.py`, `exceptions.py` are deliberate and untouched.

## Rationale

1. **Moving coordinate to data would create new violations.** coordinate has
   reverse dependencies into algorithm layer via relative imports:
   `__init__.py`'s spacetime_convert imports `..design.design_orbit` and
   `..dynamics.cr3bp_system` inside function bodies. At data layer these become
   data → algorithm, directly breaking ADR 0012's "data never imports
   algorithm" rule. Current coordinate reverse dependencies are all compliant:
   api/facade → algorithm.coordinate (api→algorithm);
   design/station_keeping/forces/dynamics → coordinate (within-algorithm). The
   earlier false positive about data/frames/gmat_fixture.py importing
   coordinate was docstring prose, not an import.
2. **Functional class and layer are two axes — don't conflate.** ADR 0021
   decision 2 verbatim: directories mirror source structure (navigation),
   markers state functional classes (what is verified). The `data` class's
   parenthetical (kernels/frames/types/IO/templates + conversion) describes
   verification content, not code location. Reading "coordinate conversion is
   in the data class" as "conversion code belongs at data layer" mistakes the
   marker axis for the layer axis.
3. **Forces test splitting violates directory mirroring; merging is cleanup,
   not refactor.** ADR 0021's migration put force-model tests at
   `tests/numerical/forces` on the intuition that models belong to numerical,
   yet left `test_atmosphere.py` at `tests/algorithm/forces` — one source
   package's tests scattered in two places. Merging moves only test files, no
   source imports — minimal risk.
4. **Dead references are ADR 0011 leftovers; one is a real bug.** Docstring
   cross-references don't affect execution, but `corrector.py`'s live import
   targets a deleted module — a must-fix defect; the shim notes (old paths kept
   compatible via shims) describe reality; deleting them would falsify.

## Consequences

### Changed

- `tests/algorithm/forces/` deleted; `test_atmosphere.py` moved to
  `tests/numerical/forces/contract/`.
- normal_form: `corrector.py` live import remapped to
  `e2m2e.algorithm.dynamics.CR3BP_System`; stale cross-references in
  `_ephemeris.py`, `fft.py`, `legendre.py`, `catalog.py`, `hamiltonian.py`,
  `coord_trans/`, `dynamical_substitution.py`, `multiple_shooting.py` remapped
  (12× `e2m2e.algorithms` → `e2m2e.algorithm`; 1×
  `e2m2e.core.SPICEManager` → `e2m2e.data.kernels.manager.SPICEManager`).
- forces/transfer/proximity: stale `e2m2e.core.*` cross-references in
  `thrust.py`, `exceptions.py`, `lowthrust_shooting.py`,
  `relative_dynamics.py` remapped to current paths.

### Unchanged

- Location of `e2m2e/algorithm/coordinate` sources and
  `tests/algorithm/coordinate` tests.
- coordinate tests' `data` markers; forces tests' `force` markers.
- ADR 0011 five-layer architecture and ADR 0012 dependency direction texts.

### Follow-ups (each independent, not with this entry)

- Forces source-layer ownership: `e2m2e/algorithm/forces` is Rust
  `e2m2e-forces`' Python config surface (self-described as parameter validation
  + to_rust_spec serialization), comparable to `e2m2e/integrators.py`;
  whether it leaves the algorithm layer needs its own ADR, constrained by the
  unresolved shape of Python code in the numerical layer. See #429.
- System/Dynamics split: `CR3BP_System` (physical system definition) vs
  `CR3BP_Dynamics` (constructing integration problems) differ in layer — a
  high-risk large migration. See #430.
- Python numerics inside normal_form/solver/transfer/family/manifold are the
  transitional state ADR 0011 explicitly migrates stepwise to Rust; document
  migration progress to avoid future audits misreading them as misplaced. See
  #431; progress ledger at
  `docs/architecture/numerics-migration-status.md`.
