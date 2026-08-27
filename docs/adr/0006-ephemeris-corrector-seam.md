# ADR 0006: Unified ephemeris-correction seam with registry dispatch

**Status**: Adopted
**Date**: 2026-06-15
**Related Issue**: #5 (architecture deepening candidate)

## Context

`correct_ephemeris_patch_points()` in
`e2m2e/algorithms/ephemeris_correction.py` uses a `method: str` parameter with
`if/elif` branches to dispatch three correction methods (standard, two_level,
homotopy). Each branch constructs its own solver, passes differently named
parameters, and re-wraps different result types into
`EphemerisCorrectionResult`. The homotopy branch also needs lazy imports to
avoid circular dependency.

Problems with this dispatch pattern:
1. **The interface is nearly as complex as the implementations**: the `method`
   string is the only abstraction; branch bodies are all of the
   implementation.
2. **Adding a method requires editing the dispatcher**: every new correction
   method adds one branch to the chain, one set of parameter translation, and
   one result re-wrap.
3. **Circular dependency**: `homotopy_correction.py` imports
   `EphemerisCorrectionResult` from `ephemeris_correction.py`, preventing the
   latter from importing the former at top level.

ADR 0005 added `TwoLevelMultipleShooting` as an independent algorithm, but the
dispatch layer remained string-based.

## Decision

1. **Define a `PatchPointCorrector` seam.**
   - Define the `PatchPointCorrector` Protocol in new module
     `ephemeris_correction_types.py`:
     ```python
     def correct(t_patch, state_patch, *, max_iter, tolerance, velocity_tolerance, verbose) -> EphemerisCorrectionResult
     ```
   - Construction parameters (`n_workers`, `kernel_dir`, `base_bodies`,
     `lambda_steps`, etc.) inject via constructors, not through the unified
     interface.
   - `EphemerisCorrectionResult` also moves into that module.

2. **Replace `if/elif` dispatch with a registry.**
   - `_REGISTRY: dict[str, Callable[..., PatchPointCorrector]]` maps method
     names to factory functions.
   - Each factory creates a private `PatchPointCorrector` implementation
     (`_StandardPatchPointCorrector`, `_TwoLevelPatchPointCorrector`,
     `_HomotopyPatchPointCorrector`).
   - Those implementations wrap existing solvers, translating unified
     parameters to solver-specific ones and re-wrapping results into
     `EphemerisCorrectionResult`.
   - `correct_ephemeris_patch_points()` becomes: look up registry → construct
     corrector → call `corrector.correct()`.

3. **Untie the circular dependency.**
   - `homotopy_correction.py` imports `EphemerisCorrectionResult` from
     `ephemeris_correction_types`.
   - `_HomotopyPatchPointCorrector` lazily imports `correct_with_homotopy`
     inside its `correct()` method (preserving original lazy semantics).

4. **Explicit error types.**
   - New `UnsupportedCorrectorMethodError(ValueError)` replaces the bare
     `ValueError`.
   - Message includes requested method name and available methods list.

5. **Existing solver interfaces unchanged.**
   - `MultipleShooting.correct()` and `TwoLevelMultipleShooting.correct()`
     signatures, return types, behavior unchanged.
   - Existing direct callers of these solvers unaffected.

## Rationale

1. **Why wrappers instead of changing solver signatures directly.** Existing
   solvers have many direct callers (tests, DRO end-to-end correction, CLI);
   changing `.correct()` signatures has too broad an impact. Wrappers separate
   seam language from solver language so both evolve stably.

2. **Why a registry rather than Protocol registration.** Python's `Protocol`
   offers no auto-registration. An explicit `_REGISTRY` dict is simple and
   explicit enough; adding a method is one lambda line.

3. **Why keep lazy imports.** `_HomotopyPatchPointCorrector` imports
   `correct_with_homotopy` lazily inside `correct()`, preserving the original
   semantic: spiceypy doesn't load when homotopy correction isn't used.

4. **Why not overturn ADR 0005.** ADR 0005 decided
   `TwoLevelMultipleShooting` as an independent algorithm. This ADR reinforces
   it: the two-level solver is consumed through the unified seam while its
   internals stay independent.

## Consequences

### Added

- `e2m2e/algorithms/ephemeris_correction_types.py`:
  `EphemerisCorrectionResult`, `PatchPointCorrector` Protocol,
  `UnsupportedCorrectorMethodError`.
- `e2m2e/algorithms/ephemeris_correction.py`: three private
  `PatchPointCorrector` implementations, `_REGISTRY`.
- `tests/algorithm/correction/test_patch_point_corrector.py`: seam protocol,
  registry, dispatch, error handling tests (20).
- `docs/adr/0006-ephemeris-corrector-seam.md`.

### Changed

- `e2m2e/algorithms/ephemeris_correction.py`: `EphemerisCorrectionResult`
  imported from `ephemeris_correction_types`;
  `correct_ephemeris_patch_points()` switched to registry dispatch.
- `e2m2e/algorithms/homotopy_correction.py`: `EphemerisCorrectionResult`
  imported from `ephemeris_correction_types`.
- `e2m2e/algorithms/__init__.py`: lazy exports for `PatchPointCorrector`,
  `UnsupportedCorrectorMethodError`.

### Unchanged

- `MultipleShooting.correct()` signature, return type, behavior.
- `TwoLevelMultipleShooting.correct()` signature, return type, behavior.
- `correct_with_homotopy()` signature and behavior.
- Public signature of `correct_ephemeris_patch_points()` (backward compatible).
- All existing tests (74 algorithm tests + 20 new = 94 all passing).

### Follow-up work

- Issue #8 (MultipleShooting parallel inline) can reuse the same
  `PatchPointCorrector` seam.
- New correction methods need only: write a `PatchPointCorrector` impl + add
  one `_REGISTRY` line.

## Revision (2026-08-13)

The `ephemeris_correction` subpackage (standard/two_level/homotopy
implementations + registry dispatch) was deleted wholesale: the design chain
unified on Rust multiple shooting (``multiple_shooting_correct_py``, default
for segmented and stable orbits); no multiple correction methods remain to
dispatch between. `EphemerisCorrectionResult` moved to
``e2m2e/algorithm/results.py`` as a domain re-wrap of Rust shooting results.
Decisions 1/2/3/4 (seam, registry, lazy import, error type) lapse accordingly;
decision 5's unchanged `MultipleShooting.correct()` interface stands (still
used by transfer/hohmann non-design chains). Related: ADR 0005 (same-batch
deletion of `TwoLevelMultipleShooting`).
