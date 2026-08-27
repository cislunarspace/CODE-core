# ADR 0013: Verification strategy — complete tasks by definition

**Status**: Adopted (implemented; test-tiering clause superseded by ADR 0021)
**Date**: 2026-07-31
**Related**: ADR 0011 (five-layer architecture)

## Context

Numerical-library verification involves two distinct questions: ① are the
numbers correct; ② did results change after a code change. Golden-file
regression comparison (against manually verified outputs) and DFH cross-
checking were once introduced as means. But golden files are someone else's
concept — essentially treating previous results as the standard, premised on
the past being right; DFH cross-checking treats another software's output as
the standard. Neither answers whether the result is correct **by definition**.

Requirement: verification means **completing tasks by definition** —
correctness adjudicated by physical definition, depending on no external
software.

## Decision

1. **Correctness is adjudicated by physical definition**: analytic-solution
   comparisons (two-body propagation closure, constant circular-orbit radius,
   Jacobi conservation, STM determinant = 1 symplectic property, Hohmann Δv
   matching theory) + physical invariants. These are definitions; a correct
   computation satisfies them naturally.
2. **Test criteria allow literature formulas/analytic values, not other
   software's runtime output.** Vallado formulas, Richardson coefficients are
   astrodynamics axioms (part of the definition); running and diffing against
   DFH etc. is another software's output and unnecessary.
3. **No golden-file comparison.**
4. **e2m2e is an independent library, never forcibly compared against other
   software.** DFH served only as development-time cross-reference (local
   manual runs diagnosing magnitudes/systematic offsets); comparison scripts
   live outside CI and release artifacts.
5. **Test tiering**: Rust units (numerics vs analytic) → Python algorithm
   units (seed shapes/control-law analytic solutions/error-model statistics)
   → integration (cross-layer chains + physical quantities) → physical
   invariants throughout. (Note: test tiering superseded by ADR 0021's
   functional-category organization.)

## Rationale

1. **Completing tasks by definition needs no external reference**: physical
   laws *are* the definition; correct computation satisfies them naturally.
2. **Golden files prove regression, not correctness**: golden files may be
   wrong themselves yet keep passing forever.
3. **Cross-software comparison introduces needless coupling**: another
   software's output does not constitute e2m2e's definition.

## Consequences

- Test assertions come from closed-form solutions, conserved quantities,
  symmetries, known constants, literature formulas.
- The golden concept is removed; existing golden-related tests/scripts leave
  with `io/`, out of the final architecture.

## Revision (2026-08-16, #426 scope decision)

Decision 4's arrangement of placing comparison scripts under `scripts/` is
withdrawn. e2m2e maintains no cross-check scripts against qiao, DFH, or other
external research pipelines; such outputs constitute no contract for the
project's operation, release, or development. Developers may investigate
differences outside the repo on their own, but that investigation should be
neither a project to-do nor a test capability.

This does not affect e2m2e's own definition-level verification. Checks relying
only on e2m2e, physical definitions, and project-shipped SPICE kernels remain
by their behavioral value; they are not recast as external cross-checks merely
because external implementations used similar methods.
