# ADR 0001: Withdraw Protocol seams

**Status**: Adopted
**Date**: 2026-05-10
**Related Issue**: #27

## Context

`e2m2e/mbse/architecture/ports.py` defines seven `@runtime_checkable`
`Protocol` classes (`SystemModel`, `EOMProvider`, `Propagator`,
`OrbitContainer`, `CorrectorStrategy`, `Optimizer`, `Visualizer`). But **not a
single place** in production code uses these `Protocol` type annotations:
`algorithms/`, `transfer/`, and `visualization/` all annotate concrete types
(`CR3BP_Dynamics`, etc.). The only `isinstance(..., Protocol)` call appears
solely in `tests/mbse/test_protocol_conformance.py`.

Meanwhile the `Dynamics` base class already provides a genuine polymorphism
mechanism via the template-method pattern (`propagate()` + `_get_eom_func()`
hook). Both `CR3BP_Dynamics` and `EphemerisDynamics` inherit from it; the
`Protocol`s therefore form a redundant parallel interface.

`TransferSearch` also demonstrably cannot use `Propagator`, because it needs
`dynamics.system.mu`, which the `Protocol` does not define — evidence that
these `Protocol` definitions are incomplete for real use anyway.

## Decision

**Withdraw**: delete `ports.py` and `test_protocol_conformance.py`. Accept a
monomorphic codebase: the `Dynamics` base class is the only polymorphism
mechanism.

## Rationale

1. **The `Protocol`s overlap the `Dynamics` base class.** `Dynamics` already
   defines `propagate()` and `equations_of_motion()`; the `Protocol`s add a
   duplicate set of contracts without runtime enforcement.

2. **Zero production usage.** No function signature anywhere in the library
   accepts a `Protocol` type. These seven `Protocol`s are pure decoration.

3. **Incomplete definitions.** `TransferSearch` needs `dynamics.system.mu`;
   `Propagator` lacks it. Making the `Protocol`s usable would mean expanding
   them — added complexity for a benefit that does not currently exist.

4. **Downstream issues proceed fine without `Protocol`s.** Issue #31 (DC →
   `dynamics.propagate()`) annotates against the `Dynamics` base class; #34
   (Transfer merge) merges directly; #32 and #33 are unaffected.

5. **MBSE metadata is decorative.** Component registries and requirement files
   reference `Protocol` names as strings, recording intent only, with no
   runtime effect.

## Consequences

### Removed

- `e2m2e/mbse/architecture/ports.py`
- `tests/mbse/test_protocol_conformance.py`
- `Protocol` exports in `mbse/architecture/__init__.py`
- `Protocol` references in MBSE component metadata

### Kept

- All public API classes (`CR3BP_Dynamics`, `Dynamics`, `Orbit`, `Transfer`,
  etc.)
- The `Dynamics` base class as the genuine polymorphism mechanism
- MBSE infrastructure (components, requirements, diagrams), references updated

### Downstream impact

| Issue | Path forward |
|-------|----------|
| #31 DC → `propagate()` | Annotate against `Dynamics` base |
| #34 Transfer + Optimizer merge | Direct merge, no `Protocol` indirection |
| #32 Stability merge | Unaffected |
| #33 Visualization flattening | Unaffected |
