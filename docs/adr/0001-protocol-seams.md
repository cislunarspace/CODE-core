# ADR 0001: Recant Protocol Seams

**Status**: Accepted
**Date**: 2026-05-10
**Issue**: #27

## Context

`e2m2e/mbse/architecture/ports.py` defined 7 `@runtime_checkable` Protocol classes (SystemModel, EOMProvider, Propagator, OrbitContainer, CorrectorStrategy, Optimizer, Visualizer). Production code used **zero** Protocol type annotations — `algorithms/`, `transfer/`, `visualization/` all annotated concrete types (`CR3BP_Dynamics` etc.). The only `isinstance(..., Protocol)` calls lived in `tests/mbse/test_protocol_conformance.py`.

Meanwhile, the `Dynamics` base class already provides the real polymorphism mechanism via Template Method pattern (`propagate()` + `_get_eom_func()` hooks). Both `CR3BP_Dynamics` and `EphemerisDynamics` inherit from it, making Protocols a redundant parallel interface layer.

TransferSearch explicitly cannot use `Propagator` because it needs `dynamics.system.mu`, which the Protocol does not define — indicating the Protocol definitions were incomplete for real usage.

## Decision

**Recant**: Delete `ports.py` and `test_protocol_conformance.py`. Accept a monomorphic codebase where `Dynamics` base class is the single polymorphism mechanism.

## Rationale

1. **Protocol and Dynamics base class overlap**. `Dynamics` already defines `propagate()` and `equations_of_motion()`. Protocol adds a redundant parallel contract with no runtime enforcement.

2. **Zero production usage**. No function signature in the library accepts a Protocol type. The 7 Protocols were decorative.

3. **Incomplete definitions**. `TransferSearch` needs `dynamics.system.mu` — not in `Propagator`. Making Protocols usable would require expanding them, adding complexity for no current benefit.

4. **Downstream issues work fine without Protocols**. Issue #31 (DC → `dynamics.propagate()`) will annotate with `Dynamics` base class. Issue #34 (Transfer merge) will merge directly. Issue #32 and #33 are unaffected.

5. **MBSE metadata is decorative**. Component registries and requirement files reference Protocol names as strings — they document intent but don't affect runtime.

## Consequences

### Removed

- `e2m2e/mbse/architecture/ports.py`
- `tests/mbse/test_protocol_conformance.py`
- Protocol exports from `mbse/architecture/__init__.py`
- Protocol references in MBSE component metadata
- Protocol entry in `CONTEXT.md`

### Preserved

- All public API classes (`CR3BP_Dynamics`, `Dynamics`, `Orbit`, `Transfer`, etc.)
- `Dynamics` base class as the real polymorphism mechanism
- MBSE infrastructure (components, requirements, diagrams) with updated references

### Downstream impact

| Issue | Path forward |
|-------|-------------|
| #31 DC → propagate() | Annotate with `Dynamics` base class |
| #34 Transfer + Optimizer merge | Direct merge, no Protocol indirection |
| #32 Stability merge | Unaffected |
| #33 Visualization flatten | Unaffected |
