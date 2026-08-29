# ADR 0012: Dependency-direction rules with CI import checks

**Status**: Adopted (implemented; dependency table and enforcement scope revised by ADR 0039)
**Date**: 2026-07-31
**Related**: ADR 0011 (five-layer architecture)

## Context

The five-layer architecture (ADR 0011) is only viable if dependency direction
is genuinely enforced. Without enforcement, inter-layer imports drift over
time: the algorithm layer quietly imports the interface layer, the data layer
starts depending on algorithm types. The reference literature's conclusion:
architecture rests on reviews and baselines, not good intentions.

## Decision

Dependency-direction rules (hard rules):

```
api/ → algorithm/ + data/
algorithm/ → data/ + _integrators
data/ → external libraries only (SPICE/r2s2/numpy)
integrators.py → _integrators
tools/ → anything (auxiliary; core never imports tools/)
```

**Two hard boundaries**:
1. The algorithm layer does not import `api/` (Pydantic lives only at the
   `api/` boundary; the algorithm layer uses numpy/dataclasses).
2. The data layer does not import `algorithm/` (types Orbit/EphemerisTable
   live in `data/types/` — produced by the data layer itself).

CI runs import checks for enforcement (custom script or lint rules checking,
e.g., that `algorithm/` does not import `api/` or `tools/`).

> **Revision (2026-08-29, ADR 0039)**: the `data/` row gains the package-root
> shared-kernel leaves (`exceptions`/`status`/`spice_ext`) as legal
> dependencies alongside external libraries — the SPICE dual-instance bridge
> (ADR 0016) reaches the Rust extension through `spice_ext`, not through the
> `integrators.py` facade, which `data/` must not import. `integrators.py`
> may additionally import the shared-kernel leaves; it still imports no
> layer. The "Pydantic only at the `api/` boundary" clause constrains the
> algorithm layer only; the non-runtime `mbse/` documentation artifact is
> outside its scope. The CI check now resolves relative imports, scans
> package-root modules, and forbids `integrators`/`mbse` as import targets —
> the original checker's blind spots left these rules unenforced (issue
> #545). All other clauses stand.

## Rationale

1. **Pydantic only at the `api/` boundary**: keeps the algorithm layer on
   numpy + exceptions, avoiding 156 tests breaking over Pydantic object
   signature changes.
2. **Self-sufficient data layer**: `data/` depends only on external libraries;
   independently testable and swappable data sources.
3. **Core never depends on tools**: `tools/` is auxiliary; the core library
   doesn't import it, keeping core pure.

## Consequences

- CI gains an import-check step.
- New code follows dependency direction; legacy code gets fixed while
  migrating.
