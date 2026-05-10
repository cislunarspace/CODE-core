# Context Map

This is a single-context repository. All domain knowledge lives at the root level.

| Context | Location | Covers |
|---------|----------|--------|
| `e2m2e` | `./CONTEXT.md` | The entire e2m2e library |

## Architecture layers

```
core/           → Foundation — CR3BP system, orbit data structures, physics models
    ↓
algorithms/     → Numerical solvers — differential correction, continuation, stability, multiple shooting
    ↓
transfer/       → Transfer design — grid search, NLP optimization
    ↓
visualization/  → Plotting — orbit families, transfer trajectories

Cross-cutting:   mbse/ — Protocol interfaces, Pydantic models, requirement tracking, diagram generation
```

## Skills consuming domain docs

These skills read `CONTEXT.md` before exploring:

- `/improve-codebase-architecture`
- `/diagnose`
- `/tdd`

These skills read `docs/adr/` for architectural decisions:

- `/grill-with-docs` — writes ADRs
- All skills above consult existing ADRs before proposing changes
