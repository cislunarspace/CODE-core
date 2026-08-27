# ADR 0024: Unified algorithm result status contract

**Status**: Adopted
**Date**: 2026-08-11
**Related**: ADR 0011 (five-layer architecture), ADR 0014 (interface Facade),
ADR 0020 (failure policy — revises its decision 3), ADR 0023 (explicit-event
SciPy exception), Issue #351

## Context

e2m2e's algorithm results once expressed outcomes via `success`, `converged`,
`correction_success`, `None`, free strings, and solver-instance state. These
conflate solver termination, candidate feasibility, and domain facts — already
causing collision candidates to be visualized as valid solutions.

The data layer's `Orbit` is an orbit container; algorithm and task-execution
outcomes belong on algorithm-layer result objects. The Facade is the task-level
public boundary and needs to translate these outcomes in a stable, machine-
decidable way.

## Decision

All e2m2e algorithm and task results uniformly carry `status: ConvergenceState`,
`cause: FailureCause`, `message: str`. `status` is the final outcome; `cause`
is a stable reason code; `message` only supplements human-readable context.
Success is fixed as `CONVERGED/NONE`; the `FailureCause`→`ConvergenceState`
mapping is unique and validated at construction. Synchronous calls never end at
`ITERATING`.

`ConvergenceState` gains `INFEASIBLE`, `COLLISION`, `FAILED`. Result objects
declare their triple and payload explicitly per domain, with no generic base
class or generic wrapper. Differential correction, continuation, and transfer
grid candidates each get concrete result objects. Non-success results may keep
approximate trajectories, candidates, or partial families but are never marked
successful.

`Orbit`, `OrbitFamily`, `TransferArc`, `EphemerisTable` keep domain-data
responsibilities. `Orbit` drops correction-process fields, retaining orbit-
geometric `closure_error`. Domain-fact booleans like `safe`, `is_periodic`,
`collision_found` fall outside this migration and remain.

Hard failures keep raising domain exceptions to cut control flow, with
exceptions also carrying the status triple + diagnostics. Soft failures return
flagged result objects. Facade responses directly contain the status triple;
"request processed successfully but scientific task infeasible" is soft failure
inside a normal response. Optional stages use `StageRecord` for applicability
and execution status — never encoding "not applicable"/"not run" as failure
causes.

Remove boolean result interfaces (`success`, `converged`, `correction_success`)
without a runtime compatibility layer. New persistence formats write only the
status triple; reading old formats fails with migration hints.

**This revises ADR 0020 decision 3**: that decision permitted boolean fields as
compatibly-retained derived properties; this one tightens to removal outright.
Beyond ADR 0020's `INFEASIBLE`/`COLLISION`, `ConvergenceState` also gains
`FAILED`.

## Trade-offs

Keeping compatibility projections of legacy booleans would reduce short-term
churn but allow state forks to regrow — rejected. Wrapping all results in a
uniform generic would mask structural differences among orbits, candidates,
families, and mission products — so unify the contract only, not the data
tiers. Splitting candidate collisions vs solver divergence into multiple status
systems would force downstream multi-way branching — both are expressed through
one final-status plus fine-grained reason codes.

## Consequences

Algorithm layer, Facade, tests, and visualization migrate to the status triple.
Python/Rust and third-party numeric libraries' native returns get translated at
the algorithm boundary without changing their external interfaces. This is a
breaking migration of public API and result persistence formats.
