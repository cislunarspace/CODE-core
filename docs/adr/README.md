# Architecture Decision Records (ADR)

This directory records e2m2e's architecture decisions. Each ADR is a decision
snapshot: it captures the context, decision, rationale, and consequences at
the time of writing. When a decision later changes, do not rewrite the
original text; instead append a revision subsection at the end, or write a new
ADR and mark the supersession in the old one.

## Status vocabulary

- **Adopted**: the decision is in effect. Parenthetical notes may indicate
  implementation progress or partial revision, e.g.: Adopted (partially
  implemented: …), Adopted (decision 3 revised by ADR 0024).
- **Rejected**: the proposal was not adopted. The body keeps the proposal and
  the rejection rationale.
- **Superseded**: the decision was wholly overturned by a later ADR; the
  status line names the successor, e.g.: Superseded (see ADR 0024). The
  original entry is kept, never deleted.

Status describes the fate of the decision itself. When the decision's object
is to veto some mechanism (e.g. ADR 0008 vetoes runtime freezing), the status
is still Adopted, with the vetoed object noted in parentheses.

When a decision is partially revised by a later ADR, both entries keep mutual
pointers: the new ADR states in its "Related" section and relevant clauses
which clauses were revised; the old ADR gets revision notes at the revised
spots. Silent overrides without pointers violate the ADR conflict-annotation
convention.

## Numbering rules

- Numbers are four digits, increasing, never reused; normally in time order.
- Back-filled historical decisions occupy vacated numbers of their era, with
  the actual decision date noted at the top (see ADR 0005).

## Template

```markdown
# ADR XXXX: Title

**Status**: see vocabulary above
**Date**: YYYY-MM-DD
**Related Issue**: #nnn
**Related**: ADR YYYY (relationship to this entry)

## Context

Why this decision must be made now. State facts and constraints clearly,
without piling up detail.

## Decision

Itemized list; each item actionable and verifiable.

## Rationale

For each decision item, why this shape and not another. Where alternatives
exist, state why they were excluded.

## Consequences

Added / changed / unchanged. Where there is a cost, state it.
```

Optional subsections: `Alternatives compared`, `Trade-offs`,
`Revision (date, reference)`. Revision subsections are appended at the end;
original text untouched. ADRs leave no TODOs: to-dos move to issues or new
ADRs.

## Index

| No. | Title | Status |
|---|---|---|
| 0001 | Withdraw Protocol seams | Adopted |
| 0002 | Rust integrator core, Python-controlled dynamics | Adopted (with multiple revisions) |
| 0003 | Axes, ITRF93 defaults, GMAT-compatible Earth orientation | Adopted |
| 0004 | ForceModel config-driven | Adopted |
| 0005 | TwoLevelMultipleShooting as an independent algorithm | Adopted (revoked 2026-08-13: implementation deleted, see revision at end) |
| 0006 | Unified ephemeris-correction seam with registry dispatch | Adopted (revoked 2026-08-13: implementation deleted, see revision at end) |
| 0007 | Dynamic-axes state injection scheme | Adopted |
| 0008 | Revoke runtime freezing of Axes / Origin / CoordinateSystem | Adopted (freezing mechanism rejected and reverted) |
| 0009 | Enable spice feature for release wheels | Adopted (implemented) |
| 0010 | r2s2 integration and TDT+GCRS ↔ TDB+EBCRS spacetime conversion | Adopted (implemented) |
| 0011 | Five-layer architecture and radical full renaming | Adopted (implemented) |
| 0012 | Dependency-direction rules with CI import checks | Adopted (implemented; dependency table and enforcement scope revised by ADR 0039) |
| 0013 | Verification strategy: complete tasks by definition | Adopted (test-tiering clause superseded by ADR 0021) |
| 0014 | Interface layer Facade/MCP/CLI | Adopted (implemented; decisions 2 and 5 revised by ADR 0043, decision 8 completed for catalog value sets by ADR 0044) |
| 0015 | NominalOrbit contract and coordinate-conversion abstraction | Adopted (implemented) |
| 0016 | EphemCache ephemeris cache architecture | Adopted |
| 0017 | Transfer grid search: purely numerical kernel pushed down to Rayon | Adopted |
| 0018 | Jacobian interface extended with ∂a/∂v; STM covers velocity dependence | Adopted |
| 0019 | Drag Rust port uses ITRF93 pxform frame rotation (replacing ITRFApproxAxes) | Adopted |
| 0020 | Failure policy: deterministic failures raise, infeasible searches return flags, no implicit degradation | Adopted (decision 3 revised by ADR 0024) |
| 0021 | Test suite organized by functional categories; speed tiering abolished | Adopted |
| 0022 | Independent physical constants management | Adopted |
| 0023 | SciPy propagation exception for explicit event inputs | Adopted |
| 0024 | Unified algorithm result status contract | Adopted |
| 0025 | Test suite convergence: external references removed, primary marker invariant, explicit backend selection | Adopted |
| 0026 | Test suite layer clarification: coordinate ownership, forces test merge, dead-reference cleanup | Adopted |
| 0027 | System/Dynamics separation retained: dynamics directory unsplit, two classes unmerged | Adopted |
| 0028 | Planar triangular libration point family via full-period pseudo-arclength continuation | Adopted (#428 seam revised by ADR 0029) |
| 0029 | Orbit family generation via unified Rust deep module | Adopted (implemented) |
| 0030 | algorithm/forces stays at algorithm layer: Python config/orchestration surface, numerics in crates | Adopted |
| 0031 | Orbit catalog: record format, storage layout, query interface | Adopted (decision 4 overturned by ADR 0045; decisions 1, 2, 5 revised by ADR 0045; decision 7 revised by ADR 0043) |
| 0032 | HJB dynamics in a new crate plus binding-layer generic entry | Adopted |
| 0033 | HJB low-thrust toolchain: value-function product contract and online query interface | Adopted |
| 0034 | Scope of the ephemeris force-model Hamiltonian | Adopted |
| 0035 | GUI sidecar stdio protocol: shared Facade envelope, large arrays over binary frames | Adopted |
| 0036 | CR3BP baseline orbit-family dataset: precomputed full-family data shipped with the package | Adopted |
| 0037 | Test suite time budget, minimal real-call coverage, and e2e test boundaries | Adopted |
| 0038 | IAS15 integrator and force-model parametric variational equations (ASSIST-derived); MERCURIUS not adopted | Adopted |
| 0039 | Shared-kernel leaf modules at the package root | Adopted (implemented) |
| 0040 | transfer_design converged trajectory: unified synodic-frame contract with trajectory_times | Adopted (implemented) |
| 0041 | spatiography — cislunar partition (Primer) analytic core: five-province taxonomy, [primer] constants, scales/classify/boundaries tools | Adopted (implemented) |
| 0042 | orbit taxonomy — 42-label classification of CR3BP periodic orbits: STK CODE vocabulary, self-defined analytic criteria, ingest stamping and response enrichment | Adopted (implemented; decision 5 tool-count clause superseded by ADR 0043/0044; label table relocated by ADR 0044) |
| 0043 | Interface class split — Facade keeps task-level methods, catalog and spatiography become their own classes | Adopted (implemented) |
| 0044 | Terminology list exposure — closed value sets leave the repository through one registered tool | Adopted |
| 0045 | Orbit record granularity — one record per trajectory, family as label | Adopted |
