# ADR 0043: Interface class split — Facade keeps task-level methods, catalog and spatiography become their own classes

**Status**: Adopted (implemented — classes split, inventory scans the exposed
set; `catalog_terminology` lands with ADR 0044, `catalog_promote` removal with
ADR 0045)
**Date**: 2026-09-01
**Related Issue**: #610
**Related**: ADR 0011 (five-layer architecture), ADR 0012 (dependency
direction), ADR 0014 (interface layer — decisions 2 and 5 revised here),
ADR 0031 (decision 7 revised here), ADR 0035 (sidecar protocol), ADR 0041,
ADR 0042 (decision 5 tool-count wording revised here), ADR 0044 (terminology
list), ADR 0045 (record granularity).

## Context

`Facade` has grown to 31 methods: 6 private helpers, `config`, and 24
`@mcp_exposed` methods (18 implemented, 6 placeholders). The class now mixes
three unrelated concerns — mission-design tasks, catalog data management, and
spatiography analysis — behind one name. A reader who wants to design an orbit
must scan catalog and partition-analysis methods to find the five that matter.

Two recorded decisions produced this shape. ADR 0014 decision 2 defines MCP
tools as the complete set of Facade methods, so every new capability had to
land on `Facade` to become callable. ADR 0031 decision 7 put the catalog query
and sweep methods there for the same reason. Neither decision anticipated that
the class itself is read by humans and is now the main comprehension cost of
the interface layer.

A related symptom: the tool count is quoted as 22 in ADR 0042 decision 5 and in
issue texts, while `tool_specs(Facade())` returns 18. A count defended as a
number drifts; only a rule survives.

## Decision

### 1. Facade keeps task-level methods only

`Facade` retains exactly the five task-level capabilities: `design_orbit`,
`control_orbit`, `transfer_design`, `orbit_propagation`, `spacetime_transform`.
Its private helpers and `config` stay.

### 2. Catalog class

A new interface-layer class holds catalog data management plus family
generation: `catalog_query`, `catalog_get`, `catalog_delete`, `catalog_tag`,
`catalog_export`, `catalog_sweep`, `catalog_promote` (removal scheduled by
ADR 0045 decision 5), `orbit_family_generation`, and the terminology-list
method introduced by ADR 0044. Family generation lands here because
`catalog_sweep` is its batch orchestration — both call the same
`run_family_sweep` kernel, and both produce catalog records.

### 3. Spatiography class

A new interface-layer class holds `spatiography_scales`,
`spatiography_classify`, `spatiography_boundaries`,
`spatiography_resonance_atlas`, `spatiography_dynamical_map`.

### 4. Placeholder methods follow their domain

The six tier-2 placeholders move to the class of their domain when they land;
they do not return to `Facade`. `orbit_stability`, `manifold_analysis` and
`relative_motion` are analysis capabilities; `transfer_search`,
`low_thrust_design` and `low_energy_transfer` are transfer capabilities. Their
final classes are decided when each is implemented, not here.

### 5. Tool inventory scans the exposed classes; the list stays single-sourced

`tool_inventory()` takes the set of exposed instances instead of one Facade and
returns one flat `ToolInfo` list. MCP registration, CLI subcommand derivation
and sidecar preflight keep consuming that one list unchanged — the mechanism of
ADR 0014 decision 2 is preserved, only its scan root widens. Tool names,
schemas and behaviour are unchanged, so MCP and CLI callers see no difference.

### 6. The tool-face growth rule replaces the tool count

ADR 0042 decision 5's "the tool count stays 22" is withdrawn as a criterion. A
new registered tool is admitted when either it is a task-level capability, or
its content is referenced by a field of an existing response and no existing
tool can supply that content (the terminology-list case, ADR 0044). Tool counts
are reported by running the inventory, never quoted from documents.

## Rationale

1. **Class split over face shrink.** Shrinking the MCP face to the five task
   tools was considered and rejected: an agent would keep the auto-ingest
   behaviour of ADR 0031 decision 8 while losing every way to read what was
   ingested, and the GUI would lose its catalog call chain. The comprehension
   cost the split addresses falls on humans reading a class, and splitting
   classes removes it without removing capability.
2. **Family generation with the catalog, not with tasks.** It produces
   many records per call and shares its kernel with `catalog_sweep`; keeping it
   next to `design_orbit` would split one mechanism across two classes.
3. **One inventory, several roots.** Any per-class registry would recreate the
   second list that ADR 0014 decision 2 exists to prevent.
4. **A rule, not a number.** The 22-versus-18 drift is evidence that counts in
   prose decay. The admission rule is checkable at review time.

## Consequences

### Added

- Two interface-layer classes (catalog, spatiography) and a widened
  `tool_inventory()` signature.

### Changed

- ADR 0014 decision 2: MCP tools = the union of `mcp_exposed` methods over the
  exposed interface classes. Decision 5 (CLI symmetry) follows the same union.
- ADR 0031 decision 7: the catalog methods live on the catalog class.
- ADR 0042 decision 5: the tool-count clause is replaced by decision 6 above.
- In-process callers that hold `Facade` for catalog or spatiography work move to
  the new classes. transfer-orbit-design's `facade_bridge` is the only known
  such caller.

### Unchanged

- Tool names, request models, envelopes, binary-frame contract, sidecar
  protocol, CLI subcommand names, and the 18-tool face at the time of writing.
- The five-layer dependency direction; all three classes stay at `api/`.

### Cost

- One breaking change for in-process callers, with no behavioural benefit to
  them — the benefit is comprehension for readers of the interface layer.
