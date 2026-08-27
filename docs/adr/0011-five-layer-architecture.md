# ADR 0011: Five-layer architecture and radical full renaming

**Status**: Adopted (implemented)
**Date**: 2026-07-31
**Related**: `docs/architecture/architecture.md`

## Context

e2m2e's current top-level layout is a product of historical evolution: `core`
(systems/frames/force models), `algorithms` (numerical algorithms),
`transfer` (transfers), `dfh` (DFH alignment layer), `io` (DFH formats),
`visualization`, `mbse`, `proximity`, `integrators`. Functionality landed
(FR1–FR5) but responsibility boundaries rest on convention, not structure:

- `dfh/` is the DFH-alignment layer, organized by provenance instead of
  domain; one functional layer straddles data, algorithm, and numerical tiers.
- `io/` sits awkwardly between the data layer and DFH specifics; its role is
  vague.
- No unified entry point (Facade): users assemble
  `CR3BP_System → Dynamics → DifferentialCorrection` themselves.
- No MCP; README's LLM+Agent-callable vision unfulfilled.

Requirement: design the final-form software architecture, leave code templates
for later implementation, and document not-yet-implemented features in docs
and README.

## Decision

**Five-layer architecture**: data layer `data/` → numerical layer `crates/`
→ algorithm layer `algorithm/` → interface layer `api/` → tools layer
`tools/`. Inner layers never sense outer layers.

**Radical full renaming**: existing `core/algorithms/transfer/dfh/io/
visualization` all migrate into the new five layers with no legacy packages
kept. `dfh/` is dismantled (five capabilities return to their domains), `core`
is dismantled (no top-level core), `algorithms` → `algorithm` (singular),
`io/` stays out of e2m2e entirely (DFH format interop is temporary scripts).

**Transition strategy**: keep old paths alive via `sys.modules` aliases;
rename in batches along dependency order (data first → numerical/algorithm →
api/tools); one commit per batch with tests green before the next. Current
HEAD is the product baseline.

## Rationale

1. **Structure-enforced replaces convention-enforced**: layering enforced by
   directory structure + dependency rules (ADR 0012), not good intentions.
2. **`dfh/` dismantled**: orbit design/station keeping/transfers/prediction/
   spacetime conversion are e2m2e's own domain capabilities, not things that
   exist to align with DFH. DFH was only a development-time reference
   (ADR 0013).
3. **`io/` kept out of e2m2e**: DFH-format interoperability was a
   development-time stopgap; final e2m2e is an independent library
   (ADR 0013).
4. **Facade/MCP/CLI**: turning the library into a toolset callable by LLMs and
   humans alike delivers on README's vision.

## Consequences

- New `docs/architecture/architecture.md` describing final form.
- Top-level layout: data/algorithm/api/tools + integrators.py + mbse +
  _integrators.
- Migration proceeds in dependency-order batches, each batch regression-tested.
