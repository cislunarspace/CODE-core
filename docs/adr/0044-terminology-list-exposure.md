# ADR 0044: Terminology list exposure — closed value sets leave the repository through one registered tool

**Status**: Adopted (implemented — constants at the data layer, single tool
registered, cross-layer sync locked by tests; the request-side `valid_ranges`
outlet remains the open half of ADR 0014 decision 8)
**Date**: 2026-09-01
**Related Issue**: #609
**Related**: ADR 0011 (five-layer architecture), ADR 0012 (dependency
direction), ADR 0014 (decision 8 — conditional value domains are public and
single-sourced; completed here), ADR 0031 (catalog records), ADR 0035 (sidecar
protocol), ADR 0042 (decisions 1 and 5 — the label table and the no-new-tool
clause), ADR 0043 (interface class split), ADR 0045 (record granularity).

## Context

ADR 0042 defined 42 taxonomy labels and stamped measured labels onto records and
responses, but decided against a new tool: the structured form of each label
stayed inside `label_legend()` in the algorithm layer, and nothing in the
repository calls it. A caller receiving `taxonomy_labels: ["halo_l2_northern"]`
has no way to ask which category, libration point or hemisphere that label
denotes.

The consequence is already visible downstream. transfer-orbit-design hardcodes
the category rule (a `resonant_` prefix plus four moon-centered labels) in its
frontend and keeps a family-name list of its own, re-checked by hand on every
e2m2e upgrade. The record-side `orbit_family` names have no single closed set in
this repository either: they are the image of the ingest mapping plus the
lower-cased generator types, spread over three places.

ADR 0014 decision 8 already forbids exactly this: value domains must be exposed
through machine-readable public interfaces, and callers must not keep local
copies. The promise was made; the outlet was never built. Labels are also
expected to grow, so a hand-copied list drifts by construction.

## Decision

### 1. Scope: the closed value sets callers need to render results

The terminology list covers the taxonomy labels in structured form (canonical
string, category, family, libration point, hemisphere, resonance), the
record-side `orbit_family` names, and the `transfer_type` values. Free text
(`note`), open identifiers (`record_id`, `source_tool`) and numeric envelopes
are not terminology and stay out.

### 2. The constants live at the data layer

The 42-label table and its structured forms move from
`algorithm/orbit_taxonomy/labels.py` to the data layer as static reference
data, joined there by the `orbit_family` closed set and the `transfer_type`
values. The classifier keeps its criteria cascade and stays in the algorithm
layer (ADR 0042 decisions 2 and 3 unchanged); it reads the table from the data
layer, which is the permitted dependency direction. The ingest expectation map
derives its family names from the same constant instead of restating them.

### 3. One registered method on the catalog class

The catalog class of ADR 0043 gains `catalog_terminology`, an
`mcp_exposed` method returning the three lists in one response. It is
registered like any other tool, so all call chains reach it: MCP, CLI,
sidecar, and in-process. It is admitted under ADR 0043 decision 6 second
clause — its content is referenced by the `taxonomy_labels` field of existing
responses and no other tool can supply it.

Named `catalog_terminology`, not `catalog_vocabulary` as proposed in #609:
CONTEXT.md's language conventions fix "terminology list" as the term and list
"vocabulary" under avoid.

### 4. The package version is the terminology version

The response carries no version field. A terminology list is frozen per package
release; a caller fetches it once per session and refreshes after upgrading
e2m2e. Callers render an unrecognised label as its canonical string, which is
self-describing by ADR 0042 decision 1.

### 5. Exposure through data files or algorithm-layer imports is not sanctioned

Callers must not read package data files or import the algorithm-layer table to
obtain terminology. Those routes serve only in-process Python and would leave
the GUI's sidecar call chain, MCP agents and the CLI to keep copies.

## Rationale

1. **A registered tool, because the call chain decides.** The sidecar admits
   only registered tools, and the GUI's catalog access is moving there
   (ADR 0035). A data-layer class or a package JSON file is unreachable from a
   separate process; response metadata would ship 42 entries with every query
   and give no way to refresh without a dummy query.
2. **Constants at the data layer, criteria at the algorithm layer.** The label
   table is a static table — reference data by nature. The classification
   criteria are analysis. Splitting them lets both the classifier and the
   interface read one table without an upward dependency.
3. **One method, not four.** Separate tools for labels, families and transfer
   types would fragment one question — what values may appear — into three
   round trips and three tools.
4. **No version field.** Package version already pins the table, and CHANGELOG
   plus this ADR's amendments already announce changes. A second version
   mechanism would need its own maintenance without answering a question the
   package version leaves open.

## Alternatives compared

- **A derived `record_kind` field** (proposal 1 of #609) was rejected. The
  single-record model of ADR 0045 removes the container ambiguity that made a
  product-kind field attractive: with one trajectory per record, the only
  content question left is mission orbit versus transfer orbit, answered by
  `transfer_type`. Field semantics and segment conventions are specified there
  instead.
- **Legend attached to `catalog_query` responses**: rejected, see rationale 1.
- **Static JSON shipped in the package**: rejected, see decision 5 — and it
  would be a second copy of a table the repository already holds.

## Consequences

### Added

- A data-layer terminology module; `catalog_terminology` on the catalog class,
  hence one more registered tool, one more CLI subcommand.

### Changed

- ADR 0014 decision 8 is satisfied for the catalog value sets.
- ADR 0042 decision 5's no-new-tool clause is superseded for this case by
  ADR 0043 decision 6.
- `algorithm/orbit_taxonomy` imports its label table from the data layer;
  `label_legend()` becomes the data layer's function.
- transfer-orbit-design deletes its hardcoded category rule and family list.

### Unchanged

- Classification criteria, stamping at ingest, response enrichment, canonical
  strings as serialization keys (ADR 0042).
- The request-side conditional ranges (`valid_ranges`) keep having no Facade
  outlet; that is the remaining half of ADR 0014 decision 8 and belongs to its
  own issue.

### Cost

- One more tool on the face, justified under the admission rule rather than by
  count.
