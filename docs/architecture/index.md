# Architecture

e2m2e is the **algorithm toolset infrastructure** in an LLM+Agent mission
planning system: the large model handles intent understanding and
orchestration; e2m2e handles precise and reliable orbit computation. This page
is the reading map for the architecture chapter.

## The five architectural designs

Full narrative in [architecture](architecture.md); summary:

| Module | Responsibility | Key decisions |
|---|---|---|
| Spacetime systems and constants | UTC/TDB/TAI/TT time scales, J2000/ITRF93/GCRS frame conversion, multiple physical constant baselines | ADR 0010, ADR 0022 |
| Rust computation | Four core crates (spice / propagation / forces / integrators), plus two HJB solver crates (levelset / hjb-dynamics, see [hjb-subsystem](hjb-subsystem.md)) | ADR 0002, ADR 0016, ADR 0032 |
| Python orchestration | Construct problem → call Rust iterators → interpret results; task-level Facade | ADR 0014, ADR 0029 |
| CI | Three-platform wheel matrix (Linux x64/ARM, Windows) + CSPICE build package distribution | ADR 0009 |
| Data management | GitHub Release ephemeris data, Git-tracked family seeds, packaged CR3BP baseline family dataset, local catalog | ADR 0031, ADR 0036 |

Division-of-labor principle (ADR 0011 five-layer architecture, ADR 0012
dependency direction): **domain decisions stay in Python, hot loops go to
Rust**. Rust never touches SPICE handles; it consumes pre-sampled injected
ephemeris cache tables. The cspice kernel is global state and cannot be used
concurrently — this constraint determines where the seam sits.

## Chapter reading map

- [architecture](architecture.md), overview: walk through the full chain of
  designing an L2 NRHO to see where each of the five modules participates.
  Read this first.
- [system-dynamics-dataflow](system-dynamics-dataflow.md), deep dive into the
  System and Dynamics class hierarchies: how data flows piece by piece through
  construction, propagation, and result caching.
- [numerics-migration-status](numerics-migration-status.md), migration ledger
  for numerical kernels across algorithm-layer submodules: sunk / migrating /
  intentionally kept in Python, each with reasons and issues.
- [hjb-subsystem](hjb-subsystem.md), target shape of the HJB subsystem:
  two-level division of labor, Hamiltonian seam, dimension ceiling, binding
  entry, verification tiering.
- [hjb-hamiltonian-dataflow](hjb-hamiltonian-dataflow.md), supporting research
  for the ephemeris force-model Hamiltonian: force-model and EphemCache
  status quo, and the data flow of one solve.

## Architecture decision records (ADR)

Each ADR is a decision snapshot recording context, decision, rationale, and
consequences; when decisions change, originals are not rewritten — revisions
are appended or a new entry written. ADRs live in `docs/adr/`, aimed at
development collaboration, not the user documentation site. For the index and
status vocabulary see [`docs/adr/README.md`](https://github.com/cislunarspace/e2m2e/blob/main/docs/adr/README.md).

The highest-impact entries worth reading first:

| ADR | Topic | Relation to this page |
|---|---|---|
| [0011](https://github.com/cislunarspace/e2m2e/blob/main/docs/adr/0011-five-layer-architecture.md) | Five-layer architecture and full renaming | Source of the layering vocabulary |
| [0012](https://github.com/cislunarspace/e2m2e/blob/main/docs/adr/0012-dependency-direction.md) | Dependency-direction rules and CI checks | Who may depend on whom between modules |
| [0002](https://github.com/cislunarspace/e2m2e/blob/main/docs/adr/0002-rust-integrator-core.md) | Rust integrator core | Origin of the Python/Rust seam |
| [0016](https://github.com/cislunarspace/e2m2e/blob/main/docs/adr/0016-ephem-cache-architecture.md) | EphemCache ephemeris cache | How Rust avoids touching cspice |
| [0014](https://github.com/cislunarspace/e2m2e/blob/main/docs/adr/0014-api-facade-mcp-cli.md) | Facade / MCP / CLI same source | Task-level interface model |
| [0024](https://github.com/cislunarspace/e2m2e/blob/main/docs/adr/0024-unified-algorithm-result-status.md) | Unified algorithm result status contract | Shared vocabulary of the result-interpretation layer |
| [0031](https://github.com/cislunarspace/e2m2e/blob/main/docs/adr/0031-orbit-catalog.md) | Orbit catalog | Realization of the data-management module |

## Known open items

Architecture is not finished work. Currently registered review items:

- Three coexisting frame-conversion paths in the spacetime system; time
  conversion responsibility chain needs clarification.
- Dual shooting paths in the Python orchestration layer; numerical residue in
  `algorithm/transfer` to be consolidated.
- Optimization kernels such as NLP and NSGA-II remain in Python and sink to
  Rust gradually per the cadence in
  [numerics-migration-status](numerics-migration-status.md).
