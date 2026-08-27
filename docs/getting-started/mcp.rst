Using e2m2e through MCP
=======================

e2m2e packages task-level capabilities (orbit design, station-keeping
simulation, transfer design, orbit catalog, etc.) into MCP (Model Context
Protocol) tools callable directly by LLM Agents (ADR 0014). In an MCP-capable
client you write no code and fill no parameter tables — describe the task in
natural language, and the model picks tools and assembles arguments from each
tool's schema (which documents units, defaults, and value domains).

Setup
~~~~~

The MCP server ships with the ``[mcp]`` extra and starts over stdio transport::

   pip install "e2m2e[mcp]"        # or uv sync --extra mcp
   e2m2e mcp-serve                 # stdio JSON-RPC, no listening port

Register it in your MCP client configuration (``mcpServers`` format of Claude
Desktop / Cursor)::

   {
     "mcpServers": {
       "e2m2e": {
         "command": "/path/to/.venv/Scripts/e2m2e.exe",
         "args": ["mcp-serve"],
         "cwd": "/path/to/e2m2e-repo"
       }
     }
   }

ZCode workspace configuration (``<repo>/.zcode/config.json``) is analogous with
nested key ``mcp.servers``::

   {
     "mcp": {
       "servers": {
         "e2m2e": {
           "type": "stdio",
           "command": "C:\\path\\to\\.venv\\Scripts\\e2m2e.exe",
           "args": ["mcp-serve"],
           "cwd": "C:\\path\\to\\e2m2e-repo"
         }
       }
     }
   }

.. note::

   Pin ``cwd`` at the repo root: the SPICE kernel directory (``kernels``) and
   catalog directory (``catalog``) default to relative paths. Alternatively set
   absolute paths via ``SPICE_KERNEL_DIR`` / ``E2M2E_CATALOG_DIR``.

Unified envelope
~~~~~~~~~~~~~~~~

Every tool returns the unified envelope ``{status, data, error, meta}``: on
success ``status="ok"`` and ``data`` is that tool's response model; on failure
``status="error"`` with structured error codes in ``error``
(``INVALID_PARAMS``, ``RECORD_NOT_FOUND``, ``PROPAGATION_FAILED``, …) — no
traceback leakage. Numerical tools additionally report algorithm convergence in
``data.status`` (converged / diverged / stagnated / max_iterations / …; ADR 0020's
soft-failure semantics: divergence is a valid result too).

Tool inventory
~~~~~~~~~~~~~~

The list derives purely from Facade method metadata; ``placeholder`` entries are
not registered (they go live automatically once implemented). Currently 13:

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - Tool
     - Tier
     - Purpose & key inputs
   * - ``design_orbit``
     - 1
     - Mission orbit design. ``orbit_type`` ∈ DRO/DPO/HALO/NRHO/LISSAJOUS/L4/L5/
       AXIAL/SPO/LPO/HORSESHOE/ELFO plus per-family shape parameters (most have
       defaults); produces an ephemeris-corrected nominal orbit, auto-ingested.
   * - ``control_orbit``
     - 1
     - Station-keeping Monte Carlo simulation. Input either ``input_record_id``
       referencing a catalog record or ``input_ephemeris`` given directly;
       control modes 1–6, navigation/thrust errors, force-model orders, etc.
   * - ``transfer_design``
     - 1
     - Transfer design. ``transfer_type`` ∈ HMN/LGA/WSB/low_thrust; ``tli_epoch``
       is the TLI epoch; LGA/WSB need ``target_ephemeris`` — frame contract in
       caveats below.
   * - ``orbit_propagation``
     - 1
     - Orbit prediction. GCRS six-dim initial state + epoch + duration (seconds).
   * - ``spacetime_transform``
     - 1
     - Spacetime conversion: synodic↔J2000, GCRS↔EBCRS. For synodic conversion
       ``times`` are nondimensional synodic times t_syn (0 = reference epoch
       ``et0_jd``); GCRS↔EBCRS uses JD_TDB and needs ``ephemeris_path``.
   * - ``orbit_family_generation``
     - 2
     - Family continuation generation (eight families HALO/NRHO/AXIAL/LISSAJOUS/
       SPO/LPO/HORSESHOE/DRO) with per-family amplitude/direction parameters;
       family records auto-ingested.
   * - ``catalog_query``
     - Catalog
     - Multi-dimensional filtered query (family, libration point, Jacobi range,
       amplitude range, tags, convergence status) returning summaries.
   * - ``catalog_get``
     - Catalog
     - Full record by ``record_id`` (incl. CR3BP & ephemeris segment arrays).
   * - ``catalog_tag``
     - Catalog
     - Teaching annotations: ``tags`` replaces wholesale; ``note`` free text.
   * - ``catalog_promote``
     - Catalog
     - Lift a family member (``member_index``) into a standalone record;
       ``source_record_id`` points at its family.
   * - ``catalog_export``
     - Catalog
     - Package-and-export by query: ``dest`` ending in ``.zip`` produces an
       archive, otherwise a directory; directly openable as a catalog.
   * - ``catalog_sweep``
     - Catalog
     - Parameter-space batch generation + ingestion: family × libration point ×
       amplitude grid / energy windows / LISSAJOUS 2-D amplitude grid.
   * - ``catalog_delete``
     - Catalog
     - Delete a record by ``record_id`` (file + index entry), irreversible.

Every artifact-producing tool auto-writes to the orbit catalog on success
(ADR 0031), returning ``record_id`` — the handle for chained cross-tool calls.

Typical workflows
~~~~~~~~~~~~~~~~~

Design → station keep → annotate::

   design_orbit(orbit_type="NRHO", north_south=2, perilune_height=3000)
     └─ yields record_id
   control_orbit(input_record_id=<previous>, control_mode=1)
     └─ station-keeping product points back — lineage intact
   catalog_tag(record_id=…, tags=["teaching"], note="…")
   catalog_export(orbit_family="nrho", dest="nrho.zip")

Family → pick member → station keep::

   orbit_family_generation(orbit_type="HALO", libration_point=2, n_orbits=10)
   catalog_query(orbit_family="halo")          # browse family records
   catalog_promote(record_id=…, member_index=3)  # lift member to standalone
   control_orbit(input_record_id=<promoted>)

Bulk-fill the catalog::

   catalog_sweep(orbit_types=["HALO","NRHO"], max_amplitudes_km=[…],
                 jacobi_windows=[[3.0,3.2]])

Natural-language example (say this inside your MCP client)::

   Design an L2 southern NRHO with perilune height 3000 km starting at epoch
   2026-01-01, then run 100 Monte Carlo station-keeping simulations on it, and
   tag it "candidate".

Caveats
~~~~~~~

1. **``transfer_design``'s ``target_ephemeris`` frame contract**: LGA/WSB expect
   physical-unit synodic rotating-frame states (km, km/s); inertial ephemerides
   produced by ``design_orbit`` / ``orbit_propagation`` must first pass through
   ``spacetime_transform(j2000_to_synodic)``, otherwise target geometry is wholly
   wrong. HMN / low_thrust interpret inputs as geocentric inertial.
2. **Units**: durations in seconds (interval-like params of ``control_orbit`` in
   days); angles in degrees; epochs accept UTC ISO strings or
   ``[year, month, day, hour, minute, second]``; keep JD_TDB vs nondimensional
   t_syn distinct per above.
3. **``catalog_delete`` is irreversible**; confirm via ``catalog_get`` before.
4. ``orbit_stability`` requires an ``Orbit`` object bound to period/system — not
   expressible through the JSON envelope yet, so unregistered; revisit once
   record-reference inputs land. Use the algorithm-layer API meanwhile
   (:doc:`../algorithms/stability`).

Next steps
~~~~~~~~~~

- :doc:`installation`: installation & the ``[mcp]`` extra
- :doc:`../api/e2m2e`: API reference for Facade / models / MCP modules
- ``docs/adr/0014-api-facade-mcp-cli.md``: interface-layer design decision (ADR,
  outside Sphinx builds)
