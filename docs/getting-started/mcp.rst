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

Claude Code registers the same binary with a one-liner::

   claude mcp add e2m2e --cwd "C:\path\to\e2m2e-repo" -- .venv/Scripts/e2m2e.exe mcp-serve

For wire-level debugging (list tools, inspect schemas, replay single calls),
the MCP Inspector drives the server interactively::

   npx @modelcontextprotocol/inspector .venv/Scripts/e2m2e.exe mcp-serve

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

Progress notifications
~~~~~~~~~~~~~~~~~~~~~

Long-running tools report MCP progress while they work. Include a
``progressToken`` in the request ``_meta`` and the server emits
``notifications/progress`` with ``progress`` as a monotone fraction in [0, 1]
(``total`` is always 1), plus a human-readable ``message``:

* ``transfer_design`` reports 0 at entry and 1 at completion; the WSB backend
  additionally maps its grid-search task counter into the (0.1, 0.9) window,
  so a large ``n_sun_phase × n_tof`` sweep produces visible intermediate ticks.
* ``orbit_family_generation`` reports stage-level start/end only (a single
  Rust call; no per-member channel yet).

Requests without a ``progressToken`` are unaffected — no notifications, no
overhead. Intermediate notifications are throttled server-side to at most one
per 100 ms (the final fraction-1.0 notification is never throttled), and a
failed delivery never affects the computation.

Tool cancellation
~~~~~~~~~~~~~~~~~~

Long-running tools (``transfer_design``, ``orbit_family_generation``)
execute in one-shot **worker subprocesses** (ADR 0014 amendment), which
makes two protocol events effective:

* ``notifications/cancelled`` for an in-flight request: the worker process
  is terminated and the request is never answered (the cancelled result is
  dropped, not turned into an error).
* client disconnect (stdio EOF): the same propagation — the server tears
  down its handlers and each one kills its worker.

Progress notifications keep flowing while the computation runs (the worker
streams them back to the server). Killing loses only the current task: the
catalog persists records atomically at the end of each tool method, so
committed records are never corrupted by a kill. A worker that dies without
returning a result surfaces as a structured ``WORKER_CRASHED`` error. All
other tools run in-process in the thread pool as before and cannot be
cancelled.

Tool inventory
~~~~~~~~~~~~~~

The list derives purely from Facade method metadata; ``placeholder`` entries are
not registered (they go live automatically once implemented). Currently 18:

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
   * - ``catalog_terminology``
     - Catalog
     - Closed value sets for rendering results: taxonomy label legend
       (canonical → category/family/point/hemisphere/resonance),
       ``orbit_family`` names, ``transfer_type`` values. No parameters; the
       package version is the terminology version (ADR 0044).
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
   * - ``spatiography_scales``
     - 2
     - Analytic scales per region: Laplace/Hill radii, resonance ladder, libration-point distances.
   * - ``spatiography_classify``
     - 2
     - Region classification for a state sample (``zone_ids`` + ``legend``; synodic nondimensional frame).
   * - ``spatiography_boundaries``
     - 2
     - Boundary geometry: synodic-planar boundary circles + Battin asymmetric curves + L1–L5, or
       osculating (a, e) curves.
   * - ``spatiography_resonance_atlas``
     - 2
     - Resonance atlas at an element point: Gallardo half-widths, secular loci, vZLK phase portrait.
   * - ``spatiography_dynamical_map``
     - 2
     - Six-region dynamical map over an (a, e) grid: MEGNO Ȳ field, eight-class fate ids,
       escape/impact diagnostics; large arrays via the sidecar frame contract.

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

Family → pick member → station keep (one record per orbit, ADR 0045; the
family is a queryable label, not a container)::

   orbit_family_generation(orbit_type="HALO", libration_point=2, n_orbits=10)
   catalog_query(family_id=…)                    # the run's member records
   control_orbit(input_record_id=<member record>)

Bulk-fill the catalog::

   catalog_sweep(orbit_types=["HALO","NRHO"], max_amplitudes_km=[…],
                 jacobi_windows=[[3.0,3.2]])

Natural-language example (say this inside your MCP client)::

   Design an L2 southern NRHO with perilune height 3000 km starting at epoch
   2026-01-01, then run 100 Monte Carlo station-keeping simulations on it, and
   tag it "candidate".

Interactive mission design (human-in-the-loop)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The intended operating mode is a reviewed conversation, not a fire-and-forget
script: you state mission intent in natural language, the agent assembles tool
calls from each tool's schema (units, defaults, value domains), and every
result comes back as a unified envelope for you to check before the next step.
Artifacts are auto-ingested into the orbit catalog, so the whole session is
reproducible — each product carries a ``record_id`` and links back to its
inputs via ``source_record_id``.

A reviewed loop looks like this:

1. **State intent** — e.g. "Design an L2 southern NRHO with 2000 km perilune
   starting 2026-01-01."
2. **Agent calls** ``design_orbit(...)`` → envelope ``status="ok"`` with
   ``data.status="converged"`` and ``data.record_id=…`` (auto-ingested).
3. **Review the envelope** — there are two status layers: the transport layer
   (``status``: ok/error with structured ``error.code``) and the numerical
   layer (``data.status``: converged / diverged / stagnated / …). Per the
   soft-failure semantics (ADR 0020) a diverged run is a valid, inspectable
   result — never chain downstream steps on a ``status="ok"`` envelope whose
   ``data.status`` is not ``converged``.
4. **Chain by reference** — "run station-keeping Monte Carlo on it" → the
   agent passes the previous ``record_id`` as ``input_record_id``; the product
   record points back automatically.
5. **Annotate and package** — ``catalog_tag`` attaches review notes;
   ``catalog_export`` bundles a tagged subset into a zip for handover.

Verified end-to-end session (13 calls, ~1 minute on a release build)::

   catalog_query({})                                  # baseline member records
   design_orbit(orbit_type="NRHO", north_south=2, perilune_height=2000,
                phase=0.5, epoch=[2026,1,1,0,0,0.0],
                duration=691200.0, output_step=7200.0)    # converged, ~22 s
   catalog_tag(record_id=…, tags=["candidate"], note="…")
   orbit_family_generation(orbit_type="HALO", libration_point=2, n_orbits=5)
   catalog_query(family_id=…)                  # member records of the run
   design_orbit(orbit_type="DRO", amplitude=15000, phase=0.5001,
                epoch=[2026,1,1,0,0,0.0],
                duration=7776000.0, output_step=7200.0)   # converged, ~17 s
   catalog_tag(record_id=…, tags=["evader"], note="…")
   design_orbit(orbit_type="HALO", collinear_point=2, amplitude=30000,
                phase=0.0, epoch=[2026,1,1,0,0,0.0], duration=3155760.0,
                output_step=3600.0, perturbation={…})     # SK nominal
   control_orbit(input_record_id=…, control_mode=1, control_interval=10.0,
                 num_controls=2, num_monte_carlo=2, perturbation={…})
   catalog_export(tags=["candidate"], dest="case.zip")

Parameter cookbook (verified recipes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convergence is parameter- and epoch-sensitive. Prefer these measured recipes
(release build, kernels as shipped) over ad-hoc scaling:

.. list-table::
   :header-rows: 1
   :widths: 18 42 12 28

   * - Tool
     - Recipe
     - Runtime
     - Notes
   * - ``design_orbit`` NRHO
     - L2 south, ``perilune_height=2000``, ``phase=0.5``, 8-day arc
       (``duration=691200``, ``output_step=7200``)
     - ~20–100 s
     - Converges at epochs 2024 and 2026. A 30-day / 5000 km arc converges
       at 2024 but not at 2026 (segmented shooting residual ~2e4 km), and
       year-scale arcs do not converge either — do not scale ``duration``
       blindly.
   * - ``design_orbit`` DRO
     - ``amplitude=15000``, ``phase=0.5001``, 90-day arc
     - ~17 s
     - The default ``amplitude=10000`` stalls the initial-guess differential
       correction (80 iterations, 29 m residual). Pick amplitudes near
       baseline-catalog members (``catalog_query(orbit_family="dro")``).
   * - ``control_orbit``
     - HALO L2 30000 km nominal (36.5-day arc) with the same perturbation
       switches as the design; ``control_mode=1``, ``control_interval=10``,
       ``num_controls=2``
     - seconds
     - Contract: control horizon + ``feedback_arc`` must fit inside the
       nominal ephemeris span (the nominal view interpolates, it does not
       extrapolate). Long-horizon station-keeping needs its own tuning —
       a 9 × 7-day DRO setup was observed to diverge (per-burn Δv
       0 → 3.5 → 6.0 → 86.5 m/s, breaching ``thrust_max``).

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
5. ``control_orbit`` reads the referenced record's ephemeris segment as the
   nominal and only interpolates within its span: the control horizon
   ``(num_controls-1) * control_interval`` plus ``feedback_arc`` must fit
   inside the designed arc, so design the nominal with enough ``duration``
   headroom.
6. Long-arc CR3BP ephemeris correction is parameter- and epoch-sensitive
   (see the cookbook above); a request that converges at one epoch may fail
   at another. Failed corrections surface as ``DESIGN_FAILED`` with the
   residual in the message — not as a crash.

Troubleshooting
~~~~~~~~~~~~~~~

- ``DESIGN_FAILED … SPICE(FRAMEDATANOTFOUND)`` (ITRF93) at future epochs:
  the body-fixed kernel list must load the predictive PCK
  ``SPICEEarthPredictedKernel.bpc`` *before* the historical
  ``earth_latest_high_prec.bpc`` (overlap prefers the later-loaded,
  higher-accuracy historical data; the predictive file extends coverage to
  ~2037). Fixed in the current tree (issue #556); on older checkouts keep
  epochs inside the historical file's coverage.
- ``data.status`` renders as ``"<ConvergenceState>"`` instead of a value like
  ``"converged"``: envelope enum degradation on the fallback serialization
  path (family / catalog_get responses), fixed in the current tree (issue
  #557).
- A fresh catalog answers ``catalog_query({})`` with the packaged CR3BP
  baseline's member records (592 at the time of writing): the baseline
  distribution bundles expand into per-member records on first open
  (ADR 0036 / ADR 0045) — that is intentional, not leftover state.
- ``pytest tests/api/test_mcp.py`` reports the whole file as skipped: the
  ``[mcp]`` extra is missing (the module uses ``importorskip``). Install with
  ``uv sync --extra mcp`` and re-run.

Next steps
~~~~~~~~~~

- :doc:`installation`: installation & the ``[mcp]`` extra
- :doc:`../api/e2m2e`: API reference for Facade / models / MCP modules
- ``docs/adr/0014-api-facade-mcp-cli.md``: interface-layer design decision (ADR,
  outside Sphinx builds)
