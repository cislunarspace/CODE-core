# ADR 0014: Interface layer — Facade / MCP / CLI

**Status**: Adopted (implemented — Facade, MCP, sidecar, and the CLI↔MCP
symmetric subcommands, #602; decisions 2 and 5 revised by ADR 0043, decision 8
completed for catalog value sets by ADR 0044)
**Date**: 2026-07-31
**Related**: ADR 0011 (five-layer architecture), README vision
(LLM+Agent-callable)

## Context

e2m2e's current external shape is loose APIs (users assemble
`CR3BP_System → Dynamics → DifferentialCorrection`) with no unified entry, no
MCP, no CLI. README's vision requires LLMs to invoke Earth-Moon orbit
algorithms as they would Lambert or C-W tools. The interface layer (`api/`)
is where that vision gets delivered.

## Decision

1. **The Facade is the sole entry point**, its methods mapping to task-level
   capabilities (coarse granularity). The algorithm layer keeps fine-grained
   APIs for experts. Two granularity tiers.
2. **Pure derivation + metadata markers**: MCP tools = the complete set of
   Facade methods; Facade methods carry `mcp_exposed: bool` metadata (tier-1/
   tier-2 True; tier-3/auxiliary False). Registration scans Facade methods;
   the list has one source of truth.
   *(Revision note 2026-09-01, #610: ADR 0043 decision 5 widens the scan root —
   MCP tools = the union of `mcp_exposed` methods over the exposed interface
   classes (Facade, catalog, spatiography). The single-source mechanism itself
   is unchanged.)*
3. **Pydantic models all hand-written**: input/output/error models carefully
   specify parameter units, defaults, value domains. They stay at the `api/`
   boundary, never entering the algorithm layer.
4. **Facade returns dedicated Pydantic models**; the MCP transport wraps a
   uniform envelope ({status, data, error, meta}). Errors are translated in
   `api/`: exceptions → structured error codes (OrbitError with code/message/
   details).
5. **CLI subcommands = Facade methods** (those with mcp_exposed=True),
   parameters generated from the same Pydantic models. CLI and MCP are fully
   symmetric.
   *(Revision note 2026-09-01, #610: subcommands derive from the same union as
   decision 2, see ADR 0043 decision 5.)*
6. **MCP deployment = in-process library as the main body + thin CLI wrapper
   `mcp-serve`**: `create_server(facade)` function + `e2m2e mcp-serve`
   subcommand. One Facade instance = one server.
   *(Revision note 2026-08-30, #588: the two long-running tools now execute
   in worker subprocesses — each call constructs its own worker Facade via
   environment inheritance; see the amendment at the end of this ADR.
   Everything else keeps the in-process path.)*
7. **config.py constructor injection**: `Facade(config=Config(...))`,
   covering only runtime environment (kernel paths/precision thresholds/
   logging); physical constants belong to data/templates/. SPICEManager's
   global handles and r2s2's process singleton are known limitations managed
   explicitly via Config.
8. **Conditional value domains are public and single-sourced**: input-model
   value domains depending on other fields must be exposed through
   machine-readable public interfaces; validators and those interfaces share
   one rule definition. GUIs, CLIs, and MCP must not parse error text, read
   validator source, or maintain local copies of ranges.
   *(Revision note 2026-09-01, #609: ADR 0044 builds the outlet for the catalog
   closed value sets (taxonomy labels, family names, transfer types) through
   `catalog_terminology`. The request-side conditional ranges (`valid_ranges`)
   still have no registered outlet — the remaining half of this decision.)*

## MCP tool list

- Tier 1 task-level (stable skeleton, will grow): design_orbit / control_orbit
  / transfer_design / orbit_propagation / spacetime_transform.
- Tier 2 subtask-level (will grow): orbit_family_generation / orbit_stability
  / transfer_search / low_thrust_design / manifold_analysis /
  low_energy_transfer / relative_motion.
- Tier 3 auxiliary (not registered): porkchop / normal_form / safety /
  visualize / format I/O.

## Rationale

1. **Pure derivation**: Facade methods are the single source; adding a
   capability = adding a Facade method + hand-written model = MCP tool + CLI
   subcommand both appear automatically; the list never drifts.
2. **Hand-written models**: tier-1 is what Agents use most; schemas need care
   (units/defaults/domains written plainly in Pydantic); where conditional
   domains exceed static schema expressiveness, model public interfaces
   supplement them — for long-term maintenance quality.
3. **CLI↔MCP symmetry**: the same Facade method serves MCP for Agents and CLI
   for humans, validated by the same model set.

## Consequences

- The `api/` layer provides Facade/config/models/mcp/cli.
- transfer-orbit-design keeps an independent repo hosting only the GUI
  (deprecating tod/generates algorithm script layers, superseded by e2m2e
  CLI); GUI parameter forms are generated from e2m2e Pydantic models and their
  conditional-domain public interfaces.

## Amendment (2026-08-30): long-running tools execute in worker subprocesses (#588, #576 Phase 2)

### Context

Long-running tools (`transfer_design`, `orbit_family_generation` — minutes at
production scales) run to completion with no way to abort: `notifications/
cancelled` and client disconnects were unobserved. The python mcp SDK 2.x
delivers both as cancellation of the handler's task scope (interrupt mode
resp. task-group teardown on EOF) — but the handler's `anyio.to_thread.
run_sync` thread cannot be killed, and cooperative checkpoints fail inside
GIL-released Rust sections (family generation is a single Rust call). A
process is the only reliable interruption unit.

### Decisions

1. **Execution-strategy table at the transport layer**: `LONG_RUNNING_TOOLS`
   in `api/mcp/server.py` routes `tools/call` for the two long-running tools
   to a one-shot worker subprocess; every other tool keeps the thread-pool
   path unchanged. Routing is a transport concern (spawn cost vs.
   computation cost), not Facade metadata — the Facade signatures stay as
   they are.
2. **Worker protocol**: `python -m e2m2e.api.mcp.worker` reads one JSON
   request line from stdin (`{"tool", "arguments"}`), writes throttled
   progress lines and one final result line (the unified envelope) to
   stdout, exits 0. The worker imports no mcp SDK (envelope/tools/facade
   only — same dependency constraint as the sidecar). Errors translate
   inside the worker; stderr passes through to the server's logs.
3. **Cancellation = kill**: the server-side await is fully cancellable; on
   cancellation (peer cancel notification or EOF) the worker is killed and
   reaped inside a shielded scope, then the cancellation propagates. The
   cancelled request is never answered (the SDK drops its result).
4. **Data safety**: catalog persistence is one atomic write per record
   (tmp + `os.replace`, ADR 0031) at the end of the tool method — a kill
   loses at most the current task's record, never corrupts committed ones.
5. **Environment via inheritance**: the worker constructs its own
   `Facade(Config())`; env vars (`E2M2E_CATALOG_DIR`, …) carry the
   environment. Records a worker commits are visible to the parent
   (SQLite index, no in-process caching).
6. **Failure envelope**: a worker exiting without a result line surfaces as
   `WORKER_CRASHED` (structured error; exit code in the message).

### Consequences

- Progress forwarding survives the process boundary: the worker emits
  progress lines; the parent forwards them from inside the event loop (the
  thread-path reporter would deadlock if driven on the loop thread).
- Windows spawn cost is ~1.3 s per worker roundtrip (interpreter boot +
  imports; measured 2026-08-30 on the dev machine) — noise against
  minutes-long computations; short tools never pay it.
- One worker process per call: no state reuse; a killed worker never
  serves another request.

## Amendment (2026-09-01): transport-neutral execution core (#601)

### Context

The two transports (MCP server, GUI sidecar) each re-derived the tool
surface and drifted: different unknown-tool error codes (TOOL_NOT_FOUND vs
UNKNOWN_TOOL), duplicated canvas-contract frame extraction, per-call
rebuild of every tool's JSON schema, and — a correctness bug — worker
subprocesses rebuilt `Config()` from the environment, silently discarding
the caller's constructor-injected Config on long-running tools.

### Decisions

1. **One execution entry**: `e2m2e/api/execution.py` owns tool-spec lookup,
   validation, error translation, and canvas frame extraction.
   `execute_tool(facade, tool, arguments, *, progress_callback, binary_dtype)
   -> (envelope, frames)` is the only execution path; the MCP server wraps
   `CallToolResult`, the sidecar wraps JSON lines + frames, the worker
   subprocess calls the same core. `preflight()` centralizes the
   unknown-tool and dtype checks so adapters can short-circuit before
   protocol framing.
2. **`LONG_RUNNING_TOOLS` moves to the execution core** (no `[mcp]` extra
   dependency): one routing list consumed by both transports. The MCP
   server keeps the async worker pump; a sidecar worker route is deferred
   until tod asks for it (needs a worker-protocol frames extension and a
   wire-level cancel message — protocol extensions belong with their
   consumer).
3. **Injected Config crosses the process boundary**: the worker request
   carries `{"tool", "arguments", "config"}`; the worker rebuilds its
   Facade from `Config.from_payload`. `from_payload` rejects unknown
   fields and non-object payloads with `INVALID_PARAMS` — no silent
   fallback to environment defaults when a config was supplied.
   **The `Config` field set is now a cross-process contract** (the
   docstring's "骨架，字段待定稿" caveat ends here): adding or removing a
   field is a protocol change and must update `to_payload`/`from_payload`
   together.
4. **Error codes unified**: unknown tool is `TOOL_NOT_FOUND` everywhere
   (sidecar's `UNKNOWN_TOOL` retired); no new codes introduced.
5. **Frame codec relocates** from `api/sidecar/frames.py` to
   `api/frames.py`: the execution core's canvas extraction encodes frames,
   so the codec must sit at core level. Byte format, magic, and the
   ADR 0035 contract are untouched.
6. **Spec lookup is on-demand**: `tools.tool_spec(facade, name)` builds one
   schema; per-call cost no longer grows with the tool count.

### Consequences

- Adding a Facade tool remains a one-place registration; both transports
  and the worker inherit it.
- `Facade` gains a read-only `config` property (the worker-request builder
  serializes it).
- `_finite_or_none` is public (`finite_or_none` in `api/catalog_ingest.py`);
  three consumers no longer import a private name across modules.
