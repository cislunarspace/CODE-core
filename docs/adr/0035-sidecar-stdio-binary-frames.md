# ADR 0035: GUI sidecar stdio protocol — shared Facade envelope, large arrays over binary frames

**Status**: Adopted
**Date**: 2026-08-22
**Related**: ADR 0014 (interface Facade/MCP/CLI), issue #518

## Context

tod (transfer-orbit-design) is migrating its UI to Tauri (Rust shell + web
frontend). e2m2e is a Python library that Rust cannot call in-process, so the
Facade must be consumed via a resident subprocess (sidecar) + stdio messaging.

Measurements (tod repo `tools/bench_serialization.py`): encoding a full halo
family (26882 members, 32M floats) as JSON takes 17.9 s and 687 MB — slower than
computing it; raw f64 binary takes ~98 ms / 258 MB. Conclusion: control messages
can be JSON text lines; large arrays must be binary frames.

## Decision

New `e2m2e serve-stdio` CLI subcommand (sibling of `mcp-serve`) as the GUI
sidecar entry. The protocol has layers:

### 1. Control layer: JSON lines reusing the unified envelope

Requests, responses, and progress events are one JSON object per line. Responses
reuse ADR 0014 §4's unified envelope `{status, data, error, meta}`, defined in
`e2m2e/api/mcp/envelope.py` — single source shared by MCP and sidecar transports;
protocol evolution maintains one place. Tool surface = Facade methods with
`mcp_exposed=True` (pure derivation, no new business logic).

### 2. Progress lines

No reuse of MCP notifications (whose semantics serve LLM clients). GUI needs only
discardable progress lines while tasks run; consumers are a match in the Rust
shell:

```json
{"status": "progress", "data": null, "error": null, "meta": {"job_id": "...", "percent": 0.42, "message": "..."}}
```

`percent` is a 0–1 float; consumers must skip unknown `status` lines without
erroring.

### 3. Binary frames: cross-repo persistent contract

When a JSON line carries `"binary_frames": N` (default 0), N binary frames follow
the newline sequentially, after which JSON-line flow resumes. **The frame format is
a tod ↔ e2m2e cross-repo persistent contract**, defined field-by-field below (all
multi-byte integers little-endian):

```
Offset Length Type      Meaning
0     4    u32   magic = 0x324D3245 (ASCII "E2M2" as LE u32; bytes 45 32 4D 32)
4     1    u8    dtype: 0 = float32, 1 = float64
5     1    u8    ndim: array dimension count, ≥ 1
6     4·ndim  u32[]  shape: element counts per dim (not bytes), LE
6+4·ndim  var  var    raw array bytes: C-contiguous (row-major), LE,
                  length = prod(shape) · element width
```

- Both f32 and f64 supported; requesters declare dtype in request parameters. tod
  canvas rendering uses f32 (Three.js `BufferAttribute` is f32 anyway — f64 gets
  truncated in-browser); intermediate quantities and disk-persisted recomputables
  stay f64.
- No version field: magic doubles as the version anchor; incompatible changes get a
  new magic, old+new coexisting as two protocol versions.
- No multi-array packed headers: N independent frames in sequence, count declared by
  the line's `binary_frames`.
- Corresponding JSON fields (e.g., flattened state lists) become `null` placeholders
  or are omitted — binary frames are the sole real body; which field maps to which
  frame is per-tool response-model convention, evolving with models.

## Rationale

1. **Envelope single-source**: `envelope.py` implements plain dicts + pure
   functions with no MCP-specific coupling — cross-transport reuse costs zero.
2. **Frame format fixed once**: cross-language ABIs are painful to change, so
   minimize fields: magic, dtype, shape, bytes — four items. Complex layouts
   (packed headers, versions, alignment padding) all rejected.
3. **Progress via envelope**: avoids a second event mechanism for GUI; `status` is
   an existing field needing only extended values.

## Consequences

- `e2m2e serve-stdio` subcommand + sidecar protocol module (frame codec decoupled
  from transport, unit-testable).
- tod's Rust shell implements peer decoding per this ADR §3; frame-format changes
  require a new ADR.

## Revision (2026-09-01, #601)

The sidecar delegates execution to the transport-neutral core
(`e2m2e/api/execution.py`, see ADR 0014 amendment): validation, error
translation, and canvas frame extraction live there; this module keeps only
wire framing (JSON lines, job_id progress lines, `binary_frames` count and
frame ordering). The frame codec moved from `api/sidecar/frames.py` to
`api/frames.py` — byte format and magic unchanged, so the cross-repo
contract is unaffected. The unknown-tool response code is `TOOL_NOT_FOUND`
(previously `UNKNOWN_TOOL` here), unified with the MCP transport.

## Amendment (2026-09-01): job cancellation and live progress (#607)

### Context

The synchronous run_loop could not observe a cancel intent: while a tool
executed, stdin went unread, and minutes-long family generation ran to
completion regardless. The deferral recorded in ADR 0014's amendment (#601)
is superseded by a maintainer decision: e2m2e is a foundational toolset —
its wire contract should be complete ahead of a concrete consumer.

### Decisions

1. **Cancel request line**: a request of the shape `{"cancel": "<job_id>"}`
   kills the in-flight long-running job (worker subprocess kill) and emits
   `{"status": "cancelled", "data": null, "error": null, "meta":
   {"job_id": ...}}`. Semantics are idempotent — cancelling an unknown or
   finished job still emits the cancelled line and changes nothing else.
   The cancelled request itself gets no result line (mirrors MCP
   semantics). Consumers already tolerate unknown `status` values
   (§2), so the extension is backward compatible.
2. **Concurrent run_loop**: long-running tools (execution core
   `LONG_RUNNING_TOOLS`) execute in a worker subprocess on a background
   thread; the reader keeps consuming stdin during computation. Short
   tools stay inline. Cancel registration happens synchronously in the
   reader (event + process slot), so a cancel line racing job startup is
   deterministic. Each job's output bytes (line + frames) are written
   atomically under a lock — frames never interleave.
3. **Live progress**: long-tool progress lines now forward the algorithm
   layer's real fractions via the worker, superseding the single fake
   percent=0 start line for those tools; short tools keep the start line.
4. **Worker subprocess protocol** (internal to e2m2e, not the cross-repo
   contract): the request may carry `binary_dtype`; the result line may
   declare `binary_frames: N` followed by N raw frames in the §3 format.
5. Multiple concurrent long jobs are allowed (keyed by `job_id`); jobs
   without `job_id` run uncancellable. No queueing policy is defined.

### Consequences

- tod gains stop-button support: send a cancel line, tolerate the
  `cancelled` status, correlate by `job_id`.
- `handle_request` remains the single-request inline seam (uncancellable);
  cancellation is a run_loop (process) feature.
