# ADR 0039: Shared-kernel leaf modules at the package root

**Status**: Adopted (implemented)
**Date**: 2026-08-29
**Related Issue**: #545
**Related**: ADR 0011 (five-layer architecture), ADR 0012 (revises its
dependency table and CI enforcement scope), ADR 0016 (dual-instance SPICE
bridge), ADR 0024 (status contract relocation)

## Context

A dependency-direction audit (issue #545) found three seams violating ADR
0012's literal rules, and — worse — that the CI import check could not see
any of them:

1. The numerics facade `e2m2e/integrators.py` imported
   `algorithm.results.ResultStatus` and `data.templates` enums: at the FFI
   boundary it normalizes raw Rust status strings into the ADR 0024 contract
   and validates the triple at construction.
2. The data layer (`data/kernels/manager.py`) imported the facade to reach
   `spice_furnsh`/`spice_unload`/ephemeris-cache switches: the Rust cspice
   and Python spiceypy instances keep independent kernel pools (ADR 0016), so
   kernel load/unload must drive both — but the data layer had no legal
   channel to the extension that did not cross the numerics facade (and,
   through it, the algorithm layer).
3. `design_orbit.py` imported `api.models.DesignOrbitRequest` under
   `TYPE_CHECKING` — runtime-neutral, but visible to any AST-level check.

The checker's three blind spots mapped one-to-one to the three seams: package
-root files were never scanned (the facade is a file, not a layer directory);
relative imports were invisible to `layer_of()` (the repository has hundreds
of them); and `integrators`/`mbse` appeared in no forbidden set, so
data → facade → algorithm laundered through untouched. ADR 0012's promised
CI enforcement was effectively zero over exactly these seams.

## Decision

1. **Shared-kernel leaf modules live at the package root.** The set is
   `exceptions.py`, `status.py`, `spice_ext.py`, and the build-generated
   `_rust_abi.py`. Every layer may import them; they import no layer
   themselves (only stdlib and each other). This formalizes the existing
   `exceptions.py` precedent ("exceptions belong to no single layer"). New
   package-root modules must be registered in the import checker.
2. **The ADR 0024 status contract moves to `e2m2e.status`.**
   `ConvergenceState`, `FailureCause`, `CAUSE_STATUS`, and `ResultStatus`
   have their single definition there; `data.templates` and
   `algorithm.results` re-export them with unchanged object identity. The
   old paths are permanent stable aliases, not deprecation targets.
   Relocation is safe: all persistence and transport use string values
   (FFI strings, npz with `allow_pickle=False`, JSON `.value`, the MCP
   envelope's enum branch), never module paths.
3. **`e2m2e.spice_ext` owns the extension ABI gate and the SPICE bridge
   surface.** It hosts `_check_rust_abi`/`_abi_ok` (process-level cache) and
   exposes `spice_furnsh`/`spice_unload`/`enable_ephem_cache`/
   `disable_ephem_cache` straight from `e2m2e._integrators` (same module,
   same symbol objects as the facade). The data layer consumes it directly;
   dual-instance bridging semantics (ADR 0016) are unchanged. The facade
   keeps a thin `require_rust_extension` shell — ABI core imported from
   `spice_ext`, symbol-presence check still against facade globals — so all
   existing algorithm-layer call sites are untouched. Both gates share one
   missing-symbol error text via `_ensure_symbols`.
4. **`design_orbit` takes a minimal `Protocol`** (`_DesignRequest`, 22
   fields actually consumed) instead of importing the api model even under
   `TYPE_CHECKING`. `api.DesignOrbitRequest` satisfies it structurally;
   mypy checks both directions (algorithm-side attribute access, facade
   call-site compatibility).
5. **The import checker enforces for real.** It resolves relative imports
   against each file's package, expands `from e2m2e import <layer>`-style
   aliases, scans package-root modules (shared-kernel leaves and the
   facade must import no layer; `__init__.py` as composition root and
   `_rust_abi.py` as a build artifact are exempt), and extends the
   forbidden table: data additionally forbids `integrators` and `mbse`;
   algorithm and api additionally forbid `mbse`.
6. **`mbse/` is a non-runtime documentation artifact**, per ADR 0011's
   "top-level attachment" placement and CONTEXT-MAP's "should not be
   depended on at runtime". It is a forbidden import target for all
   runtime layers but is not itself a checked source layer; its pydantic
   use is not subject to the "Pydantic only at the api/ boundary" clause,
   which constrains the algorithm layer only.

## Rationale

1. **Why up to the package root:** the status vocabulary is consumed by
   data, algorithm, and the numerics facade alike; a home inside either
   data or algorithm keeps the reverse dependency. A zero-dependency leaf
   below every layer is the only placement with no illegal edge. The same
   argument applies to the SPICE bridge surface the data layer needs.
2. **Why two extension gates instead of moving `require_rust_extension`:**
   its documented semantics check symbol presence in the *caller's* module
   namespace. The facade's namespace carries 90+ symbols referenced by
   name from 26+ algorithm-layer call sites; moving the function would
   force call-site rewrites or a namespace parameter that breaks the
   signature contract. Two thin shells sharing one ABI core and one error
   text preserve behavior exactly.
3. **Why re-export rather than migrate call sites:** 85 import statements
   for the enums and 17 for `ResultStatus` all go through package paths
   that remain valid. Identity-preserving re-exports make the old paths
   the permanent public surface; bulk-rewriting them buys nothing.
4. **Why mbse as forbidden target but not checked source:** "runtime
   components must not depend on documentation artifacts" is worth
   enforcing; giving mbse itself layer rules would force exceptions for
   the documentation-generation chain. The checker simulation over the
   full repository with the new rules yields exactly the three audited
   seams and zero pre-existing violations — enforcement could be tightened
   without a whitelist.

## Consequences

- Two new package-root modules (`status.py`, `spice_ext.py`); `exceptions.py`
  gains documented siblings. `integrators.py` imports only shared-kernel
  leaves and the extension.
- ADR 0012's dependency table and enforcement scope are revised (see the
  revision note in ADR 0012); the checker output message changes.
- Four test files that monkeypatched ABI state on the facade were migrated
  to `spice_ext` (patching a re-exported copy would silently stop working —
  deliberately, no compatibility re-exports of `_abi_ok` or
  `_MIN_REQUIRED_RUST_ABI` are provided).
- Old import paths remain permanently valid aliases; no migration is
  planned or wanted.
- A new architecture meta-test pins re-export identity, facade/leaf symbol
  identity, and gate/ABI-core identity (`tests/_meta/test_shared_kernel_leaves.py`).
