# ADR 0004: ForceModel config-driven construction

**Status**: Adopted
**Date**: 2026-06-15
**Related Issue**: #69

## Context

Issue #69 wants users to write one config (JSON or dict) that builds a force
model set (J2 + drag + SRP + finite burn), persist it, read it back, and
rebuild forces identical to the original. The existing container already
aggregates multiple `PhysicalModel`s and propagates via Rust `rk_step` steppers
(ADR 0002); each force model already performs its own frame conversion
(ADR 0003). Two things remain: lookup of individual forces by name — needed by
`get_force`/`remove_force`/`enable`/`disable` and by config round-trips — and
expressing each force's parameters as data, including those holding other
instances or Python functions.

Two spots are awkward. `DragModel` takes an `AtmosphereModel`,
`SolarRadiationPressure` a `ShadowModel`; both are instances, not a few
numbers. `FiniteBurn` takes a `thrust_profile` function, and its `direction`
may also be a function; in general neither can be stored to JSON and restored
faithfully.

## Decision

1. **Force identity (`name`) and toggle state (`enabled`) live on the
   container, not on `PhysicalModel` instances.** `ForceModel` maintains an
   ordered registry of `ForceEntry(name, force, enabled)`.
   `add_force(force, name=None)` auto-names via `type(force).__name__` when
   omitted, disambiguating same-type collisions automatically (`Foo`, `Foo_2`,
   `Foo_3`…); an explicit collision raises `ValueError`.
   `get_force`/`remove_force`/`enable`/`disable` operate by name;
   `list_forces()` returns the entries. The existing `forces` property stays
   (still returning `tuple[PhysicalModel, ...]`) for backward compatibility.
   Disabling a force skips it during propagation but keeps it in the container
   and in `to_config` output (`enabled: false`).

2. **The config schema is a versioned manifest of named entries.** Top level:
   `{"version": 1, "forces": [...]}`. Each entry is
   `{"name", "type", "enabled", "params"}` where `type` is the Python class
   name (also the registry key) and `params` holds constructor arguments.
   Injected dependencies (`DragModel`'s `atmosphere`, `SolarRadiationPressure`'s
   `shadow`) are nested entries shaped `{type, params}`, handled recursively;
   `null` means not injected (e.g. SRP under full sunlight). `to_config` emits
   resolved actual values — values after constructor defaults took effect — so
   `GravityField(degree=2)` and `GravityField(degree=2, order=2)` yield the
   same config.

3. **`FiniteBurn` builds from config only through a closed DSL (fixed set of
   forms).** `thrust_profile` accepts `{"kind": "constant", "thrust": N}` or
   `{"kind": "pulse", "t_start", "t_end", "thrust"}`; `direction` accepts
   `{"kind": "fixed", "vector": [x, y, z]}`. `from_config` builds closures and
   tags them (`_e2m2e_config_kind`) so `to_config` can invert. If a
   `FiniteBurn`'s callables didn't come from this DSL (e.g. user-written
   `lambda t: ...`), `to_config` raises `NotSerializableError`. It still
   propagates fine; it just cannot be serialized.

4. **Round-trip acceptance = config dict equality.**
   `to_config(from_config(config)) == config` is the round-trip acceptance
   property: exact Python dict equality, no tolerance. Trajectory equality is a
   separate physical check owned by LEO end-to-end tests.

Only one existing class is touched: `GravityField.__init__` stores the raw
`gravity_file` argument (`self._gravity_file_arg`) so custom `.gfc` paths round-
trip. This is the only intrusion into a `PhysicalModel` subclass; everything
else goes into new module `e2m2e/algorithm/forces/force_config.py`.

## Rationale

1. **Why `name`/`enabled` on the container rather than instances.**
   `PhysicalModel` instances flow through many modules (`FiniteBurn` referenced
   by thrust handling, `GravityField` by gravity paths). A `name` attribute on
   the base class drags a container-specific label into every module; and
   `enabled` would imply disabling is a property of the force itself, which it
   is not: a force can always compute acceleration, only the container decides
   whether to call it. The registry keeps labels where they are used.

2. **Why a closed DSL instead of excluding `FiniteBurn` or accepting arbitrary
   callables.** Excluding `FiniteBurn` breaks the acceptance criterion's
   coverage claim: `ImpulsiveBurn` is not a `PhysicalModel` and isn't in the
   container. A registry of named callables leaks user code into configs,
   breaking cross-session round-trips. A closed DSL covers realistic cases
   (constant thrust, pulsed burn, fixed direction) and fails loudly otherwise,
   never silently producing an unloadable config.

3. **Why class names as the type discriminator.** Only four force types exist
   today with no need yet for public stable identifiers; class names serve
   directly as registry keys without a translation layer. Aliases can be added
   later as fallback keys without breaking existing configs.

4. **Why serialize resolved values.** Two instances built differently but with
   identical effective parameters are the same force. Serializing resolved
   values makes round-trips independent of how users happened to invoke
   constructors; read-only properties already expose them.

5. **Why config-dict equality as the round-trip criterion.** Exact and cheap to
   assert. Trajectory equality depends on integrators and floats and can only be
   checked with tolerances — useful as a separate sanity check but wrong as the
   definition of round-trip success.

## Consequences

### Added

- `e2m2e/algorithm/forces/force_config.py`: type→builder and type→serializer
  dispatch, DSL builders for `FiniteBurn`, recursive atmosphere/shadow
  builders, JSON `load_force_config`/`dump_force_config`.
- `ForceModel`: `ForceEntry` registry, `add_force(name=)`,
  `remove_force(name | index)`, `get_force`, `list_forces`, `enable`,
  `disable`, `from_config` classmethod, `to_config` method.
- `GravityField._gravity_file_arg` (stores raw path).

### Changed

- `ForceModel._forces` becomes tuple `_entries` of `ForceEntry`; the `forces`
  property still returns `PhysicalModel`s for compatibility.
- Propagation paths (`_propagate_via_rust` etc.) skip `enabled=False` entries.

### Unchanged

- Physics and `compute_acceleration` signatures of the four `PhysicalModel`
  subclasses (aside from `GravityField` storing raw `gravity_file`).
- `propagate`/`propagate_maneuvers` behavior (propagation reads enabled
  entries only, so disabled forces are naturally excluded).
- Frame conversion responsibility: each force model converts itself; see
  ADR 0003.

### Follow-up work

- Extend the `FiniteBurn` DSL with new `kind` values (VNB/LVLH direction
  alignment, time-varying thrust curves), backward-compatibly.
- If public stable identifiers become necessary later, add force-type aliases
  as fallback registry keys.
- Versioned schema migration when `version` exceeds 1.
