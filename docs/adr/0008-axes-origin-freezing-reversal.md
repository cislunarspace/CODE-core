# ADR 0008: Revoke runtime freezing of Axes / Origin / CoordinateSystem

**Status**: Adopted (freezing mechanism rejected; landed freezing reverted)
**Date**: 2026-07-17
**Related Issues**: #76, #121, #122

## Context

Concrete subclasses of `Axes`, `Origin`, and `CoordinateSystem` were at one
point slated for runtime freezing (`@dataclass(frozen=True)` or equivalent).
The freezing effort went through:

- **#76** introduced `CoordinateSystem` freezing, converting it to
  `@dataclass(frozen=True)`; landed and CLOSED.
- **#121** followed up, planning to freeze static `Axes` / `Origin` and all
  their concrete subclasses.
- **#122** designed the boundary between `DynamicAxes` and frozen base classes
  (dependent on #121).

After landing, re-evaluation judged freezing **unnecessary**. This ADR records
the reversal: the freezing mechanism is rejected, `CoordinateSystem` has been
reverted to a plain class, and #121/#122 are closed as wontfix.

The problem freezing targeted — runtime swapping of `axes`/`origin` components
causing inconsistent transformation results — never actually occurred. These
classes' instances follow a read-only convention after construction: callers
should not swap `axes`/`origin` components at runtime, but the application
layer does not enforce it.

## Decision

**Reject the freezing mechanism and revert the landed #76 freeze.**

Concretely:

1. `CoordinateSystem` reverts from `@dataclass(frozen=True)` to a plain class
   (commit `62f7308`).
2. #121 (freezing static `Axes` / `Origin` and concrete subclasses) will not
   proceed; wontfix.
3. #122 (boundary design for `DynamicAxes` vs frozen base classes) is wontfix
   with #121.
4. None of these are introduced: `CoordinateSystem` freezing,
   `Axes`/`Origin` base `__init_subclass__` hooks, dataclass conversion of
   concrete subclasses.

## Rationale

1. **No real bug drives it.** The problem freezing was meant to solve —
   runtime component swaps yielding inconsistent transformations — never
   appeared in actual code. After `CoordinateSystem` freezing shipped,
   patterns like `cs.axes = X` appeared neither in core nor in forces /
   transfer / algorithms layers. The problem was imagined, not observed.

2. **Tamper protection is not the application layer's job.** If someone swaps
   `cs.axes` to another instance at runtime, that's a scenario where the
   program itself has been tampered with. Users obtain verified code copies
   from GitHub and can trust their runtime. Application-layer freezing cannot
   stop the root cause of program tampering; it merely moves checkpoints to
   runtime.

3. **Freezing carries non-trivial engineering cost.** Concrete subclasses like
   `IAU2000EqAxes` have derived fields (e.g. internal state derived from
   `time_step`); the dataclass form needs boilerplate such as
   `field(default_factory=...)` + `__post_init__` + `object.__setattr__`; the
   boundary between `DynamicAxes` (deliberately mutable per ADR 0007) and
   frozen bases requires dedicated design (composition / exemption hooks /
   parallel inheritance trees — #122 existed entirely for this). That
   complexity serves a nonexistent bug; the cost exceeds the benefit.

4. **YAGNI.** At the code-style level, "authors shouldn't write mutation" is
   already covered by the grep gatekeeper in
   `tests/algorithm/coordinate/test_coordinate_immutability.py` (scanning
   `e2m2e/algorithm/coordinate/` for assignments like `cs.axes = X`). Runtime
   swap protection differs from that and was never an application-layer
   concern anyway.

## Consequences

### Reverted / not proceeding

- `CoordinateSystem`'s `@dataclass(frozen=True)`: reverted to plain class
  (commit `62f7308`).
- Three tests in `TestCoordinateSystemFrozen`: deleted.
- #121 (full freezing of static `Axes` / `Origin` and subclasses): wontfix.
- #122 (`DynamicAxes` boundary design): wontfix.

### Kept

- `TestCoordinateSystemOrthogonality`,
  `TestCoordinateSystemTransformVector` (zero vector): unrelated to freezing;
  kept.
- The grep gatekeeper in
  `tests/algorithm/coordinate/test_coordinate_immutability.py`: kept. It
  guards against authors writing mutation (static code style), not runtime
  swapping (dynamic protection). Different responsibilities; the former has
  nothing to do with freezing.

### Impact on upstream decisions

| Issue | Original decision | State after this ADR |
|-------|--------|----------------|
| #76 | `CoordinateSystem` freezing (was CLOSED) | Reverted; original decision overturned |
| #121 | Freeze all static `Axes` / `Origin` | wontfix |
| #122 | `DynamicAxes` vs frozen-base boundary | wontfix (depended on #121) |
