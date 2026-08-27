# ADR 0007: Dynamic-axes state injection scheme

**Status**: Adopted
**Date**: 2026-06-15
**Related Issue**: Slice 12 (dynamic axes and VNB/LVLH maneuver support)

## Context

Slice 12 needs two dynamic axes types — VNB (Velocity-Normal-Binormal) and
LVLH (Local Vertical Local Horizontal) — for:
1. Specifying `FiniteBurn` thrust direction in the VNB/LVLH frame;
2. Specifying `ImpulsiveBurn` Δv in VNB/LVLH before conversion to inertial.

The fundamental difference between dynamic and static axes (ICRS, ITRF,
etc.): their rotation matrices depend not only on epoch `et` but also on
spacecraft `state` (position + velocity). VNB's x-axis lies along velocity, z
along angular momentum, y completes by cross product — all changing with
`state` in real time.

The existing `Axes` interface offers only `rotation_matrix(et)` and
`rotation_and_rate(et)`, with no `state` in signatures. The decision: modify
the `Axes` interface, or find another way while keeping current signatures?

## Alternatives compared

### Option A: modify the `Axes` interface

Add a `rotation_matrix(et, state)` overload to the `Axes` base class, or make
`state` an optional parameter. All existing subclasses (ICRS, ITRF, IAU2000Eq,
etc.) would need adaptation.

Problems:
1. **Static axes forced to accept irrelevant parameters.** ICRS is always the
   identity matrix; passing it `state` is noise.
2. **Breaks existing callers.** `CoordinateSystem.transform_state`, all tests,
   and all force-model coordinate conversions would need signature changes or
   branching.
3. **Conflates two concepts.** Static axes are pure functions of time;
   dynamic axes are functions of state + time. Stuffing both into one
   interface blurs the distinction.

### Option B: keep `Axes` signatures; add a `DynamicAxes.update` method

Leave the `Axes` interface untouched. Add abstract base class `DynamicAxes`
extending `Axes` with an `update(t, state)` method. Callers `update` first,
then take matrices when needed.

```python
class DynamicAxes(Axes):
    """State-dependent dynamic axes."""

    @abc.abstractmethod
    def update(self, t: float, state: npt.NDArray[np.floating]) -> None:
        """Refresh internal direction cache from current state."""
        raise NotImplementedError

    def rotation_matrix(self, et: float) -> npt.NDArray[np.floating]:
        """Return rotation matrix from most recent update."""
        # subclass caches R during update; return cache here
        ...
```

VNB and LVLH are implemented as `DynamicAxes` subclasses. Static axes remain
untouched.

## Decision

**Option B is chosen.** Concretely:

1. **The `Axes` interface stays unchanged.** Static axes (ICRS, ITRF,
   IAU2000Eq etc.) fully unaffected. Callers like
   `CoordinateSystem.transform_state` need no changes.

2. **New `DynamicAxes` abstract base class.** Extends `Axes`, adds abstract
   `update(t, state)`. Subclasses compute direction vectors from state inside
   `update` and cache them; `rotation_matrix` returns cached values.
   `rotation_and_rate` likewise.

3. **The state-injection points live in `ForceModel.propagate` and
   `System.update_coordinate_systems`.**
   - `ForceModel.propagate`, before each integration step, if a `DynamicAxes`
     is detected, calls `axes.update(t, y)` to refresh directions.
   - `System.update_coordinate_systems` serves as the system-level unified
     entry: before propagation starts or any computation needing dynamic axes,
     the system injects current state into all registered dynamic axes.
   The system-level entry was chosen over per-force-model injection because
   dynamic axes may be shared across force models (VNB used for both drag and
   thrust directions simultaneously); updating at system level avoids duplicate
   computation and inconsistent states.

4. **VNB axis definitions (revised 2026-08-23).**
   - x-axis (Velocity): along `v / |v|`.
   - y-axis (Normal): along orbital angular momentum `h = r × v`,
     `h / |h|`.
   - z-axis (Binormal): `x × y`, ensuring right-handedness.
   The original text placed Normal on z and wrote Binormal as `z × x`,
   contradicting implementation and tests; corrected here against
   `VNBAxes` in `standard_dynamic_axes.py`. Literature disagrees on VNB's
   y/z naming; e2m2e takes this entry as normative.

5. **LVLH axis definitions (revised 2026-08-23).**
   - x-axis (Radial): along position `r / |r|` (radially outward).
   - z-axis (Cross-track): along orbital angular momentum `h / |h|`.
   - y-axis (In-track/Local Horizontal): `z × x`, ensuring right-handedness.
   The original text had z pointing Earthward (`-r`) and y along negative
   angular momentum, contradicting implementation and tests; corrected here
   against `LVLHAxes`. LVLH conventions vary in literature (some take z as
   `-r`, Earth-pointing); e2m2e adopts this entry — the RSW convention:
   radially outward, in-track, orbit-normal.

6. **`ObjectReferencedAxes` deferred.** `ObjectReferencedAxes` (relative axes
   centered on a celestial body, e.g. Moon-centered VNB) requires body state
   queries via `System`'s ephemeris interfaces. Slice 12 only supports VNB/
   LVLH based on spacecraft's own state; `ObjectReferencedAxes` awaits a later
   slice, which will extend the `DynamicAxes.update` signature to accept body
   states.

## Rationale

1. **Why leave the `Axes` interface alone.** Static axes are the overwhelming
   majority. Changing all static axes and all callers for a few dynamic ones
   pays general costs for special-case benefits; `DynamicAxes` as a subclass
   extends generality with special cases — open-closed principle.

2. **Why update-and-cache rather than recompute every call.** `rotation_matrix`
   may be called multiple times per state (`transform_vector` and
   `transform_state` each call once). `update` concentrates the state→direction
   computation at one point; subsequent matrix reads are O(1) cache lookups.
   The `update` semantics are also explicit: "I'm about to use this state;
   prepare."

3. **Why inject state at system level.** Dynamic axes are a coordinate-system-
   layer concept, not any single force model's private property. If both
   `DragModel` and `FiniteBurn` use VNB, calling `update` inside each
   `compute_acceleration` causes:
   - duplicated direction computation within one integration step;
   - direction inconsistency if two models interpret state differently (raw
     state vs interpolated midstep state).
   Updating once at `System.update_coordinate_systems` /
   `ForceModel.propagate` guarantees all force models see the same direction
   within a step.

4. **Why fix written definitions rather than inherit literature ambiguity.**
   VNB/LVLH axis conventions differ across sources (some define LVLH z as
   `+r`, others `-r`). Decisions 4–5 pin definitions into this ADR because:
   - axis inconsistency directly changes a maneuver's physical meaning; there
     must be one written baseline;
   - tests assert against decisions 4–5's definitions, locking implementation,
     tests, and docs together;
   - users migrating missions from other software can check this entry instead
     of guessing.

5. **Why defer `ObjectReferencedAxes`.** It adds two complexities:
   - querying a body's state at epoch `et` (`System.get_body_state`),
     introducing ephemeris dependency;
   - defining relative-state semantics (spacecraft state minus body state,
     then VNB/LVLH).
   Slice 12's use case (spacecraft-own VNB/LVLH maneuvers) needs neither.
   Deferral avoids premature generalization.

## Consequences

### Added

- `e2m2e/algorithm/coordinate/dynamic_axes.py`: `DynamicAxes` abstract base.
- `e2m2e/algorithm/coordinate/standard_dynamic_axes.py`: `VNBAxes` and
  `LVLHAxes`.
- `tests/algorithm/coordinate/test_dynamic_axes.py`: direction correctness
  tests (asserting axes per decisions 4–5).

### Changed

- `ForceModel.propagate`: after propagation moved to the Rust compiled path,
  Python no longer calls `DynamicAxes.update` step-by-step; state injection of
  dynamic axes rests with the system-level entry.
- `EphemerisSystem`: adds `update_coordinate_systems(t, state)`, calling
  `update` when `coordinate_system.axes` is a `DynamicAxes`. The `System` base
  class does not define it.

### Unchanged

- `Axes` interface signatures (`rotation_matrix(et)`,
  `rotation_and_rate(et)`).
- All static axes implementations (ICRS, ITRF, IAU2000Eq, GMATITRF).
- `CoordinateSystem.transform_state` / `transform_vector` signatures and
  behavior.
- Existing behavior of `FiniteBurn`/`ImpulsiveBurn` (`FiniteBurn` gains VNB/
  LVLH support via a new `direction_frame` field without breaking existing
  APIs).

### Follow-up work

- `ObjectReferencedAxes`: implement once body-relative frames are needed;
  extends `DynamicAxes.update` to accept reference-body states.
- `FiniteBurn` config DSL extension: new `direction_kind: "vnb" | "lvlh"` plus
  `direction_vector` (components in the dynamic frame); see ADR 0004 follow-ups.
- `ImpulsiveBurn.frame` field: accept `"vnb"`, `"lvlh"`, converting via
  `CoordinateSystem.transform_vector` (matching GMAT
  `Burn::ConvertDeltaVToInertial`'s `coincident=true` pure rotation).
