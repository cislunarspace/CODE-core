# ADR 0015: NominalOrbit contract and coordinate-conversion abstraction

**Status**: Adopted (implemented)
**Date**: 2026-07-31
**Related**: ADR 0011 (five-layer architecture), Gómez "Dynamics and Mission
Design Near Libration Points" vol I §8.2.3, Soffel "Relativity in Celestial
Mechanics and Astrometry"

## Context

A data contract is needed between mission orbit design (FR1) and station
keeping (FR2): how design-produced nominal orbits are consumed by control.
Gómez vol I §8.2.3 gives the canon: **nominal orbit = equally-spaced epoch
state table + Floquet basis + projection factor table + high-order
interpolation (Lagrange r=5–6)**.

Another issue: spacetime reference frames. Existing conversion-class
interfaces vary (`SynodicJ2000System`/`GCRSEBCRSSystem`/`Axes`); a draft once
proposed a new Frame abstraction to unify them. Soffel's conclusion:
conversion-chain algorithms belong to the kernel; data (EOP/leap seconds/
ephemerides) belongs to the data layer.

## Decision

1. **NominalOrbit is the FR1↔FR2 data contract**, housed in
   `data/types/trajectory.py`: equally-spaced epoch state table + Floquet
   basis + projection factor table + high-order interpolator. **Floquet bases
   and projection factors are precomputed by FR1** (design_orbit outputs carry
   them); control interpolates throughout and never recomputes. The
   control_orbit control laws evolve from computing STMs on the fly to
   interpolating projection factors.
2. **Spacetime reference frames = strengthen existing Axes/Origin/
   CoordinateSystem abstractions** (no new Frame abstraction). All coordinate
   systems (including synodic↔J2000, GCRS↔EBCRS) express as Axes + Origin +
   CoordinateSystem; joint spacetime conversion (switching timescale
   simultaneously) becomes a CoordinateSystem extension method. To add:
   timescales unified as part of the reference frame.
3. **Timescales merge into EphemerisProvider** (no separate TimeSystem
   class). TDB as the dynamics-uniform time: algorithm/numerical layers use
   ET(TDB) or JD_TDB internally throughout; UTC appears only at interface
   boundaries.
4. **Conversion algorithms live in `algorithm/coordinate/`**; `frames/`
   keeps only data (EOP, leap seconds, ephemeris handles). Conversion
   algorithms finally stay in Python (Axes subclass methods); Rust sinking is
   later performance optimization.

## Rationale

1. **Precomputed projection factors**: station keeping needs projection
   factors at every control instant; precompute once, interpolate through the
   whole control run — clearly superior (Gómez 8.2.3).
2. **No new Frame abstraction**: ADR 0003/0007 already invested in Axes/
   Origin/CoordinateSystem; a parallel Frame abstraction duplicates — avoid
   over-abstraction (duplication is cheaper than wrong abstraction).
3. **TDB uniformity**: TCB drifts 0.47 s/year while TDB retains only <2 ms
   periodic terms (Soffel); dynamics time standardizes on TDB.

## Consequences

- `data/types/trajectory.py` defines NominalOrbit (with Floquet basis +
  projection factors + interpolator).
- `data/kernels/provider.py`'s EphemerisProvider carries time/state/frame
  methods, single-point + batch.
- `data/frames/` keeps only data; conversion algorithms in
  `algorithm/coordinate/`.
