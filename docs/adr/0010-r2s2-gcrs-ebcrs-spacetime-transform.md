# ADR 0010: r2s2 integration and TDT+GCRS ↔ TDB+EBCRS spacetime conversion

**Status**: Adopted (implemented)
**Date**: 2026-07-30
**Related**: issue #252, ADR 0003 (three-layer coordinate abstraction),
ADR 0007 (dynamic axes)

## Context

DFH alignment requirement FR5 mandates bidirectional TDT+GCRS ↔ TDB+EBCRS
spacetime conversion, uniformly via r2s2 (CAS's cislunar spacetime coordinate
library, `https://github.com/r2s2-astro/r2s2`, a mandatory dependency named by
the stakeholder). TDT is the older name for TT (Terrestrial Time); EBCRS is
the Earth-Moon barycentric celestial reference system — axes aligned with
BCRS/ICRS, origin at the Earth–Moon barycenter.

## r2s2 research findings

r2s2 (PyPI package `r2s2`, version 0.1.0) provides pairwise conversions among
six spacetime coordinates under three reference frames (TCB/TDB with BCRS
positions, TCG/TT with GCRS positions, TCL/LRT with LCRS positions), including
relativistic terms required by IAU resolutions; its API is paired functions
like `TT2TDB`/`TDB2TT`, reading ephemerides via calcephpy internally.

Coverage for this task:

- **Directly covered**: (TT, GCRS geocentric position) ↔ (TDB, BCRS SSB
  position), i.e. `TT2TDB`/`TDB2TT`. Timescale conversion (TDT↔TDB) is built
  in; e2m2e's existing SPICE time chain needs no supplement.
- **Gap 1: EBCRS origin.** r2s2's TDB-side positions are SSB-centered; there
  is no Earth-Moon barycentric frame. EBCRS differs from BCRS only by origin
  translation; fill it from the same ephemeris's EMB position:
  `x_ebcrs = xs − x_emb(t)`.
- **Gap 2: velocity.** r2s2 converts position triples only, not velocities.
  This wrapper likewise covers positions only; velocity conversion awaits a
  concrete requirement.
- **Ephemeris requirement**: a built-in time ephemeris (TT−TDB) is mandatory;
  JPL `de440t.bsp` (`t` suffix variant) recommended. The repo's existing
  `de440s.bsp` and `de430.bsp` lack time ephemerides and raise calceph
  data-missing errors in practice. INPOP21a spice-format pairs (main file +
  time ephemeris `*_time.bsp`) were verified to work as backup when JPL is
  unreachable (both are ephemeris data; the conversion algorithm is identical,
  differences lie in the ephemeris models themselves). Note INPOP native format
  (.dat) does not work: its main file and time ephemeris are two separate files,
  and calceph forbids opening multiple native INPOP files simultaneously;
  spice format has no such restriction.

## Installation and dependencies

`r2s2>=0.1.0` joins `pyproject.toml` as a hard dependency (PyPI source, no git
dependency needed). Its dependency calcephpy builds from PyPI source on
Windows (verified locally on Python 3.13); r2s2's release page also provides a
Windows prebuilt wheel as fallback.

## Wrapper decisions

New module `e2m2e/algorithm/coordinate/gcrs_ebcrs.py`, public class
`GCRSEBCRSSystem`:

```python
system = GCRSEBCRSSystem("kernels/de440t.bsp")
jd_tdb, r_ebcrs = system.gcrs_to_ebcrs(jd_tt, r_gcrs)
jd_tt, r_gcrs = system.ebcrs_to_gcrs(jd_tdb, r_ebcrs)
```

Key decisions:

1. **Shape mirrors `SynodicJ2000System`** (the converter-class precedent in
   the same package), not inheriting `Axes`. Rationale: the `Axes` abstraction
   returns rotation matrices for given et and cannot accommodate joint
   spacetime conversion: this one switches timescale and includes relativistic
   terms — not a pure rotation at any fixed instant. EBCRS's spatial part (ICRS
   axes + EM barycenter origin) is already expressible as
   `CoordinateSystem(ICRSAxes, CelestialBodyOrigin("EARTH MOON BARYCENTER"))`;
   the new class's value lies precisely outside the Axes/Origin model.
2. **EMB translation uses the same ephemeris**: querying EMB state directly
   via calcephpy keeps one dataset consistent with what r2s2 uses internally,
   avoiding cross-ephemeris error contamination by mixing SPICE's de440s.
   calcephpy is r2s2's hard dependency; this introduces nothing new.
3. **Validate time ephemeris at construction**: when the ephemeris lacks the
   TT−TDB time ephemeris, raise `CoordinateDataError` immediately naming the
   de440t variant needed, rather than leaving users to guess calceph internals.
4. **Known limitation**: `R2S2.init_E` is process-global state; instances built
   with different ephemerides overwrite each other (last constructor wins) —
   documented in the class docstring.

## Verification

`tests/algorithm/coordinate/test_gcrs_ebcrs.py`:

- Bidirectional round-trip consistency (GCRS→EBCRS→GCRS and reverse), position
  tolerance 10 m, time tolerance 1 ms (floor set jointly by r2s2's 1 ns/1 mm
  iteration precision and ~40 µs resolution of single-segment Julian-date
  floats, with an order of magnitude of margin);
- Differential quantification against same-ephemeris Newtonian translation
  reference (DFH CoordinateTransform's approach: translate the Earth-center—EMB
  offset, ignoring TT/TDB distinction): the spatial difference is precisely the
  relativistic correction — between millimeters and hundreds of meters at
  Earth-Moon distances; time difference within TDB−TT's ±1.7 ms envelope
  (assertion threshold relaxed to 3 ms);
- Differential against e2m2e's existing SPICE chain (SPICEManager + de440s),
  tolerance 1 km — meant to catch wiring errors (axes, origins, units), not to
  grade accuracy;
- Pure timescale conversion at the geocenter vs ERFA `dtdb` (Fairhead &
  Bretagnon analytic model): sub-millisecond agreement;
- Two error paths: ephemeris missing time ephemeris (de440s.bsp) and file
  missing.

## Remarks

During implementation, JPL sites (ssd.jpl.nasa.gov / naif.jpl.nasa.gov) were
unreachable so `de440t.bsp` could not be downloaded; functional verification
used IMCCE's mirror of the INPOP21a spice pair instead. Test code prefers
`de440t.bsp`, falls back to INPOP21a, and skips if neither exists (consistent
with the repo's kernel-missing skip convention for SPICE tests).

INPOP21a verification measurements (DFH main.cpp demo position, LEO):
round-trip position difference < 0.001 mm, time difference < 1 µs; spatial
difference vs same-ephemeris Newtonian reference (i.e., relativistic
correction) 0.20 m at LEO and 10.1 m at lunar distance (~365 thousand km);
TDB−TT ±0.36 ms; difference vs ERFA `dtdb` independent model at the geocenter
< 0.03 µs.
