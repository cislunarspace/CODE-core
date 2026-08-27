# e2m2e Architecture Design

## Starting from one NRHO orbit

Take the design of an Earth–Moon L2 NRHO as an example to see where each of
the five modules participates.

**Step one: the user inputs a time instant — 2024-01-01 00:00:00 UTC.** This
is human convention, but it is not the independent variable of ephemerides:
DE ephemerides use barycentric dynamical time TDB. At that moment UTC and TDB
differ by about 69 seconds (37 seconds of leap seconds + 32.184 seconds fixed
offset). Query the Moon's position with UTC taken directly as TDB: the Moon
orbits Earth at roughly 1 km/s, so the position error is about 70 km — the
lunar braking window is completely lost. Moreover UTC leaps irregularly and is
discontinuous, unfit as a dynamics argument. So e2m2e standardizes on TDB
internually, converting to UTC only at interface boundaries. STK does the
same: UTC only for input/output, TAI/TDT internally. This is why time scales
exist in the **spacetime systems and constants module**.

**Next: which coordinate frame to design in.** The CR3BP initial guess comes
in the Earth–Moon synodic frame (libration points are stationary there), the
mission specification's parameters are in the J2000 inertial frame, and
ground-station coordinates are in the ITRF93 Earth-fixed frame. The same set
of (x,y,z) numbers without conversion means another physical location: Earth
rotates 15° per hour; states in two frames differ by a rotation growing with
time. In 1999 NASA's Mars Climate Orbiter was lost to unit confusion (pound-
force vs newton); frame confusion is the same class of error, only stealthier,
and the compiler will not flag it. Frame conversion makes every state vector's
frame traceable and convertible — this is why reference-frame conversion lives
in the same module.

**Then: which constants.** Earth GM has DE440 values, DE421 values, WGS-84
values; the Earth–Moon mass ratio μ has 1965-era values, DE421 values, and
normalized-convention values. e2m2e itself stumbled here: Python and Rust each
copied a μ, two values coexisted in one crate, and computed orbit families
mismatched the literature. The trouble with constants is not having many
values but that nobody says clearly which value belongs to which baseline and
where it applies. Constant baselines (multiple sets coexisting, self-consistent
within each) exist for exactly this — the constants part of the same module.

**Initial guess in hand, shooting begins.** Take a 2-year 50-rev ephemeris
correction: 50 revs × 8 nodes per rev × 50 iterations ≈ twenty thousand
trajectory segments, thousands of steps per segment, force models evaluated
and ephemeris queried at every step — scalar operations in the billions. This
layer must be fast and parallelizable, while cspice kernels are global state
and cannot be used concurrently — so the **Rust computation module** takes no
SPICE handles, only pre-sampled injected ephemeris cache tables.

**The whole design chain is orchestration, not formulas.** DRO or Halo, how
large an initial-guess amplitude, how tight correction tolerance, propagate
one year or ten — these are domain decisions changing daily, kept in Python.
That is the **orchestration module**'s share.

**After computing, deliver.** Design engineers sit on Windows, simulation
centers run Kylin, servers are Kunpeng ARM — three-platform wheels are the
deployment reality; NAIF's site is unstable from domestic networks, so CSPICE
must ship via GitHub Release. That is the **CI module**.

**Finally, data must be kept.** Ephemeris files are tens to hundreds of MB,
updated every few years; Halo family seeds are two floats whose every change
needs review. Things differing a million-fold in size with entirely different
lifecycles cannot be put in one basket. That is the **data management module**.

The software has five architectural designs:

## 1. Spacetime systems and constants module

(1) Time scales (UTC/TDB/TAI/TT; dynamics standardized on TDB);
(2) Reference-frame conversion (J2000/ITRF93/IAU 2006, GCRS-EBCRS, dynamic
axes);
(3) Physical constant baselines (DE421/DE440/WGS84 coexisting, chosen per
scenario, single-source across Python/Rust against drift).

Together they form a self-consistent spacetime basis, actually maintained
separately. Code locations: conversion **algorithms** live in
`e2m2e/algorithm/coordinate` (per ADR 0011, formerly `core/coordinate/`);
coordinate **data** (EOP/leap seconds/ephemeris handles) in
`e2m2e/data/frames` (ADR 0015). Functional markers are assigned by verified
content under `data` — an axis independent of code layering (ADR 0026).

## 2. Rust computation module

Six crates:

(1) spice (CSPICE FFI + caching);
(2) propagation (pure math integrators);
(3) forces (force models + STM);
(4) integrators (pyo3 bindings + iterative solvers such as shooting +
parallelism);
(5) levelset (level-set / HJB solver kernel, ToolboxLS port);
(6) hjb-dynamics (Hamiltonian dynamics implementations for the HJB solver,
ADR 0032).

Iterative problems are constructed by Python and passed in; Rust only iterates
to convergence; Rust takes no SPICE handles, only pre-sampled injected
ephemeris cache tables.

Each algorithm module's migration progress (which numerics have sunk, which
remain in Python, and why) is itemized in
[numerics-migration-status](numerics-migration-status.md); consult the ledger
before drawing audit conclusions.

## 3. Python orchestration module

Only three things, no numerical iteration:

(1) construct the problem;
(2) call Rust iterators;
(3) interpret results.

Three orchestration tiers: task-level Facade (MCP/CLI derived from the same
source), mission orbit design algorithm chains (family initial guess → Rust
correction → high-fidelity propagation), and subproblem construction (family
strategies, control laws, manifold seeds).

Beyond executing input validation, the task-level API model also exposes
parameter metadata for the current task context, for GUIs, CLIs, and MCP to
generate widgets and range hints; conditional value domains and validators
share one rule definition, so external consumers keep no local copies of
parameter rules.

## 4. CI module

Builds software for Linux x64 AMD (Ubuntu, Kylin, etc.), Linux ARM (aarch64,
Kunpeng/Phytium), and Windows. Three-platform wheel matrix + sdist + GitHub
Release + PyPI; the CSPICE build package (x86_64 re-uploaded from NAIF's
official package, aarch64 self-built) ships via GitHub Release; domestic
systems like Kylin are covered by the manylinux 2_28 glibc baseline.

## 5. Data management module

(1) GitHub Release manages some build libraries, ephemeris data, and the
CSPICE build library (released by version, fetched by script);
(2) Git tracks initial values of some orbits (family seed parameters);
(3) Local computation artifacts enter the orbit catalog (ADR 0031): records =
JSON metadata + NPZ array segments (with schema version), SQLite only as a
derived index.

This module will later extend to support intelligent game-theoretic research
as its data foundation. Ephemeris serialization uses EphemerisTable as the
unified intermediate format; converters to/from typical ephemeris formats
(CCSDS OEM/ODM, SPICE BSP/PCK, etc.) will follow.

Some computational functionality still runs in Python (NLP optimization
etc.) and is migrating stepwise to the Rust core. Continuation, NSGA-II
evolutionary operators, and low-thrust direct-method numerical evaluation
kernels have sunk; Python keeps outer orchestration and reference paths.
The spacetime and constants module still needs further review (three
coexisting frame paths, time-conversion responsibility chain). The Python
orchestration module is not yet stable enough (dual shooting paths, transfer
directory numerical residue); further review and system design are required.
