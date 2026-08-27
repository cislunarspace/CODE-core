# ADR 0009: Enable spice feature for release wheels

**Status**: Adopted (implemented)
**Date**: 2026-07-29
**Related**: ADR 0002 (Rust integrator core), issue #246

## Context

The `spice` feature gates Rust-side fast paths — third-body gravity, compiled
STM propagation, shooting, etc. (roughly half the Rust code). Until now,
release wheels uniformly shipped without spice: cspice-sys builds needed a
local CSPICE or `CSPICE_DIR`; CI/release never configured either, so enabling
it failed the build. Python-side code try/excepts missing spice bindings and
silently degrades to Python slow paths — functionally correct, but users lose
the fast paths without knowing. This ADR evaluates whether shipping wheels
with spice is feasible.

## Evaluation

### Licensing

NAIF's CSPICE redistribution terms (Toolkit Redistribution clause on the
Toolkit page):

- Mirroring/forwarding the entire Toolkit: requires NAIF's prior written
  permission; do not do it unilaterally.
- **Shipping SPICE Toolkit library modules as part of a package supporting
  self-built SPICE tools: "entirely appropriate".** JPL Document Review
  issued Clearance CL05-2438 permitting SPICE products distribution via NAIF
  servers.

e2m2e wheels embedding statically linked CSPICE fall under the second case
(the wheel is a self-built SPICE tool; CSPICE is one of its components) — no
licensing obstacle. The obligation is attribution: add NOTICE in repo and
wheel metadata stating CSPICE comes from NASA/JPL NAIF.

### Size

Measured on a real server with release builds
(`libe2m2e_integrators.so`): without spice 832 KB, with spice 2681 KB —
**about +1.8 MB per wheel**. Acceptable.

### Build reliability

cspice-sys's `downloadcspice` downloads Toolkit sources from
naif.jpl.nasa.gov at build time and compiles them in place (reqwest download +
gzip/tar unpack; all work inside manylinux containers). CI has exercised this
mechanism successfully (PR #248). A pre-download + actions/cache + `CSPICE_DIR`
scheme for the release runner was considered but rejected: manylinux builds run
inside Docker where passing host cache dirs and `CSPICE_DIR` into the container
has no clean path — high complexity, little benefit. Release uses the same
downloadcspice mechanism as CI (which exercises it every run); NAIF's realtime
reachability is recorded as accepted risk, to be hardened into a caching scheme
only if it becomes a problem.

### Platform matrix

cspice-sys compiles CSPICE from source via cc, depending on no prebuilt
libraries:

- manylinux_2_17 (glibc 2.17): source build, no glibc version issues.
- Windows MSVC: officially supported by cspice-sys (downloadcspice fetches the
  PC_Windows_VisualC_64bit package).
- macOS: not in the current release matrix; not evaluated.

## Consequences

Licensing, size, and platforms all clear — **release enables spice**
(implemented):

1. release.yml's maturin build adds `--features spice`, built via
   downloadcspice (same mechanism as CI).
2. NOTICE file added at repo root attributing CSPICE to NASA/JPL NAIF;
   packaged in sdist too.
3. Installation docs clarified: wheels carry Rust fast paths; source builds
   also default to spice (Cargo default feature, see ADR 0002's 2026-08
   revision).

## Revision (2026-08: build-mechanism change)

The downloadcspice mechanism described above (build-reliability section,
platform matrix, and consequence item 1) is deprecated:

- cspice-sys removed its downloadcspice feature: builds now fail outright when
  `CSPICE_DIR` is missing instead of downloading sources from naif.jpl.nasa.gov.
  CSPICE always comes from GitHub `cspice-v1` release prebuilt packages via
  `scripts/download_cspice.py`, pointed at by `CSPICE_DIR` (annotated in root
  Cargo.toml and release.yml).
- The aarch64 prebuilt package is built from NAIF sources on native arm64
  runners by `cspice-aarch64-build.yml` and published to `cspice-v1`.
- Spice is now a default feature (`crates/*/Cargo.toml default=["spice"]`,
  declared again in pyproject `[tool.maturin]`); release.yml's maturin args no
  longer need explicit `--features spice`.
- Licensing conclusions, NOTICE attribution (item 2), and the enable-spice-for-
  releases decision are unchanged.

## Revision (2026-08-12, ADR 0020 decision 4)

After releases began shipping spice, the Python-side try/except silent
degradation for missing spice bindings was removed: without spice (environment
not set up) the library raises (issue #378) instead of silently falling back to
slow paths. Corresponding tests' `importorskip` semantics were adjusted
likewise.
