# ADR 0022: Independent physical constants management

**Status**: Adopted
**Date**: 2026-08-10
**Related**: ADR 0011 (five-layer architecture; constants live at
data/templates/), ADR 0014 (config.py covers runtime environment only;
constants belong to data/templates/), ADR 0013 (correctness by definition)

## Context

e2m2e's physical constants today: a constants layer exists nominally while
multiple numeric sets coexist in fact. `data/templates/systems.py` manages a
few Earth-Moon constants, `data/kernels/manager.py`'s `_GM_VALUES` manages body
GMs; radii, SRP, light speed, rotation rates scatter across the algorithm layer
and Rust crates — with reproducible inconsistencies already observed.

Three sources were surveyed: e2m2e's own code, GMAT R2026a sources, and a
literature corpus (Folta 2022, Vallado 2022, Soffel 2015, Topputo 2013,
Szebehely 1967, 16 works total). All three converge: **an independent,
source-traceable physical constants management is needed, and system default
baselines must be distinguished from model-carried constants.**

### Confirmed inconsistencies (9 classes)

Ordered by impact; all real in current code:

1. **Earth-Moon mass ratio μ forks across languages (worst)**. Python
   production default `data/templates/seed.py` = `0.0121506683`
   (Szebehely's 1965 value); Rust `cr3bp.rs` = `0.0121505856`; Python
   `normal_form/constants.py` = `0.012150585609624` (qiao convention). Within
   one Rust crate, `cr3bp.rs` uses 0.0121505856 while `bcr4bp.rs` reverts to
   0.0121506683.
2. **Five values for Earth GM**: `398600.4418` (systems.py, GMAT/WGS-84),
   `398600.435507` (DE440, `_GM_VALUES`), `398600.4415` (gravity_file
   default), `398600.435436` (DE430, nbody_stm test), `398600.436` (truncated
   test value).
3. **Six values for Moon GM**: `4902.800118` (DE440), `4902.8001`, `4902.8`,
   `4902.800066` (DE430), `4902.800122` (test), `4902.799967088639`
   (GRGM900C, frozen_orbit).
4. **Four sets of EM distance/characteristic length**: `384405` (systems.py,
   Cui 2025), `384400` (seed.py characteristic length), `389703`
   (transfer_optimization), `384747.981` (normal_form qiao). System-default
   scale and orbit-family scale out of sync.
5. **Four values for lunar radius**: `1737.4` (IAU mean), `1738.0` (GRGM900C
   reference), `1738.1` (transfer_optimization), `1737.1` (harmonics test).
6. **Two values for Earth radius**: `6378.1363` (systems.py), `6378.137`
   (transfer_optimization and many tests).
7. **Two values for solar radius**: `695700` (shadow/srp/Rust), `696000`
   (relativistic_correction.py). Rust `relativistic.rs` and Python
   `relativistic_correction.py` are both relativistic corrections yet each uses
   its own.
8. **Solar constant/SRP pressure only in comments**: 1367 W/m² appears only in
   `srp.py`/`ecom_srp.py` comments; light speed `2.998e8` is a comment
   approximation, true values only inside Rust `relativistic.rs` and parameter
   defaults.
9. **Time constants repeatedly defined**: `86400` ≥9 places, `365.25` ≥7,
   `36525` ≥4, all independent.

Also: `systems.py` defines `MU_EARTH` but `data/templates/__init__.py` doesn't
export it; `hohmann.py` bypasses the top-level package to import directly.

### Key fact: none of μ's three values is computed

- `0.0121506683`: recorded in Szebehely (1967)'s appendix as the 1965 value;
  historical inheritance.
- `0.0121505856`: close to DE421's exact ratio
  (`4902.8005821478 / (398600.4415 + 4902.8005821478)` = 0.01215058535) — a
  hand-copied truncated approximation from literature, **not** computed from
  DE440 GMs. DE440-computed is 0.01215058439.
- `0.012150585609624`: qiao normalization convention value.

I.e., Rust's seemingly more modern value is also just another copied
approximation. This is precisely the direct consequence of lacking independent
constants management: every author copies from whichever source is at hand and
nobody can say which set is the baseline.

## Decision

1. **Establish an independent physical constants layer `data/constants/`,
   peer to `data/templates/`.** The template layer owns mission/algorithm
   defaults (orbit family seeds, perturbation switches); the constants layer
   owns physical truth tables. Separate concerns.

2. **Constants split into two groups by mutability**:
   - **Universal physical constants** (light speed, gravitational constant,
     AU, time constants): physically unique, one set repo-wide.
   - **Body parameter catalog** (per-body GM, equatorial radius, reference
     radius, flattening, NAIF ID, rotation rate, solar radiation parameters):
     same quantity has multiple authoritative sources (DE versions / WGS-84 /
     IAU / gravity models) — **organized as coexisting baselines (datums)**.

3. **Multiple baselines coexist; programs choose per scenario.** This is the
   fundamental difference from hard unification: don't anoint a single value;
   provide several **self-consistent baseline sets** (within each set GM, μ,
   characteristic length are mutually consistent and share provenance);
   callers pick per mission. Baselines include at least:
   - `DE440`: current SPICE ephemeris — the default for ephemeris dynamics.
   - `DE421`: CR3BP/BCR4BP literature baseline (Folta 2022, Szebehely-system
     EM parameters) — default for orbit families/libration point studies.
   - `WGS84`: GNSS baseline for Earth shape & GM (radius, flattening, J2) —
     default for near-Earth missions.
   - Gravity-model self-carried constants (e.g. GRGM900C's lunar GM/radius)
     are **not merged into baselines**; they stay with their force models.

4. **Default baseline selection**:
   - EM mass ratio μ, characteristic length/time: **unify on DE ephemeris
     values**, retiring the 1965 value `0.0121506683`. CR3BP orbit families
     default to `DE421` (μ = 0.012150585350562453), matching mainstream
     literature (Folta 2022, Topputo 2013, revised Szebehely); ephemeris
     dynamics defaults to `DE440`.
   - Earth geometric parameters (radius, flattening, J2): near-Earth scenes
     default `WGS84`.
   - Solar radiation/SRP: defaults to traditional engineering value
     1367 W/m² (GMAT-compatible), with modern TSI 1361 W/m² as alternative.

5. **Single source + generated alignment; Python/Rust never hand-copy their
   own.**
   - Python side: constants defined once in `data/constants/`; algorithm layer
     and tests import exclusively; no hardcoding.
   - Rust side: shares origin with Python. Guarantee no drift via
     **build-time generation** (generating the Rust constants module from the
     single-source file) or **alignment tests** (Rust tests assert equality
     with Python values). Pick one; see Open Questions.
   - Every constant annotated with provenance (`# DE440, TDB` /
     `# IAU 2015` / `# WGS-84`) — the annotation *is* documentation.

6. **Absorb scattered sites, remove duplicate definitions.** SRP/solar
   constant, light speed, body radius tables, time constants all absorbed here;
   defined-but-unexported constants like `MU_EARTH` get exported;
   `normal_form/constants.py`'s qiao-normalized constants labeled model-carried
   (qiao convention), kept out of baselines.

7. **Verification**: new constants-consistency tests: same quantity asserted
   equal across Python/Rust; within-baseline self-consistency assertions
   (μ == GM_moon/(GM_earth+GM_moon); t* == sqrt(l³/GM_total)); rewrite
   `systems.py`'s "already unified" comments to this layer's true state.

## Recommended value choices

The table below gives recommended baseline values per quantity plus retained
alternatives. Units uniformly km/s/kg (GM in km³/s²).

### Universal physical constants (one set repo-wide)

| Constant | Value | Unit | Source |
|---|---|---|---|
| Light speed c | 299792.458 | km/s | defined (SI) |
| Gravitational constant G | 6.67430e-20 | km³/(kg·s²) | CODATA 2018 (already used in systems.py; keep) |
| Astronomical unit AU | 149597870.7 | km | IAU 2012 Resolution B2 (already used; keep) |
| Seconds per day | 86400 | s | defined |
| Julian year | 365.25 | day | defined |
| Julian century | 36525 | day | defined |
| Solar constant (flux) | 1367 (default) / 1361 (modern TSI alt.) | W/m² | GMAT IERS 1996 / Pesce 2023 |
| SRP pressure at 1 AU | derived flux/c | N/m² | derived; not separately defined |

### Earth-Moon system baseline (CR3BP/BCR4BP characteristic scales)

| Baseline | μ | Characteristic length l* (km) | Characteristic time t* (s) | Source |
|---|---|---|---|---|
| `DE421` (orbit-family default) | 0.012150585350562453 | 384400.0 | 375190.2588926273 | Folta 2022 Table 2 |
| `DE440` (ephemeris default) | 0.012150584394709708 | derived from GMs & ephemeris | derived from GMs | JPL DE440 |
| ~~1965 legacy~~ | ~~0.0121506683~~ | — | — | **retired** (Szebehely 1965) |

Note: t*, l*, GM satisfy `t* = sqrt(l*³ / GM_total)`; in-set consistency via
assertions, never filled independently. DE421's t* equals Folta 2022's
3.751902588926273e5 s.

### Body GMs (km³/s², by baseline)

| Body | DE440 | DE421 | WGS-84 |
|---|---|---|---|
| Earth | 398600.435507 | 398600.4415 | 398600.4418 |
| Moon | 4902.800118 | 4902.8005821478 | — |
| Sun | 1.32712440018e11 | 1.32712428e11 | — |
| EMB | 403503.235502 | 403503.242083 | — |

Earth's three values correspond respectively to DE440 ephemeris, DE421
ephemeris, WGS-84/EGM96 shape model. GNSS near-Earth uses WGS-84; ephemeris
dynamics uses the matching DE version. Model-carried (GRGM900C lunar GM
4902.799967088639) stays out of this table, remaining with its gravity model.

### Body radii (km; distinguishing mean/equatorial vs gravity-field reference)

| Body | Mean/equatorial | Source | Field reference radius | Source |
|---|---|---|---|---|
| Earth | 6378.137 | WGS-84 (near-Earth default); 6378.1363 is GMAT PCK historical, kept for GMAT alignment | 6378.1363 | EGM96/JGM coefficient file header |
| Moon | 1737.4 | IAU 2015 (shadow/cartography/default datum); LOLA 1737.151 alternative | 1738.0 | GRGM900C file header |
| Sun | 696000 | Vallado D-5 (unify here; retire 695700) | — | — |

**Choice rationale**: shadow/SRP/relativistic occultation and physical radii
use mean/equatorial radii; gravity fields (harmonics, solid tides) read their
reference radius from coefficient-file headers, never mixed with mean radii.
This mirrors GMAT exactly (`GmatDefaults.hpp` default radius vs gravity file
`radius` override) — e2m2e currently conflates the two.

### Earth rotation rate

| Value | Unit | Source |
|---|---|---|
| 7.292115146706979e-5 | rad/s | IERS (with LOD; used by ITRF frames; already in gmat_itrf.py) |
| 7.29211585530e-5 | rad/s | GMAT `CelestialBody` default (atmosphere/rotation models; alternative) |

GMAT itself carries both for different purposes; e2m2e follows suit per scene.

## Proposed structure

```
e2m2e/data/constants/
├── __init__.py          # unified export entry
├── universal.py         # universal constants (c/G/AU/time/solar constant)
├── datums.py            # baseline definitions (DE421/DE440/WGS84: GM/μ/l*/t*)
├── bodies.py            # body catalog (radii/flattening/NAIF ID/rotation)
└── sources.py           # datum/source enums & metadata (provenance per value)
```

- `datums.py` makes baselines first-class: `Datum.DE421.mu`,
  `Datum.DE440.earth_gm`, each internally consistent.
- `bodies.py`'s catalog replaces & extends `manager.py`'s `_GM_VALUES`
  (adding radii/flattening/rotation beyond GM); `get_gm()` queries by baseline.
- Constants migrate from `data/templates/systems.py`, which keeps re-exports
  for compatibility (or dies outright; see Open Questions).
- Rust constants module generated from single source or guarded by alignment
  tests.

## Rationale

1. **Coexisting baselines over hard unification**: literature and GMAT both
   show lunar mean radius 1737.4 vs field reference 1738.0, or Earth GM's DE-vs-
   WGS-84 split, are **each correct in their own scenario**; hard unification
   breaks model self-consistency. Today's mess isn't too many values but
   nobody saying which set a value belongs to and where it applies.
2. **Baseline sets over loose values**: μ, l*, t*, GM must share provenance to
   be self-consistent (mixing CR3BP's μ and l* across sources desyncs orbit
   families from literature). Packaging one source's values as a baseline beats
   managing constants one-by-one for internal consistency.
3. **Single source + generated alignment**: GMAT's lesson — centralization is
   necessary but insufficient: it centralized yet still grew three AU values.
   e2m2e spans Python/Rust; it needs an added cross-language single-origin
   enforcement (generation or alignment tests), or post-merge it forks again.
4. **Provenance is documentation**: μ's three-value lesson shows constants
   without stated sources equal no constants. Every value annotated with origin;
   later disputes have paper trails.

## Consequences

- New `data/constants/` layer, peer to `data/templates/`.
- μ defaults switch from 1965 legacy to DE values. **This changes CR3BP orbit
  family/libration point numerical results** — flag prominently in CHANGELOG
  and update affected test baselines uniformly (conftest, examples, normal-form).
- μ fork between Rust `cr3bp.rs` and Python `seed.py` eliminated; solar-radius
  fork between `relativistic.rs` and `relativistic_correction.py` eliminated.
- `manager.py`'s GM queries route through baselines.

## Open questions

1. **Python/Rust single-origin mechanism**: build-time generation of Rust
   constants from one source file (thorough but adds build complexity), or
   cross-language alignment tests (lightweight but test-guarded)? Leaning
   toward starting with tests.
2. **Fate of `systems.py`**: keep re-export compatibility after migration, or
   repoint all imports directly? Leaning direct repoint, no double layer.
3. **μ switch compatibility**: retain an explicit `datum="legacy1965"` option
   on CR3BP_System to reproduce old results, or clean cut? Leaning clean cut +
   CHANGELOG note; old value unsupported.
