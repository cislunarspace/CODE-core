# ADR 0041: spatiography — cislunar partition (Primer) analytic core

**Status**: Adopted (implemented)
**Date**: 2026-08-30
**Related**: ADR 0012 (layering), ADR 0014 (facade tiers), ADR 0022
(constants single source), ADR 0031 (catalog scope), ADR 0040 + Amendment
(state_frame vocabulary), transfer-orbit-design ADR 0013 (visualization
contract); Rosengren et al. 2026, *The Astrodynamics Primer on Cislunar and
Translunar Space* (hereafter "the Primer"), §5.

## Context

The Primer establishes that the Earth–Moon environment is a stratified
geography — five provinces (terrestrial → cislunar inner secular / cislunar
outer resonant → circumlunar → translunar → heliocentric) bounded by the
Laplace radius, zero-velocity gateways, a family of mutually non-equivalent
spheres of influence, resonance ladders, and the lunisolar tidal parity —
rather than one undifferentiated beyond-GEO regime. We adopt this partition
as an e2m2e analytic capability: scales and classification in
`e2m2e/algorithm/spatiography/`, exposed as MCP tools, with visualization
geometry consumed by the transfer-orbit-design canvas (region overlay layer;
the canvas computes nothing). Numerical cartography (MEGNO + fate classes,
Primer §7) is explicitly out of scope for this ADR.

Two surveys drove the design: a 16-agent close reading of the Primer with
adversarial verification (93 grounded definitions, every Table 1/2 value
independently reproduced), and a codebase reconnaissance confirming the
capability is greenfield (no SOI/Hill/Laplace/region code existed).

## Decision

### 1. Five-province taxonomy and naming discipline

`RegionId`: `terrestrial`, `cislunar_inner_secular`,
`cislunar_outer_resonant`, `circumlunar`, `translunar`, `heliocentric`
(snake_case legend strings in responses; IntEnum in the subpackage).
Per Primer §2.6:

- `cislunar` is **never** an umbrella value; the system-level umbrella stays
  in documentation (`geolunar space` / Earth–Moon system space), not in
  region names.
- No criterion may key on GEO (Primer §2.3 explicitly rejects it); the GEO
  radius exists only as an (a, e) corridor reference line.
- L1 belongs to the cislunar side, L2 to the translunar side, L3/L4/L5 to
  system equilibria.

### 2. Self-consistent `[primer]` constants section (ADR 0022)

The Primer's golden values reproduce only from its own adopted set — SPICE
GMs, Simon et al. 1994 lunar mean elements (a☾ = 383397.7725 km), GGM02
J2/R⊕, IAU au. Mixing them with `Datum.DE421` GMs or
`EARTH_MOON_DISTANCE_KM` (384405/384400 km, different quantities) shifts
scales systematically. New `constants.toml` section `[primer]` +
`data/constants/primer.py` (+ `ConstantSource` members `SIMON1994`,
`GGM02`, `LEGNARO2024`) hold the 12-value set; the subpackage assembles it
into a frozen `PrimerConstants`. Lunar radius comes from
`bodies.MOON.require_mean_radius_km()` (IAU2015 1737.4 km, consistent with
the Primer's implied value).

### 3. L1/L2 use exact root-finding; Primer values are a documented note

The Primer's tabulated (L1)^☾ = 57868 km, (L2)^☾ = 64347 km match a series
approximation γ = α(1∓α/3−α²/9) with the mass parameter taken as
GM☾/GM⊕ — not exact CR3BP roots (which differ by 1.2–2.3%). e2m2e keeps its
exact `compute_libration_points()` (scipy fsolve) as authoritative for
gateway/Jacobi analysis; Primer values are cited in docs/tests as
literature values. Exact critical Jacobi values C1..C5 (Parker convention,
matching Primer §6.1) come from `2U` at the exact points.

### 4. Reproduction traps (verified numerically; encoded in tests)

1. T☾ must be derived: 2π√(a☾³/GM⊕) = 27.34460 d. Hardcoding 27.346 breaks
   Table 1's period column.
2. Moon Hill radius 61364 km (35.32 R☾) reproduces only from the
   approximate form a☾(GM☾/3GM⊕)^{1/3} (complete form: 61114 km).
3. Table 2's 16 exterior-terrestrial resonances (selenocentric) reproduce
   only with mass factor (GM☾/GM⊕)^{1/3}; the literal Eq. (126) μ̄^{1/3} is
   systematically 0.4% low.
4. Battin Eq. (118): the ψ zero-point must be the anti-Earthward direction
   to reproduce 52009 km (Earthward) / 64201 km (anti-Earthward); the
   Primer's in-text angle definition contradicts its own numbers.
   The curve's global maximum (~66.4 Mm near ψ ≈ 78°) is not quoted in the
   Primer; only axial values are.
5. Overlap is deliberate: Table 1 itself interleaves (5:4ζ at 0.86 inside
   the L1–L2 envelope; L2 coincides with 4:5ζ; a_TP and Earth SOI lie inside
   the translunar band; Table 1's SC edge 0.34 differs from Table 4's 0.35).
   Classifiers return ordered multi-labels; `include_overlaps=False`
   selects a primary label by documented precedence.
6. Not given by the Primer (implementer-defined, to be revisited with the
   cartography batch): MEGNO thresholds, grid resolutions, lunar-impact
   geometric criterion.

### 5. Tool surface (tier-2, pure derivation)

- `spatiography_scales` — all closed-form scales, libration points,
  Jacobi criticals, full resonance ladder (Table 1/2), constants provenance.
- `spatiography_classify` — per-state five-province classification with
  diagnostics (geocentric/selenocentric distances, osculating a, Jacobi
  value, topology Case I–V with open-neck list). Requests carry an explicit
  `frame` label adopting the ADR 0040-Amendment `state_frame` vocabulary:
  `synodic_barycentric_km` (first new use) and `synodic_barycentric_nd`
  (registered by this ADR as the first dimensionless use; `gcrs_km` remains
  reserved for the ephemeris batch).
- `spatiography_boundaries` — discrete visualization geometry:
  `synodic_planar` (barycentric synodic km, z=0: seven circles, the Battin
  closed curve, L1–L5) and `ae_curves` (osculating (a, e) corridor family —
  grazing, Hill apocenter, Moon-Hill encounter branches, GEO crossing,
  resonance verticals, Tisserand contours). These corridors are element-space
  crossing diagnostics, not physical surfaces (Primer Fig. 8/11 caption).
  Response `state_frame` is `synodic_barycentric_km` or the new vocabulary
  value `element_space_ae` (a in km, e dimensionless) registered here.

The frontend consumes geometry as data and only normalizes units
("界面不碰算法，算法不进界面"); boundary/region products are not cataloged
(ADR 0031 scope discipline).

### 6. Phase 1 scope boundary

Deferred to the cartography batch (not implemented here): Gallardo
semi-analytic libration widths (Eqs. 100–104), secular-resonance loci
(Eqs. 73–78), vZLK phase portraits, MEGNO + fate classification, and the
six map domains of Primer Table 4 as *integration* configurations
(Table 4 bands are already available through
`classify_by_semi_major_axis(reference="table4")`).

## Rationale

Alternatives considered: (a) putting scales as methods on
`CR3BP_System` — rejected: r_L and the SOI family are not properties of a
CR3BP system alone (they need solar parameters), and `BCR4BPSystem` inherits
`CR3BP_System`, so method additions ripple; (b) hard-coding the Primer set
inside the subpackage — rejected by ADR 0022 (single constants source) and
by provenance auditability; (c) mutually exclusive zone classification —
rejected by the Primer's own structure (deliberate overlaps), confirmed by
adversarial verification.

## Consequences

- Consumers (tod canvas, LLM agents) get machine-readable partition
  semantics with explicit frame labels; region overlay rendering lands in
  transfer-orbit-design as a new `regions` group excluded from fit-view.
- The `[primer]` section is load-bearing for golden tests; changing any
  value shifts the whole partition — the tests pin each value with its
  `ConstantSource`.
- Table 1's 1:3☾ period (82.00 d) is internally inconsistent with its own
  caption formula (82.03 d); tests tolerate 0.05 d on that single row.
