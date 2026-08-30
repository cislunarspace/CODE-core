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

## Amendment: Phase 3 cartography batch (#578/#579/#580)

**Status**: Phase 3a (analytic layer, #578) implemented; 3b/3c registered
below as they land.

### Phase 3a — Gallardo widths, secular loci, vZLK portraits (#578)

Delivered in `resonances.py` (width half of the module) and the new
`secular.py`, exposed as `spatiography_resonance_atlas`.

1. **γ-invariance of the coplanar width.** The resonant angle is
   σ = k☾λ − kλ☾ + γ (Eq. 100). For the coplanar slice, shifting γ only
   translates the numerically averaged R(σ) horizontally, so ΔR (Eq. 103)
   and the half-width (Eq. 104) are γ-invariant; the implementation fixes
   γ = 0 and documents it. The satellite–Moon **apsidal offset**
   (varpi_offset_deg, default 180° anti-aligned, the §7.3 map convention)
   is a physical parameter and does change ΔR.
2. **Stable/unstable equilibria convention** (derived, test-locked): the
   a-Hessian of the semi-secular Hamiltonian K (Eq. 102) is
   K_aa = −(3/4)μ⊕/a³ < 0, so the stable equilibrium σ_s sits at the
   **minimum** of R(σ) and σ_u at its maximum; ΔR = R_max − R_min ≥ 0.
3. **Encounter truncation** (2ρ_H per Fig. 8 caption / Gallardo et al.
   2021): samples with |r − r☾| < 2ρ_H are excluded and the average is
   renormalized over the kept samples. `truncated_fraction` is reported
   per profile; near-Moon/exterior slices where it is large are
   encounter-dominated and their widths are not trustworthy as island
   half-widths (the corotation 1:1 overprediction — Primer §5.3, line
   959 — is the canonical example; the tool response carries the caveat).
4. **New state_frame vocabulary** registered: `element_space_ai`
   (apsidal-stationary loci, a in km, I in deg) and `vzlk_phase_plane`
   (ω in deg, y = √(1−e²)); `envelope_ae`/`vertical_ae` reuse
   `element_space_ae`.
5. **Eq. 96 vs Eq. 47 factor discipline**: the secular prefactors
   (ω_ext, ω_int, Eqs. 47/56) use K = 1 − (3/2)sin²I☾ (Eq. 48), distinct
   from the Laplace-radius characteristic rates (Eq. 96, factor
   1 − sin²I☾/2); a test pins the two apart.
6. vZLK double-averaging warning threshold α = a/a☾ > 0.8 is an
   implementer calibration (registered free parameter).

### Phase 3b — MEGNO kernel + fate diagnostics/classifier (#579)

Delivered in `crates/e2m2e-forces/src/megno.rs` (Rust kernel),
`e2m2e/algorithm/spatiography/megno.py` (re-export + scipy reference),
and `fate.py` (diagnostics + 8-class classifier).

1. **MEGNO formulation details** (Eq. 142): augmented 14-dim state
   [state(6), tangent(6), I₁, I₂]; the integrand's inner product
   δ·δ̇ is the **full phase-space** 6-dim inner product (δr·δv +
   δv·δv̇) — the Ȳ → 2 benchmark for regular motion hinges on the
   linear-in-t growth of the flow-aligned tangent component. Tangent
   renormalization at step boundaries (|δ| ∉ [1e-100, 1e100]) needs no
   accumulator compensation (integrand is scale-invariant). Step error
   control counts only the first 6 dims (same convention as
   `propagate_cr3bp_stm`).
2. **Free parameters registered** (trap ⑥ resolution, implementer
   calibration, subject to revisit):
   - MEGNO ordered/chaotic bands: |Ȳ − 2| ≤ 0.2 ordered,
     Ȳ ≥ 2 + 1.0 chaotic, in-between unclassified. Calibration
     cross-check: #578 resonance positions (SC low-inclination slice
     nearly all Ȳ ≈ 2, Primer line 1419).
   - Lunar-impact geometric criterion: surface collision
     r_sel ≤ R☾ (distinct from Moon-Hill entry, which is counted,
     non-terminal).
   - Terminal-event priority: earth reentry → moon impact → escape.
3. BCR4BP MEGNO shares the loop structure; its Python reference is not
   provided (mu_sun = 0 degenerates to CR3BP, covered by the CR3BP
   parity test).

### Phase 3c — six-domain dynamical map pipeline + MCP tool (#580)

Delivered in `crates/e2m2e-forces/src/cartography.rs` (geocentric
EM/EMS kernel), `e2m2e/algorithm/spatiography/cartography.py`
(scenario/grid/comparison orchestration), and the
`spatiography_dynamical_map` tool (sidecar `_BINARY_TOOLS`, five E2M2
frames: Ȳ / fate ids / t_escape / min selenocentric / min geocentric).

1. **EM/EMS model**: geocentric point-mass; perturbers (Moon; Sun for
   EMS) on **fixed Kepler ellipses** initialized from the scenario
   ("ephemeris-init then isolated evolution"). The Moon's 18.6-yr nodal
   regression is therefore absent — a documented deviation accepted for
   the architectural-persistence test (solar tide topology, not
   encounter phasing, is what the comparison isolates).
2. **MEGNO tangent in the map kernel is local** (δv̇ = J·δr; bodies
   fixed-model, REBOUND variational-particle semantics).
3. **Free parameters registered** (inheriting Phase 3b's, issue #580's
   own gaps): body initial phases (Moon (Ω☾, ω☾) = (311.07°, 175.84°)
   from the paper's anti-aligned convention, M☾ = 0; solar true
   longitude = lunar at the eclipse epoch), terminal early-stop
   (default on), max_step = 6 h (pericentre-dip miss guard at
   step-end-only event detection), per-zone grid resolutions (CI uses
   probe grids within ADR 0037 budget; production maps run manually via
   `scripts/spatiography_map_production.py`).
4. Golden checks test-locked: SC low-inclination slice Ȳ ≈ 2 (Primer
   line 1419, probe window), Table 4 bands bracket the #578 nominal
   resonance ladder, CG open-gateway note T☾(a☾, 0) = 3 < C1 (exact
   root-finding convention).
