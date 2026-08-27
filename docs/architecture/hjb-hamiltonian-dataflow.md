# Ephemeris Force-Model Hamiltonian: Solve-Chain Data Flow (#498)

This page is the companion research document for ADR 0034. For the
subsystem's overall architecture (two-level division of labor, Hamiltonian
family spectrum, dimension ceiling, binding entries, state semantics,
verification tiering), `docs/architecture/hjb-subsystem.md` on master is
authoritative; this page does not repeat it. It covers only two areas the
other does not: internal facts about force models and ephemeris caching, and,
once ADR 0034 decision 1's planar full-ephemeris Hamiltonian lands, which path
the data of one solve flows along. Line numbers refer to master (`c3af80f`).

## Force-model side: CompiledForce status facts

`crates/e2m2e-forces/src/forces/compiled.rs` is the compiled force-model
enum: Python serializes each force into a tuple (`to_rust_spec`), Rust
rebuilds it via `force_from_tuple`, and the integration inner loop never
crosses back to Python. Facts relevant to #498:

- The interface is pointwise `acceleration(et, state6, observer)`
  (compiled.rs:195) plus the summing `compute_total_acceleration`
  (compiled.rs:364). Batching needs no new interface: ephemeris quantities
  depend on t only, not on nodes; look up the cache once per t per RK substep
  and reuse across all grid nodes — exactly the batch semantics #498 wants.
- #498's planar full-ephemeris scope uses only the `PointMass` and
  `ThirdBody` variants. `ThirdBody`'s solar position queries the ephemeris via
  spk_accel; the cache applies in the inner loop (next section).
- **Variable-mass contract**: `SRPVariableMass { area, cr, shadow_bodies }`
  stores no mass; paired `acceleration_with_mass(et, state, mass, observer)`
  reads current mass from the augmented state. Under ADR 0034 decision 2 it
  was out of #498's scope at writing time; it existed only in uncommitted
  workspace then and has since been merged independently per decision 5
  (#507, Python side `VariableMassSolarRadiationPressure`);
  geo-nrho's `lowthrust_rs` calls this variant.

## Ephemeris side: EphemCache status facts

`crates/e2m2e-spice/src/ephem_cache.rs` (ADR 0016): before integration, needed
body states and frame matrices are pre-sampled on a uniform time grid via
cspice and stored in memory as cubic splines; during solving we look up tables
and never touch cspice. Cubic splines were chosen for C² continuity so
adaptive integrators do not shrink step sizes. Key structural facts:

- **Process-level singleton**: `static CACHE: RwLock<Option<EphemCache>>`,
  installed by `enable(cache)`. `RwLock` is deliberate design: parallel
  segments read pure numeric splines concurrently across threads, and read
  locks never block each other (around ephem_cache.rs:468). ADR 0034 decision
  4 keeps this singleton; injection per construction (ADR 0033) happens at the
  configuration level: the force-model list, time range, and epoch mapping are
  constructor parameters of the Hamiltonian.
- **Two-tier miss semantics** (ADR 0020 decision 4): when not enabled, queries
  return `Ok(None)` and fall back to cspice; after enabling, misses (outside
  range / missing target) hard-fail without exception. The cache interval must
  cover the whole HJB solve window.
- **Everything needed for a time-varying synodic frame is in the cache**:
  `lookup_body_position` / `lookup_body_velocity` give Moon-relative-to-Earth
  position and velocity; rotation rate ω(t), ω̇(t), and libration rate derive
  from them — no second ephemeris query path needed.

## Hamiltonian structure for low-thrust min-fuel

Performance index `J = ψ(x(tf)) + ∫ fuel_weight·δ dt`, control sets
`δ ∈ [0,1]`, `û ∈ S²`. Dynamics
`f = [v, a_forces(r,t) + (T·δ/m)·û, -T·δ/(Isp·g₀)]`.
Minimizing over controls yields analytic optimal laws:

- Thrust direction `û* = -p_v/‖p_v‖` (anti-covariant-velocity);
- Bang-bang throttle: switching function
  `S = fuel_weight - (T/m)·‖p_v‖ - p_m·T/(Isp·g₀)` (includes the mass costate;
  mind T/m's m/s²→km/s² unit conversion); `δ* = 1` when `S < 0`, else 0;
- With control eliminated:
  `H* = p_r·v + p_v·a_forces + min(0, S)`
  (`PlanarDoubleIntegrator` is the massless-dimensional planar precedent:
  `control_gain = fuel_weight - max_accel·‖p_v‖`,
  `H = drift + min(control_gain, 0)`, see e2m2e-hjb-dynamics'
  double_integrator.rs).

The `partial_bound` envelope derivations entered the repo with the
implementation (written into implementation doc comments):
`∂H/∂p_r = v`; `∂H/∂p_v = a_forces + thrust terms`, with the mass dimension
taking T/m at the upper end of the mass grid interval; `∂H/∂p_m` is constant
`-T/(Isp·g₀)` when `δ*=1`.

## Target state: data flow of one planar full-ephemeris HJB solve

Once ADR 0034 decision 1's implementation lands, one solve runs in three
segments.

**Preparation (Python + cspice).** The caller provides solve window
`[et0, etf]`, grid definition, terminal cost, and engine parameters.
`EphemCache::build` pre-samples (MOON, EARTH) and (SUN, EARTH) on a uniform
grid covering the window and calls `enable`; the `CompiledForce` list (two
primaries' `PointMass` + solar `ThirdBody`) and epoch mapping (solver t ↔ SPICE
et) go to the Hamiltonian impl as constructor parameters. Terminal cost ψ is
laid onto the grid via the shape module.

**Solving (Rust hot loop, zero cspice).** Each TVD-RK substep: convert t to et
via epoch mapping, query the cache once for lunar position/velocity, derive
ω(t), ω̇(t), libration rate — **one lookup per t, reused across the grid**;
then node by node over the grid: synodic-frame coordinates compute two-primary
point-mass gravity and solar third-body gravity (in-plane components),
superpose frame-transform-induced Coriolis, centrifugal, ω̇ and libration
corrections, minimize over control per the switching function to get H*;
`partial_bound` supplies the dissipation envelope; LF terms compose `dphi_dt`.
The backward solve is implemented as forward evolution under time reversal
(semantic contract: geo-nrho `hjb-dp-route.md` §2).

**Artifacts (Python).** The value-function grid persists under ADR 0033
decision 3's contract: metadata explicitly records state-dim order
`(x, y, vx, vy, m)`, nondimensionalization conventions, `times` semantics (ET
seconds), and epoch mapping parameters; stored as catalog value-function
records; consumers (#499's gradient interface, time interpolation mandatory)
depend only on that format.

## Degradation cross-checks and verification hooks

ADR 0034 decision 6's three-tier acceptance maps onto three ready anchors:

- **(a) Degradation cross-check**: replace the cache with a circularized
  stationary synthetic ephemeris (constant lunar distance, constant ω); the
  ephemeris dynamics must degrade term-by-term to `Cr3bpSynodic`. Anchored on
  the impl already cross-checked against `cr3bp_eom` in #497.
- **(b) Force consistency**: for identical (t, state), forces inside the
  Hamiltonian (post frame transformation) match direct
  `compute_total_acceleration` calls pointwise;
- **(c) Coarse-fine regression**: small grids solve both ephemeris and CR3BP
  value functions; compare magnitude and iso-surface structure (verification
  ladder tier 3).

## Extension slot: attaching spherical-harmonics/SRP experiments

If ADR 0034 decision 2's experimental items start later, the hook sits at the
solve segment's per-node force evaluation: swap `PointMass` for
`GravityField` (10×10, body-fixed frame matrices also via cache); SRP uses
`acceleration_with_mass` reading the mass-axis coordinate. Evaluate on the z=0
section, discard out-of-plane components; no seam changes, no grid changes.
After experimental results are cross-checked against CR3BP/ephemeris
point-mass solutions, a new ADR revises decision 2.

## Status gap list

This section snapshots gaps at writing time (`c3af80f`); all three have since
progressed, annotated item by item.

1. **Ephemeris Hamiltonian**: not implemented at writing time (#498 body);
   landed with #515: `EphemerisPlanar` in e2m2e-hjb-dynamics;
   `solve_hjb_py` registers the `ephemeris_planar` dynamics.
2. **Variable-mass SRP contract**: only in an uncommitted local workspace at
   writing time; independently committed per ADR 0034 decision 5 (#507):
   `SRPVariableMass` + `acceleration_with_mass` + Python class
   `VariableMassSolarRadiationPressure`. Not on #498's critical path, but
   `lowthrust_rs` depends on it.
3. **Verification ladder tiers 3–4**: tier 3 coarse-fine regression established
   with #498 acceptance
   (`tests/numerical/integrators/bindings/test_hjb_solve.py`);
   tier 4 closed-loop replay remains manual on geo-nrho's side.
