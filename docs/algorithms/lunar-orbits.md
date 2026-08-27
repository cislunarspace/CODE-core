# Lunar Orbit Design

This page explains how e2m2e designs lunar orbits. The companion example
`examples/main_lunar_orbits.py` demonstrates the whole process with four
orbits; this page clarifies what stands behind the example: what lunar orbits
are, which spacetime systems e2m2e uses, what steps computation takes, and how
data flows.

Start from a concrete task: give e2m2e four numbers — semi-major axis
7000 km, perilune altitude 200 km, inclination 75 degrees, argument of
perilune 270 degrees — and ask for this orbit's positions over the coming
days. In which coordinate frame do these four numbers make sense? Through
which conversions and computations must they pass to become a trustworthy
ephemeris? Run this task through and lunar orbit design is understood.

## 1. What lunar orbits are

A lunar orbit revolves around the Moon as its central body. By lunar distance
there are four layers; each layer has different dominant dynamics and a
different design language.

| Layer | Orbit | Scale & shape | Dominant dynamics | Stability & period |
|---|---|---|---|---|
| ① | Low lunar orbit LLO | 100–2000 km above the surface, near-circular | Lunar gravity dominant, approximates two-body Kepler | Stable, period ≈ 2 h |
| ② | Elliptical lunar frozen orbit ELFO | Low perilune (≈ 200 km), apolune thousands to tens of thousands of km, high eccentricity | Lunar gravity plus non-spherical perturbations | Period ≈ 1 day, line of apsides can freeze |
| ③ | Distant retrograde orbit DRO | 10–50 thousand km from the Moon, retrograde | Earth–Moon three-body dynamics significant, yet the orbit still closes around the Moon | Neutrally stable, period days to weeks |
| ④ | Halo | Around Earth–Moon L1 or L2, ≈ 60 thousand km from the Moon | Three-body dynamics dominant | Unstable, period ≈ half a month |

Scale is the main thread. The farther from the Moon, the larger Earth's
gravity share, with dynamics transitioning from two-body Kepler to the
three-body problem and orbits from stable to unstable.

Layer ① LLO speaks pure two-body language. Keplerian elements fully describe
the orbit; period on the order of 2 h, near-circular shape; main uses are
remote sensing and landing intermediate orbits. e2m2e has no separate LLO
type: taking the ELFO pipeline's eccentricity toward zero yields near-circular
orbits.

Layer ② ELFO is the most subtle class of lunar orbit. As a large-ellipse
orbit flies around the Moon, lunar non-spherical perturbations (mainly J2)
rotate the line of apsides secularly, perilune altitude drifts, and the orbit
stops repeating. Choose suitable inclination and argument of perilune so that
perturbations' secular effects cancel and apsidal drift approaches zero — the
orbit freezes. The e2m2e example uses inclination 75°, argument of perilune
270° (apolune pointing at Earth). Be clear: freezing is a verified property,
not a constructed one. The ELFO pipeline does not solve; the user supplies
parameters, the pipeline propagates under full perturbed dynamics and
statistics drift, answering with data whether those parameters freeze.

Layer ③ DRO flies against the Moon's orbital direction. Three-body dynamics
is already significant but the orbit still closes around the Moon. DRO is
neutrally stable — neither diverging nor converging — cheap to maintain, and a
candidate orbit for lunar orbital stations. It cannot be described by
Keplerian elements; it needs three-body language.

Layer ④ Halo moves around a libration point without directly encircling the
Moon; perilune sits thousands to tens of thousands of km up depending on
amplitude. Three-body dynamics dominates, the orbit is unstable and needs
continuous station keeping. NRHO is the near-rectilinear form of large-amplitude
Halos with perilune only a few thousand km up — NASA's Gateway station uses
one.

One phrase sums it up: lunar orbits are a family, not a single orbit. Design
must first ask which layer the orbit sits in, then pick the matching model
language.

## 2. Spacetime systems and frame conversion

### Time

UTC appears only at input/output boundaries. Dynamics use TDB as independent
variable, called ephemeris time ET in SPICE. Why: UTC has leap seconds and is
discontinuous, unfit as dynamics' variable; TDB is the unified time scale of
ephemerides and dynamics. They currently differ by about 69 s; using UTC as
TDB directly shifts the Moon's position by ~70 km (Moon orbits Earth at
~1 km/s). e2m2e interfaces accept UTC strings and convert everything internal
to ET seconds.

### Frames

Lunar orbit design passes through four frames, each with its own role:

1. Moon-centered inertial frame. Axes aligned with J2000, origin at the Moon.
   Home of Keplerian elements. ELFO's four inputs (semi-major axis, perilune
   altitude, inclination, argument of perilune) mean something only here;
   geometrical quantities like perilune/apolune/period are computed here too.

2. Earth-centered inertial frame (GCRS). J2000 axes, origin at Earth. Home of
   propagation. e2m2e's ephemeris force model is built Earth-centered;
   spacecraft and lunar states both live in this frame. The Moon-centered
   state equals spacecraft-Earth-centered state minus the Moon's Earth-
   centered state — one subtraction converts frames. Moon-centered quantities
   such as perilune altitude are extracted from Earth-centered ephemeris after
   propagation.

3. Earth–Moon synodic frame. Rotating frame, x-axis toward the Moon,
   Earth–Moon stationary, libration points fixed. Home of CR3BP. The CR3BP
   model is simplest here; initial guesses and corrections happen in it. It is
   nondimensional: lengths divided by characteristic length, times by
   characteristic time. Two origin conventions coexist: barycentric
   normalization (Moon at x = 1-mu) and geocentric normalization (Moon at
   x = 1); internal computation uses barycentric, while output ephemeris
   synodic_position uses geocentric.

4. Moon-fixed frame (MOON_PA principal axes). Expansion frame of the lunar
   gravity field's spherical harmonics. Non-spherical perturbation
   accelerations are computed here, then rotated back to inertial.

Characteristic values come from the DE421 baseline: mu = 0.01215058535
(Earth–Moon mass ratio), characteristic length 384400 km (mean Earth–Moon
distance), characteristic time 375190 s (≈ 4.34 days). The lunar gravity field
uses GRGM900C, reference radius 1738 km.

### Implementation and one pitfall

Synodic↔J2000 conversion is done by `SynodicJ2000System`, with the batched
version sunk into Rust. The synodic x-axis takes the instantaneous Moon–Earth
unit vector, z-axis the normal of the Moon–Earth orbital plane, y completing
right-handed. Synodic coordinates divide by the instantaneous Earth–Moon
distance, which varies between 360 thousand and 405 thousand km.

Hence a pitfall you must know: `synodic_position` divides each point by the
instantaneous distance, so converting nondimensional coordinates back to km
with constant 384400 introduces ~5% scale error — an 1838 km orbit computes as
1746 km in the example. For exact Moon-centered distance, subtract the Moon's
position from Earth-centered inertial position; both ephemeris position_km and
lunar ephemeris are available without touching the synodic frame. The example's
`_moon_centric_position_km` does exactly this.

## 3. Computation flow

By orbit type e2m2e dispatches to two pipelines, corresponding to two
dynamics languages.

### ELFO pipeline

`orbit_type=ELFO`; input is Keplerian elements. Six steps:

1. Convert eccentricity: perilune altitude plus lunar radius gives perilune
   distance; e = 1 - rp/a.
2. Elements to Moon-centered Cartesian. Classical two-body formulas; true
   anomaly 0, i.e., start at perilune.
3. Superpose the Moon's Earth-centered state: Moon-centered Cartesian plus
   Moon state → Earth-centered inertial initial value.
4. Full-perturbation propagation in Earth-centered frame. Rust integrator,
   relative tolerance 1e-12. Force models include Earth/Moon/Sun/major-planet
   gravity, Earth and Moon nonspherical to degree 10, cannonball SRP.
5. Extract Moon-centered elements pointwise: each point's Earth-centered state
   minus the Moon, batched conversion to Keplerian elements.
6. Drift statistics: first-last differences of eccentricity, argument of
   perilune, perilune distance over the arc; linear fit of argument of
   perilune for annual drift rate.

This pipeline performs no correction. Whether freezing happens depends on the
input parameters; the pipeline validates under real dynamics and reports
drift. The example's 75° + 270° combination shows ≈ 2° argument-of-perilune
drift over 4 days while eccentricity and perilune altitude barely move —
freezing showing itself. Annual drift rates need a longer window (default 60
days) fits to be reliable.

### CR3BP pipeline

`orbit_type=DRO`, `HALO`, etc.; input is shape parameters. Initial values are
not computed, they are generated:

1. Generate initial guess. Family generators produce periodic orbits inside
   CR3BP. DRO starts from a seed orbit with differential correction +
   continuation fixing x-axis crossings; amplitude = mean of min/max
   Moon-centered distance. Halo starts from Richardson's third-order analytic
   approximation; after differential correction it walks the family; amplitude
   is the z-coordinate at y=0 crossings, positive north, negative south.
2. Phase location: place the initial point per the phase parameter on the
   periodic orbit; DRO additionally offsets half a period.
3. Sample nodes along the orbit (patch points). Strategies differ per family:
   Halo densifies near perilune (high speed, ill-conditioned STMs); NRHO and
   others sample uniformly in time. Deleting nodes near perilune remains as a
   utility function for comparison (forcing inclusion of epoch t=0) but is no
   longer the production default for NRHO since #473.
4. Batch conversion synodic→J2000.
5. Ephemeris correction. CR3BP differs from real ephemeris: the lunar orbit is
   not circular; solar and planetary perturbations exist; an ideal periodic
   orbit no longer closes under real dynamics. Rust multiple shooting anchors
   nodes onto real ephemeris. Stable orbits (DRO) take the velocity-weighted
   two_level path — correct one rev then extrapolate freely, bounded; unstable
   orbits (Halo/NRHO) take segmented full-arc shooting, integrating segment by
   segment to fill the ephemeris grid. NRHO fixes 1 rev/segment at step 1;
   Halo allows at most 3 revs/segment.
6. Assemble the ephemeris table.

NRHO and Halo share segmented shooting; discretization defaults are
independent: uniform time + 1 rev/segment. Defaults L2 southern family,
perilune altitude 5000 km, phase 0.5, about one-month arc converge to a
nominal ephemeris as long as the time grid (#473; 5.7.2's delete-near-perilune
default still had epoch holes or non-converging merge tiers at this scale).
Closer-in (~2000 km perilune) short arcs work too.

### Where computation happens

Numerical heavy lifting of both pipelines runs in Rust: propagation, shooting,
batched frame conversion, batched ephemeris queries. Python only orchestrates:
construct request, dispatch pipeline, interpret results. All four example
orbits take well under a minute (release build); most of it is the ELFO
pipeline's strict-tolerance integration, not Python.

## 4. Data flow

Data enters as request and exits as result, passing dispatch and two pipelines
in between.

Input is `DesignOrbitRequest`, Pydantic-validated. orbit_type decides
dispatch; shape parameters validated per type with defaults filled; ELFO
requires semi-major axis, defaults inclination 75°, argument of perilune 270°,
perilune altitude 200 km, propagating 60 days; propagation parameters in
seconds, default output step 3600 s; perturbation switches and harmonic degree
overridable.

On receiving the request, `design_orbit` dispatches by orbit_type and produces
an `OrbitDesignResult`:

- `ephemeris`: nominal ephemeris in an `EphemerisTable` container. Each row
  holds UTC calendar time, GCRS position (km) and velocity (m/s), synodic
  nondimensional position. GCRS is mission data; synodic serves three-body
  perspective plots.
- `initial_state`: Earth-centered inertial 6-dim state at epoch, feedable
  directly to prediction and control chains.
- `cr3bp_orbit`, `cr3bp_jacobi`, `correction`: CR3BP reference periodic orbit,
  Jacobi constant, correction result. None/nan for ELFO scenarios.
- `drift_e`, `drift_aop_deg`, `drift_rp_km`,
  `secular_aop_rate_deg_per_year`: ELFO freeze diagnostics.
- `moon_centric_elements`: ELFO's Moon-centered element series, every point.

Restating the frame discipline when consuming ephemeris: Moon-centered
geometric quantities (perilune altitude, apolune altitude, Moon distance)
come from GCRS position minus Moon position — never reconstruct via synodic ×
constant. The synodic series exists to view the orbit's three-body perspective
shape.

## 5. Companion example

`examples/main_lunar_orbits.py` turns all of the above into four runnable
orbits matching the four layers:

- LLO: ELFO pipeline, semi-major axis 1838 km, perilune altitude 100 km,
  eccentricity near zero, propagating 2 days ≈ 24 revs.
- ELFO: semi-major axis 7000 km, perilune altitude 200 km, propagating 4 days
  ≈ 4 revs.
- DRO: amplitude 50 thousand km, period 8.4 days, propagating 9 days for over
  a full rev.
- Halo: L2, amplitude 30 thousand km, period 14.6 days, propagating 15 days
  for one full rev.

The example draws three figures. Fig 1: Moon-centric distance vs time for all
four orbits, log axis — the scale hierarchy from 1800 to 70 thousand km at a
glance. Fig 2: Moon-centered X-Y plane shapes, two subplots splitting
near-Moon and distant regions. Fig 3: ELFO freeze verification — argument of
perilune, eccentricity, perilune altitude evolving in time, watching whether
the apsidal line stays near 270°.

To run:

```bash
python examples/main_lunar_orbits.py --save          # headless: save PNGs
python examples/main_lunar_orbits.py                 # interactive plots
python examples/main_lunar_orbits.py --skip DRO      # skip one orbit
```

SPICE kernels required (`kernels/` at repo root). Each orbit prints design
parameters, elapsed time, peri/apolume, and freeze diagnostics. To dig into
single-Halo design details see `examples/main_design.py`.
