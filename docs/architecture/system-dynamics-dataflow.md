# System and Dynamics: Data Flow Across Two Class Hierarchies

Under `e2m2e/algorithm/dynamics/` live two class hierarchies: the System side
describes *what system this is* (μ, characteristic scales, libration points,
ephemeris, units, coordinate frames); the Dynamics side describes *how to
integrate it* (integrators, tolerances, STMs, events, result caches). The
package docstring summarizes: System (data context) + Dynamics (propagation
orchestration) (`e2m2e/algorithm/dynamics/__init__.py:1`). Following data's
direction of flow, this page explains each family — what they are, who reads
which members, which path data takes in one propagation. All line numbers refer
to current code.

## Starting from one propagation

The most common scenario: propagate an Earth-Moon CR3BP trajectory. Its standard
form in the test suite (`tests/conftest.py:13-25`):

```python
system = CR3BP_System(mu=Datum.DE421.mu, primary="Earth", secondary="Moon")._with_default_scales()
dynamics = CR3BP_Dynamics(system=system)
result = dynamics.propagate(state0, (0.0, 6.3))
```

Data travels four legs along this chain.

**Leg one: constructing the System.** `CR3BP_System.__init__` takes only mass
parameter μ, two body names, optional body radii; characteristic length/time,
five libration points are all set None
(`cr3bp_system.py:71-128`). Such a system can't compute yet:
`DU`/`TU`/`VU` properties raise system-not-initialized when scales unset
(`cr3bp_system.py:194-212`). `_with_default_scales()` fills characteristic
scales per body pair (`cr3bp_system.py:130-160`); libration points defer until
first need: `get_libration_point` calls `compute_libration_points` on first use
(`cr3bp_system.py:293-317`). System construction is thus two-stage: first pick
the system, then pick the scaling.

**Leg two: constructing Dynamics.** `CR3BP_Dynamics(system)` does little: store
the system reference, fill default integrator config (RK45, rtol/atol 1e-12,
max_step 0.01), clear result caches (`dynamics.py:78-97`, `544-553`). It copies
none of the system's parameters. μ is fetched live via `self.system.mu` at every
acceleration evaluation (`dynamics.py:593`).

**Leg three: propagate.** Eventless CR3BP takes the Rust fast path; Python passes
only scalars — `mu`, time span, initial state, integrator config — across FFI
(`dynamics.py:851-859`); the trajectory computes Rust-side, returns as a
`{"time", "states"}` dict while writing `self.last_trajectory` cache
(`dynamics.py:861-870`). Details in the *propagate internals* section below.

**Leg four: results enter data containers.** The design chain wraps the returned
dict in an `Orbit`, stuffing in the system reference too:
`Orbit(states=result["states"], times=result["time"],
system=dynamics.system)` (`design_orbit.py:550`). From then on this float set
travels with its own unit/frame interpreter; whoever computes next may regrow a
dynamics from `orbit.system` — exactly how `StabilityAnalysis` rebuilds on demand
(`stability.py:95-98`).

Four legs together express both families' division: System is the long-lived
model context; Dynamics is the config-and-cache-carrying propagator orbiting it.

## System: model context

### The base class promises only three members

The `System` ABC's minimal interface has three items: `frame`, `unit_system`,
`gravitational_parameter(body)` (`system.py:15-51`). Its docstring names what is
deliberately excluded: `mu`, `body_state(body, t)`, `coordinate_system` belong to
specific implementations (`system.py:25-27`). The base also carries non-abstract
`get_body_position` defaulting to NotImplementedError: ephemeris-only capability,
placed there just for a clear error site (`system.py:54-66`).

Thin interfaces have consequences: code wanting polymorphism over both systems
can rely only on those three members; everything else probes live via
`getattr`/`hasattr`. The consumers section shows this pattern occurring for real.

### CR3BP_System: nondimensional, autonomous, two-stage initialization

Construction injects five things: `mu`, primary/secondary names, optional radii
(`cr3bp_system.py:71-80`). Two validations run: positive radii if given
(`cr3bp_system.py:93-98`), μ ∈ (0, 0.5) (`cr3bp_system.py:102-106`). Post-
construction two optional init steps remain:
`set_characteristic_scales(distance, period)` derives length/time/velocity scales
and sets `is_initialized` (`cr3bp_system.py:214-235`);
`compute_libration_points()` solves three collinear points with `fsolve`, gives
two triangular analytically (`cr3bp_system.py:237-291`; numeric at `264-266`,
analytic at `268-274`).

Against the base's three members it answers: synodic (rotating) frame,
nondimensional units, GM with convention total-mass-one → primary `1 - mu`,
secondary `mu` (`cr3bp_system.py:163-191`).

Once initialized the object provides four data classes:

- Characteristic scales: properties `DU` (km), `TU` (days), `VU` (m/s), plus raw
  fields `characteristic_length/time/velocity`
  (`cr3bp_system.py:108-110`, `194-212`).
- Libration points: `L1`–`L5` and `L_points` dict (`cr3bp_system.py:112-117`).
- Jacobi constant: `get_jacobi_constant(state)`, Parker convention
  (`cr3bp_system.py:319-354`).
- Unit conversion & stability: `dimensionless_to_physical` /
  `physical_to_dimensionless` (`cr3bp_system.py:356-396`),
  `compute_stability_index` (`cr3bp_system.py:398-442`).

Radius fields (`primary_radius_km`/`secondary_radius_km`) play no dynamical role;
they feed collision detection — flow described under events & collision.

### EphemerisSystem: unified SPICE-query entry

Construction injects five things: body-name list, kernel-loaded `SPICEManager`,
reference origin (default "EARTH"), frame (default J2000), optional
`CoordinateSystem` (`ephemeris_system.py:35-55`). Against the base's three
members: construction-given frame, physical units, GM passed through
`spice.get_gm` (`ephemeris_system.py:59-103`.

Unlike CR3BP's two-stage init, stage two here is assigning `coordinate_system`,
done after construction via property setter (`ephemeris_system.py:69-75`). The
orchestration layer's idiom: construct first, attach later:
`system = EphemerisSystem(...)` then
`system.coordinate_system = CoordinateSystem(...)` (`propagation.py:118-123`).
Force-model propagation requires this field; unset gets rejected by ForceModel
(`force_model.py:57-58`).

Ephemeris data flows out via four query methods:
`gravitational_parameter`/`get_gm` (single-body GM), `get_gm_values` (array in
bodies order), `get_body_position` (position), `get_body_state` (six-dim state)
(`ephemeris_system.py:94-150`). Plus `update_coordinate_systems(t, state)` for
advancing dynamic frames (`ephemeris_system.py:77-92`); after ForceModel
propagation sank to Rust, stepwise updates happen inside Rust — no call sites left
inside e2m2e/, tests only
(`tests/numerical/forces/container/test_force_model_dynamic_axes.py:143-186`).

### BCR4BPSystem: extends CR3BP_System, adds the Sun

`BCR4BPSystem` inherits `CR3BP_System`
(`bcr4bp_system.py:23`), taking four extra sun parameters:
`sun_mass`, `sun_distance`, `sun_angular_rate`, `sun_phase0`; the first two derive
from DE440 constants + mean Earth-Sun distance when omitted
(`bcr4bp_system.py:50-106`; derivation at `89-92`). `sun_angular_rate` is special:
it depends on characteristic time; direct construction stashes None until
`set_characteristic_scales`' override derives it from Julian-year revolution
(`bcr4bp_system.py:148-161`); hence the standard entry is classmethod
`BCR4BPSystem.earth_moon()` completing construct+scale in one step
(`bcr4bp_system.py:109-132`).

Solar position never queries ephemeris — analytic `sun_position(t)`: coplanar
circle in the synodic frame (`bcr4bp_system.py:163-183`).
`gravitational_parameter` accepts "sun" beyond "primary"/"secondary"
(`bcr4bp_system.py:185-193`). Mind the docstring: BCR4BP has no Jacobi integral;
`compute_libration_points` yields the corresponding CR3BP's points, reference
positions only (`bcr4bp_system.py:41-42`).

## Dynamics: propagation orchestrators

### Three state kinds on the base class

`Dynamics.__init__` takes just `system` (`dynamics.py:78-84`), then places three
kinds of things on instances:

1. **System reference**: `self.system`, read-only across propagation
   (`dynamics.py:84`).
2. **Integrator config**: `integrator`, `rtol`, `atol`, `max_step`, defaults on
   class constants (`dynamics.py:86-91`). Public fields mutated post-construction:
   parallel search adjusts tolerance/steps per mission
   (`search_parallel.py:837-842`); multiple shooting overrides three immediately
   after construction (`multiple_shooting.py:97-101`); test fixtures loosen
   ephemeris tolerances for speed (`tests/conftest.py:92-100`).
3. **Result caches**: `last_trajectory`, `last_stm`, overwritten at each
   propagation end (`dynamics.py:93-94`; write sites below). `CR3BP_Dynamics`
   adds `jacobi_history`/`jacobi_error` monitoring caches
   (`dynamics.py:552-553`).

I.e., Dynamics is a stateful worker: configs plus latest results persist on the
instance for callers to fetch afterward (e.g., `compute_state_transition_matrix`
internally = one propagation then final STM, `dynamics.py:986-999`).

### Template method: propagate is the skeleton, subclasses fill hooks

`propagate()` is a template method: base defines the algorithm skeleton
(normalize args → validate → dispatch → assemble results); subclasses join via
two hook groups: `_get_eom_func(with_stm)` supplies ODE RHS,
`_get_max_step(t_span)` supplies step ceilings (`dynamics.py:100-125`). Dispatch
has two branches only: `_propagate_with_stm` (42-dim augmented) vs
`_propagate_state_only` (6-dim) (`dynamics.py:223-229`). Both base implementations
run scipy `solve_ivp`; all three subclasses override both branches preferring Rust
fast paths (event handling varies per subclass — below).

### Three subclasses, two inheritance choices

`CR3BP_Dynamics` (`dynamics.py:522`) and `EphemerisDynamics`
(`ephemeris_dynamics.py:47`) implement autonomous nondimensional CR3BP equations
vs time-dependent physical-unit N-body. `BCR4BP_Dynamics` extends `Dynamics`
directly (`bcr4bp_dynamics.py:42`), not `CR3BP_Dynamics`, despite its equations
being CR3BP + solar term. The reasons visible in code:

- Different Jacobian signatures: CR3BP autonomous,
  `compute_jacobian_A(state)` (`dynamics.py:611`); BCR4BP time-dependent,
  `compute_jacobian_A(t, state)` (`bcr4bp_dynamics.py:123`).
- Inverted Jacobi semantics: `CR3BP_Dynamics` carries Jacobi monitoring caches
  from construction (`dynamics.py:552-553`), while BCR4BP is time-periodic with
  no Jacobi integral — must implement both `compute_jacobi_constant` and
  `_handle_jacobi` as NotImplementedError raises (`bcr4bp_dynamics.py:457-463`),
  switching off inherited capabilities one by one.
- STM entry gains a parameter: time-dependent systems' Φ depends on start/end
  epochs; `compute_state_transition_matrix` gains `t0`
  (`bcr4bp_dynamics.py:438-455`).
- Different Rust entries: BCR4BP's propagation functions take four extra sun
  parameters (`bcr4bp_dynamics.py:293-303`).

Inheriting `CR3BP_Dynamics` would mean overriding nearly every public method and
suppressing the Jacobi machinery anyway. The repo has prior art on this exact
trap: ForceModel once nominally inherited Dynamics merely to reuse a few data
attributes while rewriting `propagate` wholesale and raising on STM/Jacobi —
adjudicated an LSP violation (fake inheritance) and split into an independent
class (`force_model.py:30-38`).

## Consumers: who reads System outside this package

Whether separation bears load depends on how much code consumes System without
constructing Dynamics. Verified case by case:

### Force-model side: reading coordinate_system, spice, origin, gravitational_parameter

`ForceModel` holds system (annotated plain `Any`), mandating
`system.coordinate_system` be set at construction (`force_model.py:45-58`);
propagation serializes each force into Rust tuples
(`force.to_rust_spec(self.system)`), reads `system.origin` as observer
(`force_model.py:281-293`); `system.spice` presence routes
resource-missing vs capability-missing (`force_model.py:226-233`). It never
constructs Dynamics and explicitly refuses to inherit it (`force_model.py:30-38`).

`PhysicalModel._resolve_mu(system)` is the most direct System consumer force-side:
when explicit μ is absent it calls `system.gravitational_parameter(self._body)`
(`physical_model.py:29-41`). `ConicalShadowModel.flux_factor(t, state, system)`
pulls coordinate_system/spice/origin via `require_inertial_frame` before querying
Sun & occluder positions for illumination fraction
(`shadow.py:166-197`; `physical_model.py:69-84`).

### Transfer & solver side: reading origin and gravitational_parameter

The low-thrust trio treats system as a parameter bag. `qlaw_guess(system, ...)`
only parses central-body μ: probe PointMassGravity among forces first, fall back
to `system.gravitational_parameter(origin)`, `origin` itself getattr-defaulted to
"EARTH" (`qlaw.py:210`, `265-282`). `LowThrustShooting` and
`LowThrustCollocation` each do two things at construction: pre-serialize forces
via `to_rust_spec(system)`, store `origin` as observer
(`lowthrust_shooting.py:125-162`; `lowthrust_collocation.py:59-88`).

`NormalFormContext` reads only μ, probed as `getattr(system, "mu", None)`:
take it when present on CR3BP_System, else fall back to a pinned constant
(`normal_form/context.py:64`, `114`, `196-201`).

### Coordinate-conversion side: reading mu, scales, spice

`SynodicJ2000System` holds a `CR3BP_System` + spice: pointwise conversion reads
`mu` (barycenter shift) and `characteristic_time` (time dimensioning); batch paths
pass two scalars — `mu` + time unit — to Rust
(`synodic_j2000.py:28-41`, `52`, `96-97`). `rho_bridge` functions take
`EphemerisSystem`: read `system.spice` to build SynodicAxes, read
`system.get_body_state("MOON", et)` for lunar states
(`rho_bridge.py:47-63`, `110`).

### Station keeping & prediction: constructing and configuring EphemerisSystem

`control_orbit` builds a ten-body EphemerisSystem, attaches coordinate_system,
hands off to Monte Carlo (`controller.py:228-247`); workers rebuild the same
system from parameters (`monte_carlo.py:644-650`), plus a scales-only
CR3BP_System for synodic conversion (`monte_carlo.py:66-72`, `707-716`).
`propagate_orbit` likewise constructs EphemerisSystem, attaches coordinate_system,
then `ForceModel.from_config(force_config, system)` (`propagation.py:118-124`).
Both consume System while propagation happens inside ForceModel — bypassing
Dynamics.

### Remaining scattered points

`PoincareSection.periapsis(center, system)` reads `mu` + body names for section
centers (`sections.py:202-228`). Inside `design_orbit`, one CR3BP system instance
feeds four consumers simultaneously: constructing `CR3BP_Dynamics`, calling
`get_jacobi_constant` directly, reading `characteristic_time` for conversions,
constructing `SynodicJ2000System` for coordinate conversion
(`design_orbit.py:983-987`, `1030-1032`).

### Polymorphic functions annotated with System

Entries annotating `system: System`: `PhysicalModel._resolve_mu` / `to_rust_spec`
(`physical_model.py:29`, `45`), `ConicalShadowModel.flux_factor`
(`shadow.py:170`), `qlaw_guess` + `make_shooter_for_qlaw` (`qlaw.py:210`, `290`),
`LowThrustShooting.__init__` + `LowThrustCollocation.__init__`
(`lowthrust_shooting.py:125`; `lowthrust_collocation.py:59`),
`NormalFormContext.__init__` (`normal_form/context.py:64`).

Spreading open what these actually read: genuinely cross-seam polymorphic access
collapses onto the single base member `gravitational_parameter` — CR3BP-side
takes "primary"/"secondary" returning nondimensional values, ephemeris-side takes
SPICE names returning km³/s² (`system.py:42-51`'s docstring documents the dual
semantics). Other entries labeled `System` actually read implementation members
(`origin`, `coordinate_system`, `spice`, `mu`) behind getattr/hasattr fallbacks.
Contract tests pin dual-system same-interface behavior: identical assertion sets
run against both implementations
(`tests/algorithm/dynamics/test_system_contract.py:53-64`).

### Duck typing: nominal System, structural reality

Beneath annotations the real contract runs thinner — three proofs:

- Data-layer `Orbit`/`OrbitFamily` store system as `Any`, docstrings stating the
  data layer doesn't depend on algorithm — capability judged by
  `hasattr(system, "get_jacobi_constant")` (`orbit.py:8-12`, `50`, `418-429`).
- Low-thrust tests pass bare `SimpleNamespace(origin="EARTH")` as system
  (`test_lowthrust_collocation.py:23`;
  `test_qlaw_failure.py:37`).
- `RelativeDynamics.linear_model` try/excepts between
  `compute_jacobian_A(t, state)` and `compute_jacobian_A(state)` signatures
  (`relative_dynamics.py:145-150`).

## propagate internals: how data moves in one call

### Entry orchestration

`Dynamics.propagate` proper (`dynamics.py:144-233`) processes input in fixed
order:

1. `initial_state` → ndarray; `max_step` fetched via hook
   (`dynamics.py:199-200`).
2. Event normalization: single callable wrapped in list; empty list ≡ no events →
   None (`dynamics.py:201-205`).
3. Collision detection on: build collision events into the list; check whether
   initial state already sits inside some body radius
   (`dynamics.py:207-209`).
4. `backend` validation: only "scipy"/"rust"; mandatory-with-events, else error
   (`dynamics.py:211-214`).
5. Initial-state-inside-radius short-circuits to a single-point trajectory +
   immediate collision flag: scipy events won't trigger on g<0 starts — must
   handle explicitly (`dynamics.py:217-221`, `474-493`).
6. Dispatch by `with_stm` into two branches (`dynamics.py:223-229`).
7. Collision detection on: extract collision info from result event segments
   (`dynamics.py:231-232`, `495-510`).

### scipy vs Rust backend paths

**scipy path** (base implementation): `_propagate_state_only` integrates the
`_get_eom_func(False)` RHS via `solve_ivp`, transposing `result.y` to (n, 6);
failures or empties raise `PropagationFailure` — empty trajectories may never
masquerade as success (`dynamics.py:291-325`). `_propagate_with_stm` flattens the
6×6 identity into the 42-dim augmented state before integrating, splitting back
into first-6/last-36 after (`dynamics.py:254-271`). Both scipy branches write
`last_trajectory` (plus `last_stm` when carrying STM) before returning
(`dynamics.py:273-274`, `327`).

**Rust paths** (subclass overrides): eventless, all three require the Rust
extension — missing raises, no scipy degradation. What crosses FFI differs per
class:

- CR3BP: single system parameter `mu`, plus time span, t_eval, initial state,
  tolerances, steps (`dynamics.py:851-859`; STM variant `dynamics.py:782-790`).
- BCR4BP: `mu` plus sun four-pack `mu_sun`/`sun_distance`/`sun_angular_rate`/
  `sun_phase0` (`bcr4bp_dynamics.py:360-370`; STM variant `293-303`).
- Ephemeris: three sequences fetched live from system — `bodies`, `origin`,
  `gm_values` (`ephemeris_dynamics.py:124-126`, `133-142`). Body positions never
  pass through Python objects: Rust queries process-local SPICE directly inside
  its integration inner loop (cache first, cspice on miss)
  (`crates/e2m2e-spice/src/spk_accel.rs:46-56`). Along this chain
  EphemerisSystem's role collapses into a parameter bag.

All Rust returns get defensive length validation (returned-count ≠ requested-count
raises) before writing `last_trajectory` (`dynamics.py:792-804`, `861-870`).

**Events' third branch**: with events and `backend="rust"`, CR3BP/BCR4BP take
the Rust event integrator `solve_ivp_events` with the RHS sunk into Rust: the
RHS is dispatched by a `RustEomKernel` identifier (`cr3bp`/`cr3bp-with-stm`/
`bcr4bp`/`bcr4bp-with-stm` + params) to the `e2m2e-forces` CR3BP/BCR4BP EOM/STM
kernels, so per-step RHS evaluation stays inside Rust; user-defined event
functions remain Python callbacks translated into `(g, terminal, direction)`
triples (`dynamics.py` `_propagate_with_stm_rust_events`/
`_propagate_state_only_rust_events`; `bcr4bp_dynamics.py` same names). Ephemeris
doesn't support events: events non-None → immediate NotImplementedError
(`ephemeris_dynamics.py:85-113`, `162-189`).

### When return-dict keys appear

`time`/`states` always present. Others conditionally:

- `stm`: when `with_stm=True` (`dynamics.py:276-280`).
- `status`/`cause`: state-only paths only — base scipy version + CR3BP's two Rust
  pure-state branches (`dynamics.py:329-334`, `872-877`, `926-931`). The STM
  path, EphemerisDynamics, and BCR4BP_Dynamics's Rust branches carry neither.
- `jacobi`/`jacobi_error`: `with_jacobi=True` and CR3BP. Base `_handle_jacobi`
  is a no-op (`dynamics.py:345-358`); `CR3BP_Dynamics` overrides to compute
  pointwise + cache (`dynamics.py:1012-1022`); BCR4BP overrides to raise
  (`bcr4bp_dynamics.py:461-463`).
- `t_events`/`y_events`: events passed; per-event trigger times + states
  (`dynamics.py:282-284`, `336-338`).
- `collision`: `collision_detection=True`; None if clean, else
  `{"body", "t", "state"}` (`dynamics.py:231-232`, `507-509`).

### Events and collision detection

Event functions follow scipy semantics `g(t, state) -> float` with optional
`terminal`/`direction` attributes; `PoincareSection.event(...)` is the main
constructor (`tests/algorithm/dynamics/test_events.py:21-26`). Under
`with_stm=True` events receive the 42-dim augmented state
(`test_events.py:86-97`).

Collision detection translates body impacts into terminal events:
`_collision_specs` reads `primary_radius_km`/`secondary_radius_km` from system
(both missing → error), reads `mu` fixing bodies at synodic [-μ,0,0] and [1-μ,0,0]
(`dynamics.py:398-428`); `_setup_collision_detection` reads `DU` converting km
radii to nondimensional, building terminal events `g = |r - center| - R` appended
after user events (`dynamics.py:447-472`). Collision events sitting last in the
list is exploited backwards by `_extract_collision`: walking tail indices to match
per-body trigger records (`dynamics.py:501-509`). So collision's data flow reads
three system fields throughout: radii, mu, DU.

### Differences among three propagation chains

- **CR3BP**: nondimensional, autonomous. EOM without explicit t; spans and t_eval
  nondimensional; system contributes only `mu`.
- **BCR4BP**: solar direct+indirect terms atop CR3BP RHS; sun position analytic
  from `system.sun_position(t)`; EOM explicitly time-dependent
  (`bcr4bp_dynamics.py:72-121`). Jacobian lower-left block adds solar partials to
  the pseudo-potential Hessian (`bcr4bp_dynamics.py:123-159`). No Jacobi integral.
- **Ephemeris**: physical units (km, km/s, et seconds), time-dependent. EOM
  queries GM per body and ephemeris positions for non-origin bodies; origin body
  yields central term, others third-body perturbation + indirect term
  (`ephemeris_dynamics.py:235-293`, `256-291`). `max_step` defaults 60 s with
  duration-adaptive tightening (`ephemeris_dynamics.py:65-83`). Python-side
  single-step acceleration crosses SPICE per body; the Rust fast path moves query
  into the Rust inner loop — Python passes parameter bags.

## What this separation protects

### One System serving multiple Dynamics and other consumers

System sharing is proven on test and production sides alike. Tests: BCR4BP
comparison experiments rebuild dynamics directly from the converter-held system:
`CR3BP_Dynamics(spice_syn_j2000.cr3bp_system)`
(`test_bcr4bp_model.py:168`). Production: design_orbit feeds one instance four
ways simultaneously (`design_orbit.py:983-1032`); `StabilityAnalysis` and
manifolds each regrow their own `CR3BP_Dynamics` from orbit-borne systems
(`stability.py:95-98`; `manifolds.py:114`). Reverse sharing holds too: the same
`earth_moon_system` fixture backs `earth_moon_dynamics`
(`tests/conftest.py:13-25`) and any test needing only system parameters.

Contrast: Dynamics configs change per task — within one search family, dynamics
get integrator/rtol/atol/max_step overridden (`search_parallel.py:837-842`). Had
configs lived on System, shared-system consumers would stomp each other;
separation gives model params and per-run integration configs separate owners.

### The polymorphism seam and contract tests

The `System` ABC is the sole type-level seam between CR3BP and ephemeris worlds,
three members wide (`system.py:15-51`). Consumer survey: genuinely cross-seam
polymorphic calls collapse onto `gravitational_parameter`; everything else probes
implementation members via getattr/hasattr. Two contract-test files pin the
landscape: `test_system_contract.py` runs identical interface assertions against
both implementations (`test_system_contract.py:53-64`);
`test_dynamics_contract.py` asserts propagate output shapes (n, 6) / (n, 6, 6)
(`test_dynamics_contract.py:28-38`).

### Lifecycle differences

Systems live with data: `Orbit` holds references surviving serialization
round-trips (`orbit.py:50`, `253`, `275`); stability analysis and manifold work
regrow dynamics anytime. Dynamics live with tasks: construct, reconfigure,
propagate, read cache, discard; `_corrected_dro_cached` and `dro_corrector`
fixtures each build independent system+dynamics pairs, sharing nothing
(`tests/algorithm/conftest.py:62-88`). An order-of-magnitude lifecycle gap — the
separation's most tangible runtime expression.

### Genuine overlaps

Separation isn't absolute; three known overlaps, each with reasons:

- **Dual Jacobi entry**. `CR3BP_System.get_jacobi_constant` is definition home
  (`cr3bp_system.py:319-354`); `CR3BP_Dynamics.compute_jacobi_constant` delegates
  one line (`dynamics.py:1001-1010`) for `_handle_jacobi`'s post-propagation
  pointwise calls. Callers wanting Jacobi sans propagation go system-side
  (`design_orbit.py:987`; `axial_initial_guess.py:166`). Tests assert both entries
  agree numerically (`test_cr3bp_model.py:66-69`).
- **A matrix constructed twice**. `CR3BP_System.compute_stability_index` assembles
  a 6×6 linearization internally for libration-point eigenanalysis
  (`cr3bp_system.py:418-423`); `CR3BP_Dynamics.compute_jacobian_A` assembles the
  isomorphic matrix for STM variational equations + continuation reuse
  (`dynamics.py:611-638`). Both share `pseudo_potential_hessian`
  (`potential.py:14-58`) — one Hessian, assembly in respective contexts: one
  belongs to system properties, one to propagation support.
- **Radii + DU cross-layer reads**. Collision detection is Dynamics duty, yet
  radii live on System and nondimensionalizing needs System's `DU`
  (`dynamics.py:398-428`, `447-472`). Radii are body attributes not integration
  config — hence data on system, event construction on dynamics.

## Appendix: file map

- System side: `system.py` (ABC), `cr3bp_system.py` (CR3BP_System +
  LibrationPoint), `ephemeris_system.py` (EphemerisSystem),
  `bcr4bp_system.py` (BCR4BPSystem).
- Dynamics side: `dynamics.py` (Dynamics base + CR3BP_Dynamics +
  `propagate_state_at_orbit_time`), `ephemeris_dynamics.py`
  (EphemerisDynamics), `bcr4bp_dynamics.py` (BCR4BP_Dynamics).
- Shared: `potential.py` (pseudo-potential Hessian; serves CR3BP/BCR4BP Jacobians
  and libration-point stability analysis alike).
- Behavioral portraits: `tests/algorithm/dynamics/` (contracts, events,
  collisions, variational equations).
