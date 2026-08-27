# System and Dynamics: Data Flow Across Two Class Hierarchies / System 与 Dynamics：两棵类层次的数据流

[English](#system-and-dynamics-data-flow-across-two-class-hierarchies) | [简体中文](#中文)

## English

Under `e2m2e/algorithm/dynamics/` live two class hierarchies: the System side
describes *what system this is* (μ, characteristic scales, libration points,
ephemeris, units, coordinate frames); the Dynamics side describes *how to
integrate it* (integrators, tolerances, STMs, events, result caches). The
package docstring summarizes: System (data context) + Dynamics (propagation
orchestration) (`e2m2e/algorithm/dynamics/__init__.py:1`). Following data's
direction of flow, this page explains each family — what they are, who reads
which members, which path data takes in one propagation. All line numbers refer
to current code.

### Starting from one propagation

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

### System: model context

#### The base class promises only three members

The `System` ABC's minimal interface has three items: `frame`, `unit_system`,
`gravitational_parameter(body)` (`system.py:15-51`). Its docstring names what is
deliberately excluded: `mu`, `body_state(body, t)`, `coordinate_system` belong to
specific implementations (`system.py:25-27`). The base also carries non-abstract
`get_body_position` defaulting to NotImplementedError: ephemeris-only capability,
placed there just for a clear error site (`system.py:54-66`).

Thin interfaces have consequences: code wanting polymorphism over both systems
can rely only on those three members; everything else probes live via
`getattr`/`hasattr`. The consumers section shows this pattern occurring for real.

#### CR3BP_System: nondimensional, autonomous, two-stage initialization

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

#### EphemerisSystem: unified SPICE-query entry

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

#### BCR4BPSystem: extends CR3BP_System, adds the Sun

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

### Dynamics: propagation orchestrators

#### Three state kinds on the base class

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

#### Template method: propagate is the skeleton, subclasses fill hooks

`propagate()` is a template method: base defines the algorithm skeleton
(normalize args → validate → dispatch → assemble results); subclasses join via
two hook groups: `_get_eom_func(with_stm)` supplies ODE RHS,
`_get_max_step(t_span)` supplies step ceilings (`dynamics.py:100-125`). Dispatch
has two branches only: `_propagate_with_stm` (42-dim augmented) vs
`_propagate_state_only` (6-dim) (`dynamics.py:223-229`). Both base implementations
run scipy `solve_ivp`; all three subclasses override both branches preferring Rust
fast paths (event handling varies per subclass — below).

#### Three subclasses, two inheritance choices

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

### Consumers: who reads System outside this package

Whether separation bears load depends on how much code consumes System without
constructing Dynamics. Verified case by case:

#### Force-model side: reading coordinate_system, spice, origin, gravitational_parameter

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

#### Transfer & solver side: reading origin and gravitational_parameter

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

#### Coordinate-conversion side: reading mu, scales, spice

`SynodicJ2000System` holds a `CR3BP_System` + spice: pointwise conversion reads
`mu` (barycenter shift) and `characteristic_time` (time dimensioning); batch paths
pass two scalars — `mu` + time unit — to Rust
(`synodic_j2000.py:28-41`, `52`, `96-97`). `rho_bridge` functions take
`EphemerisSystem`: read `system.spice` to build SynodicAxes, read
`system.get_body_state("MOON", et)` for lunar states
(`rho_bridge.py:47-63`, `110`).

#### Station keeping & prediction: constructing and configuring EphemerisSystem

`control_orbit` builds a ten-body EphemerisSystem, attaches coordinate_system,
hands off to Monte Carlo (`controller.py:228-247`); workers rebuild the same
system from parameters (`monte_carlo.py:644-650`), plus a scales-only
CR3BP_System for synodic conversion (`monte_carlo.py:66-72`, `707-716`).
`propagate_orbit` likewise constructs EphemerisSystem, attaches coordinate_system,
then `ForceModel.from_config(force_config, system)` (`propagation.py:118-124`).
Both consume System while propagation happens inside ForceModel — bypassing
Dynamics.

#### Remaining scattered points

`PoincareSection.periapsis(center, system)` reads `mu` + body names for section
centers (`sections.py:202-228`). Inside `design_orbit`, one CR3BP system instance
feeds four consumers simultaneously: constructing `CR3BP_Dynamics`, calling
`get_jacobi_constant` directly, reading `characteristic_time` for conversions,
constructing `SynodicJ2000System` for coordinate conversion
(`design_orbit.py:983-987`, `1030-1032`).

#### Polymorphic functions annotated with System

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

#### Duck typing: nominal System, structural reality

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

### propagate internals: how data moves in one call

#### Entry orchestration

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

#### scipy vs Rust backend paths

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

**Events' third branch**: with events and `backend="rust"`, CR3BP/BCR4BP take the
generic Rust integrator `solve_ivp_events`, ODE RHS still passing as Python
callback, event functions translated into `(g, terminal, direction)` triples
(`dynamics.py:700-756`, `721-723`; `bcr4bp_dynamics.py:225-271`). Ephemeris
doesn't support events: events non-None → immediate NotImplementedError
(`ephemeris_dynamics.py:85-113`, `162-189`).

#### When return-dict keys appear

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

#### Events and collision detection

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

#### Differences among three propagation chains

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

### What this separation protects

#### One System serving multiple Dynamics and other consumers

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

#### The polymorphism seam and contract tests

The `System` ABC is the sole type-level seam between CR3BP and ephemeris worlds,
three members wide (`system.py:15-51`). Consumer survey: genuinely cross-seam
polymorphic calls collapse onto `gravitational_parameter`; everything else probes
implementation members via getattr/hasattr. Two contract-test files pin the
landscape: `test_system_contract.py` runs identical interface assertions against
both implementations (`test_system_contract.py:53-64`);
`test_dynamics_contract.py` asserts propagate output shapes (n, 6) / (n, 6, 6)
(`test_dynamics_contract.py:28-38`).

#### Lifecycle differences

Systems live with data: `Orbit` holds references surviving serialization
round-trips (`orbit.py:50`, `253`, `275`); stability analysis and manifold work
regrow dynamics anytime. Dynamics live with tasks: construct, reconfigure,
propagate, read cache, discard; `_corrected_dro_cached` and `dro_corrector`
fixtures each build independent system+dynamics pairs, sharing nothing
(`tests/algorithm/conftest.py:62-88`). An order-of-magnitude lifecycle gap — the
separation's most tangible runtime expression.

#### Genuine overlaps

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

### Appendix: file map

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

## 中文

`e2m2e/algorithm/dynamics/` 下有两棵类层次：System 一侧描述这是什么系统
（μ、特征尺度、平动点、星历、单位、坐标系），Dynamics 一侧描述怎么把它积分
出来（积分器、容差、STM、事件、结果缓存）。包 docstring 概括为 System（数据
上下文）+ Dynamics（传播编排）（e2m2e/algorithm/dynamics/\_\_init\_\_.py:1）。
本文按数据流动的方向，把这两个家族各自是什么、谁读它们的什么成员、一次传播里
数据走哪条路，逐一讲清楚。文中所有行号以当前代码为准。

### 从一次传播说起

先看最常见的一个场景：传播一条地月 CR3BP 轨迹。测试套件里它的标准形态是
（tests/conftest.py:13-25）：

```python
system = CR3BP_System(mu=Datum.DE421.mu, primary="Earth", secondary="Moon")._with_default_scales()
dynamics = CR3BP_Dynamics(system=system)
result = dynamics.propagate(state0, (0.0, 6.3))
```

数据在这条链上走四段路。

**第一段，构造 System。** `CR3BP_System.__init__` 只收质量参数 μ、两个天体名和
可选的天体半径，特征长度、特征时间、五个平动点全部置为 None
（e2m2e/algorithm/dynamics/cr3bp_system.py:71-128）。此时的系统还不能参与计算：
`DU`/`TU`/`VU` 属性在尺度未设时抛系统未初始化
（cr3bp_system.py:194-212）。`_with_default_scales()` 按天体对补上特征尺度
（cr3bp_system.py:130-160），平动点则推迟到首次需要时才解算：
`get_libration_point` 发现没算过就先调 `compute_libration_points`
（cr3bp_system.py:293-317）。System 的构造因而是两段式的：先定哪个系统，
再定用什么尺度量化它。

**第二段，构造 Dynamics。** `CR3BP_Dynamics(system)` 做的事很轻：存下 system
引用，填一套默认积分器配置（RK45、rtol/atol 1e-12、max_step 0.01），把结果缓存
置空（e2m2e/algorithm/dynamics/dynamics.py:78-97、544-553）。它不复制系统的任何
参数。μ 在每次算加速度时经 `self.system.mu` 现取（dynamics.py:593）。

**第三段，propagate。** 无事件时 CR3BP 走 Rust 快速路径，Python 侧只把 `mu`、
时间区间、初值和积分配置这组标量传过 FFI（dynamics.py:851-859）；轨迹在 Rust
侧算完，以 `{"time", "states"}` 字典回来，同时写进 `self.last_trajectory` 缓存
（dynamics.py:861-870）。这一段的细节见后文propagate 内部一节。

**第四段，结果进数据容器。** 设计链路把返回字典装进 `Orbit`，并把 system 引用
一并塞进去：`Orbit(states=result["states"], times=result["time"],
system=dynamics.system)`（e2m2e/algorithm/design/design_orbit.py:550）。从此这组
浮点数带着自己的单位与坐标系解释者旅行；之后谁想再做计算，可以从
`orbit.system` 重新长出一个 dynamics，`StabilityAnalysis` 正是这样按需重建的
（e2m2e/algorithm/stability.py:95-98）。

四段路合起来就是两个家族的分工：System 是长期持有的模型上下文，Dynamics 是
围着它转的、带配置与缓存的传播者。

### System：模型上下文

### 基类只承诺三个成员

`System` ABC 定义的最小接口只有三项：`frame`（坐标框架）、`unit_system`
（单位系统）、`gravitational_parameter(body)`（引力参数）
（e2m2e/algorithm/dynamics/system.py:15-51）。docstring 点名了什么刻意不进基类：
`mu`、`body_state(body, t)`、`coordinate_system` 属于特定实现的概念
（system.py:25-27）。基类另有一个非抽象的 `get_body_position`，默认抛
NotImplementedError：星历专属能力，放在基类只是给一个明确的报错位置
（system.py:54-66）。

接口薄是有后果的：想对两种系统多态的代码，只能依赖这三个成员，其余都得用
`getattr`/`hasattr` 现探。下文消费面一节会看到这是真实发生的模式。

### CR3BP_System：无量纲、自治、两段式初始化

构造注入五样东西：`mu`、主/次天体名、可选的主/次天体半径
（cr3bp_system.py:71-80）。构造时做两道校验：半径若给必须为正
（cr3bp_system.py:93-98），μ 必须落在 (0, 0.5)（cr3bp_system.py:102-106）。
构造完成后还有两步初始化可选：`set_characteristic_scales(distance, period)` 由
距离与周期推出特征长度/时间/速度并置 `is_initialized`（cr3bp_system.py:214-235）；
`compute_libration_points()` 用 `fsolve` 解三个共线点、解析给出两个三角点
（cr3bp_system.py:237-291，其中 264-266 数值解、268-274 解析解）。

对基类三成员，它的回答是：会合（旋转）坐标系、无量纲单位、约定总质量为 1 时
primary 的 GM 是 `1 - mu`、secondary 的是 `mu`（cr3bp_system.py:163-191）。

初始化之后，这个对象向外提供四类数据：

- 特征尺度：`DU`（km）、`TU`（天）、`VU`（m/s）三个属性，外加
  `characteristic_length/time/velocity` 原始字段（cr3bp_system.py:108-110、194-212）。
- 平动点：`L1`~`L5` 与 `L_points` 字典（cr3bp_system.py:112-117）。
- Jacobi 常数：`get_jacobi_constant(state)`，Parker 约定
  （cr3bp_system.py:319-354）。
- 单位换算与稳定性：`dimensionless_to_physical` / `physical_to_dimensionless`
  （cr3bp_system.py:356-396）、`compute_stability_index`（cr3bp_system.py:398-442）。

半径字段（`primary_radius_km`/`secondary_radius_km`）本身不参与动力学，是碰撞
检测的数据来源，流向见事件与碰撞小节。

### EphemerisSystem：SPICE 查询的统一入口

构造注入五样东西：天体名列表、已完成内核加载的 `SPICEManager`、参考原点
（默认 "EARTH"）、坐标框架（默认 J2000）、可选的 `CoordinateSystem`
（e2m2e/algorithm/dynamics/ephemeris_system.py:35-55）。对基类三成员，它的回答
是：构造时给定的框架、物理单位、GM 直通 `spice.get_gm`
（ephemeris_system.py:59-103）。

与 CR3BP 的两段式初始化不同，这里的第二步是给 `coordinate_system` 赋值，且
发生在构造之后、经 property setter 完成（ephemeris_system.py:69-75）。编排层
的标准写法是先构造再补：`system = EphemerisSystem(...)` 接着
`system.coordinate_system = CoordinateSystem(...)`
（e2m2e/algorithm/propagation.py:118-123）。力模型传播依赖这个字段，不设会被
ForceModel 拒绝（e2m2e/algorithm/forces/force_model.py:57-58）。

星历数据经四个查询方法流出：`gravitational_parameter`/`get_gm`（单体 GM）、
`get_gm_values`（按 bodies 顺序的 GM 数组）、`get_body_position`（位置）、
`get_body_state`（六维状态）（ephemeris_system.py:94-150）。另有一个
`update_coordinate_systems(t, state)` 用于推进动态坐标系
（ephemeris_system.py:77-92）；ForceModel 传播下沉 Rust 后，逐步更新由 Rust
内部完成，e2m2e/ 内已无调用点，仅测试保留
（tests/numerical/forces/container/test_force_model_dynamic_axes.py:143-186）。

### BCR4BPSystem：继承 CR3BP_System，叠加太阳

`BCR4BPSystem` 继承 `CR3BP_System`
（e2m2e/algorithm/dynamics/bcr4bp_system.py:23），构造时多收四个太阳参数：
`sun_mass`、`sun_distance`、`sun_angular_rate`、`sun_phase0`；前两个缺省时按
DE440 常量和日地平均距离推导（bcr4bp_system.py:50-106，缺省推导在 89-92）。
`sun_angular_rate` 特殊：它依赖特征时间，直接构造不给时暂存 None，由
`set_characteristic_scales` 覆写方法按儒略年公转推导（bcr4bp_system.py:148-161）；
因此标准入口是类方法 `BCR4BPSystem.earth_moon()`，一步完成构造与尺度设置
（bcr4bp_system.py:109-132）。

太阳位置不查星历，是时间 t 的解析函数 `sun_position(t)`，即会合系里的共面圆周
（bcr4bp_system.py:163-183）。`gravitational_parameter` 在 "primary"/"secondary"
之外多接受 "sun"（bcr4bp_system.py:185-193）。注意 docstring 的提醒：BCR4BP 无
Jacobi 积分，`compute_libration_points` 给出的是对应 CR3BP 的平动点，仅作参考
位置（bcr4bp_system.py:41-42）。

### Dynamics：传播编排者

### 基类持有的三类状态

`Dynamics.__init__` 只收一个 `system`（dynamics.py:78-84），随后在实例上放三类
东西：

1. **system 引用**：`self.system`，传播全程只读不写（dynamics.py:84）。
2. **积分器配置**：`integrator`、`rtol`、`atol`、`max_step`，默认值挂在类常量上
   （dynamics.py:86-91）。这些是公开字段，调用方构造后直接改：并行搜索按任务
   改容差与步长（e2m2e/algorithm/transfer/search_parallel.py:837-842），多重打靶
   构造后立刻覆写三项（e2m2e/algorithm/solver/multiple_shooting.py:97-101），测试
   fixture 为提速放宽星历传播的容差（tests/conftest.py:92-100）。
3. **结果缓存**：`last_trajectory`、`last_stm`，每次传播结束时覆写
   （dynamics.py:93-94；写入点见后文）。`CR3BP_Dynamics` 再加
   `jacobi_history`/`jacobi_error` 两个 Jacobi 监测缓存（dynamics.py:552-553）。

也就是说，Dynamics 是有态工人：配置与最近一次的结果都留在实例上，供调用方
事后取（如 `compute_state_transition_matrix` 内部就是一次传播后取末态 STM，
dynamics.py:986-999）。

### 模板方法：propagate 是骨架，子类填钩子

`propagate()` 是模板方法：基类定算法骨架（参数规范化 → 校验 → 分发 → 结果
装配），子类经两组钩子参与：`_get_eom_func(with_stm)` 给出 ODE 右端，
`_get_max_step(t_span)` 给步长上限（dynamics.py:100-125）。分发只有两支：
`_propagate_with_stm`（42 维增广状态）与 `_propagate_state_only`（6 维）
（dynamics.py:223-229）。基类的两支实现都走 scipy `solve_ivp`；三个子类各把
这两支覆写为 Rust 快速路径优先（事件场景的处理因子类而异，见后文）。

### 三个子类，两种继承选择

`CR3BP_Dynamics`（dynamics.py:522）与 `EphemerisDynamics`
（e2m2e/algorithm/dynamics/ephemeris_dynamics.py:47）分别实现自治无量纲 CR3BP
方程与含时物理单位 N 体方程。`BCR4BP_Dynamics` 直接继承 `Dynamics`
（e2m2e/algorithm/dynamics/bcr4bp_dynamics.py:42），不继承 `CR3BP_Dynamics`，
尽管它的方程就是 CR3BP 加一项太阳摄动。代码里能看到这个选择的理由：

- 雅可比签名不同：CR3BP 自治，`compute_jacobian_A(state)`（dynamics.py:611）；
  BCR4BP 含时，`compute_jacobian_A(t, state)`（bcr4bp_dynamics.py:123）。
- Jacobi 语义反转：CR3BP_Dynamics 构造即带 Jacobi 监测缓存
  （dynamics.py:552-553），而 BCR4BP 是时间周期系统、无 Jacobi 积分，必须把
  `compute_jacobi_constant` 与 `_handle_jacobi` 都实现为抛 NotImplementedError
  （bcr4bp_dynamics.py:457-463），继承来的能力要逐个点掉。
- STM 入口多一个参数：含时系统的 Φ 依赖起止时刻，
  `compute_state_transition_matrix` 多一个 `t0`（bcr4bp_dynamics.py:438-455）。
- Rust 入口不同：BCR4BP 的传播函数要多带四个太阳参数
  （bcr4bp_dynamics.py:293-303）。

继承 CR3BP_Dynamics 意味着几乎每个公开方法都要覆写、还要压掉 Jacobi 机制。
仓库里有过同型教训：ForceModel 一度形式上继承 Dynamics 只为复用几个数据属性，
实则全部重写 `propagate` 并对 STM/Jacobi 抛错，被认定为 LSP 违反（假继承）而
改为独立类（e2m2e/algorithm/forces/force_model.py:30-38）。

### 消费面：System 出了这个包，被谁读

分离是否承重，取决于 System 被多少不构造 Dynamics 的代码消费。逐个核实如下。

### 力模型侧：读 coordinate_system、spice、origin、gravitational_parameter

`ForceModel` 持有 system（类型标注就是 `Any`），构造时强制
`system.coordinate_system` 已设置（force_model.py:45-58）；传播时把每个力模型
序列化为 Rust 元组（`force.to_rust_spec(self.system)`），并读 `system.origin`
作为 observer 传入（force_model.py:281-293）；`system.spice` 是否存在被用作
资源缺失还是能力缺失的分流依据（force_model.py:226-233）。它从不构造
Dynamics，也明确不继承 Dynamics（force_model.py:30-38）。

`PhysicalModel._resolve_mu(system)` 是力模型侧对 System 最直接的消费：显式 μ
缺失时调 `system.gravitational_parameter(self._body)`
（e2m2e/algorithm/forces/physical_model.py:29-41）。`ConicalShadowModel.
flux_factor(t, state, system)` 经 `require_inertial_frame` 从 system 取出
coordinate_system、spice、原点三件事，再查太阳与遮挡体位置算光照份额
（e2m2e/algorithm/forces/shadow.py:166-197；physical_model.py:69-84）。

### 转移与求解侧：读 origin 与 gravitational_parameter

低推力三件套都把 system 当参数包用。`qlaw_guess(system, ...)` 只从中解析中心体
μ：先查力模型里的 PointMassGravity，查不到再
`system.gravitational_parameter(origin)`，`origin` 本身用 `getattr` 兜底成
"EARTH"（e2m2e/algorithm/transfer/qlaw.py:210、265-282）。
`LowThrustShooting` 与 `LowThrustCollocation` 构造时各做两件事：把力模型
`to_rust_spec(system)` 预序列化，把 `origin` 存为 observer
（e2m2e/algorithm/transfer/lowthrust_shooting.py:125-162；
e2m2e/algorithm/transfer/lowthrust_collocation.py:59-88）。

`NormalFormContext` 从 system 只取 μ，且是 `getattr(system, "mu", None)` 探测：
CR3BP_System 有就取，没有就回退固化常量
（e2m2e/algorithm/normal_form/context.py:64、114、196-201）。

### 坐标转换侧：读 mu、特征尺度、spice

`SynodicJ2000System` 持有 `CR3BP_System` 与 spice：逐点转换读 `mu`（质心平移）
与 `characteristic_time`（时间量纲化），批量路径把 `mu` 与时间单位两个标量传给
Rust（e2m2e/algorithm/coordinate/synodic_j2000.py:28-41、52、96-97）。
`rho_bridge` 的一组函数以 `EphemerisSystem` 为参：读 `system.spice` 构造
SynodicAxes、读 `system.get_body_state("MOON", et)` 取月球状态
（e2m2e/algorithm/coordinate/rho_bridge.py:47-63、110）。

### 轨道保持与预报：构造并配置 EphemerisSystem

`control_orbit` 构造十体 EphemerisSystem、赋 coordinate_system，然后交给
蒙特卡洛流程（e2m2e/algorithm/station_keeping/controller.py:228-247）；工作进程里
按参数重建同一个系统（e2m2e/algorithm/station_keeping/monte_carlo.py:644-650），
另建一个仅带特征尺度的 CR3BP_System 供会合系转换用（monte_carlo.py:66-72、
707-716）。`propagate_orbit` 同样先构造 EphemerisSystem、补 coordinate_system，
再 `ForceModel.from_config(force_config, system)`（propagation.py:118-124）。
这两个模块消费 System，但传播由 ForceModel 完成，不经过 Dynamics。

### 其余散点

`PoincareSection.periapsis(center, system)` 读 `mu` 与主/次天体名来定截面中心
（e2m2e/algorithm/manifold/sections.py:202-228）。`design_orbit` 里同一个 CR3BP
system 实例同时喂给四路消费：构造 `CR3BP_Dynamics`、直接调
`get_jacobi_constant`、读 `characteristic_time` 做时间换算、构造
`SynodicJ2000System` 做坐标转换（design_orbit.py:983-987、1030-1032）。

### 按 System 抽象签名的多态函数

签名标注 `system: System` 的入口有：`PhysicalModel._resolve_mu` /
`to_rust_spec`（physical_model.py:29、45）、`ConicalShadowModel.flux_factor`
（shadow.py:170）、`qlaw_guess` 与 `make_shooter_for_qlaw`（qlaw.py:210、290）、
`LowThrustShooting.__init__` 与 `LowThrustCollocation.__init__`
（lowthrust_shooting.py:125；lowthrust_collocation.py:59）、
`NormalFormContext.__init__`（normal_form/context.py:64）。

把这些入口实际读到的成员摊开看，真正对 CR3BP 与星历两种系统都成立的多态只
剩基类那一项：`gravitational_parameter`，它在 CR3BP 侧接受 "primary"/"secondary"
返回无量纲值，星历侧接受 SPICE 天体名返回 km³/s²（system.py:42-51 的 docstring
把这个双语义写明）。其余入口虽标着 `System`，实际读的都是实现侧成员
（`origin`、`coordinate_system`、`spice`、`mu`），靠 `getattr`/`hasattr` 兜底。
契约测试把双系统同一接口这件事固定下来：同一组断言跑在 CR3BP_System 与
EphemerisSystem 两个实现上（tests/algorithm/dynamics/test_system_contract.py:53-64）。

### 鸭子类型：名义 System，实际结构

类型标注之下，真实契约比注解更薄，三处实证：

- 数据层的 `Orbit`/`OrbitFamily` 把 system 存为 `Any`，docstring 明说数据层
  不依赖算法层，只用 `hasattr(system, "get_jacobi_constant")` 判断能力
  （e2m2e/data/types/orbit.py:8-12、50、418-429）。
- 低推力测试直接传 `SimpleNamespace(origin="EARTH")` 当 system 用
  （tests/algorithm/transfer/test_lowthrust_collocation.py:23；
  tests/algorithm/transfer/test_qlaw_failure.py:37）。
- `RelativeDynamics.linear_model` 用 try/except 在
  `compute_jacobian_A(t, state)` 与 `compute_jacobian_A(state)` 两种签名间适配
  （e2m2e/algorithm/proximity/relative_dynamics.py:145-150）。

### propagate 内部：一次调用里数据怎么动

### 入口编排

`Dynamics.propagate` 本体（dynamics.py:144-233）按固定次序处理输入：

1. `initial_state` 转 ndarray，`max_step` 经钩子取得（dynamics.py:199-200）。
2. 事件规范化：单个 callable 包成列表；空列表等价于无事件，直接置 None
   （dynamics.py:201-205）。
3. 若开碰撞检测，先把碰撞事件造出来并入事件列表，同时检查初始状态是否已在
   某天体半径内（dynamics.py:207-209）。
4. `backend` 校验：只许 "scipy"/"rust"，有事件时必须显式给，否则报错
   （dynamics.py:211-214）。
5. 初始即在半径内的，短路返回单点轨迹加即时碰撞标记：scipy 事件不会对
   g<0 的起点触发，必须显式处理（dynamics.py:217-221、474-493）。
6. 按 `with_stm` 分发到两支（dynamics.py:223-229）。
7. 若开碰撞检测，从结果的事件段提取碰撞信息（dynamics.py:231-232、495-510）。

### scipy 与 Rust 两条后端路径

**scipy 路径**（基类实现）：`_propagate_state_only` 拿 `_get_eom_func(False)`
的 ODE 右端调 `solve_ivp`，`result.y` 转置成 (n, 6)；失败或空结果抛
`PropagationFailure`，不允许拿空轨迹伪装成功（dynamics.py:291-325）。
`_propagate_with_stm` 先把 6×6 单位阵展平拼成 42 维增广状态再积分，回来按
前 6 维/后 36 维拆开（dynamics.py:254-271）。两条 scipy 路径都在返回前写
`last_trajectory`（含 STM 时连 `last_stm`）（dynamics.py:273-274、327）。

**Rust 路径**（子类覆写）：无事件时三个子类都要求 Rust 扩展可用，缺失即抛错、
不降级 scipy。传过 FFI 的东西逐类不同：

- CR3BP：只有 `mu` 一个系统参数，加时间区间、t_eval、初值、容差、步长
  （dynamics.py:851-859；STM 版 dynamics.py:782-790）。
- BCR4BP：`mu` 加太阳四参数 `mu_sun`/`sun_distance`/`sun_angular_rate`/
  `sun_phase0`（bcr4bp_dynamics.py:360-370；STM 版 293-303）。
- 星历：`bodies`、`origin`、`gm_values` 三个从 system 现取的序列
  （ephemeris_dynamics.py:124-126、133-142）。天体位置不经过 Python 对象传递：
  Rust 侧在积分内环直接查进程内 SPICE（先查星历缓存，未命中回退 cspice）
  （crates/e2m2e-spice/src/spk_accel.rs:46-56）。在这条链上，EphemerisSystem
  的角色收敛为一个参数包。

Rust 路径回来后都做长度防御校验（返回点数不等于请求点数即抛错），再写
`last_trajectory`（dynamics.py:792-804、861-870）。

**事件时的第三支**：有事件且 `backend="rust"` 时，CR3BP/BCR4BP 走通用 Rust
积分器 `solve_ivp_events`，ODE 右端仍以 Python 回调形式传入，事件函数被
折算成 `(g, terminal, direction)` 三元组（dynamics.py:700-756、721-723；
bcr4bp_dynamics.py:225-271）。EphemerisDynamics 不支持事件：events 非 None
直接 NotImplementedError（ephemeris_dynamics.py:85-113、162-189）。

### 返回字典的键在何时出现

`time`/`states` 恒有。其余键的出现条件：

- `stm`：`with_stm=True` 时（dynamics.py:276-280）。
- `status`/`cause`：只在纯状态路径出现：基类 scipy 版与 CR3BP 的两个 Rust
  纯状态分支（dynamics.py:329-334、872-877、926-931）。STM 路径、
  EphemerisDynamics 与 BCR4BP_Dynamics 的 Rust 分支都不带这两个键。
- `jacobi`/`jacobi_error`：`with_jacobi=True` 且为 CR3BP 时。基类
  `_handle_jacobi` 是 no-op（dynamics.py:345-358），CR3BP_Dynamics 覆写为逐点
  计算并写缓存（dynamics.py:1012-1022），BCR4BP 覆写为抛错
  （bcr4bp_dynamics.py:461-463）。
- `t_events`/`y_events`：传了 events 时，逐事件的触发时刻与状态
  （dynamics.py:282-284、336-338）。
- `collision`：`collision_detection=True` 时，未碰撞为 None，否则是
  `{"body", "t", "state"}`（dynamics.py:231-232、507-509）。

### 事件与碰撞检测

事件函数是 scipy 语义的 `g(t, state) -> float`，可挂 `terminal`/`direction`
属性；`PoincareSection.event(...)` 是主要的构造者
（tests/algorithm/dynamics/test_events.py:21-26）。`with_stm=True` 时事件函数
收到的是 42 维增广状态（test_events.py:86-97）。

碰撞检测把撞天体翻译成终端事件：`_collision_specs` 从 system 读
`primary_radius_km`/`secondary_radius_km`（都没注入则报错）、读 `mu` 定两天体
在会合系的固定位置 [-μ,0,0] 与 [1-μ,0,0]（dynamics.py:398-428）；
`_setup_collision_detection` 再读 `DU` 把半径从 km 折成无量纲，构造
`g = |r - center| - R` 的 terminal 事件并追加到用户事件之后
（dynamics.py:447-472）。碰撞事件在事件列表末尾这一顺序被
`_extract_collision` 反向利用：按列表尾部 n 个索引回每个天体的触发记录
（dynamics.py:501-509）。这就是碰撞数据流全程读 system 的三个字段：半径、mu、
DU。

### 三条传播链的差异

- **CR3BP**：无量纲、自治。EOM 不显含 t；时间区间、t_eval 都是无量纲量；
  系统侧只贡献 `mu`。
- **BCR4BP**：在 CR3BP 右端上叠加太阳直接项与间接项，太阳位置由
  `system.sun_position(t)` 解析给出，EOM 显式含时
  （bcr4bp_dynamics.py:72-121）。雅可比左下块在伪势能 Hessian 上再加太阳项
  偏导（bcr4bp_dynamics.py:123-159）。无 Jacobi 积分。
- **星历**：物理单位（km、km/s、et 秒）、含时。EOM 对每个天体查 GM、对非原点
  天体查星历位置，原点天体出中心项、其余出第三体摄动加间接项
  （ephemeris_dynamics.py:235-293、256-291）。`max_step` 默认 60 秒且按传播
  时长自适应收紧（ephemeris_dynamics.py:65-83）。Python 侧单步算加速度要逐
  天体过 SPICE；Rust 快速路径把这个查询挪进 Rust 内环，Python 只递参数包。

### 这份分离保护了什么

### 一个 System 服务多个 Dynamics 与别的消费者

System 实例被共享的实证在测试与生产两侧都有。测试侧，BCR4BP 对照实验把坐标
转换器持有的 system 直接拿来再建一个动力学：
`CR3BP_Dynamics(spice_syn_j2000.cr3bp_system)`
（tests/algorithm/dynamics/test_bcr4bp_model.py:168）。生产侧，design_orbit 里
同一个 system 实例同时喂动力学、Jacobi 计算、时间换算、坐标转换四路
（design_orbit.py:983-1032）；`StabilityAnalysis` 与不变流形各自从 orbit 上挂的
system 重建自己的 `CR3BP_Dynamics`（stability.py:95-98；
e2m2e/algorithm/manifold/manifolds.py:114）。反向的共享同样成立：同一
`earth_moon_system` fixture 上既可以挂 `earth_moon_dynamics`
（tests/conftest.py:13-25），也可以被任何只要系统参数的测试直接消费。

与之对照，Dynamics 的配置是按任务改的：同一个搜索任务族内，dynamics 构造后
被覆写 integrator/rtol/atol/max_step（search_parallel.py:837-842）。若配置留在
System 上，共享 system 的消费者之间会互相踩配置；分离让模型参数与本次积分
配置各有其主。

### 多态缝与契约测试

`System` ABC 是 CR3BP 与星历两个世界之间唯一的类型级接缝，缝宽三个成员
（system.py:15-51）。消费面普查显示：真正跨缝多态的调用几乎都收敛到
`gravitational_parameter` 一项，其余访问都是 getattr/hasattr 探测实现侧成员。
两个契约测试文件把这个格局固定下来：`test_system_contract.py` 对两个实现跑
同一组接口断言（tests/algorithm/dynamics/test_system_contract.py:53-64），
`test_dynamics_contract.py` 对 propagate 的输出形状断言 (n, 6) 与 (n, 6, 6)
（tests/algorithm/dynamics/test_dynamics_contract.py:28-38）。

### 生命周期的差异

System 的寿命跟着数据走：`Orbit` 持有 system 引用，序列化再加载后引用还在
（orbit.py:50、253、275），稳定性分析、流形计算可以随时从它再长出 dynamics。
Dynamics 的寿命跟着任务走：构造、改配置、传播、读缓存，然后被丢弃；
`_corrected_dro_cached` 与 `dro_corrector` 两个 fixture 各自建独立的
system+dynamics 对，互不共享（tests/algorithm/conftest.py:62-88）。两类对象的
生命周期差一个量级，这是分离在运行时最直观的体现。

### 两侧的真实重叠

分离并不彻底，有三处已知的重叠，各有来由：

- **Jacobi 常数双入口**。`CR3BP_System.get_jacobi_constant` 是定义所在
  （cr3bp_system.py:319-354）；`CR3BP_Dynamics.compute_jacobi_constant` 一行委托
  （dynamics.py:1001-1010），供 `_handle_jacobi` 在传播后逐点调用。只想要 Jacobi
  值、不想碰传播的调用方走 system 侧（design_orbit.py:987；
  e2m2e/algorithm/family/axial_initial_guess.py:166）。测试断言两入口数值一致
  （tests/algorithm/dynamics/test_cr3bp_model.py:66-69）。
- **A 矩阵两处构造**。`CR3BP_System.compute_stability_index` 内部拼一份 6×6
  线性化矩阵用于平动点特征值分析（cr3bp_system.py:418-423）；
  `CR3BP_Dynamics.compute_jacobian_A` 拼同构矩阵供 STM 变分方程与延拓模块
  复用（dynamics.py:611-638）。两者共用 `pseudo_potential_hessian`
  （e2m2e/algorithm/dynamics/potential.py:14-58），Hessian 只有一份，拼装各
  归各的语境：一个在系统性质里，一个在传播配套里。
- **碰撞半径与 DU 的跨层读取**。碰撞检测是 Dynamics 的职责，但半径存在
  System 上、折无量纲要用 System 的 `DU`（dynamics.py:398-428、447-472）。
  半径是天体属性而非积分配置，故数据留在 system，事件构造留在 dynamics。

### 附：文件地图

- System 侧：`system.py`（ABC）、`cr3bp_system.py`（CR3BP_System +
  LibrationPoint）、`ephemeris_system.py`（EphemerisSystem）、
  `bcr4bp_system.py`（BCR4BPSystem）。
- Dynamics 侧：`dynamics.py`（Dynamics 基类 + CR3BP_Dynamics +
  `propagate_state_at_orbit_time`）、`ephemeris_dynamics.py`
  （EphemerisDynamics）、`bcr4bp_dynamics.py`（BCR4BP_Dynamics）。
- 共享：`potential.py`（伪势能 Hessian，供 CR3BP/BCR4BP 的雅可比与平动点
  稳定性分析共用）。
- 行为写照：`tests/algorithm/dynamics/`（契约、事件、碰撞、变分方程）。
