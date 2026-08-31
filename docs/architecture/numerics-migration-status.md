# Algorithm-Layer Numerical Kernel Migration Status Ledger

ADR 0011's decision: some computation runs in Python while migrating stepwise to
the Rust compute core. This page registers migration status per `e2m2e/algorithm/`
submodule for direct audit citation, preventing transitional states being misread
as misplaced layers (misjudgment precedent & ruling: ADR 0026).

Each entry answers three questions: **where is the numeric kernel** (Rust crate /
Python), **migration status** (sunk / migrating / intentionally in Python),
**why** (intentional-Python entries must state reasons; migrating entries must cite
work issues).

Status vocabulary is fixed at three greppable terms:

- `SUNK`: numeric kernel is Rust; Python side is thin wrapper or orchestration
- `MIGRATING`: a concrete sinking work item exists, see the issue
- `INTENTIONALLY IN PYTHON`: kept in Python on decision grounds

## Quick reference

### SUNK

| Module | Numeric kernel | Work issue |
|---|---|---|
| `algorithm/dynamics` (non-event propagation) | Rust (`e2m2e-integrators`) | — |
| `algorithm/dynamics` (rust event path EOM: CR3BP/BCR4BP, `backend="rust"` events) | Rust (`e2m2e-forces` EOM/STM kernels dispatched inside `solve_ivp_events`) | #594 |
| `algorithm/forces` (numerics) | Rust (`e2m2e-forces`) | — |
| `algorithm/transfer/lambert.py` | Rust (`e2m2e-propagation`) | — |
| `algorithm/transfer/search_parallel.py` (grid search) | Rust (`e2m2e-integrators`) | — |
| `algorithm/transfer/wsb.py` (WSB grid candidate evaluation) | Rust (`e2m2e-forces` + `e2m2e-integrators`) | #447 |
| `algorithm/transfer/low_energy.py` (manifold section-state pairing) | Rust (`e2m2e-forces` + `e2m2e-integrators`) | #447 |
| `algorithm/solver/differential_correction.py` (CR3BP numeric kernel) | Rust (`e2m2e-integrators`) | #441 |
| `algorithm/solver` (ephemeris correction path) | Rust (`e2m2e-integrators`) | — |
| `algorithm/solver/continuation.py` (PAL numeric kernel) | Rust (`e2m2e-forces`) | #443 |
| `algorithm/family` (#428 orbit-family numeric kernel) | Rust (`e2m2e-forces` + `e2m2e-integrators`) | #428 |
| `algorithm/transfer/qlaw.py` (feedback integration & Q function; Python only assembles initial guesses) | Rust (`e2m2e-forces` + `e2m2e-integrators`) | #442 |
| `algorithm/transfer/nsga2.py` (evolutionary operators; Python keeps evaluation & orchestration) | Rust (`e2m2e-integrators`) | #444 |
| `algorithm/transfer/lowthrust_shooting.py`, `lowthrust_collocation.py` (direct-method numerical evaluation) | Rust (`e2m2e-integrators`; SLSQP orchestration stays Python) | #445 |
| `algorithm/transfer/porkchop.py` (grid evaluation: terminal propagation + Lambert + ΔV) | Rust (`e2m2e-forces` + `e2m2e-integrators`; Python only problem construction & archive/query) | #446 |
| `algorithm/design` (shooting/propagation paths) | Rust (`e2m2e-integrators`) | — |
| `algorithm/coordinate/synodic_j2000.py` (batch conversion) | Rust (`e2m2e-integrators`) | — |
| `algorithm/proximity/relative_dynamics.py` (propagation integration) | Rust (`e2m2e-integrators`, `solve_ivp_events`); ephemeris-target A(t) provided by the Python Jacobian kernel | — |
| `algorithm/station_keeping/monte_carlo.py` (propagation) | Rust (`e2m2e-integrators`) | — |
| `algorithm/normal_form` (integration paths) | Rust (`e2m2e-integrators`) | #336/#340 |
| `algorithm/normal_form` (CR3BP Hamiltonian numerical construction) | Rust (`e2m2e-integrators`) | — |
| `algorithm/normal_form` (H→QF scalar polynomial projection) | Rust (`e2m2e-integrators`) | — |
| `algorithm/normal_form` (numeric polynomial kernel) | Rust (`e2m2e-integrators`) | #464 |
| `algorithm/normal_form` (complex integration + QF↔CM Lie flows) | Rust (`e2m2e-integrators`, 12-real-dim split) | #465 |
| `algorithm/normal_form` (center manifold reduction) | Rust (`e2m2e-integrators`) | #466 |
| `algorithm/manifold/manifolds.py` (seed generation & batch propagation) | Rust (`e2m2e-forces` + `e2m2e-integrators`) | #448 |

### MIGRATING

| Module | Numeric kernel | Work issue |
|---|---|---|
| `algorithm/solver/MultipleShooting` class (transfer/hohmann) | Python | pending separate migration |

### INTENTIONALLY IN PYTHON

| Module | Numeric kernel | Work issue |
|---|---|---|
| `algorithm/transfer/nlp_*`, `transfer_optimization.py` (NLP optimization & orchestration) | Python | — |
| `algorithm/family/*_initial_guess.py`, `strategies/`, `cr3bp_orbits.py` (problem construction & family orchestration) | Python (numerics already Rust) | — |
| `algorithm/family/halo_family.py` (family continuation orchestration) | Python (numerics already Rust) | — |
| `algorithm/transfer` two-body/analytic & orchestration modules | Python (Lambert already Rust) | — |
| `algorithm/transfer/search_geometry.py`, `search_progress.py`, `solution_database.py` (search helpers) | Python | — |
| `algorithm/manifold/sections.py` (section event functions) | Python | — |
| `algorithm/normal_form` (symbolic Legendre/ephemeris H, NAFF, pipeline orchestration) | Python | #449 |
| `algorithm/stability.py` | Python | — |
| `algorithm/station_keeping` (control laws) | Python (propagation already Rust) | — |
| `algorithm/coordinate` (single conversions) | Python | — |
| `algorithm/design` (orchestration) | Python (numerics already Rust) | — |
| `algorithm/design/frozen_orbit.py` (ELFO helper) | Python | — |
| `algorithm/dynamics/potential.py` (pseudo-potential Hessian) | Python | — |
| `algorithm/dynamics` event integration `backend="scipy"` path (scipy event semantics kept as production contract) | Python (scipy `solve_ivp`; Rust path only under explicit `backend="rust"`) | — |
| `algorithm/dynamics/ephemeris_dynamics.py` (acceleration + analytic Jacobian kernel) | Python | — |
| `algorithm/propagation.py` | Python (propagation already Rust) | — |
| `algorithm/proximity` (orchestration) | Python (propagation already Rust) | — |
| `algorithm/proximity/relative_dynamics.py` (ephemeris-target A(t) evaluation) | Python (reuses `ephemeris_dynamics`'s numpy Jacobian kernel) | — |
| `algorithm/levelset/value_function.py` (arbitrary-point value/gradient queries on value-function grids) | Python (scipy NdBSpline tensor splines; HJB solving numerics live in the `e2m2e-hjb-dynamics` crate, ADR 0032 decision 4) | — |
| `algorithm/catalog_sweep.py` (parameter-space scan orchestration) | Python (scan-grid construction & family-generation dispatch; batch numerics via Rust family entries, ADR 0031/0029) | — |
| `algorithm/nominal_orbit/` | Python (placeholder) | — |

## SUNK

Numeric kernels in Rust; Python constructs problems, passes args across FFI,
interprets results. When auditing Python files with numeric loops, first check
whether it's just a thin wrapper.

**`algorithm/dynamics` (non-event propagation).** Without events, propagation
numerics for CR3BP/BCR4BP/ephemeris paths live in the `e2m2e-integrators` crate
(`propagate_cr3bp_py`, `propagate_bcr4bp_py`, `propagate_with_state_py` etc.);
CR3BP takes Rust fast paths directly, `dynamics.py` only constructs problems and
interprets results. Three boundaries registered honestly:

- **Event integration**: Rust event path activates only under explicit
  `backend="rust"`; `backend="scipy"`'s and base-class events' RHS are Python
  equations of motion via scipy `solve_ivp` (explicit backend choice — not silent
  fallback per ADR 0020 decision 4). **Ruled intentionally dual-backend (#546,
  2026-08-30)**: scipy event semantics (event attributes, dense output,
  `t_events`/`y_events` contract) are the production contract —
  `spatiography/fate.py` relies on them; the Rust event path's
  semantics (in-step bisection, no dense output) remain an accepted divergence,
  and converging would require semantic alignment first. See Intentionally-in-Python.
- **BCR4BP rust event path**: **sunk (#594, executing #546's 2026-08-30
  migrating ruling)** — CR3BP and BCR4BP rust event paths dispatch the RHS to
  the `e2m2e-forces` CR3BP/BCR4BP EOM/STM kernels via the `solve_ivp_events`
  kernel entry (`RustEomKernel` identifier + params), so per-step RHS evaluation
  no longer crosses the language boundary; user-defined event functions keep
  the Python callback spec mechanism. Integration itself was already Rust; the
  generic `solve_ivp_events` + Python-callback route remains for other dynamics
  classes (ephemeris etc.) and as the equivalence reference
  (`tests/algorithm/dynamics/test_rust_events_kernel.py`,
  `tests/numerical/integrators/bindings/test_solve_ivp_events_kernel.py`).
- **Ephemeris N-body Jacobian kernel**: full numpy implementation of accelerations
  + analytic Jacobians retained in `ephemeris_dynamics.py`, produced/consumed by
  `proximity/relative_dynamics.py` (`compute_jacobian_A(t, state)` duck-typed
  seam). **Ruled intentionally Python (#546, 2026-08-30)**: sinking would require
  Rust-side SPICE ephemeris access plus a full analytic-Jacobian port — the
  largest scope of the three boundaries; the sole production consumer is
  proximity's A(t) seam. Revisit on demonstrated hot-path need. See
  Intentionally-in-Python.

Same package's `potential.py` pseudo-potential Hessian is numpy (shared by
non-propagation paths); see Intentionally-in-Python. Per ADR 0002.

**`algorithm/forces` (numerics).** Force-model numerics (harmonics, tides, SRP,
third-body, drag) in the `e2m2e-forces` crate; `force_model.py` etc. are config
surfaces doing parameter validation + to_rust_spec serialization; sources stay at
the algorithm layer (ADR 0030): Python is the orchestration-side config surface,
not a numeric core; no new Python numerics directory. Layer adjudication doesn't
change the sunk registration.

**`algorithm/transfer/lambert.py`.** Two-body Lambert (Izzo) in
`e2m2e-propagation` (`lambert.rs`); this file is a thin wrapper
(`lambert_izzo_py` / `lambert_batch_py`).

**`algorithm/transfer/search_parallel.py` (grid search).** Search-phase
evaluation units + grid distribution in `e2m2e-integrators`
(`transfer_grid_search`, Rayon parallel). Python keeps `TransferSearch`
orchestration, backend dispatch, six geometry thin-wrappers (monkeypatch seam per
ADR 0017).

**`algorithm/transfer/wsb.py` (WSB grid candidate evaluation).** TLI
parameterization, BCR4BP propagation, perilune detection/interpolation, H2,
arrival-state/Delta-v computation, candidate filtering run in `e2m2e-forces`'s
pure-Rust core distributed via `e2m2e-integrators` with Rayon. Python keeps
system/parameter validation + domain result assembly; Rust by default, Python only
as explicit equivalence comparison, never auto-fallback. Work item: #447.

**`algorithm/transfer/low_energy.py` (manifold section-state pairing).**
Cartesian products of two section-state sets, position/velocity norms, weighted
stitching costs, stable sorting run in `e2m2e-forces`' pure-Rust core exposed via
`e2m2e-integrators`. Manifold tube management, four-branch orchestration, and
ThreeBodyLambert closure remain Python; manifold seeds, STM ferrying, tube
propagation sunk by #448. Rust default; Python explicit-comparison only.
Work item: #447.

**`algorithm/transfer/qlaw.py`.** Q-law feedback-law integration, Keplerian
conversion, Gauss equations, Q function, thrust direction all in the Rust kernel
exposed via `qlaw_propagate_py` / `qlaw_segment_direction_py`. Python only parses
dynamics params, resamples continuous trajectories, assembles
`LowThrustSegment`s; standalone `rv_to_keplerian` keeps legacy-compatible behavior,
not constituting a feedback-integration degradation path.

**`algorithm/transfer/lowthrust_shooting.py`, `lowthrust_collocation.py`.**
Direct low-thrust shooting's multi-segment controlled propagation + sensitivity-
chain assembly, plus Hermite-Simpson collocation's batched defect evaluation, run
in `e2m2e-integrators`' Rust entries. Python keeps problem construction, SLSQP
outer orchestration, initial guesses, result interpretation; `backend="python"`
retains original implementation as equivalence comparison & degraded path.
Comparison tests:
`tests/algorithm/transfer/test_lowthrust_rust_backend.py`. Migrated in #445.

**`algorithm/transfer/porkchop.py`.** Grid evaluation (terminal propagation +
Lambert + ΔV assembly + Rayon distribution) in `e2m2e-forces`' Rust core, exposed
via `e2m2e-integrators`. Python only constructs problems & interprets results:
built-in terminals (`OrbitTerminal`/`StateTerminal` + unpatched `CR3BP_Dynamics`)
hand terminal specs directly to Rust (orbit-terminal propagation sunk too);
custom-terminals or patched scenarios extract state grids via the
`get_arrival_state` protocol into the same Rust core — no Python numeric fallback
(#378). SQLite archiving, interpolation queries, Pareto fronts stay Python.
Tests: `tests/algorithm/transfer/test_porkchop_rust_backend.py`. Migrated in #446.

**`algorithm/solver` (ephemeris correction path).** Multiple-shooting iteration
`multiple_shooting_correct_py` is Rust; segmented & stable-orbit correction
default there. The `MultipleShooting` class (used by transfer/hohmann) remains
Python — see Migrating (pending separate item).

**`algorithm/solver/differential_correction.py` (CR3BP numeric kernel).**
Symmetry strategies & Orbit orchestration stay Python; half-period-symmetric and
full-period closure residuals, STM Jacobians, Newton corrections, linear solves,
and convergence state machines execute via `differential_correction_cr3bp_py` in
Rust. Python no longer retains differential-correction numeric backends; CR3BP
correction entries uniformly go Rust.

**`algorithm/transfer/nsga2.py`.** Constraint-domination sorting, crowding
distance, tournament + environmental selection, SBX crossover, polynomial
mutation live in `e2m2e-integrators`' `nsga2` module via `nsga2_*_py`.
Python keeps objective callbacks, `ProcessPoolExecutor` parallel evaluation, NumPy
randomness, generational loops, `NSGA2Result` assembly; `backend="python"`
retains originals for comparison/degradation. Random sampling generated Python-side
per existing branch order and handed to Rust, so same-seed evolution is equivalent
across backends. Work item: #444.

**`algorithm/solver/continuation.py` (PAL numeric kernel).** Pseudo-arclength
continuation's XZ-symmetric constraint F/dF assembly, tangents (null space), PAL
Newton iterations in the `e2m2e-forces` crate (`pal_continuation` module) exposed
via `pal_f_df_tangent_py` / `pal_newton_step_py`. Dual-backend
`pseudo_arclength_continuation`: rust default; `backend="python"` walks numpy
reference path (comparison & degradation; equivalence tests at
`tests/algorithm/design/continuation/test_halo_pal_rust_equivalence.py`). Initial
tangent computed identically Python-reference-side across both backends (no sign
guarantee between numpy-SVD and Rust generalized cross product; first-step direction
must lock to one implementation). Outer per-orbit orchestration (physical sanity,
direction feedback, stagnation detection) stays Python — correction numeric kernel
already sunk, see SUNK #441. `family/halo_family.py` is pure orchestration with no
numeric kernel — Intentionally-in-Python.

**`algorithm/family` (#428 orbit-family numeric kernel).** Eight families unify
via `generate_cr3bp_family_py`: one Rust call performs seed construction, CR3BP
correction, PAL, step control, member filtering, structured termination, partial-family
retention. Internally reuses pure-Rust implementations behind
`differential_correction_cr3bp_py` and `planar_full_period_pal_py`; collinear-point
root-finding, linear center modes, Lissajous nonlinear center-reduced multi-point
trajectories, perilune-distance / L4-L5 radial-amplitude / out-of-plane amplitude
scans execute inside the same module; DRO family (#502) is Moon-centered, walking
x0 natural-parameter continuation from standard seeds bidirectionally across the
amplitude window, min/max lunar-distance mean measured Rust-side too. Python never
calls numeric atoms per-member, retains no numeric fallbacks — only request
validation, domain dispatch, `OrbitFamily` rewrapping. Work items: #428, #502.
catalog_sweep's energy dimension (#476) uses batch entry
`generate_cr3bp_family_windows_py`: each (family, libration point)'s continuation
trace runs once; Jacobi window filtering happens inside the single Rust call
(same tier as amplitude windows).

**`algorithm/design` (shooting/propagation paths).** Segmented corrections,
multiple shooting, segment propagation, time conversion go Rust
(`segmented_shooting_correct_py`, `multiple_shooting_correct_py`,
`propagate_segments_py`, `batch_et_to_utc_py`); `design_orbit.py` is mission
orchestration.

**`algorithm/coordinate/synodic_j2000.py` (batch conversion).** Batched
synodic↔J2000 goes Rust (`batch_synodic_to_j2000_py`,
`batch_j2000_to_synodic_py`).

**`algorithm/proximity/relative_dynamics.py` (propagation).** Relative-dynamics
propagation goes Rust (`solve_ivp_events`).

**`algorithm/station_keeping/monte_carlo.py` (propagation).** Monte-Carlo
sample propagation goes Rust (`propagate_compiled_stm_py`).

**`algorithm/normal_form` (integration paths).** Propagation integration goes
Rust (`solve_ivp_rust`, #336/#340). QF↔CM high-order Lie flows go `qf_to_cm_py`
/ `cm_to_qf_py` (#465, 12-real-dim split complex integration); `backend="python"`
explicit comparison only.

**`algorithm/normal_form` (CR3BP Hamiltonian numerical construction).**
Collinear points' `build_cr3bp_hamiltonian` goes `build_cr3bp_hamiltonian_py`
(JM `c_n` form, ms-scale); triangular points lack that input semantics — sympy
symbolic path retained.

**`algorithm/normal_form` (H→QF scalar polynomial projection).** CR3BP scalar
coefficients' `project_hamiltonian_to_qf` goes `project_hamiltonian_qf_py`
(numeric multinomial expansion); ephemeris time-series coefficients fall back to
sympy.

**`algorithm/normal_form` (numeric polynomial kernel).** `poly_poisson` /
`poly_simplify` / `polylist_simplify` + in-kernel exponent utilities
(`keys_by_order` / `trim_degree`) go `e2m2e-integrators`' `poly_*_py` bindings;
full support for scalars & 1-D time series, real/complex coefficients. Python
thin wrappers; default `backend='rust'`; `backend='python'` explicit comparison
only (never silent fallback). sympy symbolic-coefficient paths stay Python.
Work item: #464.

**`algorithm/normal_form` (complex integration + QF↔CM Lie flows).** `qf_to_cm`
/ `cm_to_qf` default through `qf_to_cm_py` / `cm_to_qf_py` (12-real-dim split +
DOP853, full real↔complex bases, all-order Lie); `backend="python"` comparison
only. Work item: #465.

**`algorithm/normal_form` (center manifold reduction).**
`CenterManifoldReducer.reduce` defaults through `center_manifold_reduce_py`:
two-step Lie homological reduction (invariant / center), real+complex frequency
W matrices, MAD suppression, `list_deriv`, all-order Poisson chains, imaginary/
real basis transforms — fully in Rust; path embeds the polynomial operations its
Poisson chains use, coexisting with #464's package-level `poly_*`.
`backend="python"` explicit comparison only; silent degradation forbidden
(ADR 0020). Work item: #466.

**`algorithm/manifold/manifolds.py` (seed generation & batch propagation).**
Monodromy eigendecomposition, STM ferrying, ±ε seeds, batch arc-propagation
dispatching in `e2m2e-forces`' `manifold` module exposed via `manifold_seeds_py`
/ `manifold_propagate_py`. Python does parameter validation, `ManifoldTube`/
`Orbit` assembly, optional post-hoc section truncation; no Python numeric
fallback. Work item: #448.

## MIGRATING

Transitional states per ADR 0011, each with an independent work item.
`MultipleShooting` is a generic Python class supporting multiple dynamics models;
it cannot reuse the CR3BP correction kernel; needs separate evaluation and
migration.

**`algorithm/solver/MultipleShooting` class.** The multiple-shooting class used
by transfer/hohmann remains generic Python; evaluate and migrate separately.

Deferred unscheduled (#449 assessment): quasi-Floquet full-matrix method (P5),
multiple-shooting Newton shell (P6); standalone FFT productization (P2): #466
already embeds the FFT particular solutions need — may not need a separate item.

## INTENTIONALLY IN PYTHON

Kept deliberately, not tech debt. On encountering these modules' Python numerics,
judge by their stated reasons; don't misreport as misplaced.

**`algorithm/transfer/nlp_core.py`, `nlp_scipy.py`, `nlp_copt.py`,
`transfer_optimization.py` (NLP optimization & orchestration).** Why:
SLSQP/COPT serial iteration is Python's strength (early architecture consensus,
ADR 0017 boundary); `transfer_optimization.py` is the search-optimize two-step's
optimization-stage high-level orchestration (construct optimizers, objectives/
constraints) — NLP territory. Home of default solvers, not a migration target.

**`algorithm/family/*_initial_guess.py`, `strategies/`, `cr3bp_orbits.py`
(problem construction & family orchestration).** Why: family selection,
direction/sampling-rule validation, rewrapping Rust results into domain objects =
orchestration duties (architecture.md §3). #428's seeds, propagation, correction,
PAL, stepping, filtering, center modes, Lissajous sampling, family metrics are all
inside one Rust call; Python offers no numeric fallback.

**`algorithm/family/halo_family.py` (family continuation orchestration).** Why:
seeds, per-orbit correction calls, direction feedback, stagnation detection,
family assembly = orchestration duties (architecture.md §3); the PAL kernel it
calls sank via `continuation.py` (see SUNK #443); this file itself has no numeric
iteration.

**`algorithm/transfer` two-body/analytic & orchestration modules.**
`hohmann.py`, `multi_impulse.py`, `lga.py`, `three_body_lambert.py`,
`mission_assessment.py`, `cost.py`, `propulsion.py`, `terminal.py`,
`transfer.py` (NLP orchestration), `config.py`. Why: analytic formulas or
orchestration, no feed-numbers-and-iterate hot paths; Lambert already Rust.
Multi-impulse NLP node optimization is NLP territory, above.

**`algorithm/transfer/search_geometry.py`, `search_progress.py`,
`solution_database.py` (search helpers).** Why: `search_geometry.py` is the
geometry kernel's numpy pure-function reference — ADR 0017's boundary-fixed
thin-wrapper / numpy benchmark (monkeypatch seam) must coexist with Rust;
`search_progress.py` wraps tqdm; `solution_database.py` queries/filters solution
stores (data-management nature).

**`algorithm/stability.py`.** Why: monodromy/Floquet analysis is numpy
eigendecomposition without performance hot paths; propagation via `dynamics`
already Rust. Not on migration lists.

**`algorithm/manifold/sections.py` (section event functions).** Why: defines
Poincaré-section event functions (crossing detection delegated to
`Dynamics.propagate(events=...)`, integrator-located within steps; propagation
already Rust); no independent numeric iteration of its own.

**`algorithm/normal_form` (symbolic Legendre/ephemeris H, NAFF, pipeline
orchestration).** Why (#449 assessment):

- **Symbolic construction isn't a numeric hot path.** sympy Legendre / ephemeris
  `build_hamiltonian` are CAS; collinear CR3BP numeric construction already Rust.
  No parity sinking benefit.
- **NAFF** wraps an external executable — never enters a Rust crate.
- **Pipeline / catalog orchestration** chains calls and assembles results;
  polynomial kernel, QF↔CM, center-manifold reduction already sunk
  (#464/#465/#466) — orchestration layers aren't migrated separately.
- All three main-chain numeric items sunk; quasi-Floquet full-matrix and MS
  Newton shell deferred unscheduled. Correctness doesn't take qiao as oracle
  (#426).

**`algorithm/design/frozen_orbit.py` (ELFO helper).** Why: classical-elements↔
Cartesian conversion is analytic formulas; drift statistics is orchestration;
propagation already Rust (queries hit sunk Rust instances).

**`algorithm/dynamics/potential.py` (pseudo-potential Hessian).** Why: analytic-
derivative formulas in numpy, shared by non-propagation paths like dynamics
equations & stability analysis; no hot path per call; propagation-path
acceleration computation already Rust.

**`algorithm/dynamics` event integration `backend="scipy"` path.** Why (#546
ruling, 2026-08-30): scipy's event semantics (event-function attributes, dense
output, `t_events`/`y_events` result contract) are the de-facto public contract
production consumers rely on (`spatiography/fate.py`);
the Rust event path (`solve_ivp_events`) locates events by in-step bisection
without dense output — an accepted divergence per ADR 0020 decision 4 — and
converging to a Rust-only event path would require aligning semantics first.
Dual backend stays explicit-choice, never silent.

**`algorithm/dynamics/ephemeris_dynamics.py` (acceleration + analytic Jacobian
kernel).** Why (#546 ruling, 2026-08-30): the full numpy kernel is consumed only
by `proximity/relative_dynamics.py`'s A(t) duck-typed seam; sinking it would
need Rust-side SPICE ephemeris access plus a port of the analytic Jacobians —
the largest scope among the dynamics-layer boundaries — with no other hot
consumer today. Revisit if relative-dynamics propagation becomes a demonstrated
production hot path.

**`algorithm/station_keeping` (control laws).** `controller.py`,
`momentum_management.py`, `special_point.py`, `target_point.py`,
`error_models.py`. Why: control laws + orchestration, no hot-loop iterations;
propagation/STM already Rust (`monte_carlo.py` uses
`propagate_compiled_stm_py`).

**`algorithm/coordinate` (single conversions).** Why: one-shot scalar conversions
have no performance hot path; batch paths already Rust (`synodic_j2000.py`).
coordinate's layer ownership was ruled by ADR 0026 decision 1 (stays algorithm),
independent of this ledger.

**`algorithm/design` (orchestration).** Why: mission orchestration duty;
numerics (shooting/propagation) already Rust, above.

**`algorithm/propagation.py`.** Why: single-segment prediction orchestration
(ADR 0011: no independent orchestrator class), pairing `ForceModel` + calling
propagation + producing `EphemerisTable`; propagation numerics already Rust.

**`algorithm/proximity` (orchestration).** `phasing.py`, `safety.py`. Why:
orchestration; relative-dynamics propagation already Rust
(`relative_dynamics.py`). Ephemeris targets' A(t) evaluation reuses
`ephemeris_dynamics`'s numpy Jacobian kernel — intentionally Python.

**`algorithm/levelset/value_function.py` (value-function grid queries).** Why: a
gridded data product distributed with catalog records; arbitrary-point value/
gradient queries via scipy NdBSpline tensor splines, no per-call hot path
(ADR 0032 decision 4); HJB solving numerics proper live in the
`e2m2e-hjb-dynamics` crate (ADR 0032/0033), not this file.

**`algorithm/catalog_sweep.py` (parameter-space scan orchestration).** Why:
amplitude/perilune/Jacobi-window scan-grid construction, per-family dispatch,
record assembly = orchestration; batch numerics via Rust family entries
(`generate_cr3bp_family_windows_py` etc., ADR 0029); Python keeps no independent
numeric path.

**`algorithm/nominal_orbit/`.** Why: placeholder currently (interpolator awaits
FR1; package docstring says so); nothing real to sink.

## Maintenance notes

- **Fixed status vocabulary**: new entries may only use SUNK / MIGRATING /
  INTENTIONALLY IN PYTHON so the whole text stays grep-enumerable.
- **Granularity**: down to files/paths. One module may split into several entries
  (e.g., `algorithm/design`: shooting/propagation sunk vs orchestration
  intentional; `algorithm/normal_form`: integration/CR3BP-Hamiltonian/H→QF/
  polynomial-kernel/QF↔CM/center-manifold sunk vs symbolic/NAFF/orchestration
  intentional) — quick-reference table paths rule.
- **Migrating entries**: when issues close (sink done or verdict changed), move
  entries into their status sections, keeping issue numbers as historical pointers.
- **Intentional entries**: must carry reasons citing ADR/document decisions; no
  "not now" temporaries.
- **New modules**: any new `e2m2e/algorithm/` submodule containing numerics must
  register here before merging.
- Related ADRs: 0011 (five-layer architecture), 0017 (grid search sinking to
  Rayon), 0026 (suite layer clarification — its follow-up item 3 is this ledger's
  origin).
