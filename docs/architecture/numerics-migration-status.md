# Algorithm-Layer Numerical Kernel Migration Status Ledger / algorithm 层数值内核迁移状态清单

[English](#algorithm-layer-numerical-kernel-migration-status-ledger) | [简体中文](#中文)

## English

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

### Quick reference

#### SUNK

| Module | Numeric kernel | Work issue |
|---|---|---|
| `algorithm/dynamics` (non-event propagation) | Rust (`e2m2e-integrators`) | — |
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

#### MIGRATING

| Module | Numeric kernel | Work issue |
|---|---|---|
| `algorithm/solver/MultipleShooting` class (transfer/hohmann) | Python | pending separate migration |

#### INTENTIONALLY IN PYTHON

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
| `algorithm/propagation.py` | Python (propagation already Rust) | — |
| `algorithm/proximity` (orchestration) | Python (propagation already Rust) | — |
| `algorithm/proximity/relative_dynamics.py` (ephemeris-target A(t) evaluation) | Python (reuses `ephemeris_dynamics`'s numpy Jacobian kernel) | — |
| `algorithm/levelset/value_function.py` (arbitrary-point value/gradient queries on value-function grids) | Python (scipy NdBSpline tensor splines; HJB solving numerics live in the `e2m2e-hjb-dynamics` crate, ADR 0032 decision 4) | — |
| `algorithm/catalog_sweep.py` (parameter-space scan orchestration) | Python (scan-grid construction & family-generation dispatch; batch numerics via Rust family entries, ADR 0031/0029) | — |
| `algorithm/nominal_orbit/` | Python (placeholder) | — |

### SUNK

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
  fallback per ADR 0020 decision 4).
- **BCR4BP rust event path**: integration in Rust but RHS fed as Python callback
  into `solve_ivp_events` — cross-language callback overhead exists.
- **Ephemeris N-body Jacobian kernel**: full numpy implementation of accelerations
  + analytic Jacobians retained in `ephemeris_dynamics.py`, produced/consumed by
  `proximity/relative_dynamics.py` (`compute_jacobian_A(t, state)` duck-typed seam).

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

### MIGRATING

Transitional states per ADR 0011, each with an independent work item.
`MultipleShooting` is a generic Python class supporting multiple dynamics models;
it cannot reuse the CR3BP correction kernel; needs separate evaluation and
migration.

**`algorithm/solver/MultipleShooting` class.** The multiple-shooting class used
by transfer/hohmann remains generic Python; evaluate and migrate separately.

Deferred unscheduled (#449 assessment): quasi-Floquet full-matrix method (P5),
multiple-shooting Newton shell (P6); standalone FFT productization (P2): #466
already embeds the FFT particular solutions need — may not need a separate item.

### INTENTIONALLY IN PYTHON

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

### Maintenance notes

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

## 中文

ADR 0011 决策：部分计算功能由 Python 执行，正在逐步迁移至 Rust 计算核心。
本文按 `e2m2e/algorithm/` 的子模块逐项登记迁移状态，供审计直接引用，避免把
过渡状态误读成放错层（误判先例与裁决见 ADR 0026）。

每个登记项回答三个问题：**数值内核在哪**（Rust crate / Python）、**迁移状态**
（已下沉 / 迁移中 / 有意留 Python）、**为什么**（有意留 Python 必写理由；
迁移中必附工作 issue）。

状态词是固定的三个，全文可 grep：

- `已下沉`：数值内核在 Rust，Python 侧是薄封装或编排
- `迁移中`：有明确的下沉工作项，见对应 issue
- `有意留 Python`：有决策依据地保留在 Python

### 速查表

### 已下沉

| 模块 | 数值内核 | 工作 issue |
|---|---|---|
| `algorithm/dynamics`（无事件传播） | Rust（`e2m2e-integrators`） | 无 |
| `algorithm/forces`（数值） | Rust（`e2m2e-forces`） | 无 |
| `algorithm/transfer/lambert.py` | Rust（`e2m2e-propagation`） | 无 |
| `algorithm/transfer/search_parallel.py`（网格搜索） | Rust（`e2m2e-integrators`） | 无 |
| `algorithm/transfer/wsb.py`（WSB 网格候选评估） | Rust（`e2m2e-forces` + `e2m2e-integrators`） | #447 |
| `algorithm/transfer/low_energy.py`（流形截面态配对） | Rust（`e2m2e-forces` + `e2m2e-integrators`） | #447 |
| `algorithm/solver/differential_correction.py`（CR3BP 数值内核） | Rust（`e2m2e-integrators`） | #441 |
| `algorithm/solver`（星历修正路径） | Rust（`e2m2e-integrators`） | 无 |
| `algorithm/solver/continuation.py`（PAL 数值内核） | Rust（`e2m2e-forces`） | #443 |
| `algorithm/family`（#428 轨道族数值内核） | Rust（`e2m2e-forces` + `e2m2e-integrators`） | #428 |
| `algorithm/transfer/qlaw.py`（反馈积分与 Q 函数；Python 仅组装初猜） | Rust（`e2m2e-forces` + `e2m2e-integrators`） | #442 |
| `algorithm/transfer/nsga2.py`（演化算子；Python 保留评估与编排） | Rust（`e2m2e-integrators`） | #444 |
| `algorithm/transfer/lowthrust_shooting.py`、`lowthrust_collocation.py`（直接法数值评估） | Rust（`e2m2e-integrators`；SLSQP 编排留 Python） | #445 |
| `algorithm/transfer/porkchop.py`（网格评估：终端传播 + Lambert + ΔV） | Rust（`e2m2e-forces` + `e2m2e-integrators`；Python 仅问题构造与存档/查询） | #446 |
| `algorithm/design`（打靶/传播路径） | Rust（`e2m2e-integrators`） | 无 |
| `algorithm/coordinate/synodic_j2000.py`（批量转换） | Rust（`e2m2e-integrators`） | 无 |
| `algorithm/proximity/relative_dynamics.py`（传播积分） | Rust（`e2m2e-integrators`，`solve_ivp_events`）；星历目标的 A(t) 由 Python 雅可比内核提供 | 无 |
| `algorithm/station_keeping/monte_carlo.py`（传播） | Rust（`e2m2e-integrators`） | 无 |
| `algorithm/normal_form`（积分路径） | Rust（`e2m2e-integrators`） | #336/#340 |
| `algorithm/normal_form`（CR3BP Hamiltonian 数值构造） | Rust（`e2m2e-integrators`） | 无 |
| `algorithm/normal_form`（H→QF 标量多项式投影） | Rust（`e2m2e-integrators`） | 无 |
| `algorithm/normal_form`（数值多项式核） | Rust（`e2m2e-integrators`） | #464 |
| `algorithm/normal_form`（复值积分 + QF↔CM Lie 流） | Rust（`e2m2e-integrators`，12 实维分裂） | #465 |
| `algorithm/normal_form`（中心流形化简） | Rust（`e2m2e-integrators`） | #466 |
| `algorithm/manifold/manifolds.py`（种子生成与批量传播） | Rust（`e2m2e-forces` + `e2m2e-integrators`） | #448 |

### 迁移中

| 模块 | 数值内核 | 工作 issue |
|---|---|---|
| `algorithm/solver/MultipleShooting` 类（transfer/hohmann） | Python | 待单独迁移 |

### 有意留 Python

| 模块 | 数值内核 | 工作 issue |
|---|---|---|
| `algorithm/transfer/nlp_*`、`transfer_optimization.py`（NLP 优化与编排） | Python | 无 |
| `algorithm/family/*_initial_guess.py`、`strategies/`、`cr3bp_orbits.py`（问题构造与族编排） | Python（数值已 Rust） | 无 |
| `algorithm/family/halo_family.py`（族延拓编排） | Python（数值已 Rust） | 无 |
| `algorithm/transfer` 二体/解析与编排模块 | Python（Lambert 已 Rust） | 无 |
| `algorithm/transfer/search_geometry.py`、`search_progress.py`、`solution_database.py`（搜索辅助） | Python | 无 |
| `algorithm/manifold/sections.py`（截面事件函数） | Python | 无 |
| `algorithm/normal_form`（符号 Legendre/星历 H、NAFF、pipeline 编排） | Python | #449 |
| `algorithm/stability.py` | Python | 无 |
| `algorithm/station_keeping`（控制律） | Python（传播已 Rust） | 无 |
| `algorithm/coordinate`（单次转换） | Python | 无 |
| `algorithm/design`（编排） | Python（数值已 Rust） | 无 |
| `algorithm/design/frozen_orbit.py`（ELFO 辅助） | Python | 无 |
| `algorithm/dynamics/potential.py`（伪势能 Hessian） | Python | 无 |
| `algorithm/propagation.py` | Python（传播已 Rust） | 无 |
| `algorithm/proximity`（编排） | Python（传播已 Rust） | 无 |
| `algorithm/proximity/relative_dynamics.py`（星历目标的 A(t) 求值） | Python（复用 `ephemeris_dynamics` 的 numpy 雅可比内核） | 无 |
| `algorithm/levelset/value_function.py`（值函数网格任意点值/梯度查询） | Python（scipy NdBSpline 张量样条；HJB 求解数值已在 `e2m2e-hjb-dynamics` crate，ADR 0032 决策 4） | 无 |
| `algorithm/catalog_sweep.py`（参数空间扫描编排） | Python（扫描网格构造与族生成分派；批量数值走 Rust 族生成入口，ADR 0031/ADR 0029） | 无 |
| `algorithm/nominal_orbit/` | Python（占位） | 无 |

### 已下沉

数值内核在 Rust，Python 侧只构造问题、传参过 FFI、解释结果。审计时若看到
Python 文件里有数值循环，先查它是否只是薄封装。

**`algorithm/dynamics`（无事件传播）。** 无事件时 CR3BP/BCR4BP/星历路径的
传播数值在 `e2m2e-integrators` crate（`propagate_cr3bp_py`、
`propagate_bcr4bp_py`、`propagate_with_state_py` 等），CR3BP 直接走 Rust
快速路径，`dynamics.py` 只做问题构造与结果解释。三处边界如实登记：

- **events 事件积分**：Rust 事件路径只在显式 `backend="rust"` 时启用；
  `backend="scipy"` 与基类事件的右端函数是 Python 运动方程，走 scipy
  `solve_ivp`（显式后端选择，按 ADR 0020 决策 4 不构成静默回退）。
- **BCR4BP rust 事件路径**：积分数值在 Rust，右端函数以 Python 回调喂入
  `solve_ivp_events`，存在跨语言回调开销。
- **星历 N 体雅可比内核**：加速度与解析雅可比的全套 numpy 实现保留在
  `ephemeris_dynamics.py`，并被 `proximity/relative_dynamics.py` 生产消费
  （`compute_jacobian_A(t, state)` 鸭子类型缝）。

同包 `potential.py` 的伪势能
Hessian 是 numpy 实现（供非传播路径共用），见有意留 Python 节。
依据 ADR 0002。

**`algorithm/forces`（数值）。** 力模型数值（球谐、潮汐、SRP、三体、大气）
在 `e2m2e-forces` crate；`force_model.py` 等 Python 文件是做参数验证与
to_rust_spec 序列化的配置面，源码留在 algorithm 层（ADR 0030）：Python 是
编排侧配置面，不是数值核；不新建 Python 数值目录。层级裁决不改变
数值已下沉的登记。

**`algorithm/transfer/lambert.py`。** 二体 Lambert（Izzo）在
`e2m2e-propagation` crate（`lambert.rs`），本文件是薄封装
（`lambert_izzo_py` / `lambert_batch_py`）。

**`algorithm/transfer/search_parallel.py`（网格搜索）。** 搜索阶段的评估单元
与网格分发在 `e2m2e-integrators`（`transfer_grid_search`，Rayon 并行）。
Python 侧保留 `TransferSearch` 编排、后端分发与 6 个几何 thin-wrapper
（monkeypatch 缝，见 ADR 0017）。

**`algorithm/transfer/wsb.py`（WSB 网格候选评估）。** TLI 参数化、BCR4BP
传播、近月点检测与插值、H2、到达态/Delta-v 计算和候选筛选在
`e2m2e-forces` 的纯 Rust 核执行，经 `e2m2e-integrators` 用 Rayon 分发。
Python 保留系统/参数校验和领域结果组装；默认 Rust，Python 仅作显式等价性
对照，绝不自动回退。工作项：#447。

**`algorithm/transfer/low_energy.py`（流形截面态配对）。** 两组截面态的
笛卡尔积、位置/速度范数、加权拼接代价和稳定排序在 `e2m2e-forces` 的
纯 Rust 核执行，经 `e2m2e-integrators` 暴露。流形管管理、四分支编排和
ThreeBodyLambert 闭合仍在 Python；流形种子、STM 转运和管传播已由 #448 下沉。
默认 Rust，Python 仅作显式等价性对照。工作项：#447。

**`algorithm/transfer/qlaw.py`。** Q-law 反馈律积分、开普勒根数转换、Gauss
方程、Q 函数与推力方向在 Rust 内核完成，经 `qlaw_propagate_py` 与
`qlaw_segment_direction_py` 暴露。Python 侧只解析动力学参数、从连续轨迹
重采样并组装 `LowThrustSegment`；独立公开的 `rv_to_keplerian` 保持既有兼容
行为，不构成反馈积分降级路径。

**`algorithm/transfer/lowthrust_shooting.py`、`lowthrust_collocation.py`。**
低推力直接打靶的多段受控传播、灵敏度链式组装，以及 Hermite-Simpson 配点的
批量缺陷求值在 `e2m2e-integrators` 的 Rust 入口完成。Python 侧保留问题构造、
SLSQP 外层编排、初猜与结果解释；`backend="python"` 提供原有实现作为等价性
对照和降级路径。对照测试见
`tests/algorithm/transfer/test_lowthrust_rust_backend.py`。迁移完成于 #445。

**`algorithm/transfer/porkchop.py`。** 网格评估（终端传播 + Lambert +
ΔV 组装 + Rayon 分发）在 `e2m2e-forces` 的 Rust 核完成，经
`e2m2e-integrators` 暴露。Python 只做问题构造与结果解释：内置终端
（`OrbitTerminal`/`StateTerminal` + 未 patch 的 `CR3BP_Dynamics`）把终端
规格直接交给 Rust（轨道终端传播同步下沉）；自定义终端或 patch 场景经
`get_arrival_state` 协议提取状态网格后交同一 Rust 核，无 Python 数值
回退（#378）。SQLite 存档、插值查询、Pareto 前沿留 Python。对照测试见
`tests/algorithm/transfer/test_porkchop_rust_backend.py`。迁移完成于 #446。

**`algorithm/solver`（星历修正路径）。** 多重打靶迭代
`multiple_shooting_correct_py` 在 Rust，segmented 与稳定轨道修正默认走它。
`MultipleShooting` 类（transfer/hohmann 仍使用）本身还是 Python 实现，见
迁移中（待单独立项）。

**`algorithm/solver/differential_correction.py`（CR3BP 数值内核）。** 对称性
策略与 Orbit 编排留在 Python；半周期对称和全周期闭合的残差、STM 雅可比、
Newton 修正、线性求解与收敛状态机经
`differential_correction_cr3bp_py` 在 Rust 执行。Python 不再保留微分修正
数值后端；CR3BP 修正入口统一走 Rust。

**`algorithm/transfer/nsga2.py`。** 约束非支配排序、拥挤度距离、锦标赛和
环境选择、SBX 交叉及多项式变异在 `e2m2e-integrators` 的 `nsga2` 模块，经
`nsga2_*_py` 暴露。Python 保留目标函数回调、`ProcessPoolExecutor` 并行评估、
NumPy 随机数生成、逐代评估与 `NSGA2Result` 组装；`backend="python"` 保留原
实现作对照与降级。随机抽样由 Python 按既有条件分支顺序生成并交给 Rust，因此
两后端同种子演化等价。工作项：#444。

**`algorithm/solver/continuation.py`（PAL 数值内核）。** 伪弧长延拓的 XZ
对称约束 F/dF 组装、切向量（零空间）与 PAL 牛顿迭代在 `e2m2e-forces`
crate（`pal_continuation` 模块），经 `pal_f_df_tangent_py` /
`pal_newton_step_py` 暴露。`pseudo_arclength_continuation` 双后端：默认
rust，`backend="python"` 走 numpy 参照路径（对照与降级；等价性对照见
`tests/algorithm/design/continuation/test_halo_pal_rust_equivalence.py`）。
初始切向量两后端统一由 Python 参照计算（零空间符号约定在 SVD 与 Rust
广义叉积间无保证，首步延拓方向须由同一实现锁定）。外层逐轨编排（物理合理性
检查、方向反馈、停滞检测）留 Python，微分修正数值内核已下沉，见已下沉
#441。`family/halo_family.py` 是纯编排，无独立数值内核，登记在有意留
Python 节。

**`algorithm/family`（#428 轨道族数值内核）。** 八族统一走
`generate_cr3bp_family_py`：一次 Rust 调用完成种子构造、CR3BP 修正、PAL、
步长控制、成员筛选、结构化终止和部分族保留。其内部复用
`differential_correction_cr3bp_py` 与 `planar_full_period_pal_py` 背后的纯
Rust 实现；共线点求根、线性中心模态、Lissajous 非线性中心约化多点轨迹，
以及近月距、
L4/L5 径向振幅与面外振幅扫描也在同一模块内执行；DRO 族（#502）是月心族，
从标准种子做 x0 自然参数延拓（跨种子振幅窗口双向行走），距月心距离
min/max 均值测量同在 Rust 侧。Python 不逐成员调用
数值原子，不保留数值回退，只负责请求校验、领域分派和 `OrbitFamily` 重包。
工作项：#428、#502。catalog_sweep 的能量维度（#476）经
`generate_cr3bp_family_windows_py` 批量入口：同一（族、平动点）的延拓
trace 只走一次，Jacobi 窗口筛选留在 Rust 单次调用内（与振幅窗口同层）。

**`algorithm/design`（打靶/传播路径）。** 分段修正、多重打靶、段传播、
时间转换走 Rust（`segmented_shooting_correct_py`、
`multiple_shooting_correct_py`、`propagate_segments_py`、
`batch_et_to_utc_py`）；`design_orbit.py` 是任务编排。

**`algorithm/coordinate/synodic_j2000.py`（批量转换）。** 会合系↔J2000
批量转换走 Rust（`batch_synodic_to_j2000_py`、`batch_j2000_to_synodic_py`）。

**`algorithm/proximity/relative_dynamics.py`（传播）。** 相对动力学传播走
Rust（`solve_ivp_events`）。

**`algorithm/station_keeping/monte_carlo.py`（传播）。** 蒙特卡洛采样的
传播走 Rust（`propagate_compiled_stm_py`）。

**`algorithm/normal_form`（积分路径）。** 传播积分走 Rust（`solve_ivp_rust`，
见 #336/#340）。QF↔CM 高阶 Lie 流走 `qf_to_cm_py` / `cm_to_qf_py`（#465，
12 实维分裂复积分）；`backend="python"` 仅作显式对照。

**`algorithm/normal_form`（CR3BP Hamiltonian 数值构造）。** 共线点
`build_cr3bp_hamiltonian` 走 `build_cr3bp_hamiltonian_py`（JM `c_n` 形式，
ms 级）；三角点无该输入语义，保留 sympy 符号路径。

**`algorithm/normal_form`（H→QF 标量多项式投影）。** CR3BP 标量系数的
`project_hamiltonian_to_qf` 走 `project_hamiltonian_qf_py`（multinomial
数值展开）；星历时间序列系数回退 sympy。

**`algorithm/normal_form`（数值多项式核）。** `poly_poisson` /
`poly_simplify` / `polylist_simplify` 及核内幂次工具（`keys_by_order` /
`trim_degree`）走 `e2m2e-integrators` 的 `poly_*_py` 绑定；完整支持标量与
一维时间序列、实/复系数。Python 侧为薄封装，默认 `backend='rust'`，
`backend='python'` 仅作显式等价性对照（不静默回退）。sympy 符号系数路径
仍留 Python。工作项：#464。

**`algorithm/normal_form`（复值积分 + QF↔CM Lie 流）。** `qf_to_cm` /
`cm_to_qf` 默认走 `qf_to_cm_py` / `cm_to_qf_py`（12 实维分裂 + DOP853，
完整实↔复基底与全阶 Lie）；`backend="python"` 仅作显式对照。工作项：#465。

**`algorithm/normal_form`（中心流形化简）。** `CenterManifoldReducer.reduce`
默认走 `center_manifold_reduce_py`：两步 Lie 同调（invariant / center）、
实/复两套频域 W、MAD 抑制、`list_deriv`、全阶 Poisson 链与虚/实基底变换
完整在 Rust；路径内嵌 Poisson 链所用多项式运算，并与 #464 包级 `poly_*`
并存。`backend="python"` 仅显式对照，禁止静默降级（ADR 0020）。工作项：
#466。

**`algorithm/manifold/manifolds.py`（种子生成与批量传播）。** 单值矩阵特征
分解、STM 转运、±ε 种子与批量弧传播调度在 `e2m2e-forces` 的 `manifold`
模块，经 `manifold_seeds_py` / `manifold_propagate_py` 暴露。Python 只做
参数校验、`ManifoldTube`/`Orbit` 组装与可选的事后截面截断；不保留 Python
数值回退。工作项：#448。

### 迁移中

ADR 0011 明示的过渡状态，每个条目有独立工作项。`MultipleShooting` 是支持多种
动力学模型的泛型 Python 类，不能复用 CR3BP 微分修正内核；后续需单独评估和迁移。

**`algorithm/solver/MultipleShooting` 类。** transfer/hohmann 使用的多重
打靶类仍是泛型 Python 实现，后续需单独评估和迁移。

后置未派发（#449 评估）：quasi-Floquet 全矩阵法（P5）、多重打靶 Newton
壳（P6）；独立 FFT 产品化（P2）：#466 已内嵌特解所需 FFT，可不单开。

### 有意留 Python

有决策依据地保留在 Python，不是技术债。审计时看到这些模块的 Python 数值，
按对应理由判定，不误报放错层。

**`algorithm/transfer/nlp_core.py`、`nlp_scipy.py`、`nlp_copt.py`、
`transfer_optimization.py`（NLP 优化与编排）。** 理由：SLSQP/COPT 串行迭代是
Python 强项（早期架构讨论共识，ADR 0017 边界固化）；
`transfer_optimization.py` 是搜索-优化两步法优化阶段的高层编排（构造优化器、
计算目标/约束），属 NLP 范畴。这是默认求解器所在，不是迁移目标。

**`algorithm/family/*_initial_guess.py`、`strategies/`、`cr3bp_orbits.py`
（问题构造与族编排）。** 理由：选择轨道族、校验固定方向/采样规则并把
Rust 结果重包为领域对象属于编排模块职责（architecture.md 第 3 节）。
#428 的种子、传播、修正、PAL、步长、筛选、中心模态、Lissajous 采样和族
度量均已收进单次 Rust 调用；Python 不提供数值回退。

**`algorithm/family/halo_family.py`（族延拓编排）。** 理由：种子生成、
逐轨微分修正调用、方向反馈、停滞检测与族组装是编排职责
（architecture.md 第 3 节）；调用的 PAL 数值内核经 `continuation.py`
已下沉（见已下沉 #443），本文件自身无数值迭代。

**`algorithm/transfer` 二体/解析与编排模块。** `hohmann.py`、`multi_impulse.py`、
`lga.py`、`three_body_lambert.py`、`mission_assessment.py`、`cost.py`、
`propulsion.py`、`terminal.py`、`transfer.py`（NLP 编排）、`config.py`。
理由：解析公式或编排，无喂进数字就迭代的热路径；其中 Lambert 求解已
Rust。多脉冲 NLP 的节点优化属 NLP 范畴，见上。

**`algorithm/transfer/search_geometry.py`、`search_progress.py`、
`solution_database.py`（搜索辅助）。** 理由：`search_geometry.py` 是几何核的
numpy 纯函数实现，ADR 0017 边界固化的 thin-wrapper / numpy 对照基准
（monkeypatch 缝），必须与 Rust 几何核并存；`search_progress.py` 是 tqdm
进度封装；`solution_database.py` 是解库查询与筛选（数据管理性质）。

**`algorithm/stability.py`。** 理由：单值矩阵/Floquet 乘子分析是 numpy 特征
分解，无性能热路径；传播经 `dynamics` 已 Rust。未列入迁移清单。

**`algorithm/manifold/sections.py`（截面事件函数）。** 理由：庞加莱截面事件
函数定义（穿越检测交给 `Dynamics.propagate(events=...)`，由积分器步内
定位，传播已 Rust），本身无独立数值迭代。

**`algorithm/normal_form`（符号 Legendre/星历 H、NAFF、pipeline 编排）。**
理由（#449 评估）：

- **符号构造不是数值热路径。** sympy Legendre / 星历 `build_hamiltonian`
  属 CAS；共线点 CR3BP 数值构造已 Rust。无对等下沉收益。
- **NAFF** 是外部可执行文件封装，不进 Rust crate。
- **pipeline / catalog 编排** 是串联与结果组装；多项式核、QF↔CM 与
  中心流形化简已下沉（#464/#465/#466），不单独迁编排层。
- 数值主链三项均已下沉；quasi-Floquet 全矩阵法与 MS Newton 壳后置
  未派发。正确性不以 qiao 为 oracle（#426）。

**`algorithm/design/frozen_orbit.py`（ELFO 辅助）。** 理由：经典根数↔笛卡尔
转换是解析公式，漂移统计是编排；传播已 Rust（查询已下沉 Rust 实例）。

**`algorithm/dynamics/potential.py`（伪势能 Hessian）。** 理由：解析导数公式
的 numpy 实现，供动力学方程与稳定性分析等非传播路径共用，单次调用无热
路径；传播路径的加速度计算已 Rust。

**`algorithm/station_keeping`（控制律）。** `controller.py`、
`momentum_management.py`、`special_point.py`、`target_point.py`、
`error_models.py`。理由：控制律与编排，无热路径数值迭代；传播/STM 已 Rust
（`monte_carlo.py` 用 `propagate_compiled_stm_py`）。

**`algorithm/coordinate`（单次转换）。** 理由：单次标量转换无性能热路径；
批量路径已 Rust（`synodic_j2000.py`）。coordinate 的层级归属由 ADR 0026
决策 1 裁决（留在 algorithm 层），与本清单无关。

**`algorithm/design`（编排）。** 理由：任务编排职责；数值（打靶/传播）已
Rust，见上。

**`algorithm/propagation.py`。** 理由：单段预报编排（ADR 0011：不建独立
编排器），配 `ForceModel` + 调传播 + 输出 `EphemerisTable`；传播数值已 Rust。

**`algorithm/proximity`（编排）。** `phasing.py`、`safety.py`。理由：编排；
相对动力学传播已 Rust（`relative_dynamics.py`）。其中星历目标下
A(t) 的求值复用 `ephemeris_dynamics` 的 numpy 雅可比内核，属有意留 Python。

**`algorithm/levelset/value_function.py`（值函数网格查询）。** 理由：随轨道库
记录分发的网格数据产品，任意点值/梯度查询用 scipy NdBSpline 张量样条评估，
单次调用无热路径（ADR 0032 决策 4）；HJB 求解数值本体在
`e2m2e-hjb-dynamics` crate（ADR 0032/0033），不在本文件。

**`algorithm/catalog_sweep.py`（参数空间扫描编排）。** 理由：振幅/近月点/
Jacobi 窗口等扫描网格构造、按族分派与记录组装是编排职责；批量数值经
Rust 族生成入口（`generate_cr3bp_family_windows_py` 等，ADR 0029）完成，
Python 不保留独立数值路径。

**`algorithm/nominal_orbit/`。** 理由：当前为占位实现（插值器待 FR1 落地，
包 docstring 明示），无实际数值可下沉。

### 维护说明

- **状态词固定**：新登记项只能用已下沉 / 迁移中 / 有意留 Python三词，
  保证全文 grep 可枚举。
- **粒度**：登记粒度到文件/路径。同一模块可拆多条（如 `algorithm/design`：
  打靶/传播路径已下沉、编排有意留 Python；`algorithm/normal_form`：积分/
  CR3BP Hamiltonian/H→QF/多项式核/QF↔CM/中心流形已下沉，符号与
  NAFF/编排有意留 Python），以速查表路径为准。
- **迁移中条目**：issue 关闭（下沉完成或改判）时，把条目移到对应状态节，
  保留 issue 编号作为历史指针。
- **有意留 Python 条目**：必须带理由，理由应引用 ADR 或文档决策，不写
  暂时不迁这类临时话。
- **新模块**：`e2m2e/algorithm/` 下出现含数值实现的新子模块时，在本清单
  登记后再合入。
- 关联 ADR：0011（五层架构）、0017（网格搜索下沉 Rayon）、0026（测试
  套件层级澄清，后续工作第三条即本清单的由来）。
