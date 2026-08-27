# HJB Subsystem Architecture

Start from a concrete scenario. A spacecraft performs a low-thrust transfer in
cislunar space; flight control must answer at every step: in the current
state, which direction to thrust and how hard. Direct and indirect methods
re-solve an optimal control problem at every query — slow and unstable. The
HJB route does it differently: offline, solve the Hamilton-Jacobi equation
once on a state grid to obtain a numerical table of the value function V(x,
t); online, look up the value-function gradient and compute the control
directly from optimality conditions. One solve, fast queries. That is what
this subsystem does.

This page describes the target shape of e2m2e's HJB subsystem: who provides
what, where the seams are, and how far deepening goes. The demand-side
two-level dynamic programming scheme lives in the geo-nrho project's
docs/hjb-dependency-architecture.md; this is the corresponding supply-side
architecture. Decision snapshots: ADR 0032, 0033, 0034.

## 1. Subsystem positioning and two-level division of labor

Low-thrust trajectory rapid planning has two levels. Level one solves the HJB
equation backwards on structured grids, producing a low-dimensional value
function table. Level two approximates the value function with a neural
network over the full seven-dimensional state (six state dims plus time) for
online control. Level two exists because level one cannot escape the curse of
dimensionality: structured-grid storage grows exponentially with dimension;
seven dims on a grid is infeasible.

e2m2e delivers level one plus verification tooling, concretely four things:

- Grid solver: e2m2e-levelset, the Rust port of ToolboxLS, providing upwind
  schemes, Lax-Friedrichs dissipation, TVD Runge-Kutta time integration.
- Dynamics: a set of structs implementing the Hamiltonian trait that plug a
  concrete problem's vector field into the solver.
- Python bindings: solve entry points exposed via e2m2e-integrators so
  downstream orchestration can run offline solves.
- Verification tools: propagators and compiled force models for closed-loop
  replay and coarse-vs-fine model comparison.

e2m2e does not deliver level two. Neural network training, online policy, and
mission-layer terminal constraints belong to downstream projects (currently
geo-nrho). This boundary deserves emphasis: level one's output is a value
function grid file; how level two consumes it is governed by e2m2e only as
file format and state semantics — training and inference are out of scope.

## 2. The Hamiltonian seam and the dynamics family

The seam between solver and dynamics is e2m2e-levelset's Hamiltonian trait,
corresponding to ToolboxLS's `hamFunc` and `partialFunc` callbacks, with two
methods:

- `hamiltonian(t, grid, phi, p)`: computes H(x, t, p) over the whole grid,
  with the control already analytically eliminated by optimality conditions.
- `partial_bound(t, grid, phi, p_min, p_max, dim)`: per-dimension dissipation
  coefficient envelope max|∂H/∂p_dim|.

Dynamics parameters become fields of the implementing struct, fixed at
construction — no callbacks, no Python inside Rust hot loops. The trait is
dimension-agnostic and carries an explicit time parameter; these two
properties are structural preconditions for later deepening.

Dynamics is not one implementation but a family, a spectrum ordered by
fidelity:

| Impl | State dims | Autonomy | Force model | Landing |
|---|---|---|---|---|
| Cr3bpSynodic | 4: x, y, vx, vy | autonomous | two primaries point-mass, centrifugal, Coriolis | #497 |
| Mass extension | 5: add m | autonomous | same, control becomes thrust, acceleration decays with mass | mass axis landed with #498 (ephemeris convention); CR3BP convention follow-up issue |
| Bcr4bp | 4 or 5 | non-autonomous, explicit t | adds solar gravity | follow-up issue |
| Ephemeris force model | 4 or 5 | non-autonomous | e2m2e-forces compiled force models, ephemeris via EphemCache lookup (ADR 0016) | #498 (ADR 0034) |

New family members do not replace old ones. Coarse models hold two permanent
values: computation is cheap, suited to tuning grid and scheme parameters; and
they serve as regression baselines for fine models — an ephemeris force-model
solution should first match the CR3BP solution in magnitude and structure;
mismatch means implementation error, caught before comparing against mission
data.

Code ownership: family members live in the new crate e2m2e-hjb-dynamics
(Apache-2.0), not in e2m2e-levelset. Two reasons. Licensing: levelset inherits
ToolboxLS's ACM non-commercial license wholesale, so original code placed
there would fall under the same terms. Positioning: levelset keeps the purity
of faithful porting, every module maintainable against its MATLAB original;
dynamics is e2m2e's original layer — different concerns. Dependency direction
is one-way: hjb-dynamics depends on levelset's traits; under the `ephemeris`
feature it additionally depends on e2m2e-forces / e2m2e-spice ephemeris cache
types; reverse dependency forbidden, consistent with ADR 0012's dependency
direction spirit.

## 3. Dimension ceiling

Structured-grid storage = node count × bytes per node. With 40 nodes per
dim, double precision, single array:

| State dims | Nodes | Single array memory |
|---|---|---|
| 4 | 2.6×10⁶ | ≈ 0.02 GB |
| 5 | 1.0×10⁸ | ≈ 0.8 GB |
| 6 | 4.1×10⁹ | ≈ 33 GB |

The solver holds several same-shaped arrays simultaneously (φ, per-dim
gradients, dissipation coefficients), so real usage is several times one
array. The conclusion is hard: **grid-layer state dimensions cap at five**.
Six-dimensional infeasibility is arithmetic fact, not something engineering
optimization can route around.

This ceiling decides where high-fidelity forces go. High-order gravity of
Earth and Moon (e.g. 10×10) and SRP are intrinsically three-dimensional
forces: zonal terms cannot be faithfully projected into planar synodic frames,
and SRP direction depends on the Sun's three-dimensional position.
Three-dimensional forces mean six-dimensional states, beyond the ceiling.
Hence three-dimensional high-fidelity problems are not solved at the grid
layer; they are carried by level-two neural networks whose training data comes
from level-one low-dimensional solutions and mission trajectories. The
grid-layer force-model fidelity ceiling is forces expressible planarly in the
synodic frame: two-primary point masses and solar third-body gravity. Level
one's solutions thereby mean optimal value functions under approximate models
— priors and training signals for level two, not final products.

## 4. Python binding layer

Bindings are exposed through e2m2e-integrators' pyo3 cdylib; large arrays
cross as flat Vec<f64> plus shape; ABI stamps increment per abi-version.txt —
existing processes, unchanged.

Per ADR 0032 decision 3, the entry shape was changed to the generic entry
`solve_hjb_py`. Parameters come in four groups: terminal conditions and
integration controls (terminal cost, time interval, CFL number, step ceiling),
grid definition (per-dim bounds and node counts), dynamics identifier (string:
planar_double_integrator, cr3bp_synodic, ephemeris_planar), and dynamics
parameter table (`HashMap<String, f64>`, values keyed). Rust constructs the
corresponding Hamiltonian impl from the identifier; expected dimension follows
from it; missing keys or invalid values raise explicit errors at the binding
layer.

`solve_planar_lowthrust_hjb_py`, previously used by geo-nrho, remains as a
compat wrapper with signatures pinned to the double integrator (drift_accel,
max_accel, fuel_weight listed individually), forwarding internally to
solve_hjb_py. The generic entry pins the ABI change to that one addition;
afterwards new dynamics only touch Rust-side construction branches. The cost:
the parameter table is weakly typed across the FFI boundary; misspelled keys
surface only at runtime, so the binding layer must validate existence and
values rather than silently ignoring them.

## 5. State semantics and coordinate conventions

Levels one and two hand off through value function grid files; both sides'
interpretation of state must agree verbatim. Conventions below are documented
rather than living only in code:

- Nondimensionalization: characteristic length is the primary-to-secondary
  distance, characteristic time the inverse of synodic angular velocity, so
  angular speed is identically 1 and is not a dynamics parameter. The mass
  ratio μ is the sole nondimensional dynamics parameter, fixed when
  constructing the Hamiltonian, matching `CR3BP_System` in
  e2m2e/algorithm/dynamics.
- State order: (x, y, vx, vy). Synodic frame origin at the barycenter, x-axis
  from primary toward secondary, rotating with the system. Matches STATE_ORDER
  in geo-nrho's algorithm/dp.py.
- Time semantics: in autonomous systems time is only integration direction and
  the value function may take a time-independent form (e.g. TTR); in
  non-autonomous systems time is genuine dependence of the value function. The
  binding entry expresses both cases with the same time-interval parameter.
- Lifting convention: the four-dimensional planar state is the z = vz = 0
  section of three-dimensional space states. Lifting to mission states goes
  through frame rotation (EphemCache's frame matrices, ADR 0016) from the
  synodic frame to the mission frame. Level two depends on this convention
  when using level-one solutions as training data.

## 6. Verification tiering

Verification has four tiers, inside-out:

1. Solver self-contained gates (existing): upwind scheme convergence orders,
   Burgers equation against Hopf-Lax exact solution, dual-integrator reachable
   set against analytic solutions — 22 cases total, see the e2m2e-levelset
   README. This tier depends on no external dynamics.
2. Dynamics correctness: a new Hamiltonian's zero-control vector field matches
   propagate_cr3bp_py's CR3BP dynamics pointwise on sampled states; closed-loop
   behavior checked against Lagrange point stability and orbit periods. #497's
   verification section is this tier.
3. Coarse-fine regression: once the ephemeris force-model Hamiltonian
   (EphemerisPlanar, #498) lands, CR3BP solutions serve as regression baseline,
   checking consistency of value function magnitudes and iso-surface structure.
   Coarse models are fine models' first assertion.
4. Closed-loop replay: value-function gradients generate control laws fed into
   e2m2e-propagation's compiled force models; verify the closed-loop trajectory
   reaches the terminal set. This tier corresponds to the verification
   constraints of geo-nrho's architecture doc, binding the HJB solution to an
   independent integrator so scheme-dissipation drift cannot silently
   accumulate.

None of the four tiers rely on MATLAB reference data, consistent with
e2m2e-levelset's self-contained verification principle during porting and with
ADR 0013's verification-by-definition strategy.
