# e2m2e-levelset

Level-set methods and Hamilton-Jacobi (HJ) reachability solvers, ported from
Ian M. Mitchell's (UBC) [Toolbox of Level Set
Methods](https://www.cs.ubc.ca/~mitchell/ToolboxLS/) (ToolboxLS 1.1, MATLAB).
It solves time-dependent HJ PDEs `D_t φ = -H(x, t, φ, ∇φ)`, providing a grid
PDE kernel for reachability sets, shortest time-to-reach (TTR), and
pursuit-evasion game computations for low-thrust trajectories.

Division of labor with e2m2e's other numerical crates: e2m2e-propagation /
e2m2e-forces integrate ODEs along a **single trajectory**, while this crate
evolves PDEs on **structured grids**; dynamics enter via the
[`Hamiltonian`](src/hamiltonian.rs) trait (e.g. the CR3BP relative-motion
Hamiltonian) without depending on e2m2e's integrators. Pure math crate: no
SPICE, no pyo3; external exposure goes through e2m2e-integrators under the
unified ABI-stamp process.

## License (important)

This crate derives from ToolboxLS, whose original license is the ACM
non-commercial license (full text in `LICENSE-ToolboxLS`). Academic and
research use is royalty-free, but **redistribution must retain the original
copyright and license terms**, and commercial products are not permitted.
Hence this crate's `license-file` deviates from the workspace Apache-2.0. If
a future e2m2e release ships binaries including this crate, release notes
must state this difference; commercial use requires contacting the original
author (mitchell@cs.ubc.ca).

## Module mapping

| Rust module | ToolboxLS source | Notes |
|---|---|---|
| `grid` | `Grids/processGrid.m` | Grid struct, complete |
| `boundary` | `BoundaryCondition/addGhost*.m` | Ghost-cell fill (5 boundary conditions) |
| `derivative` | `SpatialDerivative/UpwindFirst/upwindFirst*.m` | First-order / ENO2 / ENO3 / WENO5 upwind schemes |
| `hamiltonian` | user callbacks `hamFunc` / `partialFunc` | Two function handles merged into one trait |
| `dissipation` | `Dissipation/artificialDissipationGLF/LLF/LLLF.m` | LF dissipation coefficients and CFL step bounds |
| `term` | `Term/termLaxFriedrichs/Normal/Reinit/Sum.m` | HJ temporal term |
| `integrator` | `Integrators/odeCFL1/2/3.m` + `odeCFLset.m` | TVD Runge-Kutta + CFL step control |
| `integrator` (Post modules) | `Helper/PostTimestep/postTimestepMask/Reinit/TTR.m` | Per-step post-processing hooks |
| `integrator` (TerminalEvent) | `Helper/TerminalEvent/terminalEventConverge.m` | Convergence terminal event |
| `shape` | `InitialConditions/BasicShapes` + `SetOperations` | Implicit shape initial data and set operations |
| `signed_distance` | `Helper/SignedDistance/signedDistanceIterative.m` | Iterative signed distance function |

Not ported: vector level sets (`Examples/Vector`, multi-component cells),
plotting/animation (`Helper/Visualization` and the various `animate*`
examples), and pure-convection shortcuts such as `termConvection` (expressible
equivalently with an LF term plus linear Hamiltonians).

## Protocol mapping (core MATLAB → Rust design decisions)

ToolboxLS's kernel is a web of function-handle protocols; the weakly-typed
`schemeData` struct stuffs grids, scheme choices, and user parameters into
every callback. On the Rust side, **each "protocol role" is materialized as a
trait, and `schemeData` fields become struct fields**:

| MATLAB protocol | Rust counterpart | Notes |
|---|---|---|
| `schemeData.derivFunc` function handle | [`UpwindDerivative`](src/derivative.rs) trait | `[derivL, derivR] = derivFunc(grid, data, dim)` evaluated per dimension, preserved |
| `schemeData.hamFunc` + `partialFunc` | [`Hamiltonian`](src/hamiltonian.rs) trait (two methods) | The two callbacks always appear as a pair; merging avoids implementing half a protocol |
| `schemeData.dissFunc` | [`Dissipation`](src/dissipation.rs) trait | Returns `(diss, stepBound)`, same as MATLAB |
| `schemeFunc(t, y, schemeData)` | [`Term`](src/term.rs) trait | `ydot`/`stepBound` packed into a `TermRhs` struct |
| `hamFunc` mutating `schemeData` in place probed via `nargout` | removed | Dynamics parameters live in fields of the implementing struct; evaluated with `&self`; `Term::rhs(&mut self)` keeps evolvable state |
| Free-form user fields of `schemeData` | own fields of the `Hamiltonian` impl | Where parameters like air3D turning-rate bounds go |
| `grid.bdry{dim}` function handle + `bdryData` | [`BoundaryCondition`](src/grid.rs) enum | A finite set of five; enums are more testable than closures |
| Derived fields `xs`/`shape` of the `grid` struct | `Grid::axis()`/`shape()` methods | No redundant serialization |
| Cell arrays (one entry per dim) | `Vec<ArrayD<f64>>` | Vector level sets (cell of cell) excepted |
| `y(:)` vectorization / `reshape` round trips | `ArrayD<f64>` throughout | Serialized only when crossing to Python |
| MATLAB 1-based dimension indices | 0-based | Uniform across the library |

Data conventions: grid function shape = node counts `n` per dimension; nodes
are cell centers `min + (i + 0.5) * dx` (as in the original,
`dx = (max - min) / n`).

## Implementation and verification status

All four phases are implemented; verification does not rely on MATLAB
reference data but uses two self-contained gates — analytic solutions and
convergence orders (22 test cases all passing: 10 unit + 12 integration):

| Phase | Content | Verification gates (measured) |
|---|---|---|
| 1 | Ghost cells ×5, first-order upwind, `ode_cfl1` | Bit-exact assertions for periodic wrap / constants / slopes / extrapolation; advecting-circle centroid error < 0.02 and area-ratio deviation < 5% (81² grid) |
| 2 | ENO2/ENO3/WENO5, GLF/LLF/LLLF, `termLaxFriedrichs`, `ode_cfl2/3` | Sin-field derivative convergence-order gates at 0.8/1.8/2.5/4.0 (measured 1.00/2.00/3.00/5.00, see `derivative.rs` unit tests); Burgers equation t=0.5 and Hopf–Lax exact solution L∞ error < 0.02 (ENO2+GLF, N=401) with convergence under refinement |
| 3 | `ReinitTerm` (with Russo-Smereka subgrid fix), `signed_distance_iterative`, all shapes and set operations | Reinitialization by 0.3× the distance function returns true distances with max error < 2.5 dx and mean < 0.5 dx (81²); Zalesak disk slit topology preserved |
| 4 | `RestrictUpdateTerm`, `PostTimestepTtrRecorder`, `TerminalEvent`, dual-integrator TTR | Reachable-set boundary misclassification vs analytic solution < 2% (101²); TTR max error < 0.45, visibly reduced at 51→101 resolution; convergence events terminate early |

The TTR's order-of-0.45 magnitude is inherent LF-family dissipation (the dual
integrator's ∂H/∂p is p-independent, making GLF/LLF/LLLF equivalent): the
analytic TTR has gradient magnitude about 2–3, so dissipation lag
∫diss·dt accumulates to exactly this scale; doubling the grid brings it down
to 0.28, consistent with the MATLAB original at equal resolution. The
WENO5/odeCFL3 combination can substantially reduce this error at higher cost.

Two error-prone spots fixed during the port (regression-guarded in tests):
- The third-order mixed-difference scale factor is `Δ²D2 / (3·dx)`, not
  `dx·Δ²D2 / 3` (coincidentally equal on h=1 grids; convergence-order tests
  catch this);
- `termReinit` assembles δ, and the ODE right-hand side must take the sign
  flip (missing it diverges exponentially).

## Integration path

- Python exposure: through e2m2e-integrators' pyo3 cdylib with a new submodule
  (e.g. `_levelset`), ABI stamp incremented in `abi-version.txt`; large arrays
  (100³ grid ≈ 8 MB/component) cross as flat `Vec<f64>` + shape, consistent
  with `_integrators`, without pyo3-numpy.
- Visualization: Python-side matplotlib contours, kept out of Rust.
