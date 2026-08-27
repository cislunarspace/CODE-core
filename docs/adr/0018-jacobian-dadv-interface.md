# ADR 0018: Jacobian interface extended with ∂a/∂v; STM covers velocity dependence

**Status**: Adopted
**Date**: 2026-08-07
**Related**: ADR 0002 (Rust integrator core), ADR 0003 (ITRF93 defaults),
ADR 0017 (grid search Rayon), issue #317
**Related code**: `crates/e2m2e-forces/src/forces/{compiled,nbody_stm,augmented_state,compiled_stm}.rs`,
`crates/e2m2e-integrators/src/multiple_shooting.rs`

## Context

The state transition matrix (STM) satisfies the variational equation
``Φ̇ = A·Φ`` where

```text
A = | 0₃ₓ₃   I₃ₓ₃ |
    | ∂a/∂r  ∂a/∂v |
```

The lower-left block ``∂a/∂r`` (acceleration vs position) sums per-force
Jacobians; the lower-right block ``∂a/∂v`` (acceleration vs velocity) is zero
for pure N-body gravity: all body gravities depend only on position. This
repo's STM propagation (CR3BP / EphemerisDynamics / N-body STM / compiled STM /
multiple shooting) long rested on that assumption.

Atmospheric drag breaks it: ``a_drag = −½·ρ(|r|)·BC·|v|·v`` depends on both
position (via density ρ) and velocity (relative velocity itself). Once drag
enters STM propagation, ``A``'s ``∂a/∂v`` block is nonzero, and the old
implicit ``∂a/∂v = 0`` assumption makes the STM silently wrong: no error, just
wrong sensitivities.

The drag Rust port (issue #315 series) brings ``DragModel`` into compiled force
models — the first velocity-dependent force on the STM path. This forces the
force-Jacobian interface to extend from returning ``∂a/∂r`` to returning
``(∂a/∂r, ∂a/∂v)``.

## Decision

Extend the Rust force-model Jacobian interface's return from the pair
``(acc, ∂a/∂r)`` to a **triple** ``(acc, ∂a/∂r, ∂a/∂v)``, threading ``∂a/∂v``
through the whole chain into the variational equation.

1. **`acceleration_and_jacobian` returns the triple.**
   - Type alias ``AccelJacobiResult = Result<([f64;3], [[f64;3];3], [[f64;3];3]), String>``
     (compiled.rs).
   - Each `CompiledForce` variant supplies `∂a/∂v` under its own semantics:
     - `PointMass` / `ThirdBody` / `IndirectTerm` / `GravityField` / `SRP`:
       analytic ``[[0.0;3];3]`` (position-type forces, velocity-independent).
     - `Drag`: central-difference FD homologous with its `∂a/∂r`, perturbing
       all 6 J2000 components (12 accel evaluations); `∂a/∂v` is true-valued.
   - `compute_total_acceleration_and_jacobian` accumulates `total_dadv`
     across forces (isomorphic to `total_jac`).

2. **`stm_derivative` accepts `dadv`.**
   - Signature ``stm_derivative(stm, jac_da_dr, dadv) -> [f64;36]``;
     bottom 3 rows of ``A`` use ``∂a/∂r·Φ[:3] + ∂a/∂v·Φ[3:]``.
   - `nbody_stm`, `augmented_state`, `compiled_stm`, and `multiple_shooting`
     all update their call signatures accordingly.

3. **N-body path has zero `dadv`.**
   - `compute_nbody_acceleration_and_jacobian` returns
     ``dadv = [[0.0;3];3]`` (N-body gravity is velocity-independent). The
     path's behavior is unchanged; the triple merely writes out the implicit
     zero.

4. **Python-side `ForceModel` STM fallback synced (issue #317 item 2.1).**
   - `_compute_total_jacobian` returns ``(∂a/∂r, ∂a/∂v)``;
     `_eom_func_with_stm` assembles ``A[3:,3:] = ∂a/∂v``.
   - Analytic-Jacobian forces (`compute_jacobian` not None): `∂a/∂v = 0`;
     forces without analytic Jacobians take finite differences perturbing both
     position and velocity for true values.
   - This eliminates the future hazard of velocity-dependent forces silently
     erring without SPICE. Issue #378 later removed the Python fallback
     wholesale (no more Python RHS fallbacks, ADR 0020); this clause's code was
     removed with it.

## Impact surface

Extending the interface to triples touches all STM-related Rust modules; ABI
unchanged (internal signatures, not pyfunction boundaries):

| Module | Change |
|---|---|
| `compiled.rs` | `AccelJacobiResult` alias; triple returns from `acceleration_and_jacobian` / `compute_total_acceleration_and_jacobian` |
| `nbody_stm.rs` | `compute_nbody_acceleration_and_jacobian` returns constant-zero `dadv`; `stm_derivative` gains `dadv` parameter |
| `augmented_state.rs` | 42-dim augmented RHS forwards `dadv` to `stm_derivative` |
| `compiled_stm.rs` | `augmented_eom` destructures triple, forwards `dadv` |
| `multiple_shooting.rs` | STM shooting segments forward `dadv` |
| `e2m2e/algorithm/forces/force_model.py` | Python fallback sync (item 2.1; removed later by issue #378) |

## Precedents and positioning

- **More fundamental than ADR 0017.** 0017 is a parallel-execution strategy
  for grid search (Rayon), touching one algorithm path; this ADR is the
  force-Jacobian contract underpinning every STM propagation (shooting,
  correction, STM sampling, grid-search integration). 0017's cited
  precedents are themselves consumers of this contract.
- **Relation to ADR 0002.** Its revision 2 established SPICE-related
  propagation compiling into the Rust extension (cspice pool singleton). This
  ADR doesn't move that boundary; it refines the in-Rust data contract from two
  matrices to three — internal to ADR 0002's kernel, no seam crossing.
- **Relation to ADR 0003.** drag's nonzero `∂a/∂v` stems from its physics
  (relative velocity), frame-independent; the coordinate side's ITRF93 choice
  is recorded separately in ADR 0019.

## Why not other shapes

- **Not struct fields.** The ``(acc, ∂a/∂r, ∂a/∂v)`` triple matches GMAT's
  `CompleteDerivativeCalculations` division: forces supply Ã's lower-left
  block only, here refined from ``3×3`` into ``(∂a/∂r, ∂a/∂v)``. A struct
  (``AccelJacobi { acc, dadr, dadv }``) reads slightly better but touches all
  destructure points; the triple aligns with existing `AccelDrag` /
  `AccelJacobiResult` style, minimal churn.
- **Not optional `∂a/∂v`.** An optional ``Option<[[f64;3];3]>`` would let
  velocity-dependent forces omit filling it — recreating exactly the silent-
  error mode this ADR kills. Mandatory triples fail compilation when missing.

## Consequences

### Added

- Force-Jacobian interface explicitly includes `∂a/∂v`; the STM variational
  equation handles velocity-dependent forces correctly.
- Drag entering compiled-STM paths no longer corrupts sensitivities silently.
- Python `ForceModel` fallback sync (item 2.1), removing a future hazard
  sans-SPICE (path removed by #378).

### Unchanged

- N-body / CR3BP / EphemerisDynamics STM behavior (`dadv` identically zero —
  equivalent to old implicit zero).
- pyfunction ABI (`propagate_compiled_stm_py` etc.; argument/return shapes
  unchanged).
- `PhysicalModel.compute_jacobian` still returns only `∂a/∂r` (3×3); Python
  analytic-Jacobian contract unchanged; `∂a/∂v` supplied at `ForceModel` level
  as zeros for analytic forces / FD otherwise. (That Python contract was later
  removed by issue #378.)

### Trade-offs

- **FD evaluation count doubles (drag).** `drag_accel_and_jacobian` grows from
  perturbing 3 position components (6 evals) to all 6 (12 evals). Drag FD was
  already the heaviest single force on the STM path, but STM cost is dominated
  by RK stepping rather than per-call Jacobians; drag magnitudes are small
  (fast altitude decay) — imperceptible in practice.
- **Python FD path cost.** Forces without analytic Jacobians expand Python-FD
  from 6 evals (position) to 12 (position+velocity). That was a degraded path,
  drag unreachable there; sole consumer then was spherical-harmonics
  `GravityField` (position-type; velocity FD gave zeros), so the extra 6 evals
  were acceptable. (Path removed by issue #378.)
