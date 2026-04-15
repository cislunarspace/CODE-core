---
title: "Periodic Orbit Correction"
---

# Periodic Orbit Correction: DifferentialCorrection

> **File**: `e2m2e/algorithms/differential_correction.py`

Differential correction linearizes the periodicity conditions and uses Newton-Raphson iteration to converge a rough initial guess into a precise periodic orbit.

## How to Correct to a Precise Periodic Orbit

End-to-end workflow:

```python
import numpy as np
from e2m2e.core import CR3BP_System, CR3BP_Dynamics
from e2m2e.algorithms import DifferentialCorrection

# 1. Initialize
system = CR3BP_System.from_known_system("earth_moon")
system.set_characteristic_scales(distance=384400, period=27.32*86400)
system.compute_libration_points()
dynamics = CR3BP_Dynamics(system)

# 2. Create the corrector with a symmetry configuration
dc = DifferentialCorrection(dynamic=dynamics)
dc.setup_2D_symmetric_x_fixed_x0(x0=0.8)

# 3. Provide an initial guess and iterate
initial_state = np.array([0.8, 0.0, 0.0, 0.0, 0.5, 0.0])
orbit, result = dc.iterate_correction(initial_state, t_half=1.6)

print(f"Converged: {result['converged']}, Iterations: {result['iterations']}, Error: {result['error']:.2e}")
```

## Which Symmetry Configuration to Choose?

Different orbit types require different symmetry configurations. Choosing the wrong one will lead to non-convergence or incorrect orbits.

| What You Want to Do | Which Configuration | Description |
|---------------------|-------------------|-------------|
| Design DRO or Lyapunov (fixed x0) | `setup_2D_symmetric_x_fixed_x0(x0)` | Fix initial x coordinate, solve for vy and half-period |
| Design DRO (fixed period) | `setup_2D_symmetric_x_fixed_t(t_half)` | Fix half-period, solve for x0 and vy |
| Design DRO (fixed y0) | `setup_2D_symmetric_y_fixed_y0(y0)` | Fix initial y coordinate |
| Design Halo (fixed z0 amplitude) | `setup_halo_orbit_fixed_z0(z0, lp)` | Most common Halo configuration |
| Design Halo (fixed x0) | `setup_halo_orbit_fixed_x0(x0, lp)` | Alternative Halo configuration |
| General 3D XZ-symmetric orbit | `setup_3D_symmetric_xz_fixed_x0(x0)` | General-purpose 3D symmetric |

**Selection Tips**:

- **2D vs 3D**: If the orbit lies in the $z=0$ plane (DRO, Lyapunov), use a 2D configuration; if the orbit has a $z$-direction component (Halo), use a 3D configuration.
- **Which parameter to fix**: This depends on which quantity you have a good prior estimate for. For example, if Richardson's approximation gives an estimate for $z_0$, use `fixed_z0`.

## Where to Get Initial Guesses

A good initial guess is key to convergence. Several approaches:

### Richardson Third-Order Approximation (Halo-specific)

```python
from e2m2e.algorithms import compute_halo_initial_guess

initial_state, t_half = compute_halo_initial_guess(
    system=system, libration_point=1, amplitude_z=0.1
)
```

This is an analytical approximation based on the Lindstedt-Poincare method and is very accurate for small-amplitude Halos.

### From an Existing Orbit via Continuation

If you already have a converged orbit, you can use [Continuation](continuation.md) to step along the family curve and obtain an initial guess for a new orbit.

### Manual Construction

For DROs, a typical initial guess structure:

```python
# DRO initial guess: [x0, 0, 0, 0, vy0, 0]
initial_state = np.array([x0, 0.0, 0.0, 0.0, vy_guess, 0.0])
t_half = period_guess  # half-period
```

## What to Do When Convergence Fails

| Symptom | Possible Cause | Remedy |
|---------|---------------|--------|
| Iteration diverges | Initial guess too far from true orbit | Start from a closer known orbit, or use Richardson's approximation |
| Oscillating without converging | Step size too large | Check `max_iter` (default 50), increase iteration count |
| Converges to wrong orbit | Symmetry configuration mismatch | Check that the orbit type matches the `setup_*` method |
| Abnormal Jacobi constant | Insufficient integration accuracy | Confirm `rtol=atol=1e-12`, do not increase step sizes |

## Mathematical Foundation

The periodicity condition requires $\mathbf{x}(T) - \mathbf{x}(0) = \mathbf{0}$. By adding a phase condition $\phi$, the correction equation becomes:

$$\mathbf{F}(\mathbf{x}, \lambda) = \begin{pmatrix} \mathbf{x}(T; \mathbf{x}_0, \lambda) - \mathbf{x}_0 \\ \phi(\mathbf{x}_0, \lambda) \end{pmatrix} = \mathbf{0}$$

Solved via Newton-Raphson iteration:

$$\mathbf{J} \Delta \mathbf{x} = -\mathbf{F}$$

where $\mathbf{J}$ is the Jacobian matrix constructed from the STM (state transition matrix).

## API Quick Reference

| Method | Description |
|--------|-------------|
| `setup_2D_symmetric_x_fixed_x0(x0)` | 2D symmetric, fixed x0 |
| `setup_2D_symmetric_x_fixed_t(t_half)` | 2D symmetric, fixed half-period |
| `setup_2D_symmetric_y_fixed_y0(y0)` | 2D symmetric, fixed y0 |
| `setup_3D_symmetric_x_fixed_x0(x0)` | 3D symmetric, fixed x0 |
| `setup_3D_symmetric_xz_fixed_x0(x0)` | 3D XZ symmetric, fixed x0 |
| `setup_3D_symmetric_xz_fixed_z0(z0)` | 3D XZ symmetric, fixed z0 |
| `setup_halo_orbit_fixed_z0(z0, lp)` | Halo-specific, fixed z0 |
| `setup_halo_orbit_fixed_x0(x0, lp)` | Halo-specific, fixed x0 |
| `iterate_correction(initial_guess, t_half)` | Execute iterative correction, returns `(Orbit, dict)` |

For the complete API documentation, see [API Reference](../reference/api-reference.md).
