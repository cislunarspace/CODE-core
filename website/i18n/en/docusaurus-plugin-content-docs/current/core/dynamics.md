---
title: 'Dynamics Integration: CR3BP_Dynamics'
---

# Dynamics Integration: CR3BP_Dynamics

> **File**: `e2m2e/core/dynamics.py`

`CR3BP_Dynamics` implements the equations of motion and numerical integration for the CR3BP. It is the core engine for all orbital computations — differential correction, orbit propagation, and state transition matrix calculations all depend on it.

## How to Create a Dynamics Model

```python
from e2m2e.core import CR3BP_System, CR3BP_Dynamics

system = CR3BP_System.from_known_system("earth_moon")
dynamics = CR3BP_Dynamics(system)
```

A `CR3BP_Dynamics` object is bound to a `CR3BP_System`, and all subsequent propagations use the parameters of that system.

## How to Propagate an Orbit

### Basic Propagation

```python
import numpy as np

initial_state = np.array([0.8, 0.0, 0.0, 0.0, 1.0, 0.0])
result = dynamics.propagate(initial_state, t_span=(0, 10.0))

states = result["states"]       # [n, 6] state array
times = result["times"]         # [n] time array
jacobi = result["jacobi"]       # [n] Jacobi constant array
```

The integrator uses scipy `solve_ivp` with precision `rtol=atol=1e-12`.

### Propagation with STM

When you need to compute the State Transition Matrix (required by algorithms such as differential correction and stability analysis), set `with_stm=True`:

```python
result = dynamics.propagate(initial_state, t_span=(0, 10.0), with_stm=True)

stm = result["stm"]  # [n, 6, 6] state transition matrix array
```

This simultaneously integrates the 42-dimensional augmented state (6-dimensional state + 36-dimensional STM components).

### Monitoring the Jacobi Constant

The Jacobi constant is a conserved quantity in the CR3BP. If the Jacobi constant changes significantly during propagation, it indicates insufficient integration accuracy:

```python
result = dynamics.propagate(initial_state, t_span=(0, 100.0))
jacobi_drift = abs(result["jacobi"][-1] - result["jacobi"][0])
print(f"Jacobi drift: {jacobi_drift:.2e}")
# Should be < 1e-10 under normal conditions
```

## How to Compute the State Transition Matrix

Compute the STM at a specific time independently:

```python
stm = dynamics.compute_state_transition_matrix(initial_state, t=5.0)
print(stm.shape)  # (6, 6)
```

The STM is the core tool for differential correction — it tells you "how small changes in the initial state affect the final state."

-> See [Differential Correction - How to Correct Precise Periodic Orbits](../algorithms/differential_correction_en.md)

## API Quick Reference

| Method | Description |
|--------|-------------|
| `propagate(initial_state, t_span, with_stm, with_jacobi)` | Integrate an orbit (core method) |
| `compute_state_transition_matrix(initial_state, t)` | Compute the 6x6 state transition matrix |
| `equations_of_motion(t, state)` | 6-dimensional equations of motion (overridable) |
| `equations_with_stm(t, augmented_state)` | 42-dimensional augmented equations |
| `compute_jacobi_constant(state)` | Compute the Jacobi constant |
| `check_cross_section(state, plane, value)` | Detect Poincare section crossings |

For the full API documentation, see [API Reference](../reference/api-reference_en.md).

## Equations of Motion

The dimensionless equations of motion for the CR3BP in the rotating frame:

$$\begin{cases}
\dot{x} = v_x \\
\dot{y} = v_y \\
\dot{z} = v_z \\
\dot{v}_x = 2v_y + x - \dfrac{(1-\mu)(x+\mu)}{r_1^3} - \dfrac{\mu(x-1+\mu)}{r_2^3} \\
\dot{v}_y = -2v_x + y - \dfrac{(1-\mu)y}{r_1^3} - \dfrac{\mu y}{r_2^3} \\
\dot{v}_z = -\dfrac{(1-\mu)z}{r_1^3} - \dfrac{\mu z}{r_2^3}
\end{cases}$$

where $r_1 = \sqrt{(x+\mu)^2 + y^2 + z^2}$ and $r_2 = \sqrt{(x-1+\mu)^2 + y^2 + z^2}$.

The acceleration terms include: Coriolis force ($\pm 2v$), centrifugal force ($x, y$), and gravitational attraction from both bodies.
