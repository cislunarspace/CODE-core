---
title: MultipleShooting - Multiple Shooting Method
---

# MultipleShooting - Multiple Shooting Method

The `MultipleShooting` class implements a multiple shooting numerical corrector suitable for complex constraints and long-period orbit correction.

## Overview

The multiple shooting method divides a trajectory into N nodes and n_seg = N-1 arc segments. After independently integrating each segment, a residual vector is constructed by matching endpoint states of adjacent segments. A least-squares correction is then performed using the Jacobian matrix (which contains the state transition matrices), iterating until the residual satisfies the tolerance.

### Algorithm Features
- **Segmented integration**: Divides long-period orbits into multiple short arcs, improving numerical stability
- **Boundary matching**: Constructs constraint conditions by matching endpoint states of adjacent segments
- **Free-time support**: Optionally treats time nodes as free variables
- **State transition matrix**: Uses STMs to construct the Jacobian matrix, improving convergence speed

## Class Definition

```python
class MultipleShooting:
    """Multiple shooting corrector.

    Divides a trajectory into N nodes and n_seg = N-1 arc segments. After
    independently integrating each segment, a residual vector is constructed
    by matching endpoint states of adjacent segments. A least-squares correction
    is then performed using the Jacobian matrix (containing STMs), iterating
    until the residual satisfies the tolerance.

    When var_time=True, time nodes are also treated as free variables in the
    correction (suitable for free-time problems).
    """
```

## Auxiliary Classes

### `MultipleShootingResult`
Container class for correction results.

**Attributes**:
- `t_patch`: Corrected time node array, shape (N,)
- `state_patch`: Corrected state array, shape (N, 6)
- `converged`: Whether convergence was achieved within the maximum iteration count
- `iterations`: Actual number of iterations
- `max_residual`: Maximum residual in the final iteration
- `residual_history`: History of maximum residuals per iteration

## Main Methods

### `__init__(dynamics)`
Initialize the multiple shooting corrector.

**Parameters**:
- `dynamics`: Dynamics model object, must provide the following interfaces:
  - `propagate(state, time_span, with_stm=True)`: Integrate and propagate, returning a dict containing "states" and "stm"
  - `equations_of_motion(t, state)`: Compute state derivatives (right-hand side function values)

### `correct(t_patch, state_patch, max_iter=100, tol=1e-10, var_time=False, verbose=False)`
Execute multiple shooting correction.

**Parameters**:
- `t_patch`: Initial time node array, shape (N,)
- `state_patch`: Initial state array, shape (N, 6)
- `max_iter`: Maximum number of iterations
- `tol`: Convergence tolerance
- `var_time`: Whether to treat time nodes as free variables
- `verbose`: Whether to print iteration information

**Returns**:
- `MultipleShootingResult`: Correction result

### `compute_residual(t_patch, state_patch)`
Compute the residual vector at the current shooting points.

**Parameters**:
- `t_patch`: Time node array
- `state_patch`: State array

**Returns**:
- `np.ndarray`: Residual vector

### `compute_jacobian(t_patch, state_patch, var_time=False)`
Compute the Jacobian matrix.

**Parameters**:
- `t_patch`: Time node array
- `state_patch`: State array
- `var_time`: Whether to include time variable derivatives

**Returns**:
- `np.ndarray`: Jacobian matrix

## Utility Functions

### `sample_patch_points(orbit, n_segments, method='uniform')`
Uniformly sample shooting points from an orbit.

**Parameters**:
- `orbit`: `Orbit` object
- `n_segments`: Number of arc segments (number of nodes = n_segments + 1)
- `method`: Sampling method, 'uniform' or 'adaptive'

**Returns**:
- `tuple`: (t_patch, state_patch)

### `convert_to_j2000(state_patch, system)`
Convert states to the J2000 inertial frame.

**Parameters**:
- `state_patch`: State array, shape (N, 6)
- `system`: `CR3BP_System` or `EphemerisSystem` object

**Returns**:
- `np.ndarray`: States in the J2000 inertial frame

## Usage Examples

### Basic Usage
```python
from e2m2e.algorithms import MultipleShooting, sample_patch_points
from e2m2e.core import CR3BP_System, CR3BP_Dynamics, Orbit

# Create system and dynamics
system = CR3BP_System.from_known_system("earth_moon")
dynamics = CR3BP_Dynamics(system=system)

# Create initial orbit (example)
initial_state = [0.8, 0, 0, 0, 0.5, 0]
orbit = Orbit(states=[initial_state], times=[0])
orbit.period = 3.0

# Sample shooting points
t_patch, state_patch = sample_patch_points(
    orbit=orbit,
    n_segments=5  # Divide into 5 arc segments
)

# Create multiple shooting corrector
multiple_shooting = MultipleShooting(dynamics=dynamics)

# Execute correction
result = multiple_shooting.correct(
    t_patch=t_patch,
    state_patch=state_patch,
    max_iter=50,
    tol=1e-10,
    var_time=True,  # Allow time nodes to vary
    verbose=True
)

# Check results
if result.converged:
    print(f"Converged in {result.iterations} iterations")
    print(f"Maximum residual: {result.max_residual}")
    print(f"Corrected time nodes: {result.t_patch}")
    print(f"Corrected states: {result.state_patch}")
else:
    print("Did not converge")
    print(f"Final residual: {result.max_residual}")
```

### Combined with Ephemeris Dynamics
```python
from e2m2e.core import EphemerisSystem, EphemerisDynamics
from e2m2e.core.spice import SPICEManager

# Initialize SPICE
spice_manager = SPICEManager()
spice_manager.load_kernels_from_directory("./kernels/")

# Create ephemeris system and dynamics
ephemeris_system = EphemerisSystem(
    bodies=["EARTH", "MOON", "SUN"],
    reference_epoch="2025-06-21T11:00:06"
)
ephemeris_dynamics = EphemerisDynamics(system=ephemeris_system)

# Use multiple shooting to correct ephemeris orbit
multiple_shooting = MultipleShooting(dynamics=ephemeris_dynamics)

# Execute correction (fixed time)
result = multiple_shooting.correct(
    t_patch=t_patch,
    state_patch=state_patch,
    max_iter=100,
    tol=1e-12,
    var_time=False,  # Fixed time nodes
    verbose=True
)
```

### Convert to J2000 Inertial Frame
```python
from e2m2e.algorithms import convert_to_j2000

# Convert corrected states to J2000 inertial frame
state_j2000 = convert_to_j2000(
    state_patch=result.state_patch,
    system=system  # or ephemeris_system
)

print(f"J2000 inertial frame state shape: {state_j2000.shape}")
```

## Algorithm Theory

### 1. Problem Statement
For orbit correction problems, we need to find orbits satisfying boundary conditions. The multiple shooting method discretizes the continuous problem into multiple arc segments.

### 2. Variable Definitions
- Time nodes: $t_0, t_1, \dots, t_N$
- State nodes: $\mathbf{x}_0, \mathbf{x}_1, \dots, \mathbf{x}_N$
- Number of arc segments: $N$ (number of nodes = $N+1$)

### 3. Constraint Conditions
For each arc segment $i$, integrating from $\mathbf{x}_i$ to $\mathbf{x}_{i+1}$ must satisfy the equations of motion:
$$\mathbf{F}_i = \Phi(t_i, t_{i+1}; \mathbf{x}_i) - \mathbf{x}_{i+1} = \mathbf{0}$$
where $\Phi$ is the flow map.

### 4. Jacobian Matrix
The Jacobian matrix is composed of state transition matrices:
$$J = \begin{bmatrix}
\Phi_1 & -I & 0 & \cdots & 0 \\
0 & \Phi_2 & -I & \cdots & 0 \\
\vdots & \vdots & \ddots & \ddots & \vdots \\
0 & 0 & \cdots & \Phi_N & -I
\end{bmatrix}$$
where $\Phi_i$ is the state transition matrix for the $i$-th arc segment.

### 5. Iterative Correction
Using Newton's method for iteration:
$$\Delta \mathbf{X} = -J^{-1} \mathbf{F}$$
$$\mathbf{X}^{(k+1)} = \mathbf{X}^{(k)} + \Delta \mathbf{X}$$

## Parameter Selection Guidelines

### 1. Number of Arc Segments
- **Short-period orbits**: 3-5 arc segments
- **Medium-period orbits**: 5-10 arc segments
- **Long-period orbits**: 10-20 arc segments
- **Very long orbits**: 20+ arc segments

### 2. Convergence Tolerance
- **General precision**: `tol=1e-8`
- **High precision**: `tol=1e-12`
- **Ultra-high precision**: `tol=1e-14`

### 3. Maximum Iteration Count
- **Simple problems**: `max_iter=50`
- **Medium problems**: `max_iter=100`
- **Difficult problems**: `max_iter=200`

### 4. Time Variables
- **Fixed-time problems**: `var_time=False`
- **Free-time problems**: `var_time=True`
- **Mixed problems**: Some times fixed, some free

## Application Scenarios

### 1. Long-Period Orbit Correction
```python
# For orbits with very long periods, single-pass integration may be numerically unstable
# Use multiple shooting for segmented integration
result = multiple_shooting.correct(
    t_patch=t_patch,
    state_patch=state_patch,
    max_iter=100,
    tol=1e-10,
    var_time=True
)
```

### 2. Complex Boundary Conditions
```python
# Orbits that must satisfy multiple intermediate constraints
# Additional constraints can be added at specific nodes
def add_intermediate_constraint(state):
    """Add intermediate point constraint"""
    # Example: require passing through a specific position
    return state[0] - target_x  # x-coordinate constraint
```

## Performance Optimization

### 1. Parallel Computation
```python
# Arc segment integration can be executed in parallel
from concurrent.futures import ThreadPoolExecutor

def integrate_segment(args):
    """Parallel integration of a single arc segment"""
    i, dynamics, state_i, t_span = args
    result = dynamics.propagate(state_i, t_span, with_stm=True)
    return i, result

# Use thread pool for parallel integration
with ThreadPoolExecutor() as executor:
    futures = []
    for i in range(n_segments):
        args = (i, dynamics, state_patch[i], [t_patch[i], t_patch[i+1]])
        futures.append(executor.submit(integrate_segment, args))

    results = [f.result() for f in futures]
```

### 2. Sparse Matrices
For a large number of arc segments, the Jacobian matrix is sparse; sparse matrix storage and solvers can be used.

### 3. Initial Guesses
Good initial guesses can significantly improve convergence speed:
- Use analytical approximations
- Extrapolate from previous step results
- Use low-precision results as high-precision initial guesses

## Common Issues

### 1. Non-Convergence
**Possible causes**:
- Poor initial guess
- Unreasonable arc segment partitioning
- Tolerance set too tight
- Overly complex dynamics model

**Solutions**:
- Improve initial guess
- Adjust number of arc segments
- Relax tolerance, then gradually tighten
- Use homotopy to simplify the problem

### 2. Numerical Instability
**Possible causes**:
- Large differences in arc segment lengths
- Inappropriate integrator tolerance
- State transition matrix computation errors

**Solutions**:
- Uniformly partition time nodes
- Adjust integrator parameters
- Use a higher-precision integrator

### 3. Insufficient Memory
**Possible causes**:
- Too many arc segments
- High state dimensionality
- Storing complete history data

**Solutions**:
- Reduce number of arc segments
- Use sparse matrices
- Store only essential data

## Related Algorithms

- [`DifferentialCorrection`](differential_correction.md): Differential correction method (single-pass integration)
- [`Continuation`](continuation.md): Orbit family continuation
- [`StabilityAnalysis`](stability.md): Stability analysis

## References

1. Stoer, J., & Bulirsch, R. (2002). Introduction to Numerical Analysis. Springer.
2. Betts, J. T. (2010). Practical Methods for Optimal Control and Estimation Using Nonlinear Programming. SIAM.
3. Ascher, U. M., & Petzold, L. R. (1998). Computer Methods for Ordinary Differential Equations and Differential-Algebraic Equations. SIAM.
