---
title: 'EphemerisDynamics - Ephemeris Dynamics'
---

# EphemerisDynamics - Ephemeris Dynamics

The `EphemerisDynamics` class inherits from the `Dynamics` base class and provides precise multi-body gravitational computation based on NASA SPICE kernels.

## Class Definition

```python
class EphemerisDynamics(Dynamics):
    """Ephemeris dynamics based on SPICE kernels
    
    Computes gravitational acceleration from multiple bodies on a spacecraft,
    considering actual non-spherical gravity and precise ephemeris positions.
    
    Args:
        system: EphemerisSystem object defining the bodies participating in computation
    """
```

## Inheritance Hierarchy

```
Dynamics (base class)
    +-- CR3BP_Dynamics (CR3BP dynamics)
    +-- EphemerisDynamics (ephemeris dynamics)
```

## Main Methods

### `__init__(system)`
Initialize ephemeris dynamics.

**Parameters**:
- `system`: `EphemerisSystem` object

### `equations_of_motion(t, state)`
Compute the state derivatives (right-hand side function values).

**Parameters**:
- `t`: Time (seconds), relative to the reference epoch
- `state`: 6D state vector [x, y, z, vx, vy, vz] (km, km/s)

**Returns**:
- `np.ndarray`: 6D state derivatives [vx, vy, vz, ax, ay, az]

**Physical Model**:
The acceleration is computed as:
$$a(r, t) = \sum_{i=1}^{N} \mu_i \frac{r_i - r}{\|r_i - r\|^3}$$
where:
- $r$ is the spacecraft position
- $r_i$ is the position of the $i$-th body
- $\mu_i$ is the gravitational parameter of the $i$-th body

### `propagate(initial_state, time_span, **kwargs)`
Propagate an orbit.

**Parameters**:
- `initial_state`: Initial state vector
- `time_span`: Time interval [t_start, t_end] or list of integration times
- `**kwargs`: Additional parameters passed to `solve_ivp`

**Returns**:
- `dict`: A dictionary containing the following keys:
  - `states`: State history, shape (n_times, 6)
  - `times`: Array of time points
  - `stm`: State transition matrix history (if requested)

## Usage Examples

### Basic Usage
```python
from e2m2e.core import EphemerisSystem, EphemerisDynamics
from e2m2e.core.spice import SPICEManager

# Initialize SPICE
spice_manager = SPICEManager()
spice_manager.load_kernels_from_directory("./kernels/")

# Create system
system = EphemerisSystem(
    bodies=["EARTH", "MOON", "SUN"],
    reference_epoch="2025-06-21T11:00:06"
)

# Create dynamics
dynamics = EphemerisDynamics(system=system)

# Define initial state (near Earth-Moon L2 point)
initial_state = [1.1556 * 384400, 0, 0, 0, 1.023, 0]  # km, km/s

# Propagate orbit
result = dynamics.propagate(
    initial_state=initial_state,
    time_span=[0, 7 * 86400],  # Propagate 7 days
    method='DOP853',
    rtol=1e-12,
    atol=1e-14
)

# Retrieve results
states = result["states"]
times = result["times"]
```

### Computing the State Transition Matrix
```python
# Propagate and compute STM
result_with_stm = dynamics.propagate(
    initial_state=initial_state,
    time_span=[0, 86400],
    with_stm=True
)

stm_history = result_with_stm["stm"]  # Shape (n_times, 6, 6)
```

### Using with Algorithms
```python
from e2m2e.algorithms import DifferentialCorrection

# Create differential corrector
corrector = DifferentialCorrection(dynamic=dynamics)

# Configure symmetry (example)
corrector.setup_2D_symmetric_x_fixed_x0(x0=1.1556)

# Execute correction
initial_guess = Orbit(states=[initial_state], times=[0])
initial_guess.period = 6.8 * 86400  # Estimated period (seconds)

corrected_orbit = corrector.iterate_correction(initial_guess=initial_guess)
```

## Performance Considerations

1. **Computational Cost**: Ephemeris dynamics computation is more expensive than CR3BP because it requires:
   - Querying precise positions of each body
   - Computing gravitational acceleration from each body
   - Processing non-spherical gravity terms (if supported by kernels)

2. **Integrator Selection**: High-order integrators are recommended:
   - `'DOP853'`: 8th-order explicit Runge-Kutta, suitable for high-accuracy requirements
   - `'RK45'`: 5th-order Runge-Kutta, balancing accuracy and speed

3. **Tolerance Settings**: Smaller tolerances are recommended:
   - `rtol=1e-12`: Relative tolerance
   - `atol=1e-14`: Absolute tolerance

## Important Notes

1. **SPICE Kernels**: Required kernel files must be loaded beforehand
2. **Time System**: Uses Ephemeris Time (ET); be mindful of UTC conversion
3. **Unit Consistency**: Ensure all inputs use consistent units (km, km/s)
4. **Memory Usage**: Long-duration propagation may generate large amounts of data; manage memory accordingly

## Related Classes

- [`EphemerisSystem`](ephemeris_system.md): Ephemeris system definition
- [`CR3BP_Dynamics`](dynamics.md): CR3BP dynamics (simplified model)
- [`Dynamics`](dynamics.md): Dynamics base class
