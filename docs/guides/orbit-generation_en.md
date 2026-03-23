# Orbit Generation Guide

## Overview

This guide introduces how to generate various periodic orbits using E2M2E: DRO, Halo, Lyapunov, etc.

## General Process

The general steps for all orbit generation:

```python
import numpy as np
from e2m2e.core.system import CR3BP_System
from e2m2e.core.dynamics import CR3BP_Dynamics
from e2m2e.algorithms.differential_correction import DifferentialCorrection

# 1. Initialize system
system = CR3BP_System.from_known_system("earth_moon")
system.set_characteristic_scales(distance=384400, period=27.32*86400)
system.compute_libration_points()

# 2. Create dynamics model
dynamics = CR3BP_Dynamics(system)

# 3. Create differential corrector
dc = DifferentialCorrection(dynamics)

# 4. Configure and solve
dc.setup_2D_symmetric_x_fixed_x0(x0=0.8)  # or other configurations
orbit, result = dc.iterate_correction(initial_guess, t_half_guess)
```

---

## Distant Retrograde Orbit (DRO)

### Theoretical Background

DRO is a large-amplitude retrograde orbit around the Moon, appearing as a teardrop shape in the rotating frame. Its characteristics:
- Symmetric about x-axis
- Initial conditions: $[x_0, 0, 0, 0, \dot{y}_0, 0]$
- Half-period conditions: $y=0$, $\dot{x}=0$

### Generation Steps

```python
# Configure 2D symmetric x-axis fixed x0
dc.setup_2D_symmetric_x_fixed_x0(x0=0.8)

# Initial guess (near x0=0.8)
initial_state = np.array([0.8, 0.0, 0.0, 0.0, 0.5, 0.0])
t_half_guess = 1.6  # Half-period guess

# Iterate to correct
orbit, result = dc.iterate_correction(initial_state, t_half_guess, verbose=True)
```

### Parameter Ranges

| Parameter | Typical Range | Description |
|-----------|---------------|-------------|
| $x_0$ | 0.6 - 0.95 | Initial x coordinate (relative to L1/L2) |
| $\dot{y}_0$ | 0.3 - 0.8 | Initial y-direction velocity |
| $T/2$ | 1.4 - 1.8 | Half-period (dimensionless) |

### Complete Example

```python
def generate_dro(x0=0.8, y_dot_guess=0.5, t_half_guess=1.6):
    """Generate DRO orbit"""
    system = CR3BP_System.from_known_system("earth_moon")
    system.compute_libration_points()
    dynamics = CR3BP_Dynamics(system)
    dc = DifferentialCorrection(dynamics)
    
    dc.setup_2D_symmetric_x_fixed_x0(x0=x0)
    initial_state = np.array([x0, 0.0, 0.0, 0.0, y_dot_guess, 0.0])
    
    orbit, result = dc.iterate_correction(initial_state, t_half_guess)
    return orbit, result, system
```

---

## Halo Orbit

### Theoretical Background

Halo orbits are three-dimensional periodic orbits around libration points (L1 or L2), appearing as distorted "figure-8" or horseshoe shapes.

### 3D Symmetric Configuration

```python
# Configure 3D symmetric x-axis fixed x0
dc.setup_3D_symmetric_x_fixed_x0(x0=0.8)

# Initial guess (with z-direction component)
initial_state = np.array([0.8, 0.0, 0.1, 0.0, 0.5, 0.0])
t_half_guess = 1.6

# Iterate to correct
orbit, result = dc.iterate_correction(initial_state, t_half_guess, verbose=True)
```

### L1 vs L2 Halo

| Property | L1 Halo | L2 Halo |
|----------|---------|---------|
| Location | Near L1 point | Near L2 point |
| $x_0$ range | $0.8 < x_0 < 1.0$ | $1.0 < x_0 < 1.2$ |
| Amplitude | Usually smaller | Usually larger |

```python
# L1 Halo
L1_x0 = system.L1[0] + 0.01  # Right of L1
dc.setup_3D_symmetric_x_fixed_x0(x0=L1_x0)

# L2 Halo
L2_x0 = system.L2[0] - 0.01  # Left of L2
dc.setup_3D_symmetric_x_fixed_x0(x0=L2_x0)
```

---

## Lyapunov Orbit

### Theoretical Background

Lyapunov orbits are two-dimensional periodic orbits in the libration point plane (z=0), appearing as elliptical or banana-shaped.

### Configuration

```python
# Use 2D symmetric configuration
dc.setup_2D_symmetric_x_fixed_x0(x0=0.8)

# Initial guess
initial_state = np.array([0.8, 0.0, 0.0, 0.0, 0.3, 0.0])
t_half_guess = 1.5
```

### Difference from DRO

| Property | Lyapunov | DRO |
|----------|----------|-----|
| Location | Near L1/L2 | Near Moon |
| z amplitude | 0 | > 0 (3D) |
| Period | Shorter | Longer |

---

## Orbit Family Continuation

### Natural Parameter Continuation

```python
from e2m2e.algorithms.continuation import Continuation

# Create continuer
continuation = Continuation(dc, step=0.005)

# Continue from seed orbit
family = continuation.natural_continuation(
    seed_orbit=seed_orbit,
    param_range=(0.6, 0.95),  # x0 range
    step_size=0.005,
    verbose=True
)
```

### Bidirectional Continuation

```python
# Forward continuation (increasing x0)
family_forward = continuation.natural_continuation(
    seed_orbit=seed_orbit,
    param_range=(seed_x0, 0.95),
    step_size=0.005
)

# Backward continuation (decreasing x0)
family_backward = continuation.natural_continuation(
    seed_orbit=seed_orbit,
    param_range=(seed_x0, 0.6),
    step_size=-0.005  # Negative step size
)
```

### Pseudo-Arclength Continuation (Bypassing Turning Points)

```python
# Use when natural continuation fails at turning points
family = continuation.pseudo_arclength_continuation(
    seed_state=seed_state,
    seed_t_half=seed_t_half,
    n_orbits=100,
    verbose=True
)
```

---

## Save and Load

### Saving Orbit Family

```python
# Save to JSON file
family.save_to_file("output/dro_family.json")

# Or single orbit
orbit.save_to_file("output/dro_single.json")
```

### Loading Orbit Family

```python
from e2m2e.core.orbit import OrbitFamily, Orbit

# Load orbit family
family = OrbitFamily.load_from_file("output/dro_family.json")

# Load single orbit
orbit = Orbit.load_from_file("output/dro_single.json")
```

---

## Frequently Asked Questions

### 1. How to Determine Initial Guess?

**Rules of thumb**:
- $x_0$: Start from target region center, e.g., L1+0.01
- $\dot{y}_0$: Start from 0.5, adjust based on error
- $T/2$: Start from 1.5, adjust based on period estimate

**Debugging tips**:
- First propagate the initial guess to check if it's close to periodic
- Observe final errors in $y$ and $\dot{x}$

### 2. What if Iteration Doesn't Converge?

**Check items**:
1. Is the initial guess in a reasonable range?
2. Is the integrator tolerance sufficient (currently 1e-12)?
3. Try different initial guesses

**Adaptive damping**:
The algorithm has built-in adaptive damping; if it still doesn't converge, manually reduce the step size.

### 3. How to Generate Large-Amplitude Orbits?

**Strategy**:
1. Start from small-amplitude orbits
2. Use pseudo-arclength continuation to pass through turning points
3. Gradually increase amplitude

---

## References

- See [Technical Documentation - Differential Correction](e2m2e_technical_documentation.md#21-differentialcorrection)
- See [CR3BP Algorithms - Orbit Family Continuation](cr3bp_algorithms.md#5-orbit-family-continuation-algorithm)
