---
title: 'System Definition: CR3BP_System'
---

# System Definition: CR3BP_System

> **File**: `e2m2e/core/system.py`

`CR3BP_System` encapsulates the system parameters of the Circular Restricted Three-Body Problem (CR3BP) — mass parameter, characteristic scales, libration point locations, and Jacobi constant. It is the first step when using e2m2e: all dynamics models, orbit algorithms, and visualization require a system object.

## How to Create a Celestial System

### Built-in Systems (Recommended)

```python
from e2m2e.core import CR3BP_System

system = CR3BP_System.from_known_system("earth_moon")
# Other options: "sun_earth", "sun_jupiter"
```

`from_known_system` automatically sets the mass parameter $\mu$ and body names.

### Custom System

```python
system = CR3BP_System(mu=0.01215, primary_body="Earth", secondary_body="Moon")
```

### Setting Characteristic Scales

To convert between dimensionless and physical units, you must first set the characteristic scales:

```python
system.set_characteristic_scales(
    distance=384400,       # Earth-Moon distance (km)
    period=27.32 * 86400,  # Moon orbital period (s)
)
```

After setting, you can use `dimensionless_to_physical()` and `physical_to_dimensionless()` for unit conversion.

## How to Use System Information

### Computing Libration Points

```python
system.compute_libration_points()

# Get L1 position
L1 = system.L1  # or system.get_libration_point(LibrationPoint.L1)
print(f"L1: {L1}")
```

The five libration points `L1` through `L5` are available as attributes after computation.

### Computing the Jacobi Constant

The Jacobi constant is a conserved quantity in the CR3BP, used to measure the energy level of an orbit:

```python
import numpy as np

state = np.array([0.8, 0.1, 0.0, 0.0, 0.2, 0.0])
C = system.get_jacobi_constant(state)
print(f"Jacobi constant: {C:.6f}")
```

A larger Jacobi constant corresponds to lower orbital energy. The Jacobi constants at the libration points are critical thresholds that distinguish different regions of motion.

### Unit Conversion

```python
# Dimensionless → Physical (km, km/s)
physical = system.dimensionless_to_physical(state)

# Physical → Dimensionless
dimensionless = system.physical_to_dimensionless(physical)
```

## API Quick Reference

| Method | Description |
|--------|-------------|
| `from_known_system(name)` | Create a built-in system ("earth_moon" / "sun_earth" / "sun_jupiter") |
| `set_characteristic_scales(distance, period)` | Set characteristic scales (prerequisite for physical unit conversion) |
| `compute_libration_points()` | Compute the five libration point locations |
| `get_libration_point(point)` | Get the coordinates of a specified libration point |
| `get_jacobi_constant(state)` | Compute the Jacobi constant |
| `dimensionless_to_physical(state)` | Dimensionless → physical units |
| `physical_to_dimensionless(state)` | Physical units → dimensionless |
| `compute_stability_index(L_point)` | Compute the linearized stability index at a libration point |
| `info(mode)` | Print system information |

For the full API documentation, see [API Reference](../reference/api-reference_en.md).

## Mathematical Background

### Mass Parameter

$$\mu = \frac{m_2}{m_1 + m_2}$$

For the Earth-Moon system, $\mu \approx 0.01215$.

### Libration Points

Libration points satisfy the condition that the gradient of the effective potential function is zero: $\nabla U(\mathbf{r}) = 0$, where

$$U = \frac{x^2 + y^2}{2} + \frac{1-\mu}{r_1} + \frac{\mu}{r_2}$$

L1, L2, L3 lie on the $x$-axis (collinear libration points), while L4 and L5 form an equilateral triangle.

### Jacobi Constant

$$C = 2U - v^2 = x^2 + y^2 + \frac{2(1-\mu)}{r_1} + \frac{2\mu}{r_2} - (v_x^2 + v_y^2 + v_z^2)$$

In the dimensionless rotating frame, the Jacobi constant is conserved along the trajectory.
