# CR3BP_System

**File**: `e2m2e/core/system.py`

**Class Signature**:
```python
class CR3BP_System:
    """Circular Restricted Three-Body Problem system parameters"""
```

## Design Principles

The `CR3BP_System` class encapsulates system parameters for the Circular Restricted Three-Body Problem. In the CR3BP model:
- Two massive bodies (primary $m_1$ and secondary $m_2$) move in circular orbits around their common center of mass under mutual gravitational attraction
- A small mass body (spacecraft) moves in the gravitational field of the two massive bodies, without affecting their motion

The mass parameter is defined as:
$$\mu = \frac{m_2}{m_1 + m_2}$$

For the Earth-Moon system, $\mu \approx 0.01215$

## Mathematical Foundation

### Libration Points (Lagrange Points) Calculation
Libration points are special points that remain stationary relative to the two massive bodies:
$$\nabla U(\mathbf{r}) = 0$$

Where $U$ is the effective potential function:
$$U = \frac{x^2 + y^2}{2} + \frac{1-\mu}{r_1} + \frac{\mu}{r_2}$$

### Jacobi Constant
$$C = 2U - v^2 = x^2 + y^2 + \frac{2(1-\mu)}{r_1} + \frac{2\mu}{r_2} - (v_x^2 + v_y^2 + v_z^2)$$

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `mu` | `float` | Mass parameter $\mu = m_2/(m_1+m_2)$ |
| `primary_body` | `str` | Primary body name |
| `secondary_body` | `str` | Secondary body name |
| `L1-L5` | `np.ndarray` | Coordinates of five libration points |
| `characteristic_length` | `float` | Characteristic length (distance between two bodies) |
| `characteristic_time` | `float` | Characteristic time |
| `characteristic_velocity` | `float` | Characteristic velocity |

## Core Methods

| Method | Description |
|--------|-------------|
| `compute_libration_points()` | Compute positions of five libration points |
| `get_libration_point(point)` | Get coordinates of specified libration point |
| `get_jacobi_constant(state)` | Calculate Jacobi constant |
| `dimensionless_to_physical(state)` | Dimensionless → Physical units |
| `physical_to_dimensionless(state)` | Physical units → Dimensionless |
| `compute_stability_index(L_point)` | Compute stability index of libration point |

## Usage Example

```python
from e2m2e.core.system import CR3BP_System, LibrationPoint

# Create from known system
system = CR3BP_System.from_known_system("earth_moon")
system.set_characteristic_scales(distance=384400, period=27.32*86400)
system.compute_libration_points()

# Get libration point
L1 = system.get_libration_point(LibrationPoint.L1)
print(f"L1 position: {L1}")

# Calculate Jacobi constant
state = np.array([0.8, 0, 0, 0, 1.5, 0])
C = system.get_jacobi_constant(state)
```
