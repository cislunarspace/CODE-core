# Orbit & OrbitFamily

**File**: `e2m2e/core/orbit.py`

## Design Principles

The `Orbit` class encapsulates orbit data management, including period detection and stability analysis. The `OrbitFamily` class manages a family of orbit data.

## Orbit Class

### Core Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `states` | `np.ndarray` | Orbit state sequence [n, 6] |
| `times` | `np.ndarray` | Corresponding time sequence |
| `period` | `float` | Orbit period (detected) |
| `stability_index` | `float` | Stability index |
| `jacobi_constant` | `float` | Jacobi constant |

### Core Methods

| Method | Description |
|--------|-------------|
| `detect_period()` | Detect orbit period |
| `interpolate_at_time(t)` | Time interpolation to get state |
| `compute_stability()` | Compute Floquet multipliers |
| `compute_jacobi()` | Calculate Jacobi constant |

## OrbitFamily Class

### Core Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `orbits` | `List[Orbit]` | List of orbits |
| `parameter_name` | `str` | Parameter name (e.g., "x0", "mu") |
| `parameter_values` | `List[float]` | Parameter value sequence |

### Core Methods

| Method | Description |
|--------|-------------|
| `add_orbit(orbit, param_value)` | Add orbit to family |
| `get_orbit_at(param_value)` | Get orbit at specified parameter |
| `filter_by_stability(stability_type)` | Filter by stability |

## Usage Example

```python
from e2m2e.core.orbit import Orbit, OrbitFamily

# Create orbit
orbit = Orbit(states=states, times=times)
orbit.detect_period()

# Create orbit family
family = OrbitFamily(parameter_name="x0")
family.add_orbit(orbit, param_value=0.8)
```
