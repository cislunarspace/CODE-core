---
title: CoordinateTransformation & ReferenceFrame
---

# CoordinateTransformation & ReferenceFrame

**File**: `e2m2e/core/coordinate.py`

## Design Principles

Provides transformations between rotating coordinate system (synodic frame) and inertial coordinate system.

## Rotating ↔ Inertial Transformation

### Rotating to Inertial

```python
def rotating_to_inertial(state_rotating: np.ndarray, theta: float) -> np.ndarray:
    """
    Transform rotating frame state to inertial frame
    
    Parameters:
        state_rotating: Rotating frame state [x, y, z, vx, vy, vz]
        theta: Rotation angle (rad)
    
    Returns:
        Inertial frame state
    """
    # Position transformation
    x_rot = state_rotating[0]
    y_rot = state_rotating[1]
    x_inert = x_rot * np.cos(theta) - y_rot * np.sin(theta)
    y_inert = x_rot * np.sin(theta) + y_rot * np.cos(theta)
    z_inert = state_rotating[2]
    
    # Velocity transformation
    vx_rot = state_rotating[3]
    vy_rot = state_rotating[4]
    vx_inert = vx_rot * np.cos(theta) - vy_rot * np.sin(theta) - y_inert
    vy_inert = vx_rot * np.sin(theta) + vy_rot * np.cos(theta) + x_inert
    vz_inert = state_rotating[5]
    
    return np.array([x_inert, y_inert, z_inert, vx_inert, vy_inert, vz_inert])
```

## Core Methods

| Method | Description |
|--------|-------------|
| `rotating_to_inertial(state, theta)` | Rotating → Inertial |
| `inertial_to_rotating(state, theta)` | Inertial → Rotating |
| `compute_rotation_angle(t)` | Compute rotation angle |

## ReferenceFrame Enum

```python
class ReferenceFrame(Enum):
    ROTATING = "rotating"      # Rotating frame (synodic)
    INERTIAL = "inertial"      # Inertial frame
```
