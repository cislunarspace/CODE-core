---
title: 'EphemerisSystem - Ephemeris System'
---

# EphemerisSystem - Ephemeris System

The `EphemerisSystem` class defines an ephemeris system based on NASA SPICE kernels, supporting multi-body gravitational computation.

## Class Definition

```python
class EphemerisSystem:
    """Ephemeris system based on SPICE kernels
    
    Defines the list of bodies participating in gravitational computation and the reference epoch,
    supporting precise multi-body ephemeris calculations.
    
    Args:
        bodies: List of body names, e.g. ["EARTH", "MOON", "SUN"]
        reference_epoch: Reference epoch string in "YYYY-MM-DDTHH:MM:SS" format
    """
```

## Main Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `bodies` | `List[str]` | List of body names participating in gravitational computation |
| `reference_epoch` | `str` | Reference epoch for initializing SPICE time |
| `epoch_et` | `float` | Ephemeris time (seconds) corresponding to the reference epoch |
| `body_ids` | `Dict[str, int]` | Mapping from body names to NAIF IDs |

## Main Methods

### `__init__(bodies, reference_epoch)`
Initialize the ephemeris system.

**Parameters**:
- `bodies`: List of body names using standard NAIF names (e.g., "EARTH", "MOON", "SUN")
- `reference_epoch`: Reference epoch as an ISO 8601 string

**Example**:
```python
from e2m2e.core import EphemerisSystem

system = EphemerisSystem(
    bodies=["EARTH", "MOON", "SUN"],
    reference_epoch="2025-06-21T11:00:06"
)
```

### `get_body_position(body_name, et)`
Get the position of a specified body at a given ephemeris time.

**Parameters**:
- `body_name`: Body name
- `et`: Ephemeris time (seconds)

**Returns**:
- `np.ndarray`: 3D position vector (km) in the J2000 inertial frame

### `get_body_state(body_name, et)`
Get the state (position and velocity) of a specified body at a given ephemeris time.

**Parameters**:
- `body_name`: Body name
- `et`: Ephemeris time (seconds)

**Returns**:
- `np.ndarray`: 6D state vector [x, y, z, vx, vy, vz] (km, km/s)

## Usage Examples

### Basic Usage
```python
from e2m2e.core import EphemerisSystem
from e2m2e.core.spice import SPICEManager

# Initialize SPICE manager and load kernels
spice_manager = SPICEManager()
spice_manager.load_kernels_from_directory("./kernels/")

# Create ephemeris system
system = EphemerisSystem(
    bodies=["EARTH", "MOON", "SUN"],
    reference_epoch="2025-06-21T11:00:06"
)

# Get body positions
et = system.epoch_et + 86400  # 1 day after reference epoch
earth_pos = system.get_body_position("EARTH", et)
moon_pos = system.get_body_position("MOON", et)
sun_pos = system.get_body_position("SUN", et)
```

### Using with Dynamics
```python
from e2m2e.core import EphemerisDynamics

# Create ephemeris dynamics
dynamics = EphemerisDynamics(system=system)

# Propagate orbit
initial_state = [384400, 0, 0, 0, 1023, 0]  # Initial state (km, km/s)
result = dynamics.propagate(initial_state, time_span=[0, 86400])
```

## Important Notes

1. **SPICE Kernels**: Required SPICE kernel files must be loaded before use
2. **Time System**: Uses Ephemeris Time (ET) in seconds
3. **Coordinate Frame**: All positions and velocities are in the J2000 inertial frame
4. **Units**: Position in km, velocity in km/s

## Related Classes

- [`EphemerisDynamics`](ephemeris_dynamics.md): Ephemeris dynamics computation
- [`SPICEManager`](spice.md): SPICE kernel management
