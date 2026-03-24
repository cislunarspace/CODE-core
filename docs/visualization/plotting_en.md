# PlottingFunctions

**File**: `e2m2e/visualization/plotting.py`

## Design Principles

Provides visualization functions for orbits and transfer trajectories, supporting 2D and 3D plotting.

## Core Plotting Functions

### Orbit Plotting

| Function | Description |
|----------|-------------|
| `plot_orbit_2d(orbit, ax, **kwargs)` | 2D orbit plotting |
| `plot_orbit_3d(orbit_or_family, ax, **kwargs)` | 3D orbit plotting |
| `plot_orbit_family(family, ax, **kwargs)` | Plot orbit family |
| `plot_poincare_section(family, section, ax)` | Poincaré section |

### Transfer Trajectory Plotting

| Function | Description |
|----------|-------------|
| `plot_transfer_2d(transfer, ax, **kwargs)` | 2D transfer trajectory |
| `plot_transfer_3d(transfer, ax, **kwargs)` | 3D transfer trajectory |
| `plot_transfer_trajectory(transfer, ax)` | Plot transfer path |
| `plot_delta_v_budget(transfer, ax)` | $\Delta v$ budget plot |

### System Plotting

| Function | Description |
|----------|-------------|
| `plot_system_geometry(system, ax, ...)` | System geometry |
| `plot_libration_points(system, ax)` | Plot libration points |
| `plot_lagrange_surfaces(system, ax)` | Plot Lagrange equipotential surfaces |

## Plot Style Configuration

```python
# Color configuration
COLORS = {
    "earth": "#1E90FF",      # Blue
    "moon": "#808080",       # Gray
    "dro": "#FF6B6B",        # Red
    "ro": "#4ECDC4",         # Cyan
    "transfer": "#FFE66D",   # Yellow
}

# Line style configuration
LINE_STYLES = {
    "stable": "-",           # Solid
    "unstable": "--",        # Dashed
}
```

## Usage Example

```python
import matplotlib.pyplot as plt
from e2m2e.visualization.plotting import plot_orbit_3d, plot_transfer_3d

# Create 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot DRO
plot_orbit_3d(dro_orbit, ax, color='red', label='DRO')

# Plot RO
plot_orbit_3d(ro_orbit, ax, color='cyan', label='RO')

# Plot transfer trajectory
plot_transfer_3d(transfer, ax, color='yellow', label='Transfer')

ax.legend()
plt.show()
```

## Submodule Index

| Submodule | File | Main Classes/Functions |
|-----------|------|------------------------|
| `core.system` | `core/system.py` | `CR3BP_System` |
| `core.dynamics` | `core/dynamics.py` | `CR3BP_Dynamics` |
| `core.orbit` | `core/orbit.py` | `Orbit`, `OrbitFamily` |
| `core.coordinate` | `core/coordinate.py` | `CoordinateTransformation` |
| `algorithms.continuation` | `algorithms/continuation.py` | `ContinuationMethod` |
| `algorithms.differential_correction` | `algorithms/differential_correction.py` | `DifferentialCorrection` |
| `algorithms.stability` | `algorithms/stability.py` | `StabilityAnalysis` |
| `transfer.inter_orbit` | `transfer/inter_orbit.py` | `DROROTransferSearch` |
| `transfer.earth_moon` | `transfer/earth_moon.py` | `EarthMoonTransfer` |
| `transfer.moon_earth` | `transfer/moon_earth.py` | `MoonEarthTransfer` |
| `visualization.plotting` | `visualization/plotting.py` | Plotting functions |
