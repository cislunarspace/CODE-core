---
title: e2m2e Visualization Module User Guide
---

# e2m2e Visualization Module User Guide

## Overview

The `e2m2e.visualization` module provides various visualization capabilities for orbits in the Circular Restricted Three-Body Problem (CR3BP). The module uses a layered architecture, splitting visualization functionality into independent components for maintainability and extensibility.

### Module Structure

```
e2m2e/visualization/
├── __init__.py        # Public API exports
├── config.py          # PlotConfig dataclass — all styling/figure settings
├── base.py            # OrbitVisualizer base class — atomic plot operations
├── family.py          # FamilyPlotter — high-level orbit family visualization
├── transfer.py        # TransferPlotter — transfer orbit visualization
├── stability.py       # compute_stability_for_family — parallel computation
└── plotting.py        # Backward-compatible re-exports & configure_academic_fonts()
```

### Class Hierarchy

```
PlotConfig (dataclass)
    ↓ passed to
OrbitVisualizer (base.py)
    ├── FamilyPlotter (family.py)
    └── TransferPlotter (transfer.py)

compute_stability_for_family (stability.py)  — standalone function
```

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Classes & Configuration](#core-classes--configuration)
3. [Basic Visualization (OrbitVisualizer)](#basic-visualization-orbitvisualizer)
4. [Orbit Family Visualization (FamilyPlotter)](#orbit-family-visualization-familyplotter)
5. [Transfer Orbit Visualization (TransferPlotter)](#transfer-orbit-visualization-transferplotter)
6. [Stability Computation](#stability-computation)
7. [Customization Settings (PlotConfig)](#customization-settings-plotconfig)
8. [Frequently Asked Questions](#frequently-asked-questions)
9. [Example Code](#example-code)

## Quick Start

### Install Dependencies

```bash
pip install numpy matplotlib
```

### Basic Usage Example

```python
import numpy as np
from e2m2e.core.system import CR3BP_System
from e2m2e.visualization import OrbitVisualizer

# 1. Create Earth-Moon system
system = CR3BP_System.from_known_system("earth_moon")
system.set_characteristic_scales(distance=384400, period=27.32 * 86400)
system.compute_libration_points()

# 2. Create visualizer
viz = OrbitVisualizer(system)

# 3. Create example orbit data (using simple circular orbit as example)
n_points = 100
t = np.linspace(0, 2*np.pi, n_points)
x = 0.8 + 0.1 * np.cos(t)
y = 0.1 * np.sin(t)
z = np.zeros_like(t)
vx = -0.1 * np.sin(t)
vy = 0.1 * np.cos(t)
vz = np.zeros_like(t)

orbit_states = np.column_stack([x, y, z, vx, vy, vz])

# 4. Plot 2D projection
viz.plot_2d_projection(orbit_states, plane='xy', color='blue', label='Test Orbit')
viz.plot_primary_bodies()
viz.plot_libration_points()
viz.show()
```

## Core Classes & Configuration

### PlotConfig — Style Configuration Dataclass

`PlotConfig` is a `dataclass` that centrally manages all figure styling parameters. Construct it and pass it to any visualizer.

```python
from e2m2e.visualization import PlotConfig

# Default configuration
config = PlotConfig()

# Custom configuration
config = PlotConfig(
    colormap="plasma",
    orbit_linewidth=2.0,
    orbit_alpha=0.9,
    figsize_2d=(14, 10),
    figsize_3d=(16, 10),
    dpi=150,
    primary_body_color="blue",
    primary_body_size=200,
    secondary_body_color="silver",
    secondary_body_size=100,
)

# Apply global font settings
config.apply_rcparams()
```

**PlotConfig Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `colormap` | `str` | `"coolwarm"` | Orbit family colormap |
| `orbit_linewidth` | `float` | `1.5` | Orbit line width |
| `orbit_alpha` | `float` | `0.8` | Orbit transparency |
| `figsize_2d` | `tuple` | `(12, 10)` | 2D figure size |
| `figsize_3d` | `tuple` | `(14, 10)` | 3D figure size |
| `figsize_dual` | `tuple` | `(12, 7)` | Dual-axis figure size |
| `figsize_overview` | `tuple` | `(18, 14)` | Overview figure size |
| `dpi` | `int` | `100` | Resolution |
| `primary_body_color` | `str` | `"blue"` | Primary body color |
| `primary_body_size` | `int` | `200` | Primary body marker size |
| `secondary_body_color` | `str` | `"silver"` | Secondary body color |
| `secondary_body_size` | `int` | `100` | Secondary body marker size |
| `lp_colors` | `List[str]` | `["gray"]*5` | Libration point colors |
| `lp_markers` | `List[str]` | `["^"]*5` | Libration point markers |
| `lp_sizes` | `List[int]` | `[60]*5` | Libration point marker sizes |
| `title` / `label` / `tick` / `legend` | `float` | various | Font size controls |

### OrbitVisualizer — Base Class

```python
from e2m2e.visualization import OrbitVisualizer, PlotConfig

config = PlotConfig(dpi=150)
viz = OrbitVisualizer(system, config=config)
```

**Parameters:**
- `system`: CR3BP_System object, required. Used to obtain system parameters and libration point positions.
- `config`: PlotConfig object, optional. Defaults to `PlotConfig()` if not provided.

### FamilyPlotter — Orbit Family Visualization

```python
from e2m2e.visualization import FamilyPlotter, PlotConfig

config = PlotConfig(colormap="viridis")
plotter = FamilyPlotter(system, config=config)
```

Inherits from `OrbitVisualizer`, adding high-level methods: `plot_family_2d`, `plot_family_3d`, `plot_jacobi_period_stability`, `plot_family_overview`.

### TransferPlotter — Transfer Orbit Visualization

```python
from e2m2e.visualization import TransferPlotter

plotter = TransferPlotter(system)
```

Inherits from `OrbitVisualizer`, providing transfer-specific methods: `plot_solution_plane`, `plot_transfer_orbit`.

### Method Overview

| Method | Class | Description | Key Parameters |
|--------|-------|-------------|----------------|
| `plot_3d_orbit()` | OrbitVisualizer | Plot 3D orbit | `orbit, color, label, ax, show_start` |
| `plot_2d_projection()` | OrbitVisualizer | Plot 2D projection | `orbit, plane, color, label, ax, show_start` |
| `plot_libration_points()` | OrbitVisualizer | Plot libration points | `ax, show_labels, is_3d` |
| `plot_primary_bodies()` | OrbitVisualizer | Plot celestial bodies | `ax, is_3d` |
| `plot_family_2d()` | FamilyPlotter | Plot family 2D view | `family_result, jacobi_values, title, plane, ...` |
| `plot_family_3d()` | FamilyPlotter | Plot family 3D view | `family_result, jacobi_values, title, center, ...` |
| `plot_jacobi_period_stability()` | FamilyPlotter | Plot Jacobi-period-stability | `jacobi_values, periods, stability_values, ...` |
| `plot_family_overview()` | FamilyPlotter | Plot 4-subplot overview | `family_result, jacobi_values, periods, stability_values, ...` |
| `plot_solution_plane()` | TransferPlotter | Plot solution space scatter | `results, color_by, ax, ...` |
| `plot_transfer_orbit()` | TransferPlotter | Plot transfer orbit in 3D | `departure_orbit, arrival_orbit, transfer_trajectory, ...` |
| `show()` | OrbitVisualizer | Display figure | None |
| `save()` | OrbitVisualizer | Save figure | `filename, dpi` |

## Basic Visualization (OrbitVisualizer)

### 1. 3D Orbit Visualization

```python
from e2m2e.visualization import OrbitVisualizer

viz = OrbitVisualizer(system)
ax = viz.plot_3d_orbit(orbit, color='blue', label='3D Orbit')

viz.plot_primary_bodies(ax=ax, is_3d=True)
viz.plot_libration_points(ax=ax, is_3d=True)

ax.legend()
viz.show()
```

### 2. 2D Projections

```python
# XY plane projection
viz.plot_2d_projection(orbit, plane='xy', color='red', label='XY Projection')
viz.plot_primary_bodies()
viz.plot_libration_points()
viz.show()

# XZ plane projection
viz.plot_2d_projection(orbit, plane='xz', color='green', label='XZ Projection')
viz.show()

# YZ plane projection
viz.plot_2d_projection(orbit, plane='yz', color='purple', label='YZ Projection')
viz.show()
```

### 3. Celestial Bodies and Libration Points

```python
viz.plot_primary_bodies()
viz.plot_libration_points()
viz.show()
```

## Orbit Family Visualization (FamilyPlotter)

`FamilyPlotter` extends `OrbitVisualizer` with one-click orbit family visualization, automatically colored by Jacobi constant.

### 1. Family 2D View

```python
from e2m2e.visualization import FamilyPlotter, PlotConfig

plotter = FamilyPlotter(system, config=PlotConfig(colormap="viridis"))

jacobi_values = [orbit.jacobi_constant for orbit in family_result]

fig, ax = plotter.plot_family_2d(
    family_result,
    jacobi_values,
    title="L1 Lyapunov Family",
    plane="xy",
    show_bodies=True,
    show_libration=True,
    show_colorbar=True,
    save_path="family_2d.png",
)
```

### 2. Family 3D View

```python
fig, ax = plotter.plot_family_3d(
    family_result,
    jacobi_values,
    title="L1 Lyapunov Family (3D)",
    center=(0.5, 0.0, 0.0),
    radius=0.65,
    elev=20,
    azim=-60,
    save_path="family_3d.png",
)
```

### 3. Jacobi-Period-Stability Plot

```python
from e2m2e.visualization import FamilyPlotter, compute_stability_for_family

plotter = FamilyPlotter(system)
periods = [orbit.period for orbit in family_result]
stability_values = compute_stability_for_family(family_result, system)

fig, ax = plotter.plot_jacobi_period_stability(
    jacobi_values,
    periods,
    stability_values,
    title="Period & Stability vs Jacobi Constant",
    target_period=6.0,
    save_path="jacobi_period_stability.png",
)
```

### 4. Family Overview

`plot_family_overview` generates a four-subplot overview in one call: global 2D view, zoomed 2D view, Jacobi-period-stability plot, and 3D view.

```python
fig = plotter.plot_family_overview(
    family_result,
    jacobi_values,
    periods,
    stability_values,
    suptitle="L1 Lyapunov Family Overview",
    plane="xy",
    zoom_xlim=(0.4, 0.6),
    zoom_ylim=(-0.15, 0.15),
    center_3d=(0.5, 0.0, 0.0),
    radius_3d=0.3,
    target_period=6.0,
    save_path="family_overview.png",
)
```

## Transfer Orbit Visualization (TransferPlotter)

### 1. Solution Plane Scatter Plot

```python
from e2m2e.visualization import TransferPlotter

transfer_plotter = TransferPlotter(system)

# Color by transfer type
ax = transfer_plotter.plot_solution_plane(
    results,
    color_by="transfer_type",
)
```

### 2. Transfer Orbit 3D Plot

```python
ax = transfer_plotter.plot_transfer_orbit(
    departure_orbit=dro,
    arrival_orbit=ro,
    transfer_trajectory=transfer_states,
    departure_state=dep_state,
    insertion_state=ins_state,
    label="Transfer",
    color="red",
)
```

## Stability Computation

`compute_stability_for_family` is a standalone function that uses multiprocessing to compute the stability index (maximum eigenvalue magnitude) for each orbit in a family in parallel.

```python
from e2m2e.visualization import compute_stability_for_family

# Uses all CPU cores by default
stability_values = compute_stability_for_family(family_result, system)

# Limit parallel workers
stability_values = compute_stability_for_family(family_result, system, max_workers=4)
```

**Parameters:**
- `family_result`: List of orbit results
- `system`: CR3BP_System object
- `max_workers`: Maximum parallel processes, defaults to `min(cpu_count, len(family))`

**Returns:** `List[float]` — stability index for each orbit

## Customization Settings (PlotConfig)

### Using the PlotConfig Dataclass

```python
from e2m2e.visualization import PlotConfig, FamilyPlotter

config = PlotConfig(
    # Figure sizes
    figsize_2d=(14, 10),
    figsize_3d=(16, 10),
    dpi=150,

    # Orbit style
    orbit_linewidth=2.0,
    orbit_alpha=0.9,

    # Colormap
    colormap="plasma",

    # Celestial body style
    primary_body_color="blue",
    primary_body_size=300,
    secondary_body_color="gray",
    secondary_body_size=150,

    # Libration point style
    lp_colors=["darkred", "darkblue", "darkgreen", "darkviolet", "darkorange"],
    lp_markers=["^", "s", "D", "o", "v"],
    lp_sizes=[120, 120, 120, 180, 180],
)

plotter = FamilyPlotter(system, config=config)
```

### Applying Global Font Settings

```python
config = PlotConfig()
config.apply_rcparams()  # Sets Times New Roman + STIX math fonts
```

### Using Existing Axes

```python
import matplotlib.pyplot as plt
from e2m2e.visualization import OrbitVisualizer

viz = OrbitVisualizer(system)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

viz.plot_2d_projection(orbit1, plane='xy', color='blue', ax=ax1, label='Orbit 1')
viz.plot_primary_bodies(ax=ax1)
ax1.set_title('Orbit 1 - XY Projection')

viz.plot_2d_projection(orbit2, plane='xy', color='red', ax=ax2, label='Orbit 2')
viz.plot_primary_bodies(ax=ax2)
ax2.set_title('Orbit 2 - XY Projection')

plt.tight_layout()
plt.show()
```

## Frequently Asked Questions

### 1. Figure Not Displaying

**Problem:** Figure doesn't display after calling `viz.show()`.

**Solutions:**
```python
# In script usage
viz.show()

# In Jupyter notebook usage
%matplotlib inline
viz.show()

# Or use interactive mode
%matplotlib notebook
viz.show()
```

### 2. Libration Points Not Displaying

**Problem:** Libration points don't appear in the figure.

**Solution:** Ensure the system has computed libration points.
```python
system.compute_libration_points()
viz = OrbitVisualizer(system)
```

### 3. Incorrect Axis Scale

**Problem:** 2D projection axis scale is not 1:1.

**Solution:** `plot_2d_projection` automatically sets equal aspect axes. If manually modified, reset:
```python
ax.set_aspect('equal')
```

### 4. Poor Saved Figure Quality

**Problem:** Saved figure has low resolution.

**Solution:** Increase dpi parameter.
```python
viz.save('orbit.png', dpi=300)  # High resolution
viz.save('orbit.pdf')           # Vector graphics, infinite resolution
```

### 5. Multiple Orbits Overlay

**Problem:** How to plot multiple orbits on the same figure.

**Solution:** Call plotting functions multiple times.
```python
viz.plot_2d_projection(orbit1, plane='xy', color='blue', label='Orbit 1')
viz.plot_2d_projection(orbit2, plane='xy', color='red', label='Orbit 2')

viz.plot_primary_bodies()
viz.plot_libration_points()

viz.axes.legend()
viz.show()
```

## Example Code

### Complete Example: Earth-Moon Lyapunov Orbit Visualization

```python
import numpy as np
from e2m2e.core.system import CR3BP_System
from e2m2e.core.dynamics import CR3BP_Dynamics
from e2m2e.visualization import PlotConfig, OrbitVisualizer

def visualize_lyapunov_orbit():
    """Visualize Lyapunov orbit in Earth-Moon system"""

    # 1. Create Earth-Moon system
    print("Creating Earth-Moon system...")
    system = CR3BP_System.from_known_system("earth_moon")
    system.set_characteristic_scales(384400, 27.32 * 86400)
    system.compute_libration_points()

    # 2. Generate example orbit data
    print("Generating example orbit data...")
    dynamics = CR3BP_Dynamics(system)

    n_points = 200
    t = np.linspace(0, 2*np.pi, n_points)
    amplitude = 0.02

    x0 = system.L1[0]
    x = x0 + amplitude * np.cos(t)
    y = amplitude * np.sin(t)
    z = np.zeros_like(t)
    vx = -amplitude * np.sin(t)
    vy = amplitude * np.cos(t)
    vz = np.zeros_like(t)

    orbit_states = np.column_stack([x, y, z, vx, vy, vz])

    # 3. Create visualizer with custom config
    print("Creating visualizer...")
    config = PlotConfig(dpi=150, orbit_linewidth=2.0)
    viz = OrbitVisualizer(system, config=config)

    # 4. Plot XY projection
    print("Generating XY projection...")
    viz.plot_2d_projection(orbit_states, plane='xy', color='blue', label='Lyapunov Orbit')
    viz.plot_primary_bodies()
    viz.plot_libration_points()
    viz.axes.legend()
    viz.save('lyapunov_orbit_xy.png', dpi=300)
    viz.show()

    # 5. Plot 3D view
    print("Generating 3D view...")
    viz.plot_3d_orbit(orbit_states, color='red', label='3D View')
    viz.plot_primary_bodies(ax=viz.axes_3d, is_3d=True)
    viz.plot_libration_points(ax=viz.axes_3d, is_3d=True)
    viz.axes_3d.legend()
    viz.save('lyapunov_orbit_3d.png', dpi=300)
    viz.show()

    print("Visualization complete!")

if __name__ == "__main__":
    visualize_lyapunov_orbit()
```

### Complete Example: Orbit Family Full Pipeline

```python
from e2m2e.core.system import CR3BP_System
from e2m2e.visualization import (
    PlotConfig, FamilyPlotter, compute_stability_for_family,
)

def visualize_orbit_family():
    """Full orbit family visualization: 2D, 3D, stability, overview"""

    system = CR3BP_System.from_known_system("earth_moon")
    system.compute_libration_points()

    # family_result comes from the Continuation algorithm
    # ... continuation algorithm call omitted ...

    # Prepare data
    jacobi_values = [orbit.jacobi_constant for orbit in family_result]
    periods = [orbit.period for orbit in family_result]
    stability_values = compute_stability_for_family(family_result, system)

    # Create FamilyPlotter
    config = PlotConfig(colormap="coolwarm", dpi=150)
    plotter = FamilyPlotter(system, config=config)

    # 2D family view
    plotter.plot_family_2d(
        family_result, jacobi_values,
        title="L1 Lyapunov Family",
        plane="xy",
        save_path="family_2d.png",
    )

    # 3D family view
    plotter.plot_family_3d(
        family_result, jacobi_values,
        title="L1 Lyapunov Family (3D)",
        elev=20, azim=-60,
        save_path="family_3d.png",
    )

    # Jacobi-period-stability plot
    plotter.plot_jacobi_period_stability(
        jacobi_values, periods, stability_values,
        title="Period & Stability vs Jacobi Constant",
        target_period=6.0,
        save_path="stability.png",
    )

    # One-click 4-subplot overview
    plotter.plot_family_overview(
        family_result, jacobi_values, periods, stability_values,
        suptitle="L1 Lyapunov Family Overview",
        zoom_xlim=(0.4, 0.6),
        zoom_ylim=(-0.15, 0.15),
        save_path="family_overview.png",
    )

if __name__ == "__main__":
    visualize_orbit_family()
```

### Example: Comparing Multiple Orbits

```python
import numpy as np
from e2m2e.core.system import CR3BP_System
from e2m2e.visualization import OrbitVisualizer

def compare_orbits():
    """Compare multiple orbits"""

    system = CR3BP_System.from_known_system("earth_moon")
    system.set_characteristic_scales(384400, 27.32 * 86400)
    system.compute_libration_points()

    viz = OrbitVisualizer(system)

    n_points = 100
    t = np.linspace(0, 2*np.pi, n_points)

    # Orbit 1: Small amplitude
    amp1 = 0.01
    x1 = system.L1[0] + amp1 * np.cos(t)
    y1 = amp1 * np.sin(t)
    orbit1 = np.column_stack([x1, y1, np.zeros(n_points),
                              -amp1*np.sin(t), amp1*np.cos(t), np.zeros(n_points)])

    # Orbit 2: Medium amplitude
    amp2 = 0.02
    x2 = system.L1[0] + amp2 * np.cos(t)
    y2 = amp2 * np.sin(t)
    orbit2 = np.column_stack([x2, y2, np.zeros(n_points),
                              -amp2*np.sin(t), amp2*np.cos(t), np.zeros(n_points)])

    # Orbit 3: Large amplitude
    amp3 = 0.03
    x3 = system.L1[0] + amp3 * np.cos(t)
    y3 = amp3 * np.sin(t)
    orbit3 = np.column_stack([x3, y3, np.zeros(n_points),
                              -amp3*np.sin(t), amp3*np.cos(t), np.zeros(n_points)])

    viz.plot_2d_projection(orbit1, plane='xy', color='blue', label=f'Amplitude={amp1}')
    viz.plot_2d_projection(orbit2, plane='xy', color='green', label=f'Amplitude={amp2}')
    viz.plot_2d_projection(orbit3, plane='xy', color='red', label=f'Amplitude={amp3}')

    viz.plot_primary_bodies()
    viz.plot_libration_points()

    viz.axes.legend(title='Lyapunov Orbits')
    viz.axes.set_title('Comparison of Lyapunov Orbits with Different Amplitudes')

    viz.show()
    viz.save('orbit_comparison.png', dpi=300)
```

## Summary

The `e2m2e.visualization` module uses a layered architecture to provide powerful orbit visualization capabilities:

1. **PlotConfig**: A centralized dataclass for all styling parameters
2. **OrbitVisualizer**: Base class with atomic plot operations — 3D orbits, 2D projections, bodies, libration points
3. **FamilyPlotter**: Purpose-built for orbit families — one-click 2D/3D/stability/overview plots
4. **TransferPlotter**: Purpose-built for transfer orbits — solution plane scatter and 3D trajectory plots
5. **compute_stability_for_family**: Multiprocessing-based parallel stability computation

By using these features, you can:
- Quickly verify correctness of orbit design
- Intuitively understand orbital dynamics characteristics
- Generate high-quality academic figures
- Perform systematic analysis of orbit families

For more questions, please refer to the docstrings in the module source code or contact the project maintainers.
