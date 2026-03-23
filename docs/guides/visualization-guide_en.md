# e2m2e Visualization Module User Guide

## Overview

The `e2m2e.visualization.plotting` module provides various visualization capabilities for orbits in the Circular Restricted Three-Body Problem (CR3BP). This guide provides detailed instructions on how to use these features.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Class: OrbitVisualizer](#core-classorbitvisualizer)
3. [Basic Visualization Features](#basic-visualization-features)
4. [Advanced Visualization Features](#advanced-visualization-features)
5. [Customization Settings](#customization-settings)
6. [Frequently Asked Questions](#frequently-asked-questions)
7. [Example Code](#example-code)

## Quick Start

### Install Dependencies

```bash
pip install numpy matplotlib
```

### Basic Usage Example

```python
import numpy as np
from e2m2e.core.system import CR3BP_System
from e2m2e.visualization.plotting import OrbitVisualizer

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

## Core Class: OrbitVisualizer

### Initialization

```python
viz = OrbitVisualizer(system)
```

**Parameters:**
- `system`: CR3BP_System object, required. Used to obtain system parameters and libration point positions.

**Attributes:**
- `figsize`: Figure size, default `(12, 8)`
- `dpi`: Resolution, default `100`
- `orbit_linewidth`: Orbit line width, default `1.5`
- `orbit_alpha`: Orbit transparency, default `0.8`
- `primary_body_color`: Primary body color, default `"gold"`
- `secondary_body_color`: Secondary body color, default `"silver"`

### Method Overview

| Method | Description | Common Parameters |
|--------|-------------|-------------------|
| `plot_3d_orbit()` | Plot 3D orbit | `orbit, color, label, ax, show_start` |
| `plot_2d_projection()` | Plot 2D projection | `orbit, plane, color, label, ax, show_start` |
| `plot_libration_points()` | Plot libration points | `ax, show_labels, is_3d` |
| `plot_primary_bodies()` | Plot celestial bodies | `ax, is_3d` |
| `plot_orbit_family()` | Plot orbit family | `family_result, plane, colormap, ax` |
| `plot_poincare_section()` | Plot Poincaré section | `orbits, plane, value, ax` |
| `plot_jacobi_constant()` | Plot Jacobi constant | `orbit, ax` |
| `plot_stability_diagram()` | Plot stability diagram | `family_result, ax` |
| `create_overview_plot()` | Create overview plot | `orbit` |
| `show()` | Display figure | None |
| `save()` | Save figure | `filename, dpi` |

## Basic Visualization Features

### 1. 3D Orbit Visualization

```python
# Plot 3D orbit
ax = viz.plot_3d_orbit(orbit, color='blue', label='3D Orbit')

# Add celestial bodies and libration points
viz.plot_primary_bodies(ax=ax, is_3d=True)
viz.plot_libration_points(ax=ax, is_3d=True)

# Add legend and display
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
# Plot only celestial bodies
viz.plot_primary_bodies()
viz.show()

# Plot only libration points
viz.plot_libration_points()
viz.show()

# Plot both
viz.plot_primary_bodies()
viz.plot_libration_points()
viz.show()
```

## Advanced Visualization Features

### 1. Orbit Family Visualization

```python
# Assuming family_result is the result returned by Continuation algorithm
viz.plot_orbit_family(family_result, plane='xy', colormap='viridis')
viz.show()
```

### 2. Poincaré Section

```python
# Plot Poincaré section at y=0
viz.plot_poincare_section(orbit, plane='y', value=0.0)
viz.show()

# Plot Poincaré section at x=0.5
viz.plot_poincare_section(orbit, plane='x', value=0.5)
viz.show()
```

### 3. Jacobi Constant Variation

```python
# Plot Jacobi constant over time
viz.plot_jacobi_constant(orbit)
viz.show()
```

### 4. Stability Analysis

```python
# Plot orbit family period variation diagram
viz.plot_stability_diagram(family_result)
viz.show()
```

### 5. Comprehensive Overview Plot

```python
# Create overview plot with four subplots
fig = viz.create_overview_plot(orbit)
viz.show()

# Save overview plot
viz.save('orbit_overview.png', dpi=300)
```

## Customization Settings

### Modify Figure Style

```python
# Modify figure size and resolution
viz.figsize = (10, 6)
viz.dpi = 150

# Modify orbit style
viz.orbit_linewidth = 2.0
viz.orbit_alpha = 0.9

# Modify celestial body style
viz.primary_body_color = 'orange'
viz.primary_body_size = 300
viz.secondary_body_color = 'gray'
viz.secondary_body_size = 150

# Modify libration point style
viz.libration_point_colors = ['darkred', 'darkblue', 'darkgreen', 'darkviolet', 'darkorange']
viz.libration_point_sizes = [120, 120, 120, 180, 180]
```

### Using Existing Axes

```python
import matplotlib.pyplot as plt

# Create custom figure layout
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot XY projection on first axes
viz.plot_2d_projection(orbit1, plane='xy', color='blue', ax=ax1, label='Orbit 1')
viz.plot_primary_bodies(ax=ax1)
ax1.set_title('Orbit 1 - XY Projection')

# Plot XY projection on second axes
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
# Plot first orbit
viz.plot_2d_projection(orbit1, plane='xy', color='blue', label='Orbit 1')

# Plot second orbit
viz.plot_2d_projection(orbit2, plane='xy', color='red', label='Orbit 2')

# Add celestial bodies and libration points
viz.plot_primary_bodies()
viz.plot_libration_points()

# Display legend and figure
viz.axes.legend()
viz.show()
```

## Example Code

### Complete Example: Earth-Moon Lyapunov Orbit Visualization

```python
import numpy as np
from e2m2e.core.system import CR3BP_System
from e2m2e.core.dynamics import CR3BP_Dynamics
from e2m2e.algorithms.differential_correction import DifferentialCorrection
from e2m2e.visualization.plotting import OrbitVisualizer

def visualize_lyapunov_orbit():
    """Visualize Lyapunov orbit in Earth-Moon system"""
    
    # 1. Create Earth-Moon system
    print("Creating Earth-Moon system...")
    system = CR3BP_System.from_known_system("earth_moon")
    system.set_characteristic_scales(384400, 27.32 * 86400)
    system.compute_libration_points()
    
    # 2. Design Lyapunov orbit (simplified example)
    print("Generating example orbit data...")
    dynamics = CR3BP_Dynamics(system)
    
    # Create example orbit (approximate Lyapunov orbit around L1)
    n_points = 200
    t = np.linspace(0, 2*np.pi, n_points)
    amplitude = 0.02
    
    # Orbit parameters
    x0 = system.L1[0]  # x coordinate of L1 point
    y_amplitude = amplitude
    z_amplitude = 0.0  # 2D orbit
    
    # Generate orbit states
    x = x0 + amplitude * np.cos(t)
    y = y_amplitude * np.sin(t)
    z = np.zeros_like(t)
    vx = -amplitude * np.sin(t)
    vy = y_amplitude * np.cos(t)
    vz = np.zeros_like(t)
    
    orbit_states = np.column_stack([x, y, z, vx, vy, vz])
    
    # 3. Create visualizer
    print("Creating visualizer...")
    viz = OrbitVisualizer(system)
    
    # 4. Create overview plot
    print("Generating overview plot...")
    viz.create_overview_plot(orbit_states)
    viz.save('lyapunov_orbit_overview.png', dpi=300)
    viz.show()
    
    # 5. Plot XY projection separately
    print("Generating XY projection...")
    viz.plot_2d_projection(orbit_states, plane='xy', color='blue', label='Lyapunov Orbit')
    viz.plot_primary_bodies()
    viz.plot_libration_points()
    viz.axes.legend()
    viz.save('lyapunov_orbit_xy.png', dpi=300)
    viz.show()
    
    # 6. Plot 3D view
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

### Example: Comparing Multiple Orbits

```python
def compare_orbits():
    """Compare multiple orbits"""
    
    # Create system
    system = CR3BP_System.from_known_system("earth_moon")
    system.set_characteristic_scales(384400, 27.32 * 86400)
    system.compute_libration_points()
    
    # Create visualizer
    viz = OrbitVisualizer(system)
    
    # Generate multiple example orbits
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
    
    # Plot all orbits
    viz.plot_2d_projection(orbit1, plane='xy', color='blue', label=f'Amplitude={amp1}')
    viz.plot_2d_projection(orbit2, plane='xy', color='green', label=f'Amplitude={amp2}')
    viz.plot_2d_projection(orbit3, plane='xy', color='red', label=f'Amplitude={amp3}')
    
    # Add celestial bodies and libration points
    viz.plot_primary_bodies()
    viz.plot_libration_points()
    
    # Add legend and title
    viz.axes.legend(title='Lyapunov Orbits')
    viz.axes.set_title('Comparison of Lyapunov Orbits with Different Amplitudes')
    
    # Display and save
    viz.show()
    viz.save('orbit_comparison.png', dpi=300)
```

## Summary

The `e2m2e.visualization.plotting` module provides powerful orbit visualization capabilities, including:

1. **Basic visualization**: 3D orbits, 2D projections, celestial bodies and libration points
2. **Advanced analysis**: Orbit families, Poincaré sections, Jacobi constants, stability
3. **Customization**: Flexible style settings, axis control, figure saving
4. **Usability**: Clear API, detailed error messages, rich examples

By using these features reasonably, you can:
- Quickly verify correctness of orbit design
- Intuitively understand orbital dynamics characteristics
- Generate high-quality academic figures
- Perform systematic analysis of orbit families

For more questions, please refer to the docstrings in the module source code or contact the project maintainers.
