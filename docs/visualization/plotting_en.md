# Visualization Subpackage

**Directory**: `e2m2e/visualization/`

The original monolithic `plotting.py` (1650 lines) has been refactored into a 6-module subpackage.

## Module Structure

```
visualization/
├── __init__.py        # Public exports
├── config.py          # PlotConfig dataclass
├── base.py            # OrbitVisualizer base class
├── family.py          # FamilyPlotter for orbit families
├── transfer.py        # TransferPlotter for transfer trajectories
├── stability.py       # Parallel stability computation
└── plotting.py        # Backward-compat re-export shim
```

## Module Details

### `config.py` — Plot Configuration

`PlotConfig` dataclass that centralizes all visualization parameters:

| Attribute | Description |
|-----------|-------------|
| `title_fontsize` / `label_fontsize` / `tick_fontsize` | Title, axis label, tick font sizes |
| `legend_fontsize` / `colorbar_fontsize` | Legend and colorbar font sizes |
| `suptitle_fontsize` / `lp_label_fontsize` | Suptitle and libration-point label font sizes |
| `colormap` | Colormap name |
| `body_colors` / `body_sizes` | Primary body colors and sizes |
| `figure_sizes` | 2D / 3D figure dimensions |
| `orbit_linewidth` / `orbit_alpha` | Orbit line width and transparency |
| `title_y_offsets` | Title y-offset values |

**Methods:**

| Method | Description |
|--------|-------------|
| `apply_rcparams()` | Apply all matplotlib rc settings (including academic fonts: Times New Roman, stix math) |
| `get_cmap()` | Return the colormap instance for the current configuration |

### `base.py` — Visualizer Base Class

`OrbitVisualizer` base class + `ProjectionPlane` enum.

**`ProjectionPlane` enum values:** `XY`, `XZ`, `YZ`

**`OrbitVisualizer` methods:**

| Method | Description |
|--------|-------------|
| `plot_2d_projection(orbit, plane, ax)` | 2D projection plot |
| `plot_3d_orbit(orbit, ax)` | 3D orbit plot |
| `plot_primary_bodies(system, ax)` | Plot primary bodies |
| `plot_libration_points(system, ax)` | Plot libration points |
| `show()` | Display the figure |
| `save(path)` | Save the figure to file |
| `_extract_states(orbit)` | Extract state array from orbit object |
| `_sort_points_by_nearest_neighbor(points)` | Nearest-neighbor sort (prevents line scrambling) |

### `family.py` — Orbit Family Plotting

`FamilyPlotter(OrbitVisualizer)` — full visualization of orbit families.

**Public methods:**

| Method | Description |
|--------|-------------|
| `plot_family_2d(family, plane, ax)` | Orbit family 2D projection |
| `plot_family_3d(family, ax)` | Orbit family 3D plot |
| `plot_jacobi_period_stability(family, ax)` | Jacobi constant / period / stability composite plot |
| `plot_family_overview(family)` | Orbit family overview figure |

**Internal helpers:**

| Method | Description |
|--------|-------------|
| `_draw_orbit_loop_2d(orbit, plane, ax)` | Draw single orbit 2D closed loop |
| `_draw_orbit_loop_3d(orbit, ax)` | Draw single orbit 3D closed loop |
| `_add_colorbar(mappable, ax)` | Add colorbar |
| `_style_2d_ax(ax, plane)` | Style 2D axes |
| `_style_3d_ax(ax)` | Style 3D axes |
| `_get_jacobi_norm(family)` | Get normalized Jacobi constant array |

### `transfer.py` — Transfer Trajectory Plotting

`TransferPlotter(OrbitVisualizer)` — visualization of transfer orbits.

| Method | Description |
|--------|-------------|
| `plot_solution_plane(transfer, ax)` | Plot the solution plane |
| `plot_transfer_orbit(transfer, ax)` | Plot the transfer orbit |

### `stability.py` — Stability Computation

| Function | Description |
|----------|-------------|
| `compute_stability_for_family(family)` | Compute stability indices for an orbit family using `ProcessPoolExecutor` for parallel execution |

### `plotting.py` — Backward-Compatibility Shim

Re-exports the old API for backward compatibility and exposes:

| Function | Description |
|----------|-------------|
| `configure_academic_fonts()` | Configure publication-quality fonts (Times New Roman + stix math) |

### `__init__.py` — Public Exports

```python
from e2m2e.visualization import (
    PlotConfig,
    OrbitVisualizer,
    ProjectionPlane,
    FamilyPlotter,
    TransferPlotter,
    compute_stability_for_family,
)
```

## Usage Example

```python
from e2m2e.visualization import PlotConfig, FamilyPlotter, TransferPlotter

# Apply academic fonts and styles
config = PlotConfig()
config.apply_rcparams()

# Orbit family visualization
fp = FamilyPlotter(config=config)
fp.plot_family_overview(family)
fp.plot_jacobi_period_stability(family)
fp.show()

# Transfer trajectory visualization
tp = TransferPlotter(config=config)
tp.plot_transfer_orbit(transfer)
tp.save("transfer.png")
```

## Backward Compatibility

The old `from e2m2e.visualization.plotting import ...` import path still works — the `plotting.py` shim automatically re-exports to the new modules.
