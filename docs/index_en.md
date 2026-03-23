# E2M2E Documentation

> Complete documentation for the Earth-to-Moon-to-Earth (E2M2E) orbital mechanics library

## Documentation Structure

```
docs/
├── index.md              # This file (documentation index)
├── guides/               # User guides
│   ├── system-overview.md    # System architecture and design
│   ├── orbit-generation.md  # Orbit generation tutorial
│   └── visualization-guide.md # Visualization tutorial
├── core/                 # Core modules
│   ├── system.md         # CR3BP_System - System parameters
│   ├── dynamics.md       # CR3BP_Dynamics - Dynamics
│   ├── orbit.md          # Orbit, OrbitFamily - Orbits
│   └── coordinate.md     # CoordinateTransformation - Coordinate transforms
├── algorithms/           # Algorithm modules
│   ├── continuation.md       # Continuation method
│   ├── differential_correction.md  # Differential correction
│   └── stability.md          # Stability analysis
├── visualization/       # Visualization module
│   └── plotting.md          # Plotting functions
└── reference/           # Technical reference
    ├── api-reference.md     # Complete API documentation
    └── algorithms.md        # Algorithm technical details
```

## Quick Navigation

### User Guides (guides)

| Document | Description |
|----------|-------------|
| [System Overview](guides/system-overview.md) | Architecture design, module responsibilities, data flow, typical workflows |
| [Orbit Generation](guides/orbit-generation.md) | DRO, Halo, Lissajous orbit generation tutorials |
| [Visualization Guide](guides/visualization-guide.md) | Plotting features, 2D/3D visualization |

### Core Modules (core)

| Class/Module | File | Description |
|--------------|------|-------------|
| `CR3BP_System` | [core/system.md](core/system.md) | System parameters and libration point calculation |
| `CR3BP_Dynamics` | [core/dynamics.md](core/dynamics.md) | Equations of motion and numerical integration |
| `Orbit` | [core/orbit.md](core/orbit.md) | Orbit data management |
| `OrbitFamily` | [core/orbit.md](core/orbit.md) | Orbit family management |
| `CoordinateTransformation` | [core/coordinate.md](core/coordinate.md) | Coordinate system transforms |

### Algorithm Modules (algorithms)

| Class/Module | File | Description |
|--------------|------|-------------|
| `ContinuationMethod` | [algorithms/continuation.md](algorithms/continuation.md) | Arc-length continuation method |
| `DifferentialCorrection` | [algorithms/differential_correction.md](algorithms/differential_correction.md) | Periodic orbit correction |
| `StabilityAnalysis` | [algorithms/stability.md](algorithms/stability.md) | Floquet stability analysis |

### Visualization Module (visualization)

| Function | File | Description |
|----------|------|-------------|
| `plot_orbit_2d/3d` | [visualization/plotting.md](visualization/plotting.md) | Orbit plotting |
| `plot_transfer_2d/3d` | [visualization/plotting.md](visualization/plotting.md) | Transfer trajectory plotting |
| `plot_system_geometry` | [visualization/plotting.md](visualization/plotting.md) | System geometry plotting |

### Technical Reference (reference)

| Document | Description |
|----------|-------------|
| [API Reference](reference/api-reference.md) | Complete API documentation, class and method descriptions |
| [Algorithm Reference](reference/algorithms.md) | CR3BP theory, algorithm mathematical foundations |

## Quick Start

```python
from e2m2e.core.system import CR3BP_System
from e2m2e.core.dynamics import CR3BP_Dynamics
from e2m2e.core.orbit import Orbit

# Create system
system = CR3BP_System.from_known_system("earth_moon")

# Create dynamics model
dynamics = CR3BP_Dynamics(system)

# Propagate orbit
result = dynamics.propagate(initial_state=state, t_span=(0, 10.0))
```

## Resources

- [API Reference](reference/api-reference.md) - Complete API documentation
- [Example Code](../examples/) - Practical usage examples
- [Test Cases](../tests/) - Unit tests

### Common Tasks

1. **Design DRO orbit** → Refer to [Orbit Generation - DRO](guides/orbit-generation.md#distant-retrograde-orbit-dro)
2. **Design Halo orbit** → Refer to [Orbit Generation - Halo](guides/orbit-generation.md#halo-orbit)
3. **Generate orbit family** → Refer to [Orbit Family Continuation](reference/algorithms.md#5-orbit-family-continuation-algorithm)
4. **Analyze stability** → Refer to [Stability Analysis](reference/algorithms.md#7-stability-analysis)

## Physical Background

E2M2E implements orbit design based on the **Circular Restricted Three-Body Problem (CR3BP)**. In the Earth-Moon system:
- Mass parameter $\mu \approx 0.01215$
- Characteristic distance: 384,400 km (Earth-Moon distance)
- Characteristic period: 27.32 days

See [CR3BP Theory](reference/algorithms.md#1-overview) and [System Overview](guides/system-overview.md) for details.
