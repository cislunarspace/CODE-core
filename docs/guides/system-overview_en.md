# System Overview

## Architecture Design

E2M2E adopts a modular design with four core modules:

```
┌─────────────────────────────────────────────────────────────┐
│                         e2m2e                               │
├─────────────┬─────────────┬─────────────┬─────────────────┤
│    core     │ algorithms  │  transfer   │  visualization  │
├─────────────┼─────────────┼─────────────┼─────────────────┤
│ system.py   │ differential │ earth_moon  │   plotting.py   │
│ dynamics.py │ correction.py│ moon_earth  │                 │
│ orbit.py    │continuation.py│inter_orbit │                 │
│coordinate.py│ stability.py │             │                 │
└─────────────┴─────────────┴─────────────┴─────────────────┘
```

## Module Responsibilities

### Core Module

Provides basic building blocks for CR3BP orbital mechanics:

| File | Class/Function | Responsibilities |
|------|---------------|------------------|
| `system.py` | `CR3BP_System` | System parameters, libration point calculation, coordinate transforms |
| `dynamics.py` | `CR3BP_Dynamics` | Equations of motion, numerical integration, STM propagation |
| `orbit.py` | `Orbit`, `OrbitFamily` | Orbit data management, period detection, stability analysis |
| `coordinate.py` | `CoordinateTransformation` | Coordinate system transforms (rotating↔inertial) |

### Algorithms Module

Implements numerical algorithms required for periodic orbit design:

| File | Class/Function | Responsibilities |
|------|---------------|------------------|
| `differential_correction.py` | `DifferentialCorrection` | Newton iteration for periodic orbit solution |
| `continuation.py` | `Continuation` | Orbit family continuation (natural/pseudo-arclength) |
| `stability.py` | `StabilityAnalysis` | Floquet multipliers, stability determination, bifurcation detection |

### Transfer Module

Implements orbit transfer design based on core modules:

| File | Class | Responsibilities |
|------|-------|------------------|
| `transfer.py` | `Transfer` | Simplified chainable API |
| `transfer_search.py` | `TransferSearch` | DRO→RO planar transfer grid search (parallel) |
| `transfer_optimization.py` | `DROTRONLPOptimizer` | NLP optimization |

### Visualization Module

| File | Class/Function | Responsibilities |
|------|---------------|------------------|
| `plotting.py` | `OrbitVisualizer` | 2D/3D orbit plotting, Poincaré sections, overview plots |
| `plotting.py` | `compute_stability_for_family` | Orbit family stability computation |

## Data Flow

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│ CR3BP_System │────▶│ CR3BP_Dynamics│────▶│    Orbit     │
└──────────────┘     └───────────────┘     └──────────────┘
                             │                     │
                             ▼                     ▼
                      ┌───────────────┐     ┌──────────────┐
                      │Differential   │     │   Orbit      │
                      │Correction     │────▶│   Family     │
                      └───────────────┘     └──────────────┘
                             │
                             ▼
                      ┌───────────────┐     ┌──────────────┐
                      │ Continuation  │────▶│  Transfer    │
                      └───────────────┘     │  Design      │
                                             └──────────────┘
```

## Typical Workflows

### 1. Periodic Orbit Design

```python
# 1. Create system
system = CR3BP_System.from_known_system("earth_moon")
system.compute_libration_points()

# 2. Create dynamics model
dynamics = CR3BP_Dynamics(system)

# 3. Configure differential corrector
dc = DifferentialCorrection(dynamics)
dc.setup_2D_symmetric_x_fixed_x0(x0=0.8)

# 4. Iterate to solve
orbit, result = dc.iterate_correction(initial_state, t_half=1.5)
```

### 2. Orbit Family Continuation

```python
# 5. Continue orbit family
continuation = Continuation(dc, step=0.01)
family = continuation.natural_continuation(
    seed_orbit=orbit,
    param_range=(0.8, 1.2),
    step_size=0.01
)
```

### 3. Transfer Design

```python
# 6. Design transfer orbit
transfer = InterOrbitTransfer(system, dynamics)
result = transfer.design_heteroclinic_transfer(orbit_L1, orbit_L2)
```

## Design Principles

1. **Physics-driven**: All algorithms based on rigorous CR3BP mathematical models
2. **Modular**: Each module is independently testable with clear interfaces
3. **Numerically robust**: Uses high-order integrators, adaptive step sizes, convergence detection
4. **User-friendly**: Rich comments and error messages

## Extension Guide

### Adding New Orbit Types

1. Add new symmetry configurations in `differential_correction.py`
2. Implement corresponding `setup_*` methods
3. Add test cases

### Adding New Transfer Strategies

1. Create new module under `transfer/` directory
2. Inherit base class to implement specific strategy
3. Export in `__init__.py`

See [Contributing Guide](../CONTRIBUTING.md) for details.
