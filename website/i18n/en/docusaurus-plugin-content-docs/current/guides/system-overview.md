---
title: System Overview
---

# System Overview

> E2M2E architecture design, module responsibilities, and extension guide.

## Four-Layer Architecture

```
core/           Foundation — data structures and physics models
  ↓
algorithms/     Algorithm layer — differential correction, continuation, stability, multiple shooting
  ↓
transfer/       Design layer — grid search, NLP optimization
  ↓
visualization/  Presentation layer — orbit family plotting, transfer trajectory visualization
```

Dependencies between layers are strictly unidirectional: upper layers may use lower-layer functionality, but lower layers cannot reference upper layers.

## Module Responsibilities

### Core

| File | Class | Purpose |
|------|-------|---------|
| `system.py` | `CR3BP_System` | System parameters, libration points, Jacobi constant, unit conversion |
| `dynamics.py` | `CR3BP_Dynamics` | Equations of motion, numerical integration, STM propagation |
| `orbit.py` | `Orbit`, `OrbitFamily` | Orbit data containers, period detection, JSON serialization |
| `coordinate.py` | `CoordinateTransformation` | Rotating <-> inertial frame coordinate transformation |
| `spice.py` | `SPICEManager` | SPICE kernel management |
| `ephemeris_system.py` | `EphemerisSystem` | Multi-body ephemeris system |
| `ephemeris_dynamics.py` | `EphemerisDynamics` | SPICE-based N-body dynamics |

### Algorithms

| File | Class | Purpose |
|------|-------|---------|
| `differential_correction.py` | `DifferentialCorrection` | Newton-Raphson iteration for periodic orbit correction |
| `continuation.py` | `Continuation` | Natural / pseudo-arclength orbit family continuation |
| `stability.py` | `StabilityAnalysis` | Floquet multipliers, bifurcation detection |
| `multiple_shooting.py` | `MultipleShooting` | Multiple shooting method, complex constraint correction |
| `strategies/` | `CorrectionConfig` + strategy functions | Immutable correction config dataclass + 8 symmetry strategies |

### Transfer

| File | Class | Purpose |
|------|-------|---------|
| `transfer.py` | `Transfer` | Chaining API: `set_orbit().optimize()` |
| `transfer_search.py` | `DROTransferSearch` | DRO->RO planar transfer grid search (parallel) |
| `transfer_optimization.py` | `DROTRONLPOptimizer` | NLP optimization (optional COPT solver) |
| `search_config.py` | `SearchConfig` | Search/optimization parameter configuration dataclass |

### Visualization

| File | Class | Purpose |
|------|-------|---------|
| `config.py` | `PlotConfig` | Style configuration (fonts, colors, sizes) |
| `base.py` | `OrbitVisualizer` | 2D/3D orbit plotting base class |
| `family.py` | `FamilyPlotter` | Orbit family visualization (Jacobi coloring) |
| `transfer.py` | `TransferPlotter` | Transfer trajectory visualization |
| `stability.py` | `compute_stability_for_family` | Batch stability computation for orbit families |

## Data Flow

```
CR3BP_System -> CR3BP_Dynamics -> Orbit/OrbitFamily
                    ↓                    ↓
          DifferentialCorrection ->  Transfer Design
                    ↓
              Continuation -> Visualization
```

## Key Conventions

- **State vector order** is always `[x, y, z, vx, vy, vz]`, consistent throughout
- **Numerical precision**: integrator `rtol=atol=1e-12`; finite difference step sizes must not be increased
- **Dimensionless units**: DU (distance), TU (time), VU (velocity); call `set_characteristic_scales()` before physical calculations
- **Interface stability**: public method signatures must not break backward compatibility; new parameters must have default values

## Extension Guide

### Adding a New Orbit Type

Add the corresponding `setup_*` method and symmetry configuration in `differential_correction.py`.

### Adding a New Dynamics Model

Create a subclass of `Dynamics`, implementing `equations_of_motion()` and `propagate()`. Do not modify the base class.

### Adding a New Algorithm

Create a new module in the `algorithms/` directory, follow the existing interface design, and export it in `__init__.py`.

-> For detailed usage of each module, see the corresponding documentation page.
