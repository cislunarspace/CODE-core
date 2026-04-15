---
slug: /
title: E2M2E — Cislunar Transfer Orbit Design Library
---

# E2M2E — Cislunar Transfer Orbit Design Library

An orbital mechanics toolkit based on the Circular Restricted Three-Body Problem (CR3BP) for designing periodic and transfer orbits in cislunar space.

## Getting Started in 30 Seconds

```bash
pip install e2m2e
```

```python
from e2m2e.core import CR3BP_System, CR3BP_Dynamics
from e2m2e.algorithms import DifferentialCorrection
import numpy as np

# Create an Earth-Moon system and design a DRO
system = CR3BP_System.from_known_system("earth_moon")
system.set_characteristic_scales(distance=384400, period=27.32*86400)
system.compute_libration_points()

dynamics = CR3BP_Dynamics(system)
dc = DifferentialCorrection(dynamics)
dc.setup_2D_symmetric_x_fixed_x0(x0=0.8)

initial_state = np.array([0.8, 0.0, 0.0, 0.0, 0.5, 0.0])
orbit, result = dc.iterate_correction(initial_state, t_half=1.6)

print(f"Orbit period: {orbit.period:.4f}")
print(f"Jacobi constant: {orbit.jacobi_constant:.6f}")
```

## What You Can Do

### Design Periodic Orbits

DRO, Halo, Lyapunov — from initial guesses to converged orbits, to full family continuation.

- [Orbit Generation Tutorial](guides/orbit-generation.md) — DRO / Halo / Lyapunov generation workflow
- [Differential Correction](algorithms/differential_correction.md) — how to correct to a precise periodic orbit
- [Orbit Family Continuation](algorithms/continuation.md) — how to generate a family of periodic orbits
- [Halo Orbits](algorithms/halo.md) — Richardson initial guess, pseudo-arclength continuation, CLI scripts

### Analyze Orbit Stability

Floquet multipliers, bifurcation detection, stability indices — understand the dynamical properties of orbits.

- [Stability Analysis](algorithms/stability.md) — how to determine orbit stability and find bifurcation points

### Visualize Orbit Families and Transfer Trajectories

2D/3D projections, Jacobi coloring, stability maps — generate high-quality orbital mechanics plots.

- [Visualization Guide](guides/visualization-guide.md) — the full visualization workflow from single orbits to orbit families

## Library Structure

```
core/           Foundation — system definition, equations of motion, orbit data
  ↓
algorithms/     Algorithms — differential correction, continuation, stability analysis, multiple shooting
  ↓
transfer/       Design — DRO→RO transfer search, NLP optimization
  ↓
visualization/  Presentation — orbit family plotting, transfer trajectory visualization

mbse/           Cross-cutting — Protocol interfaces, Pydantic models, requirement tracing, diagram generation
```

| Layer | What It Does | Entry Classes |
|-------|-------------|---------------|
| `core` | Defines celestial systems, integrates equations of motion, manages orbit data | `CR3BP_System`, `CR3BP_Dynamics`, `Orbit` |
| `algorithms` | Corrects periodic orbits, continues orbit families, analyzes stability | `DifferentialCorrection`, `Continuation`, `StabilityAnalysis` |
| `transfer` | Searches and optimizes orbit transfers | `Transfer`, `DROTransferSearch` |
| `visualization` | Plots orbits, families, and transfer trajectories | `OrbitVisualizer`, `FamilyPlotter`, `TransferPlotter` |
| `mbse` | Protocol interfaces, data models, requirement tracing | Protocols, Pydantic models, `RequirementRegistry` |

## Typical Workflow

```python
from e2m2e.core import CR3BP_System, CR3BP_Dynamics, Orbit
from e2m2e.algorithms import DifferentialCorrection, Continuation
from e2m2e.visualization import OrbitVisualizer

# 1. Define the system
system = CR3BP_System.from_known_system("earth_moon")
system.set_characteristic_scales(distance=384400, period=27.32*86400)
system.compute_libration_points()

# 2. Create the dynamics model
dynamics = CR3BP_Dynamics(system)

# 3. Design a periodic orbit
dc = DifferentialCorrection(dynamics)
dc.setup_2D_symmetric_x_fixed_x0(x0=0.8)
orbit, result = dc.iterate_correction(
    np.array([0.8, 0.0, 0.0, 0.0, 0.5, 0.0]), t_half=1.6
)

# 4. Continue to an orbit family
cont = Continuation(corrector=dc, step=0.01)
family = cont.natural_continuation(
    seed_orbit=orbit, param_range=(0.8, 0.95), step_size=0.01
)

# 5. Visualize
viz = OrbitVisualizer(system)
for orb in family:
    viz.plot_2d_projection(orb, plane="xy")
viz.show()
```

## Learn More

| What You Might Want to Know | Where to Look |
|----------------------------|---------------|
| Physical background and mathematical formulations of the CR3BP | [Algorithm Details](reference/algorithms.md) |
| Complete list of all classes and methods | [API Reference](reference/api-reference.md) |
| Coordinate transforms (rotating ↔ inertial frame) | [Coordinate Transforms](core/coordinate.md) |
| Ephemeris dynamics and SPICE management | [Ephemeris System](core/ephemeris_system.md) |
| Multiple shooting method | [Multiple Shooting](algorithms/multiple_shooting.md) |
| Detailed project architecture design | [System Overview](guides/system-overview.md) |
