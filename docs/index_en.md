# E2M2E — Cislunar Transfer Orbit Design Library

An orbital mechanics toolkit based on the Circular Restricted Three-Body Problem (CR3BP) for designing periodic orbits and transfer trajectories in the Earth-Moon space.

## Get Started in 30 Seconds

```bash
pip install e2m2e
```

```python
from e2m2e.core import CR3BP_System, CR3BP_Dynamics
from e2m2e.algorithms import DifferentialCorrection
import numpy as np

# Create the Earth-Moon system and design a DRO
system = CR3BP_System.from_known_system("earth_moon")
system.set_characteristic_scales(distance=384400, period=27.32*86400)
system.compute_libration_points()

dynamics = CR3BP_Dynamics(system)
dc = DifferentialCorrection(dynamics)
dc.setup_2D_symmetric_x_fixed_x0(x0=0.8)

initial_state = np.array([0.8, 0.0, 0.0, 0.0, 0.5, 0.0])
orbit, result = dc.iterate_correction(initial_state, t_half=1.6)

print(f"Period: {orbit.period:.4f}")
print(f"Jacobi constant: {orbit.jacobi_constant:.6f}")
```

## What You Can Do

### Design Periodic Orbits

DRO, Halo, Lyapunov — from initial guess to converged orbit, then continue to an entire family.

- [Orbit Generation Tutorial](guides/orbit-generation_en.md) — DRO / Halo / Lyapunov generation workflows
- [Differential Correction](algorithms/differential_correction_en.md) — How to correct a seed into a precise periodic orbit
- [Continuation](algorithms/continuation_en.md) — How to generate a family of periodic orbits
- [Halo Orbits](algorithms/halo_en.md) — Richardson initial guess, pseudo-arclength continuation, CLI scripts

### Analyze Orbital Stability

Floquet multipliers, bifurcation detection, stability indices — understand the dynamical properties of your orbits.

- [Stability Analysis](algorithms/stability_en.md) — How to assess stability and find bifurcation points

### Visualize Orbit Families and Transfer Trajectories

2D/3D projections, Jacobi-colored plots, stability diagrams — publication-quality orbital mechanics figures.

- [Visualization Guide](guides/visualization-guide_en.md) — Full workflow from single orbits to family plots

## Library Structure

```
core/           Foundation — system definition, equations of motion, orbit data
  ↓
algorithms/     Numerical — differential correction, continuation, stability, multiple shooting
  ↓
transfer/       Design — DRO→RO transfer search, NLP optimization
  ↓
visualization/  Display — orbit family plots, transfer trajectory visualization

mbse/           Cross-cutting — Protocol interfaces, Pydantic models, requirement tracing, diagram generation
```

| Layer | Purpose | Entry classes |
|-------|---------|---------------|
| `core` | Define celestial systems, integrate equations, manage orbit data | `CR3BP_System`, `CR3BP_Dynamics`, `Orbit` |
| `algorithms` | Correct periodic orbits, continue families, analyze stability | `DifferentialCorrection`, `Continuation`, `StabilityAnalysis` |
| `transfer` | Search and optimize orbit transfers | `Transfer`, `DROTransferSearch` |
| `visualization` | Plot orbits, families, and transfer trajectories | `OrbitVisualizer`, `FamilyPlotter`, `TransferPlotter` |
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

# 4. Continue to get an orbit family
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

| What you might want to know | Where to look |
|-----------------------------|---------------|
| CR3BP physics and math | [Algorithm Details](reference/algorithms_en.md) |
| Complete list of classes and methods | [API Reference](reference/api-reference_en.md) |
| Coordinate transformations (rotating ↔ inertial) | [Coordinate Transformation](core/coordinate_en.md) |
| Project architecture details | [System Overview](guides/system-overview_en.md) |
