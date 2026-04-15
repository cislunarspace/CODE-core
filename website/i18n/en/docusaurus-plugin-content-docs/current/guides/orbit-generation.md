---
title: Orbit Generation Guide
---

# Orbit Generation Guide

> From initial guess to converged periodic orbit, and on to a complete orbit family.

## General Procedure

All periodic orbit generation follows the same steps:

```python
import numpy as np
from e2m2e.core import CR3BP_System, CR3BP_Dynamics
from e2m2e.algorithms import DifferentialCorrection

# 1. Create the system
system = CR3BP_System.from_known_system("earth_moon")
system.set_characteristic_scales(distance=384400, period=27.32*86400)
system.compute_libration_points()

# 2. Create the dynamics model
dynamics = CR3BP_Dynamics(system)

# 3. Create the corrector and select a symmetry configuration
dc = DifferentialCorrection(dynamic=dynamics)
```

The next step depends on the orbit type you want to design. Continue reading and choose the relevant section.

---

## DRO (Distant Retrograde Orbit)

A DRO is a large-amplitude retrograde orbit around the Moon, forming a teardrop shape in the rotating frame.

**Characteristics**: Symmetric about the x-axis, initial conditions $[x_0, 0, 0, 0, \dot{y}_0, 0]$, with $y=0, \dot{x}=0$ at half-period.

```python
dc.setup_2D_symmetric_x_fixed_x0(x0=0.8)

initial_state = np.array([0.8, 0.0, 0.0, 0.0, 0.5, 0.0])
t_half_guess = 1.6

orbit, result = dc.iterate_correction(initial_state, t_half=t_half_guess)
print(f"Period: {orbit.period:.4f}, Jacobi: {orbit.jacobi_constant:.6f}")
```

### Parameter Range Reference

| Parameter | Typical Range | Description |
|-----------|--------------|-------------|
| $x_0$ | 0.6 - 0.95 | Initial x-coordinate (relative to L1/L2) |
| $\dot{y}_0$ | 0.3 - 0.8 | Initial y-direction velocity |
| $T/2$ | 1.4 - 1.8 | Half-period (dimensionless) |

### DRO with Fixed Period

If you need to specify the period exactly, use `setup_2D_symmetric_x_fixed_t`:

```python
dc.setup_2D_symmetric_x_fixed_t(t_half=1.6)
# Here x0 and vy are automatically solved by the corrector
```

→ For detailed differential correction configuration, see [Differential Correction](../algorithms/differential_correction.md)

---

## Halo Orbits

Halo orbits are three-dimensional periodic orbits around the L1/L2 libration points.

### Generate with Richardson Initial Guess

```python
from e2m2e.algorithms import compute_halo_initial_guess

initial_state, t_half = compute_halo_initial_guess(
    system=system, libration_point=1, amplitude_z=0.1
)

dc.setup_halo_orbit_fixed_z0(z0=initial_state[2], libration_point=1)
orbit, result = dc.iterate_correction(initial_state, t_half=t_half)
```

### Generate a Halo Orbit Family

```python
from e2m2e.algorithms import Continuation

cont = Continuation(corrector=dc)
seed = cont.generate_halo_seed_orbit(
    libration_point=1, amplitude_z=0.23, halo_class=0,
)
family = cont.halo_pseudo_arclength_continuation(
    seed_orbit=seed, n_orbits=10, direction="both",
    step_size=0.0045, verbose=True,
)
```

### L1 vs L2

| Property | L1 Halo | L2 Halo |
|----------|---------|---------|
| Location | Near L1 point | Near L2 point |
| $x_0$ range | $0.8 < x_0 < 1.0$ | $1.0 < x_0 < 1.2$ |
| Amplitude | Generally smaller | Generally larger |

→ For detailed Halo documentation (Richardson initial guess, PAL implementation, MATLAB comparison, command-line scripts), see [Halo Orbits](../algorithms/halo.md)

---

## Lyapunov Orbits

Lyapunov orbits are two-dimensional periodic orbits in the libration-point plane ($z=0$).

```python
dc.setup_2D_symmetric_x_fixed_x0(x0=system.L1[0] + 0.01)

initial_state = np.array([system.L1[0] + 0.01, 0.0, 0.0, 0.0, 0.3, 0.0])
orbit, result = dc.iterate_correction(initial_state, t_half=1.5)
```

The difference between Lyapunov and DRO: Lyapunov orbits are near L1/L2 with shorter periods; DRO orbits are near the Moon with longer periods.

---

## Orbit Family Continuation

Starting from a converged seed orbit, generate a family of orbits:

```python
from e2m2e.algorithms import Continuation

cont = Continuation(corrector=dc, step=0.01)

# Natural continuation (use when parameters change monotonically)
family = cont.natural_continuation(
    seed_orbit=orbit,
    param_range=(0.8, 0.95),
    step_size=0.01,
)
```

**When to use natural continuation vs pseudo-arclength continuation**, along with parameter details, see [Orbit Family Continuation](../algorithms/continuation.md).

---

## Save and Load

```python
# Save
family.save_to_file("output/dro_family.json")
orbit.save_to_file("output/dro_single.json")

# Load
from e2m2e.core.orbit import OrbitFamily, Orbit
family = OrbitFamily.load_from_file("output/dro_family.json", system=system)
orbit = Orbit.load_from_file("output/dro_single.json", system=system)
```

---

## Troubleshooting

### Initial Value Does Not Converge

- First propagate the initial value and check whether it is close to periodic ($y$ and $\dot{x}$ should be near 0 at half-period)
- Start from small amplitude / short period and gradually continue
- For Halo orbits, use `compute_halo_initial_guess` rather than manual construction

### Large-Amplitude Orbits Needed

Start from a small-amplitude seed and use the [continuation algorithm](../algorithms/continuation.md) to gradually increase the amplitude. Natural continuation will fail at turning points of the family curve; switch to pseudo-arclength continuation at that point.

---

## Reference

- [Differential Correction](../algorithms/differential_correction.md) — Symmetry configuration selection, convergence troubleshooting
- [Orbit Family Continuation](../algorithms/continuation.md) — Natural vs pseudo-arclength continuation
- [Halo Orbits](../algorithms/halo.md) — Richardson initial guess, PAL, MATLAB comparison
- [Stability Analysis](../algorithms/stability.md) — Floquet analysis, bifurcation detection
- [Visualization Guide](visualization-guide.md) — Orbit family and transfer trajectory visualization
