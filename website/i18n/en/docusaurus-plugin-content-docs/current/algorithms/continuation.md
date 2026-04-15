---
title: "Orbit Family Continuation"
---

# Orbit Family Continuation: Continuation

> **File**: `e2m2e/algorithms/continuation.py`

Continuation starts from a converged periodic orbit (the seed) and traces along a parameter direction step by step to generate a family of orbits. This is the core tool for systematically exploring orbit space.

## How to Generate a Family of Periodic Orbits

```python
from e2m2e.core import CR3BP_System, CR3BP_Dynamics
from e2m2e.algorithms import DifferentialCorrection, Continuation

# 1. First obtain a seed orbit
system = CR3BP_System.from_known_system("earth_moon")
system.set_characteristic_scales(distance=384400, period=27.32*86400)
system.compute_libration_points()
dynamics = CR3BP_Dynamics(system)

dc = DifferentialCorrection(dynamic=dynamics)
dc.setup_2D_symmetric_x_fixed_x0(x0=0.8)
seed_orbit, _ = dc.iterate_correction(
    np.array([0.8, 0.0, 0.0, 0.0, 0.5, 0.0]), t_half=1.6
)

# 2. Continue along the parameter direction
cont = Continuation(corrector=dc, step=0.01)
family = cont.natural_continuation(
    seed_orbit=seed_orbit,
    param_range=(0.8, 0.95),  # x0 range
    step_size=0.01,
    verbose=True,
)

# 3. Save
family.save_to_file("output/dro_family.json")
print(f"Generated {len(family)} orbits")
```

## Natural Continuation vs. Pseudo-Arclength Continuation

| | Natural Continuation | Pseudo-Arclength Continuation |
|---|---------|-----------|
| **Principle** | Fix one parameter and vary it step by step | Trace along the tangent to the curve in parameter space |
| **Advantage** | Simple, few parameters | Can navigate around turning points |
| **Disadvantage** | Fails at turning points | More parameters, harder to tune |
| **Best for** | Families with monotonically changing parameters | Families with inflection/fold points (e.g., Halo families) |

**Recommendation**: Try natural continuation first. If it fails at a certain parameter value (error or orbit jumping), switch to pseudo-arclength continuation.

### Pseudo-Arclength Continuation

```python
family = cont.pseudo_arclength_continuation(
    seed_orbit=seed_orbit,
    n_orbits=50,
    direction="positive",  # or "negative"
    verbose=True,
)
```

Key parameters:
- `n_orbits`: number of orbits to generate
- `direction`: continuation direction ("positive" or "negative")
- Initial step size is set via `Continuation(step=...)`; the algorithm adjusts adaptively

## How to Generate a Halo Orbit Family

Halo orbit families typically use pseudo-arclength continuation because the family curve contains turning points:

```python
from e2m2e.algorithms import Continuation, DifferentialCorrection

dc = DifferentialCorrection(dynamic=dynamics)
cont = Continuation(corrector=dc)

# Generate seed orbit
seed = cont.generate_halo_seed_orbit(
    libration_point=1,   # L1
    amplitude_z=0.23,    # z-direction amplitude
    halo_class=0,        # Northern halo
)

# Pseudo-arclength continuation to generate family
family = cont.halo_pseudo_arclength_continuation(
    seed_orbit=seed,
    n_orbits=10,
    direction="both",    # bidirectional continuation
    step_size=0.0045,
    verbose=True,
)
```

See [Halo Orbits](halo.md) for details (including Richardson initial guess, PAL details, CLI scripts)

## API Quick Reference

| Method | Description |
|------|------|
| `natural_continuation(seed_orbit, param_range, step_size)` | Natural parameter continuation |
| `pseudo_arclength_continuation(seed_orbit, n_orbits, direction)` | Pseudo-arclength continuation |
| `generate_halo_seed_orbit(libration_point, amplitude_z, halo_class)` | Generate a Halo seed orbit |
| `generate_halo_family(seed_orbit, ...)` | Halo family by amplitude stepping |
| `halo_pseudo_arclength_continuation(seed_orbit, n_orbits, direction)` | Halo-specific pseudo-arclength continuation |

For the complete API documentation, see [API Reference](../reference/api-reference.md).
