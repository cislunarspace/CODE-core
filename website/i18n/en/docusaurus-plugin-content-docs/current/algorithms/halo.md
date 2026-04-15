---
format: md
title: Halo Orbit Generation and Orbit Family Continuation
---

# Halo Orbit Generation and Orbit Family Continuation

> Halo orbits are three-dimensional periodic orbits around the L1/L2 libration points. This page covers Richardson initial guesses, differential correction, pseudo-arclength continuation, and CLI scripts.

## Quick Start: Generate a Halo Orbit from Scratch

```python
import numpy as np
from e2m2e.core import CR3BP_System, CR3BP_Dynamics
from e2m2e.algorithms import DifferentialCorrection

system = CR3BP_System.from_known_system("earth_moon")
system.set_characteristic_scales(distance=384400, period=27.32*86400)
system.compute_libration_points()
dynamics = CR3BP_Dynamics(system)

# Richardson third-order approximation provides the initial guess
from e2m2e.algorithms import compute_halo_initial_guess
initial_state, t_half = compute_halo_initial_guess(
    system=system, libration_point=1, amplitude_z=0.1
)

# Differential correction
dc = DifferentialCorrection(dynamic=dynamics)
dc.setup_halo_orbit_fixed_z0(z0=initial_state[2], libration_point=1)
orbit, result = dc.iterate_correction(initial_state, t_half=t_half)

print(f"Period: {orbit.period:.4f}, Jacobi: {orbit.jacobi_constant:.6f}")
```

## Quick Start: Generate a Halo Orbit Family

```python
from e2m2e.algorithms import Continuation

dc = DifferentialCorrection(dynamic=dynamics)
cont = Continuation(corrector=dc)

# Generate seed orbit
seed = cont.generate_halo_seed_orbit(
    libration_point=1, amplitude_z=0.23, halo_class=0,
)

# Pseudo-arclength continuation
family = cont.halo_pseudo_arclength_continuation(
    seed_orbit=seed,
    n_orbits=10,
    direction="both",
    step_size=0.0045,
    verbose=True,
)

family.save_to_file("output/halo_family.json")
print(f"Generated {len(family)} Halo orbits")
```

---

**Related Source Code**:

| Module | Path |
|--------|------|
| Continuation and Halo families | `e2m2e/algorithms/continuation.py` |
| Richardson initial guess and Halo differential correction | `e2m2e/algorithms/differential_correction.py` |
| Single orbit / family generation scripts | `scripts/generate/generate_halo_orbit.py`, `scripts/generate/generate_halo_family.py` |
| Plotting scripts | `scripts/plot/plot_halo_orbit.py`, `scripts/plot/plot_halo_family.py` |
| Analytical initial guess tests | `tests/algorithms/test_analytical_halo.py` |

---

## Feature Overview

1. **Single Halo periodic orbit**: Richardson third-order approximation initial guess + `DifferentialCorrection` (fixed `z0` or `x0`).
2. **Seed orbit**: `Continuation.generate_halo_seed_orbit` — same as the single-orbit workflow, with `parameters` populated (`libration_point`, `amplitude_z`, `halo_class`).
3. **Pseudo-arclength continuation (XZ symmetric)**: `Continuation.pseudo_arclength_continuation` — free variables \(\mathbf{X}=[r_x, r_z, \dot{y}, T/2]\), isomorphic to `continuation_PAL_CR3BP` (with `plane=13`) from the `CR3BP_MATLAB_Library`.
4. **Halo orbit family**: `Continuation.halo_pseudo_arclength_continuation` — built on PAL with positive/negative branch step sizes, `DirectionalIncrement`, and target component configuration aligned with the script `FAMILY_L1Halo_North.m`; differential correction strategy is selectable (see below).

---

## Core API

### `DifferentialCorrection`

| Method | Purpose |
|--------|---------|
| `compute_halo_initial_guess(mu, z_amplitude, L, halo_class)` | Richardson/MATLAB-scaled initial guess (`x0`, `vy0`, `T_half`, etc.) |
| `setup_halo_orbit_fixed_z0(z0, libration_point)` | Fix initial \(z_0\), free variables \((x_0, \dot{y}_0, T/2)\) |
| `setup_halo_orbit_fixed_x0(x0, libration_point)` | Fix initial \(x_0\), free variables \((z_0, \dot{y}_0, T/2)\) |

Halo convergence results validate a lower bound on the full period \(T\) (to avoid spurious roots with \(T\to 0\)); see the handling of `halo_orbit_fixed_z0` / `halo_orbit_fixed_x0` in `iterate_correction`.

### `Continuation`

| Method | Purpose |
|--------|---------|
| `generate_halo_seed_orbit(libration_point, amplitude_z, halo_class, ...)` | Generate and correct a single seed Halo |
| `generate_halo_family(seed_orbit, ...)` | Independent Richardson initial guesses by `amplitude_z` stepping (natural parameter, not PAL) |
| `pseudo_arclength_continuation(seed_orbit, ...)` | General XZ-symmetric pseudo-arclength continuation (single direction: `positive` / `negative`) |
| `halo_pseudo_arclength_continuation(seed_orbit, ...)` | Halo-specific: bidirectional branches, default step sizes and optional parameters aligned with MATLAB scripts |

**`pseudo_arclength_continuation` key parameters**:

- `step_size`: \(|\Delta S|\) (positive; direction determined by `direction`).
- `dc_scheme`: `adaptive` (switches between 3D symmetric fixed-x / fixed-z based on \(\Delta x\), \(\Delta z\)), `matlab_halo_type1` (always `setup_halo_orbit_fixed_x0`), `matlab_halo_type2` (switches between fixed-x / fixed-z based on amplitude).
- `directional_increment`, `target_vector` (0-indexed: \(0=r_x,1=r_z,2=\dot y,3=T/2\)), `target_direction`: consistent with MATLAB `DirectionalIncrement` / `TargetVector` / `TargetDirection`.

**`halo_pseudo_arclength_continuation` key points**:

- `step_size` / `step_size_negative`: correspond to the positive branch `DeltaS~0.0045` and negative branch `|DeltaS|~0.009` in the script.
- Default `dc_scheme='adaptive'`: more robust when PAL initial guesses under Python STM Newton behave differently than MATLAB's fixed `x0` approach; to align with MATLAB `type=1`, set `matlab_halo_type1` (the implementation can retry fixed-z after fixed-x failure).

---

## PAL Implementation Notes (Differences from MATLAB and Safeguards)

- **Inner Newton termination**: Consistent with `continuation_PAL_CR3BP.m`, when \(\|F\|\) is already below tolerance, an extra Newton step on \(\mathbf{X}_{new}\) is **not** taken, to avoid pushing the point away from the physical solution.
- **Newton step clamping**: An upper bound is imposed on \(\Delta\mathbf{X}\) components to mitigate the risk of jumping to a different branch of \(F=0\) (e.g., \(|r_x|\gg 1\)).
- **Physical solution filtering**: If the PAL endpoint clearly deviates from the typical L1 Halo range, it **falls back** to the Euler prediction \(\mathbf{X}+\Delta S\,\dot{\mathbf{X}}\) followed by differential correction.
- **MATLAB inner loop uses fixed `X` for `F`**: This library computes \(F\) using the current iteration's \(\mathbf{X}_{new}\) in the PAL inner loop, which is theoretically more self-consistent; reproducing MATLAB numerics line-by-line requires a separate branch.

---

## CLI Scripts

| Script | Description |
|--------|-------------|
| `scripts/generate/generate_halo_orbit.py` | Single Halo, with configurable `libration_point`, `amplitude_z`, `halo_class` |
| `scripts/generate/generate_halo_family.py` | Calls `halo_pseudo_arclength_continuation` from a seed, outputs `output/halo/*.json` |
| `scripts/plot/plot_halo_orbit.py` | Single orbit or multi-orbit plotting from JSON |
| `scripts/plot/plot_halo_family.py` | Orbit family JSON: 2D/3D, Jacobi, stability, etc. |

Common constants (e.g., \(\mu\)) are in `scripts/utils/common.py`.

---

## References and Comparison

- Richardson, D. L. (1980). Analytic construction of periodic orbits about the collinear points. *Celestial Mechanics*.
- Local reference implementation: `CR3BP_MATLAB_Library` — `continuation_PAL_CR3BP.m`, `examples/FAMILY_L1Halo_North.m`.

---

## See Also

- [Orbit Generation Guide](../guides/orbit-generation.md) — tutorial entry point
- [Continuation Module Overview](continuation.md) — `Continuation` class index
- [Differential Correction](differential_correction.md) — symmetry configuration details
