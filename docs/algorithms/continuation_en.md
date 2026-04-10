# Continuation（Orbit Family Continuation）

**File**: `e2m2e/algorithms/continuation.py`

**Class Signature**:
```python
class Continuation:
    """Orbit family continuation: natural parameter, pseudo-arclength (XZ symmetric), Halo seed & family generation"""
```

## Design Principles

Continuation starts from a known periodic orbit (seed) and incrementally generates orbit families along parameter or pseudo-arclength directions. Natural parameter continuation is simple and effective when the parameter varies monotonically; pseudo-arclength continuation introduces tangent and constraint terms on free variables \(\mathbf{X}=[r_x,r_z,\dot y,T/2]\) to trace family curves with fold-back points.

## Core Methods

| Method | Description |
|--------|-------------|
| `natural_continuation(...)` | Natural parameter continuation |
| `pseudo_arclength_continuation(seed_orbit, ...)` | XZ symmetric pseudo-arclength continuation (direction: `positive` / `negative`) |
| `generate_halo_seed_orbit(...)` | Generate Halo seed orbit |
| `generate_halo_family(...)` | Halo family by `amplitude_z` stepping (independent Richardson initial guess) |
| `halo_pseudo_arclength_continuation(...)` | Halo-specific: bi-directional branches, step size & MATLAB script alignment |

For Halo initial guess, PAL details, script entry points and MATLAB comparison, see **[Halo Orbit Algorithm Documentation](halo.md)**.

## Usage Example

```python
from e2m2e.algorithms.continuation import Continuation

continuation = Continuation(corrector, step=0.01)
family = continuation.natural_continuation(
    seed_orbit=seed_orbit,
    param_range=(0.8, 1.0),
    step_size=0.01,
    verbose=True,
)
```

For pseudo-arclength and Halo family examples, see [Halo](halo.md) and [Orbit Generation Guide - Halo](../guides/orbit-generation.md).
