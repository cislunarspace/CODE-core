---
title: 'Stability Analysis: StabilityAnalysis'
---

# Stability Analysis: StabilityAnalysis

> **File**: `e2m2e/algorithms/stability.py`

Stability analysis uses Floquet multipliers to determine the local stability of periodic orbits and detect bifurcation points. This is a key tool for understanding the topological structure of orbit families.

## How to Determine Whether an Orbit Is Stable

```python
from e2m2e.core import CR3BP_System, CR3BP_Dynamics, Orbit
from e2m2e.algorithms import StabilityAnalysis

system = CR3BP_System.from_known_system("earth_moon")
dynamics = CR3BP_Dynamics(system)

# Assume orbit is an existing periodic orbit
analyzer = StabilityAnalysis(orbit=orbit, dynamics=dynamics)
result = analyzer.analyze()

print(f"Stability type: {result.stability_type}")
print(f"Max multiplier magnitude: {result.max_multiplier_magnitude:.6f}")
```

**How to interpret**:

- Max multiplier magnitude $\approx 1.0$: orbit is stable (multipliers on the unit circle)
- Max multiplier magnitude $> 1.0$: orbit is unstable (exponential divergence direction exists)
- Multiplier exactly crossing the unit circle: possible bifurcation

## How to Find Bifurcation Points in an Orbit Family

Bifurcation points are "turning points" of the orbit family -- at these points, the topological structure of the orbits changes (e.g., a planar orbit bifurcates into a 3D orbit).

```python
from e2m2e.algorithms import StabilityAnalysis, BifurcationType

# Analyze stability for each orbit in the family
for orbit in family:
    analyzer = StabilityAnalysis(orbit=orbit, dynamics=dynamics)
    result = analyzer.analyze()

    if result.bifurcation_type != BifurcationType.NONE:
        print(f"Jacobi={orbit.jacobi_constant:.4f}: detected {result.bifurcation_type}")
```

### Common Bifurcation Types

| Bifurcation Type | Physical Meaning | Impact on Continuation |
|-----------------|------------------|----------------------|
| `SADDLE_NODE` | Saddle-node bifurcation: endpoint of the family curve | Continuation terminates here; change direction |
| `PERIOD_DOUBLING` | Period-doubling bifurcation: period doubles | Produces a new period-doubled family |
| `PITCHFORK` | Pitchfork bifurcation: symmetry breaking | Produces two symmetric new families |
| `TORUS` | Torus bifurcation: multipliers leave the unit circle | Produces quasi-periodic motion |

### Stability Type Quick Reference

| Type | Meaning |
|------|---------|
| `STABLE` | All multipliers on the unit circle, Lyapunov stable |
| `UNSTABLE` | Multipliers exist outside the unit circle |
| `HYPERBOLIC` | Hyperbolic: multipliers not on the unit circle |
| `ELLIPTIC` | Elliptic: all multipliers on the unit circle |
| `MARGINALLY_STABLE` | Marginally stable |
| `PARABOLIC` | Parabolic: multipliers exactly equal to 1 |

## Batch Stability Computation

Batch-compute stability for an orbit family (the visualization module provides a parallel version):

```python
from e2m2e.visualization import compute_stability_for_family

stability_values = compute_stability_for_family(family, system)
# Returns the max multiplier magnitude for each orbit
```

See [Visualization Guide](../guides/visualization-guide_en.md#stability-computation) for details.

## API Quick Reference

| Method | Description |
|--------|-------------|
| `analyze()` | Full analysis: stability type + bifurcation detection |
| `compute_stability_index()` | Compute stability index (max multiplier magnitude) |
| `classify_stability()` | Classify stability type |
| `detect_bifurcation()` | Detect bifurcation type |

For the full API documentation, see [API Reference](../reference/api-reference_en.md).

## Mathematical Background

### Floquet Theory

For a periodic orbit $\mathbf{x}(t)$, small perturbations nearby satisfy:

$$\Delta \dot{\mathbf{x}}(t) = \mathbf{A}(t) \Delta \mathbf{x}(t)$$

where $\mathbf{A}(t)$ is a periodic coefficient matrix. The eigenvalues of the Monodromy matrix $\boldsymbol{\Phi}(T)$ are the Floquet multipliers:

$$\boldsymbol{\Phi}(T) \mathbf{v} = \lambda \mathbf{v}$$

Multipliers inside/outside/on the unit circle correspond to stable/unstable/critical states, respectively.

### Stability Index

$$\nu = \max_i |\lambda_i|$$

where $\lambda_i$ are the Floquet multipliers. $\nu > 1$ indicates instability.
