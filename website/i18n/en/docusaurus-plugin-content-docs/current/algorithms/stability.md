---
title: "Stability Analysis"
---

# Stability Analysis: StabilityAnalysis

> **File**: `e2m2e/algorithms/stability.py`

Stability analysis determines the local stability of periodic orbits through Floquet multipliers and detects bifurcation points. This is the key tool for understanding the topological structure of orbit families.

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

**How to interpret the results**:

- Max multiplier magnitude $\approx 1.0$: orbit is stable (multipliers lie on the unit circle)
- Max multiplier magnitude $> 1.0$: orbit is unstable (there is an exponentially divergent direction)
- Multipliers crossing the unit circle: bifurcation may be present

## How to Find Bifurcation Points in an Orbit Family

Bifurcation points are "inflection points" of an orbit family — at these points, the topological structure of the orbit changes (e.g., a 3D orbit branches off from a planar orbit).

```python
from e2m2e.algorithms import StabilityAnalysis, BifurcationType

# Analyze stability for each orbit in the family
for orbit in family:
    analyzer = StabilityAnalysis(orbit=orbit, dynamics=dynamics)
    result = analyzer.analyze()

    if result.bifurcation_type != BifurcationType.NONE:
        print(f"Jacobi={orbit.jacobi_constant:.4f}: Detected {result.bifurcation_type}")
```

### Common Bifurcation Types

| Bifurcation Type | Physical Meaning | Impact on Continuation |
|-----------------|-----------------|----------------------|
| `SADDLE_NODE` | Saddle-node bifurcation: endpoint of the family curve | Continuation terminates here; change direction |
| `PERIOD_DOUBLING` | Period-doubling bifurcation: period doubles | Produces a new doubled-period family |
| `PITCHFORK` | Pitchfork bifurcation: symmetry breaking | Produces two symmetric new families |
| `TORUS` | Neimark-Sacker (torus) bifurcation: multipliers leave the unit circle | Produces quasi-periodic motion |

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

Compute stability for an entire orbit family in batch (the visualization module provides a parallel version):

```python
from e2m2e.visualization import compute_stability_for_family

stability_values = compute_stability_for_family(family, system)
# Returns the maximum multiplier magnitude for each orbit
```

See [Visualization Guide](../guides/visualization-guide.md#stability-computation)

## API Quick Reference

| Method | Description |
|--------|-------------|
| `analyze()` | Full analysis: stability type + bifurcation detection |
| `compute_stability_index()` | Compute stability index (max multiplier magnitude) |
| `classify_stability()` | Classify stability type |
| `detect_bifurcation()` | Detect bifurcation type |

For the complete API documentation, see [API Reference](../reference/api-reference.md).

## Mathematical Background

### Floquet Theory

For a periodic orbit $\mathbf{x}(t)$, small perturbations in its vicinity satisfy:

$$\Delta \dot{\mathbf{x}}(t) = \mathbf{A}(t) \Delta \mathbf{x}(t)$$

where $\mathbf{A}(t)$ is a periodic coefficient matrix. The eigenvalues of the Monodromy matrix $\boldsymbol{\Phi}(T)$ are the Floquet multipliers:

$$\boldsymbol{\Phi}(T) \mathbf{v} = \lambda \mathbf{v}$$

Multipliers inside/on/outside the unit circle correspond to stable/unstable/marginal states, respectively.

### Stability Index

$$\nu = \max_i |\lambda_i|$$

where $\lambda_i$ are the Floquet multipliers. $\nu > 1$ indicates instability.
