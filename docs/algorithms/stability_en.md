# StabilityAnalysis

**File**: `e2m2e/algorithms/stability.py`

**Class Signature**:
```python
class StabilityAnalysis:
    """Orbit stability analysis"""
```

## Floquet Theory

For a periodic orbit $\mathbf{x}(t)$, small perturbations nearby satisfy:
$$\Delta \dot{\mathbf{x}}(t) = \mathbf{A}(t) \Delta \mathbf{x}(t)$$

Where $\mathbf{A}(t)$ is the state transition matrix, periodic: $\mathbf{A}(t+T) = \mathbf{A}(t)$.

### Floquet Multipliers
$$\boldsymbol{\Phi}(T) \mathbf{v} = \lambda \mathbf{v}$$

$\boldsymbol{\Phi}(T)$ is the monodromy matrix, $\lambda$ is the Floquet multiplier.

## Stability Classification

| Stability Type | Multiplier Characteristics | Orbit Property |
|----------------|---------------------------|----------------|
| Stable | All $\|\lambda\| = 1$ | Lyapunov stable |
| Unstable | Exists $\|\lambda\| > 1$ | Exponential divergence |
| Elliptic | Multipliers on unit circle | KAM applicable |
| Parabolic | Multipliers $= 1$ | Critical case |

## Core Methods

| Method | Description |
|--------|-------------|
| `compute_floquet_multipliers(orbit)` | Compute Floquet multipliers |
| `classify_stability(multipliers)` | Classify stability |
| `compute_stability_index(multipliers)` | Compute stability index |
| `analyze_lyapunov(orbit, dt, n_orbits)` | Lyapunov exponent analysis |

## Stability Index

$$\nu = \frac{1}{n} \sum_{i=1}^{n} \ln |\lambda_i|$$

Where $\lambda_i$ are Floquet multipliers.

## Usage Example

```python
from e2m2e.algorithms.stability import StabilityAnalysis

analyzer = StabilityAnalysis(system, dynamics)

# Analyze orbit stability
multipliers = analyzer.compute_floquet_multipliers(orbit)
stability_type, index = analyzer.classify_stability(multipliers)

print(f"Stability: {stability_type}, Index: {index}")
```
