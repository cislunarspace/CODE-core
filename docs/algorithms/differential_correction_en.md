# DifferentialCorrection

**File**: `e2m2e/algorithms/differential_correction.py`

**Class Signature**:
```python
class DifferentialCorrection:
    """Differential correction algorithm"""
```

## Design Principles

Differential correction finds precise periodic orbits by linearizing the Poincaré map. For periodic orbit problems, the state must satisfy periodic conditions:
$$\mathbf{x}(T) - \mathbf{x}(0) = \mathbf{0}$$

## Single Parameter Correction Method

Applicable to correcting single parameters (e.g., $C_J$) in family continuation:

1. **Construct correction equation**
$$\mathbf{F}(\mathbf{x}, \lambda) = \begin{pmatrix} \mathbf{x}(T; \mathbf{x}_0, \lambda) - \mathbf{x}_0 \\ \phi(\mathbf{x}_0, \lambda) \end{pmatrix} = \mathbf{0}$$

Where $\phi$ is an artificially added phase condition.

2. **Solve correction equation**
Using Newton-Raphson iteration:
$$\mathbf{J} \Delta \mathbf{x} = -\mathbf{F}$$

## Core Methods

| Method | Description |
|--------|-------------|
| `correct_period(orbit, target_state)` | Periodic orbit correction |
| `correct_poincare(state, section)` | Poincaré section correction |
| `compute_poincare_map(state, section)` | Compute Poincaré map |
| `compute_monodromy(state)` | Compute monodromy matrix |

## Periodic Orbit Detection

Period detection condition:
$$\|\mathbf{x}(T) - \mathbf{x}(0)\| < \epsilon_{period}$$

## Usage Example

```python
from e2m2e.algorithms.differential_correction import DifferentialCorrection

corrector = DifferentialCorrection(system, dynamics)

# Correct periodic orbit
corrected_orbit = corrector.correct_period(
    initial_guess=orbit,
    max_iterations=50,
    tolerance=1e-10
)
```
