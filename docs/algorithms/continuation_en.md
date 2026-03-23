# ContinuationMethod

**File**: `e2m2e/algorithms/continuation.py`

**Class Signature**:
```python
class ContinuationMethod(Enum):
    NATURAL = "natural"           # Natural continuation
    PREDICTOR_CORRECTOR = "predictor_corrector"  # Predictor-corrector
    ARC_LENGTH = "arc_length"     # Arc-length continuation
```

## Design Principles

Continuation methods are used to trace solution curves in parameter space, especially suitable for handling solution bifurcations and turning points.

## Arc-Length Continuation

Core idea: Treat parameter $\lambda$ as a function of arc length $s$, and advance along the solution curve through predictor-corrector steps.

### Predictor Step
$$\begin{pmatrix} \mathbf{f}(\mathbf{x}_k) \\ \mathbf{g}(\mathbf{x}_k, s_k) \end{pmatrix} = \mathbf{0}$$

Where $\mathbf{g}$ is the arc-length constraint condition.

### Corrector Step
Use Newton-Raphson iteration to solve:
$$\mathbf{J} \Delta \mathbf{x} = -\mathbf{f}$$

Where $\mathbf{J}$ is the extended Jacobian matrix.

## Core Methods

| Method | Description |
|--------|-------------|
| `predict(state, tangent, ds)` | Predict next point |
| `correct(state, constraints)` | Corrector solve |
| `compute_tangent(jacobian)` | Compute tangent vector |
| `find_bifurcation(points)` | Detect bifurcation points |

## Usage Example

```python
from e2m2e.algorithms.continuation import ContinuationMethod

# Continue along parameter curve
continuer = ArcLengthContinuation(f, jacobian)
curve = continuer.continue_curve(x0, lambda_range, ds=0.01)
```
