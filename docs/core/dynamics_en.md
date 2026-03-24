# CR3BP_Dynamics

**File**: `e2m2e/core/dynamics.py`

**Class Signature**:
```python
class CR3BP_Dynamics:
    """CR3BP dynamics equations"""
```

## Design Principles

The `CR3BP_Dynamics` class encapsulates CR3BP equations of motion and numerical integration methods. The equations of motion in rotating coordinates (dimensionless form):

$$\begin{cases}
\dot{x} = v_x \\
\dot{y} = v_y \\
\dot{z} = v_z \\
\dot{v}_x = 2v_y + x - \frac{(1-\mu)(x+\mu)}{r_1^3} - \frac{\mu(x-1+\mu)}{r_2^3} \\
\dot{v}_y = -2v_x + y - \frac{(1-\mu)y}{r_1^3} - \frac{\mu y}{r_2^3} \\
\dot{v}_z = -\frac{(1-\mu)z}{r_1^3} - \frac{\mu z}{r_2^3}
\end{cases}$$

Where:
$$r_1 = \sqrt{(x+\mu)^2 + y^2 + z^2}, \quad r_2 = \sqrt{(x-1+\mu)^2 + y^2 + z^2}$$

## Core Features

1. **State Propagation**: The `propagate()` method uses scipy's `solve_ivp` for numerical integration
2. **State Transition Matrix (STM)**: The 42-dimensional augmented state is integrated simultaneously via `equations_with_stm()`
3. **Jacobi Constant Monitoring**: Real-time calculation of Jacobi constant for accuracy verification

## Core Methods

| Method | Description |
|--------|-------------|
| `equations_of_motion(t, state)` | 6-dimensional equations of motion |
| `equations_with_stm(t, augmented_state)` | 42-dimensional augmented equations (with STM) |
| `propagate(initial_state, t_span, with_stm=False)` | Propagate trajectory |
| `compute_state_transition_matrix(initial_state, t)` | Compute STM |
| `compute_jacobi_constant(state)` | Calculate Jacobi constant |
| `check_cross_section(state, plane, value)` | Detect section crossing |

## Usage Example

```python
from e2m2e.core.dynamics import CR3BP_Dynamics

# Create dynamics model
dynamics = CR3BP_Dynamics(system)

# Propagate trajectory
result = dynamics.propagate(
    initial_state=np.array([0.8, 0, 0, 0, 1.0, 0]),
    t_span=(0, 10.0)
)

print(f"Number of trajectory points: {len(result['states'])}")
print(f"Final state: {result['states'][-1]}")
```
