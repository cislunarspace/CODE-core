# CR3BP Orbital Mechanics Algorithm Technical Documentation

## 1. Overview

This document describes the Circular Restricted Three-Body Problem (CR3BP) orbital mechanics algorithms implemented in the Earth-to-Moon-to-Earth (E2M2E) project. These algorithms are used to generate Distant Retrograde Orbits (DRO) and other periodic orbits in the Earth-Moon system.

### 1.1 CR3BP Model Introduction

CR3BP describes the motion of a small mass body under the gravitational attraction of two large mass bodies (primary and secondary). In the Earth-Moon system:
- Primary body: Earth
- Secondary body: Moon
- Small body: Spacecraft

In the rotating coordinate system (synodic frame), the dimensionless equations of motion are:

$$\ddot{x} - 2\dot{y} = \frac{\partial U}{\partial x}$$
$$\ddot{y} + 2\dot{x} = \frac{\partial U}{\partial y}$$
$$\ddot{z} = \frac{\partial U}{\partial z}$$

Where $U = \frac{1}{2}(x^2 + y^2) + \frac{1-\mu}{r_1} + \frac{\mu}{r_2}$ is the artificial potential energy function, and $r_1$ and $r_2$ are the distances from the spacecraft to the primary and secondary bodies respectively.

### 1.2 Key Parameters (Based on Paper Table 1)

| Parameter | Symbol | Value | Unit |
|------|------|-----|------|
| Earth-Moon mass ratio | $\mu$ | $1.21506683 \times 10^{-2}$ | - |
| Distance unit | $D_U$ | $3.84405 \times 10^5$ | km |
| Time unit | $T_U$ | $4.34811305$ | days |
| Velocity unit | $V_U$ | $1023.23281$ | m/s |

---

## 2. System Parameters Module

### 2.1 CR3BP_System Class

**File**: `e2m2e/core/system.py`

**Function**: Define CR3BP system parameters and libration point calculations.

**Main Attributes**:
- `mu`: Mass parameter $\mu = m_{moon} / (m_{earth} + m_{moon})$
- `primary_body`: Primary body name
- `secondary_body`: Secondary body name
- `L1-L5`: Five libration point coordinates

**Main Methods**:

```python
class CR3BP_System:
    def compute_libration_points() -> List[Tuple[float, float]]
        """Compute five libration point positions (L1-L5)"""
        
    def get_jacobi_constant(state: np.ndarray) -> float
        """Calculate Jacobi constant C_J = 2U - v²"""
        
    def dimensionless_to_physical(state: np.ndarray) -> Dict[str, np.ndarray]
        """Transform dimensionless state to physical units"""
        
    def physical_to_dimensionless(state: Dict) -> np.ndarray
        """Transform physical units to dimensionless state"""
```

**Libration Point Stability**: L1, L2, L3 points are dynamically unstable (basis for horseshoe orbits), L4 and L5 points are dynamically stable (Trojan celestial bodies).

---

## 3. Dynamics Model

### 3.1 CR3BP_Dynamics Class

**File**: `e2m2e/core/dynamics.py`

**Function**: Implement CR3BP dynamics equations numerical integration, supporting State Transition Matrix (STM) computation.

#### 3.1.1 Equations of Motion

```python
def equations_of_motion(self, t: float, state: np.ndarray) -> np.ndarray:
    """
    6-dimensional state vector equations of motion
    
    State vector: [x, y, z, vx, vy, vz]
    Returns derivative: [vx, vy, vz, ax, ay, az]
    """
    mu = self.system.mu
    x, y, z, vx, vy, vz = state
    
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)  # Distance to primary
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)  # Distance to secondary
    
    ax = 2*vy + x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
    ay = -2*vx + y - (1-mu)*y/r1**3 - mu*y/r2**3
    az = -(1-mu)*z/r1**3 - mu*z/r2**3
    
    return np.array([vx, vy, vz, ax, ay, az])
```

#### 3.1.2 State Transition Matrix (STM)

For periodic orbit search, linearized dynamics and State Transition Matrix $\Phi(t, t_0)$ need to be computed:

$$\dot{\Phi}(t, t_0) = A(t) \cdot \Phi(t, t_0), \quad \Phi(t_0, t_0) = I$$

Where $A(t)$ is the Jacobian matrix of the dynamics equations:

$$A = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix} 
0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 \\
U_{xx} & U_{xy} & U_{xz} & 0 & 2 & 0 \\
U_{xy} & U_{yy} & U_{yz} & -2 & 0 & 0 \\
U_{xz} & U_{yz} & U_{zz} & 0 & 0 & 0
\end{bmatrix}$$

```python
def equations_with_stm(self, t: float, augmented_state: np.ndarray) -> np.ndarray:
    """
    42-dimensional augmented state vector equations (6 states + 36 STM elements)
    """
    # ... compute Jacobian matrix A ...
    stm_dot = A @ stm  # STM propagation
    return np.concatenate([state_derivative, stm_dot.flatten()])
```

#### 3.1.3 Jacobi Constant

The Jacobi constant is the energy integral of CR3BP:

$$C_J = 2U - |\mathbf{v}|^2 = x^2 + y^2 + \frac{2(1-\mu)}{r_1} + \frac{2\mu}{r_2} - (vx^2 + vy^2 + vz^2)$$

For bounded orbital motion, must satisfy $C_J > 0$ (effective potential energy greater than kinetic energy).

#### 3.1.4 Trajectory Propagation

```python
def propagate(
    initial_state: np.ndarray,
    t_span: Tuple[float, float],
    t_eval: Optional[np.ndarray] = None,
    with_stm: bool = False
) -> Dict[str, Any]:
    """
    Numerical integration propagation
    
    Parameters:
        initial_state: Initial state vector [x, y, z, vx, vy, vz]
        t_span: Time interval [t0, tf]
        with_stm: Whether to compute state transition matrix simultaneously
    
    Returns:
        Dictionary containing trajectory time, states, STM, etc.
    """
```

**Integrator Settings**:
- Default integrator: RK45 (4(5) order Runge-Kutta)
- Relative tolerance: $1 \times 10^{-12}$
- Absolute tolerance: $1 \times 10^{-12}$

---

## 4. Differential Correction Algorithm

### 4.1 DifferentialCorrection Class

**File**: `e2m2e/algorithms/differential_correction.py`

**Function**: Iteratively correct initial conditions to refine approximate periodic orbits into exact periodic orbits.

#### 4.1.1 Algorithm Principles

For periodic orbit problems, initial state $\mathbf{x}_0$ and period $T$ must satisfy periodic conditions:

$$\mathbf{x}(T) - \mathbf{x}_0 = \mathbf{0}$$

This constitutes a system of nonlinear equations $\mathbf{F}(\mathbf{X}) = \mathbf{0}$, where $\mathbf{X}$ is the initial parameters to solve.

Using Newton-Raphson iteration:

$$\mathbf{X}_{k+1} = \mathbf{X}_k - \mathbf{J}^{-1} \mathbf{F}(\mathbf{X}_k)$$

Where $\mathbf{J} = \frac{\partial \mathbf{F}}{\partial \mathbf{X}}$ is the Jacobian matrix, computed using STM:

$$\frac{\partial \mathbf{x}(T)}{\partial \mathbf{x}_0} = \Phi(T, 0)$$

#### 4.1.2 2D Symmetric Orbit Configuration (setup_2D_symmetric_x_fixed_x0)

For DRO (Distant Retrograde Orbit) symmetric about the x-axis, symmetry is used to reduce parameters to solve:

**Symmetry Conditions**:
- Initial state: $[x_0, 0, 0, 0, \dot{y}_0, 0]$ (departing vertically from x-axis)
- Half-period conditions: $y(T/2) = 0$, $\dot{x}(T/2) = 0$ (again crossing x-axis vertically)

**Free variables**: $[\dot{y}_0, T/2]$ (initial y-direction velocity and half-period)

**Target constraints**: $[y(T/2), \dot{x}(T/2)] = [0, 0]$

```python
def setup_2D_symmetric_x_fixed_x0(self, x0: float):
    """
    Configure 2D symmetric orbit differential correction
    
    Parameters:
        x0: Fixed initial x coordinate
    
    Configuration:
        - Free variables: [y_dot0, T_half]
        - Constraint conditions: [y(T/2)=0, x_dot(T/2)=0]
        - State indices: y=1, x_dot=3
    """
    self.setup_type = "2D_symmetric_x_fixed_x0"
    self.free_variables = ["y_dot0", "T_half"]
    self.constraint_indices = [1, 3]  # y, x_dot
    self.target_conditions = {"y": 0.0, "x_dot": 0.0}
```

#### 4.1.3 Iterative Correction Process

```python
def iterate_correction(self, initial_guess: Orbit, verbose: bool = False) -> Optional[Orbit]:
    """
    Execute differential correction iteration
    
    Process:
    1. Propagate orbit to half-period
    2. Compute constraint errors
    3. Construct Jacobian matrix (using STM)
    4. Solve for corrections
    5. Update state
    6. Repeat until convergence
    
    Convergence conditions:
        - All constraint errors < tolerance (1e-12)
        - Or maximum iterations reached (50)
    """
```

**Adaptive Damping**: Prevent overshoot and oscillation

$$X_{k+1} = X_k + \alpha \cdot \delta X_k$$

Where $\alpha \in [0.1, 2.0]$ is the adaptive damping factor.

---

## 5. Orbit Family Continuation Algorithm

### 5.1 Continuation Class

**File**: `e2m2e/algorithms/continuation.py`

**Function**: Starting from a known seed orbit, generate complete orbit families through continuous parameter variation.

#### 5.1.1 Natural Parameter Continuation

```python
def natural_continuation(
    seed_orbit: Orbit,
    param_range: Tuple[float, float],
    step_size: float,
    verbose: bool = False
) -> OrbitFamily:
    """
    Natural parameter continuation algorithm
    
    Principles:
        Start from seed orbit, fix continuation parameter direction,
        gradually change parameter value, use previous orbit as
        initial guess for each step to perform differential correction.
    
    Parameters:
        seed_orbit: Seed orbit (exact periodic orbit)
        param_range: Parameter range (param_min, param_max)
        step_size: Continuation step size
    
    Continuation directions:
        - Forward: Increasing parameter
        - Backward: Decreasing parameter
        - Bidirectional: param_min < seed < param_max
    """
```

#### 5.1.2 Continuation Parameter Semantics: `fixed_parameters` vs `free_variables`

The continuation parameters in `Continuation` come from `DifferentialCorrection`'s `fixed_parameters`, not `free_variables`. Their semantics at corrector level vs continuation level:

| Level | `fixed_parameters` | `free_variables` |
|---|---|---|
| **DifferentialCorrection** | Values **fixed** during single correction iteration (not adjusted) | Variables **adjusted** during correction iteration |
| **Continuation** | **Continuation parameters** (varying along orbit family, used to parameterize orbit families) | Non-continuation parameters |

**Configuration Example** (using `setup_2D_symmetric_x_fixed_x0`):

```python
corrector.setup_2D_symmetric_x_fixed_x0(x0=0.8)
# fixed_parameters = {"x0": 0.8}  →  x0 varying along orbit family is the continuation parameter
# free_variables  = ["y_dot0", "T_half"]  →  variables adjusted during correction iteration
```

`Continuation` automatically obtains continuation parameter names from `corrector.fixed_parameters` during initialization, then maps to state vector indices via `_infer_param_index()` for parameter stepping in natural continuation:

```python
self.continuation_parameter = next(iter(corrector.fixed_parameters))
```

#### 5.1.3 Pseudo-Arclength Continuation (Alternative)

When natural continuation encounters branch or turning points, pseudo-arclength continuation can stably pass through these regions:

$$ds = \sqrt{dx_0^2 + dT^2}$$

---

## 6. Orbit Classes

### 6.1 Orbit Class

**File**: `e2m2e/core/orbit.py`

**Function**: Represents a single periodic orbit, containing state sequence and time information.

**Attributes**:
- `states`: State array (n × 6)
- `times`: Time array (n,)
- `period`: Orbit period
- `jacobi_constant`: Jacobi constant
- `stability_index`: Stability index

### 6.2 OrbitFamily Class

**File**: `e2m2e/core/orbit.py`

**Function**: Represents an orbit family, managing collections of multiple orbits.

**Methods**:
```python
class OrbitFamily:
    def add_orbit(self, orbit: Orbit) -> None:
        """Add orbit to family"""
        
    def save_to_file(self, filename: str) -> None:
        """Save orbit family to JSON file"""
        
    def load_from_file(self, filename: str) -> "OrbitFamily":
        """Load orbit family from JSON file"""
        
    def compute_stability_indices(self) -> np.ndarray:
        """Compute stability indices for all orbits in family"""
```

---

## 7. Stability Analysis

### 7.1 Stability Index

Determine orbit stability through eigenvalues of the Monodromy Matrix $M = \Phi(T, 0)$:

$$M = \begin{bmatrix} \Phi_{xx} & \Phi_{xy} \\ \Phi_{yx} & \Phi_{yy} \end{bmatrix}$$

**Floquet Multipliers**: Eigenvalues $\lambda_i$, satisfying $\lambda_1 \cdot \lambda_2 \cdot \lambda_3 \cdot \lambda_4 = 1$

**Stability Determination**:
- Stable orbit: All eigenvalues on unit circle (pure imaginary conjugate pairs)
- Unstable orbit: At least one eigenvalue outside unit circle

### 7.2 Bifurcation Detection

**File**: `e2m2e/algorithms/stability.py`

**Function**: Detect bifurcation points in orbit families.

#### 7.2.1 Bifurcation Types

| Bifurcation Type | Characteristics | Recognition Method |
|----------|------|----------|
| **Saddle-Node** | Eigenvalue $\lambda = 1$ | Detect eigenvalues close to +1 |
| **Period-Doubling** | Eigenvalue $\lambda = -1$ | Detect eigenvalues close to -1 |
| **Secondary Hopf** | Eigenvalues on unit circle conjugate | Detect complex eigenvalues with modulus close to 1 |

#### 7.2.2 `detect_bifurcation_in_family` Method

Traverse each orbit in the family, compute Floquet multipliers, detect eigenvalues close to +1 to identify saddle-node bifurcations:

```python
@staticmethod
def detect_bifurcation_in_family(
    orbits: List[Orbit],
    dynamics: CR3BP_Dynamics,
    tolerance: float = 1e-8,
) -> List[Dict[str, Any]]:
    """Detect bifurcation points in orbit family

    Parameters:
        orbits: List of Orbit objects (orbit family)
        dynamics: CR3BP_Dynamics object
        tolerance: Eigenvalue proximity to +1 tolerance, default 1e-8

    Returns:
        List[Dict[str, Any]]: List of bifurcation points, each containing:
            - orbit_index: Orbit index in family
            - orbit: Orbit object
            - eigenvalues: Eigenvalue array
            - eigenvalue_diff: Minimum |λ - 1|
            - bifurcation_type: Bifurcation type
    """
```

#### 7.2.3 `find_nearest_bifurcation` Method

Locate bifurcation closest to target parameter in orbit family:

```python
@staticmethod
def find_nearest_bifurcation(
    orbits: List[Orbit],
    dynamics: CR3BP_Dynamics,
    target_x0: Optional[float] = None,
    tolerance: float = 1e-4,
) -> Optional[Dict[str, Any]]:
    """Find bifurcation closest to target parameter in orbit family

    Parameters:
        orbits: List of Orbit objects
        dynamics: CR3BP_Dynamics object
        target_x0: Target x0 coordinate (optional)
        tolerance: Search tolerance

    Returns:
        Bifurcation dictionary, or None if not found
    """
```

#### 7.2.4 Usage Example

```python
from e2m2e.algorithms.stability import StabilityAnalysis

# Load orbit family
family = OrbitFamily.load_from_file("output/dro/dro_family.json")

# Create dynamics model
system = CR3BP_System(mu=0.0121506683)
dynamics = CR3BP_Dynamics(system)

# Detect bifurcation points
bifurcations = StabilityAnalysis.detect_bifurcation_in_family(
    orbits=family.orbits,
    dynamics=dynamics,
    tolerance=1e-8
)

# Find bifurcation closest to x0=0.8
nearest = StabilityAnalysis.find_nearest_bifurcation(
    orbits=family.orbits,
    dynamics=dynamics,
    target_x0=0.8,
    tolerance=1e-4
)
```

---

## 8. DRO Generation Complete Process

### 8.1 Algorithm Flowchart

```
┌─────────────────────────────────────────┐
│         1. Initialize CR3BP System      │
│    mu = 0.0121506683, create system     │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│         2. Create Seed Orbit Initial Guess│
│   x0 = 0.79188556619742                │
│   vy0 = 0.53682                        │
│   T_guess = 3.472526005624708          │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│      3. Differential Correction for Exact Seed Orbit │
│   setup_2D_symmetric_x_fixed_x0(x0)    │
│   iterate_correction()                  │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│      4. Natural Continuation to Generate Orbit Family│
│   natural_continuation(seed,           │
│       param_range=(0.2, 0.9),          │
│       step_size=0.0005)                 │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│         5. Save Results to JSON         │
│   dro_family_{params}_{timestamp}.json  │
└─────────────────────────────────────────┘
```

### 8.2 Code Example

```python
import e2m2e
from e2m2e.core import Orbit, OrbitFamily

# 1. Create system
system = e2m2e.core.system.CR3BP_System(mu=0.0121506683, 
                                         primary="earth", 
                                         secondary="moon")

# 2. Create dynamics model
dynamics = e2m2e.core.dynamics.CR3BP_Dynamics(system)

# 3. Seed orbit initial guess
x0 = 0.79188556619742
vy0 = 0.53682
initial_state = [x0, 0.0, 0.0, 0.0, vy0, 0.0]
seed_orbit = Orbit(states=[initial_state], times=[0])
seed_orbit.period = 3.472526005624708

# 4. Differential correction
corrector = e2m2e.algorithms.DifferentialCorrection(dynamics)
corrector.setup_2D_symmetric_x_fixed_x0(x0)
seed_dro = corrector.iterate_correction(seed_orbit)

# 5. Continue to generate family
continuation = e2m2e.algorithms.Continuation(corrector)
family = continuation.natural_continuation(
    seed_dro,
    param_range=(0.2, 0.9),
    step_size=0.0005
)

# 6. Save
family.save_to_file("output/dro/dro_family.json")
```

---

## 9. References

1. Broucke, R. A. (1968). Periodic orbits in the restricted three body problem with Earth-moon masses. NASA JPL.

2. Cui, P., et al. (2025). Two-Impulse Transfers from Lunar Distant Retrograde Orbits to Resonant Orbits. Journal of Guidance, Control, and Dynamics, Vol. 48, No. 6.

3. Koon, W. S., et al. (2011). Dynamical Systems, the Three-Body Problem and Space Mission Design. Springer.

---

## 10. Appendix: File Structure

```
e2m2e/
├── e2m2e/
│   ├── core/
│   │   ├── system.py          # CR3BP system parameters
│   │   ├── dynamics.py        # Dynamics equations
│   │   ├── orbit.py          # Orbit and OrbitFamily classes
│   │   └── coordinate.py     # Coordinate transformations
│   └── algorithms/
│       ├── differential_correction.py  # Differential correction
│       ├── continuation.py            # Orbit family continuation
│       └── stability.py              # Stability analysis
└── docs/
    └── cr3bp_algorithms.md   # This document
```
