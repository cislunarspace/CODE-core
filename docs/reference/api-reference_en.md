# E2M2E Technical Documentation

> This document is the complete technical documentation for the Earth-to-Moon-to-Earth (E2M2E) orbital mechanics library, covering design principles, mathematical foundations, API interfaces and usage guides for all core classes.

---

## Table of Contents

- [E2M2E Technical Documentation](#e2m2e-technical-documentation)
  - [Table of Contents](#table-of-contents)
  - [1. Core Module](#1-core-module)
    - [1.1 CR3BP\_System](#11-cr3bp_system)
      - [Design Principles](#design-principles)
      - [Mathematical Foundation](#mathematical-foundation)
      - [Attributes](#attributes)
      - [Core Methods](#core-methods)
      - [Usage Example](#usage-example)
    - [1.2 CR3BP\_Dynamics](#12-cr3bp_dynamics)
      - [Design Principles](#design-principles-1)
      - [Core Features](#core-features)
      - [Core Methods](#core-methods-1)
      - [Usage Example](#usage-example-1)
    - [1.3 Orbit](#13-orbit)
      - [Design Principles](#design-principles-2)
      - [Orbit Period Detection](#orbit-period-detection)
      - [Monodromy Matrix and Stability](#monodromy-matrix-and-stability)
      - [Core Attributes](#core-attributes)
      - [Core Methods](#core-methods-2)
    - [1.4 OrbitFamily](#14-orbitfamily)
      - [Design Principles](#design-principles-3)
      - [Core Attributes](#core-attributes-1)
      - [Core Methods](#core-methods-3)
    - [1.5 CoordinateTransformation \& ReferenceFrame](#15-coordinatetransformation--referenceframe)
      - [Rotating ↔ Inertial Transformation](#rotating--inertial-transformation)
      - [Core Methods](#core-methods-4)
  - [2. Algorithms Module](#2-algorithms-module)
    - [2.1 DifferentialCorrection](#21-differentialcorrection)
      - [Design Principles](#design-principles-4)
      - [Supported Symmetry Configurations](#supported-symmetry-configurations)
      - [Configuration Methods](#configuration-methods)
      - [Core Methods](#core-methods-5)
    - [2.2 Continuation \& ContinuationMethod](#22-continuation--continuationmethod)
      - [Natural Parameter Continuation](#natural-parameter-continuation)
      - [Pseudo-Arclength Continuation](#pseudo-arclength-continuation)
      - [Core Methods](#core-methods-6)
      - [Usage Example](#usage-example-2)
    - [2.3 StabilityAnalysis, StabilityType \& BifurcationType](#23-stabilityanalysis-stabilitytype--bifurcationtype)
      - [Stability Analysis Mathematical Foundation](#stability-analysis-mathematical-foundation)
      - [Stability Index](#stability-index)
      - [Core Methods](#core-methods-7)
  - [3. Transfer Module](#3-transfer-module)
    - [3.1 EarthMoonTransfer](#31-earthmoontransfer)
      - [Design Principles](#design-principles-5)
      - [Transfer Strategies](#transfer-strategies)
      - [Core Methods](#core-methods-8)
    - [3.2 MoonEarthTransfer](#32-moonearthtransfer)
      - [Core Methods](#core-methods-9)
    - [3.3 InterOrbitTransfer](#33-interorbittransfer)
      - [Transfer Types](#transfer-types)
      - [Poincaré Section Method](#poincare-section-method)
      - [Core Methods](#core-methods-10)
  - [4. Visualization Module](#4-visualization-module)
    - [4.1 OrbitVisualizer \& ProjectionPlane](#41-orbitvisualizer--projectionplane)
      - [Feature List](#feature-list)
      - [Usage Example](#usage-example-3)
    - [4.2 compute\_stability\_for\_family](#42-compute_stability_for_family)
      - [Feature Description](#feature-description)
  - [Appendix](#appendix)
    - [A. Physical Constants](#a-physical-constants)
    - [B. Known System Presets](#b-known-system-presets)
    - [C. State Vector Indexing](#c-state-vector-indexing)

---

## 1. Core Module

### 1.1 CR3BP_System

**File**: `e2m2e/core/system.py`

**Class Signature**:
```python
class CR3BP_System:
    """Circular Restricted Three-Body Problem system parameters"""
```

#### Design Principles

The `CR3BP_System` class encapsulates system parameters for the Circular Restricted Three-Body Problem. In the CR3BP model:
- Two massive bodies (primary $m_1$ and secondary $m_2$) move in circular orbits around their common center of mass under mutual gravitational attraction
- A small mass body (spacecraft) moves in the gravitational field of the two massive bodies, without affecting their motion

The mass parameter is defined as:
$$\mu = \frac{m_2}{m_1 + m_2}$$

For the Earth-Moon system, $\mu \approx 0.01215$

#### Mathematical Foundation

**Libration Points (Lagrange Points) Calculation**:
Libration points are special points that remain stationary relative to the two massive bodies:
$$\nabla U(\mathbf{r}) = 0$$

Where $U$ is the effective potential function:
$$U = \frac{x^2 + y^2}{2} + \frac{1-\mu}{r_1} + \frac{\mu}{r_2}$$

**Jacobi Constant**:
$$C = 2U - v^2 = x^2 + y^2 + \frac{2(1-\mu)}{r_1} + \frac{2\mu}{r_2} - (v_x^2 + v_y^2 + v_z^2)$$

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `mu` | `float` | Mass parameter $\mu = m_2/(m_1+m_2)$ |
| `primary_body` | `str` | Primary body name |
| `secondary_body` | `str` | Secondary body name |
| `L1-L5` | `np.ndarray` | Coordinates of five libration points |
| `characteristic_length` | `float` | Characteristic length (distance between two bodies) |
| `characteristic_time` | `float` | Characteristic time |
| `characteristic_velocity` | `float` | Characteristic velocity |

#### Core Methods

| Method | Description |
|--------|-------------|
| `compute_libration_points()` | Compute positions of five libration points |
| `get_libration_point(point)` | Get coordinates of specified libration point |
| `get_jacobi_constant(state)` | Calculate Jacobi constant |
| `dimensionless_to_physical(state)` | Dimensionless → Physical units |
| `physical_to_dimensionless(state)` | Physical units → Dimensionless |
| `compute_stability_index(L_point)` | Compute stability index of libration point |

#### Usage Example

```python
from e2m2e.core.system import CR3BP_System, LibrationPoint

# Create from known system
system = CR3BP_System.from_known_system("earth_moon")
system.set_characteristic_scales(distance=384400, period=27.32*86400)
system.compute_libration_points()

# Get libration point
L1 = system.get_libration_point(LibrationPoint.L1)
print(f"L1 position: {L1}")

# Calculate Jacobi constant
state = np.array([0.8, 0, 0, 0, 1.5, 0])
C = system.get_jacobi_constant(state)
```

---

### 1.2 CR3BP_Dynamics

**File**: `e2m2e/core/dynamics.py`

**Class Signature**:
```python
class CR3BP_Dynamics:
    """CR3BP dynamics equations"""
```

#### Design Principles

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

#### Core Features

1. **State Propagation**: The `propagate()` method uses scipy's `solve_ivp` for numerical integration
2. **State Transition Matrix (STM)**: The 42-dimensional augmented state is integrated simultaneously via `equations_with_stm()`
3. **Jacobi Constant Monitoring**: Real-time calculation of Jacobi constant for accuracy verification

#### Core Methods

| Method | Description |
|--------|-------------|
| `equations_of_motion(t, state)` | 6-dimensional equations of motion |
| `equations_with_stm(t, augmented_state)` | 42-dimensional augmented equations (with STM) |
| `propagate(initial_state, t_span, with_stm=False)` | Propagate trajectory |
| `compute_state_transition_matrix(initial_state, t)` | Compute STM |
| `compute_jacobi_constant(state)` | Calculate Jacobi constant |
| `check_cross_section(state, plane, value)` | Detect section crossing |

#### Usage Example

```python
from e2m2e.core.dynamics import CR3BP_Dynamics

dynamics = CR3BP_Dynamics(system)

# Simple propagation
result = dynamics.propagate(initial_state, t_span=(0, 10), t_eval=np.linspace(0, 10, 1000))

# Propagation with STM (for differential correction)
result_with_stm = dynamics.propagate(initial_state, t_span=(0, T), with_stm=True)
stm = result_with_stm['stm'][-1]  # STM at final time
```

---

### 1.3 Orbit

**File**: `e2m2e/core/orbit.py`

**Class Signature**:
```python
class Orbit:
    """Orbit data and processing"""
```

#### Design Principles

The `Orbit` class represents a complete orbit in CR3BP. Orbit data includes:
- **states**: State sequence, shape `(n, 6)`, each row `[x, y, z, vx, vy, vz]`
- **times**: Corresponding time sequence

#### Orbit Period Detection

Orbit period is estimated by detecting zero crossings of the x component:

1. Find sign change points of x coordinate relative to orbit center $\bar{x}$
2. Use time difference $\Delta t$ between adjacent zero crossings to estimate half-period: $T_{half} = \Delta t$
3. Full period: $T = 2 \times T_{half}$

#### Monodromy Matrix and Stability

**Monodromy Matrix**: The state transition matrix integrated along a closed orbit for one period:
$$M = \Phi(T, 0)$$
Where $\Phi$ is the state transition matrix.

**Stability Determination**:
- If all Floquet multipliers (STM eigenvalues) $\lambda_i$ satisfy $|\lambda_i| = 1$, the orbit is linearly stable
- If any $|\lambda_i| > 1$ exists, the orbit is unstable

#### Core Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `states` | `np.ndarray` | State sequence (n, 6) |
| `times` | `np.ndarray` | Time sequence (n,) |
| `period` | `float` | Orbit period |
| `amplitudes` | `dict` | Amplitudes in each direction |
| `monodromy_matrix` | `np.ndarray` | Monodromy matrix (6, 6) |
| `eigenvalues` | `np.ndarray` | Eigenvalues |
| `stability` | `str` | Stability label |
| `is_periodic` | `bool` | Whether it is a periodic orbit |

#### Core Methods

| Method | Description |
|--------|-------------|
| `compute_monodromy_matrix(dynamics)` | Compute monodromy matrix |
| `compute_stability(dynamics)` | Compute stability |
| `interpolate_at_time(t)` | Time interpolation |
| `get_amplitude(direction)` | Get amplitude |
| `save_to_file(filename)` | Save to file |
| `load_from_file(filename)` | Load from file |

---

### 1.4 OrbitFamily

**File**: `e2m2e/core/orbit.py`

**Class Signature**:
```python
class OrbitFamily:
    """Orbit family container"""
```

#### Design Principles

`OrbitFamily` is used to store and manage a family of related orbits (such as the same family of Halo orbits, Lyapunov orbits, etc.). Supports batch operations like getting all initial states, periods, Jacobi constants, etc.

#### Core Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `orbits` | `List[Orbit]` | List of Orbit objects |
| `family_type` | `str` | Orbit family type |
| `system` | `CR3BP_System` | Associated system |
| `states` | `np.ndarray` | property: All initial states (n, 6) |
| `periods` | `np.ndarray` | property: All periods (n,) |

#### Core Methods

| Method | Description |
|--------|-------------|
| `add_orbit(orbit)` | Add orbit |
| `get_states()` | Get initial state array |
| `get_periods()` | Get period array |
| `get_jacobi_constants()` | Get Jacobi constant array |
| `save_to_file(filename)` | Save orbit family |
| `load_from_file(filename)` | Load orbit family |

---

### 1.5 CoordinateTransformation & ReferenceFrame

**File**: `e2m2e/core/coordinate.py`

**Class Signature**:
```python
class ReferenceFrame(Enum):
    ROTATING = "rotating"      # Rotating frame
    INERTIAL = "inertial"     # Inertial frame
    BARYCENTRIC = "barycentric"  # Barycentric frame
    PRIMARY_CENTERED = "primary_centered"  # Primary-centered frame
    SECONDARY_CENTERED = "secondary_centered"  # Secondary-centered frame
    SYNODIC = "synodic"        # Synodic frame

class CoordinateTransformation:
    """Coordinate system transformation"""
```

#### Rotating ↔ Inertial Transformation

The transformation matrix from rotating frame (synodic) to inertial frame is:

$$\mathbf{r}_{inertial} = R_z(\theta)^T \mathbf{r}_{rotating}$$

Where $R_z(\theta)$ is the rotation matrix about z-axis:
$$R_z(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

Velocity transformation includes Coriolis terms:
$$\mathbf{v}_{inertial} = R_z(\theta)^T \mathbf{v}_{rotating} + \dot{R}_z(\theta)^T \mathbf{r}_{rotating}$$

#### Core Methods

| Method | Description |
|--------|-------------|
| `rotating_to_inertial(state, time)` | Rotating → Inertial |
| `inertial_to_rotating(state, time)` | Inertial → Rotating |
| `barycentric_to_primary(state)` | Barycentric → Primary-centered |
| `primary_to_barycentric(state)` | Primary-centered → Barycentric |
| `compute_rotation_matrix(time)` | Compute rotation matrix |

---

## 2. Algorithms Module

### 2.1 DifferentialCorrection

**File**: `e2m2e/algorithms/differential_correction.py`

**Class Signature**:
```python
class DifferentialCorrection:
    """Differential correction algorithm"""
```

#### Design Principles

The differential correction algorithm solves for initial conditions of periodic orbits. Core idea:
1. Assume initial condition $\mathbf{x}_0$ and half-period $T/2$ as free variables
2. Integrate to get final state $\mathbf{x}(T/2)$
3. Construct constraint equation $\mathbf{F}(\mathbf{x}_0, T/2) = 0$
4. Use Newton iteration to solve:
$$\begin{pmatrix} \delta\mathbf{x}_0 \\ \delta(T/2) \end{pmatrix} = -J^{-1} \mathbf{F}$$

Where $J$ is the Jacobian matrix of the constraint equation with respect to free variables.

#### Supported Symmetry Configurations

| Configuration Type | Free Variables | Target Constraints |
|---------|---------|---------|
| `2D_symmetric_x_fixed_x0` | $[v_{y0}, T_{half}]$ | $y=0, \dot{x}=0$ |
| `2D_symmetric_x_fixed_t` | $[x_0, v_{y0}]$ | $y=0, \dot{x}=0$ |
| `2D_symmetric_y_fixed_y0` | $[\dot{x}_0, T_{half}]$ | $x=0, \dot{x}=0$ |
| `3D_symmetric_x_fixed_x0` | $[z_0, v_{y0}, T_{half}]$ | $y=0, \dot{x}=0, \dot{z}=0$ |
| `3D_symmetric_xz_fixed_x0` | $[z_0, v_{y0}, T_{half}]$ | $y=0, \dot{x}=0, \dot{z}=0$ |
| `3D_symmetric_xz_fixed_z0` | $[x_0, v_{y0}, T_{half}]$ | $y=0, \dot{x}=0, \dot{z}=0$ |

#### Configuration Methods

```python
# 2D symmetric x-axis fixed x0 configuration
corrector.setup_2D_symmetric_x_fixed_x0(x0=0.8)

# 3D Halo orbit configuration
corrector.setup_3D_symmetric_x_fixed_x0(x0=0.8)
```

#### Core Methods

| Method | Description |
|--------|-------------|
| `setup_2D_symmetric_x_fixed_x0(x0)` | Configure 2D symmetric x-axis search |
| `setup_2D_symmetric_x_fixed_t(t_half)` | Configure 2D fixed period search |
| `setup_3D_symmetric_x_fixed_x0(x0)` | Configure 3D symmetric search |
| `iterate_correction(initial_guess, t_half, verbose)` | Execute iterative correction |
| `_compute_jacobian_finite_diff()` | Finite difference Jacobian computation |

---

### 2.2 Continuation & ContinuationMethod

**File**: `e2m2e/algorithms/continuation.py`

**Class Signature**:
```python
class ContinuationMethod(Enum):
    NATURAL = "natural"
    PSEUDO_ARCLENGTH = "pseudo_arclength"

class Continuation:
    """Orbit family continuation"""
```

#### Natural Parameter Continuation

Continue along a selected parameter (such as $x_0$, $z_0$, period, etc.):

1. Start from seed orbit
2. Apply step $\Delta s$ in parameter direction
3. Use previous orbit state as initial guess
4. Call differential corrector to solve
5. Repeat until parameter range boundary

#### Pseudo-Arclength Continuation

When the orbit family has fold (turning point), natural continuation fails. Pseudo-arclength method introduces arc-length parameter $s$:

$$\frac{d\mathbf{u}}{ds} = \frac{\mathbf{t}}{\|\mathbf{t}\|}$$

Where $\mathbf{t}$ is the tangent vector, $\mathbf{u} = [\mathbf{x}; T/2]$ is the state vector.

#### Core Methods

| Method | Description |
|--------|-------------|
| `natural_continuation(seed_orbit, param_range, step_size, verbose)` | Natural parameter continuation |
| `pseudo_arclength_continuation(seed_state, seed_t_half, n_orbits, verbose)` | Pseudo-arclength continuation |

#### Usage Example

```python
from e2m2e.algorithms.continuation import Continuation

# Create continuer
continuation = Continuation(corrector, step=0.01)

# Natural parameter continuation
family = continuation.natural_continuation(
    seed_orbit=seed_orbit,
    param_range=(0.8, 1.2),
    step_size=0.01,
    verbose=True
)
```

---

### 2.3 StabilityAnalysis, StabilityType & BifurcationType

**File**: `e2m2e/algorithms/stability.py`

**Class Signature**:
```python
class StabilityType(Enum):
    STABLE = "stable"
    UNSTABLE = "unstable"
    MARGINALLY_STABLE = "marginally_stable"
    HYPERBOLIC = "hyperbolic"
    ELLIPTIC = "elliptic"
    PARABOLIC = "parabolic"

class BifurcationType(Enum):
    NONE = "none"
    PERIOD_DOUBLING = "period_doubling"
    SADDLE_NODE = "saddle_node"
    TORUS = "torus"
    PITCHFORK = "pitchfork"
    TRANSCRITICAL = "transcritical"
    SECONDARY_HOPF = "secondary_hopf"

class StabilityAnalysis:
    """Orbit stability analysis"""
```

#### Stability Analysis Mathematical Foundation

For periodic orbits, eigenvalues (Floquet multipliers) $\lambda_i$ of the monodromy matrix $M$ satisfy:
- $|\lambda_i| = 1$: Neutrally stable (elliptic orbit)
- $|\lambda_i| < 1$: Asymptotically stable
- $|\lambda_i| > 1$: Unstable (hyperbolic orbit)

Due to properties of symplectic matrices, eigenvalues appear in reciprocal pairs:
$$\lambda_1 \lambda_2 = 1, \quad \lambda_3 \lambda_4 = 1$$

#### Stability Index

Commonly used stability index definition:

**Broucke Stability Index**:
$$\nu = \frac{|\lambda_1| + |\lambda_2| + |\lambda_3| + |\lambda_4|}{4}$$

For stable orbits, $\nu = 1$.

#### Core Methods

| Method | Description |
|--------|-------------|
| `compute_monodromy()` | Compute monodromy matrix |
| `compute_floquet_multipliers()` | Compute Floquet multipliers |
| `analyze_stability()` | Analyze stability type |
| `detect_bifurcation()` | Detect bifurcation type |

---

## 3. Visualization Module

### 3.1 OrbitVisualizer & ProjectionPlane

**File**: `e2m2e/visualization/plotting.py`

**Class Signature**:
```python
class ProjectionPlane(Enum):
    XY = "xy"
    XZ = "xz"
    YZ = "yz"

class OrbitVisualizer:
    """Orbit visualizer"""
```

#### Feature List

| Feature | Method |
|---------|--------|
| 3D orbit plotting | `plot_3d_orbit()` |
| 2D projection | `plot_2d_projection()` |
| Primary/secondary bodies | `plot_primary_bodies()` |
| Libration point annotation | `plot_libration_points()` |
| Orbit family | `plot_orbit_family()` |
| Poincaré section | `plot_poincare_section()` |
| Jacobi constant | `plot_jacobi_constant()` |
| Stability analysis | `plot_stability_indices()` |
| Overview plot | `create_overview_plot()` |

#### Usage Example

```python
from e2m2e.visualization.plotting import OrbitVisualizer

viz = OrbitVisualizer(system)

# Create overview plot
fig = viz.create_overview_plot(orbit)

# Save
viz.save('orbit.png', dpi=300)
```

---

### 3.2 compute_stability_for_family

**File**: `e2m2e/visualization/plotting.py`

**Function Signature**:
```python
def compute_stability_for_family(family_result, system) -> list:
    """Compute stability indices for orbit family"""
```

#### Feature Description

Compute stability index for each orbit in the family:
1. Compute monodromy matrix $M$
2. Compute eigenvalues $\lambda_i$
3. Take maximum modulus $\nu = \max|\lambda_i|$

Stability determination:
- $\nu = 1$: Neutrally stable
- $\nu < 1$: Asymptotically stable
- $\nu > 1$: Unstable

---

## Appendix

### A. Physical Constants

| Constant | Value | Unit |
|------|-----|-----|
| $G$ | $6.67430 \times 10^{-20}$ | km³/kg/s² |
| AU | $149,597,870.7$ | km |
| Earth-Moon distance | 384,400 | km |
| Earth-Moon period | 27.32 | day |

### B. Known System Presets

```python
KNOWN_SYSTEMS = {
    "earth_moon": {"mu": 0.01215, "distance": 384400, "period": 27.32*86400},
    "sun_earth": {"mu": 3.0039e-6, "distance": 1*AU, "period": 365.25*86400},
    "sun_jupiter": {"mu": 0.0009535, "distance": 5.2*AU, "period": 11.86*365.25*86400}
}
```

### C. State Vector Indexing

| Index | Component | Description |
|------|------|-------------|
| 0 | x | x coordinate |
| 1 | y | y coordinate |
| 2 | z | z coordinate |
| 3 | $\dot{x}$ | x-direction velocity |
| 4 | $\dot{y}$ | y-direction velocity |
| 5 | $\dot{z}$ | z-direction velocity |
