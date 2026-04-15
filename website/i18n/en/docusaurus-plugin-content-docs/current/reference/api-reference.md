---
title: E2M2E Technical Documentation
---

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
      - [Design Principles](#design-principles_1)
      - [Core Features](#core-features)
      - [Core Methods](#core-methods_1)
      - [Usage Example](#usage-example_1)
    - [1.3 Orbit](#13-orbit)
      - [Design Principles](#design-principles_2)
      - [Orbit Period Detection](#orbit-period-detection)
      - [Monodromy Matrix and Stability](#monodromy-matrix-and-stability)
      - [Core Attributes](#core-attributes)
      - [Core Methods](#core-methods_2)
    - [1.4 OrbitFamily](#14-orbitfamily)
      - [Design Principles](#design-principles_3)
      - [Core Attributes](#core-attributes_1)
      - [Core Methods](#core-methods_3)
    - [1.5 CoordinateTransformation \& ReferenceFrame](#15-coordinatetransformation-referenceframe)
      - [Rotating ↔ Inertial Transformation](#rotating-inertial-transformation)
      - [Core Methods](#core-methods_4)
  - [2. Algorithms Module](#2-algorithms-module)
    - [2.1 DifferentialCorrection](#21-differentialcorrection)
      - [Design Principles](#design-principles_4)
      - [Supported Symmetry Configurations](#supported-symmetry-configurations)
      - [Configuration Methods](#configuration-methods)
      - [Core Methods](#core-methods_5)
    - [2.2 Continuation](#22-continuation)
      - [Natural Parameter Continuation](#natural-parameter-continuation)
      - [Pseudo-Arclength Continuation](#pseudo-arclength-continuation)
      - [Core Methods](#core-methods_6)
      - [Usage Example](#usage-example_2)
    - [2.3 StabilityAnalysis, StabilityType \& BifurcationType](#23-stabilityanalysis-stabilitytype-bifurcationtype)
      - [Stability Analysis Mathematical Foundation](#stability-analysis-mathematical-foundation)
      - [Stability Index](#stability-index)
      - [Core Methods](#core-methods_7)
    - [2.4 CorrectionConfig & Strategy Functions](#24-correctionconfig-strategy-functions)
  - [3. Transfer Module](#3-transfer-module)
    - [3.1 TransferSearch / DROTransferSearch](#31-transfersearch-drotransfersearch)
      - [Design Principles](#design-principles_5)
      - [Search Parameters](#search-parameters)
      - [Core Methods](#core-methods_8)
    - [3.2 DROTRONLPOptimizer](#32-drotronlpoptimizer)
      - [Design Principles](#design-principles_6)
      - [TransferType Enum](#transfertype-enum)
      - [Core Methods](#core-methods_9)
    - [3.3 SearchConfig](#33-searchconfig)
    - [3.4 Transfer (Simplified API)](#34-transfer-simplified-api)
    - [3.5 Utility Functions](#35-utility-functions)
  - [4. Visualization Module](#4-visualization-module)
    - [4.1 PlotConfig](#41-plotconfig)
    - [4.2 OrbitVisualizer \& ProjectionPlane](#42-orbitvisualizer-projectionplane)
      - [Feature List](#feature-list)
      - [Usage Example](#usage-example_3)
    - [4.3 FamilyPlotter](#43-familyplotter)
    - [4.4 TransferPlotter](#44-transferplotter)
    - [4.5 compute\_stability\_for\_family](#45-compute_stability_for_family)
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
| `compute_basic_properties()` | Compute basic orbit properties |
| `compute_monodromy_matrix(dynamics)` | Compute monodromy matrix |
| `compute_stability(dynamics)` | Compute stability |
| `get_period()` | Get orbit period |
| `get_amplitude(direction)` | Get amplitude |
| `save_to_file(filename)` | Save to file |
| `load_from_file(filename)` | Load from file (classmethod) |
| `copy()` | Create a copy |

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

### 2.2 Continuation

**File**: `e2m2e/algorithms/continuation.py`

**Class Signature**:
```python
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
| `pseudo_arclength_continuation(seed_orbit, n_orbits, step_size, direction, ..., dc_scheme, ...)` | XZ symmetric pseudo-arclength continuation (`direction`: `positive` / `negative`) |
| `generate_halo_seed_orbit(libration_point, amplitude_z, halo_class, ...)` | Generate Halo seed orbit |
| `generate_halo_family(seed_orbit, n_orbits, direction, step_size)` | Halo family by `amplitude_z` natural parameter stepping |
| `halo_pseudo_arclength_continuation(seed_orbit, n_orbits, direction, step_size, step_size_negative, ...)` | Halo pseudo-arclength family (bi-directional, optional MATLAB alignment params) |

See [Halo Algorithm Documentation](../algorithms/halo.md) for details.

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
| `compute_stability_index()` | Compute stability index (nu1, nu2, nu3, broucke) |
| `classify_orbit()` | Classify stability type |
| `analyze_bifurcation()` | Detect bifurcation type |
| `full_analysis()` | Run complete analysis |
| `detect_bifurcation_in_family(orbits, dynamics)` | Static: detect bifurcations in orbit family |
| `find_nearest_bifurcation(orbits, dynamics, target_x0)` | Static: find nearest bifurcation point |

---

### 2.4 CorrectionConfig & Strategy Functions

**File**: `e2m2e/algorithms/strategies/`

The strategy pattern introduced in v3.2 separates correction configuration logic from the `DifferentialCorrection` class into independent immutable configs and strategy functions.

#### CorrectionConfig

**Class Signature**:
```python
@dataclass(frozen=True)
class CorrectionConfig:
    """Immutable configuration for a differential correction strategy"""
```

| Field | Type | Description |
|-------|------|-------------|
| `setup_type` | `str` | Correction setup type identifier |
| `symmetry_condition` | `str` | Symmetry exploited by the correction (e.g. `'x_axis'`) |
| `fixed_parameters` | `Dict[str, float]` | Parameter values held constant during correction |
| `free_variables` | `List[str]` | Variable names the Newton solver adjusts |
| `free_variable_indices` | `List[int]` | State-vector indices corresponding to free variables |
| `target_conditions` | `Dict[str, float]` | Constraint names mapped to their target values |
| `constraint_indices` | `List[int]` | State-vector indices for constraint evaluation |
| `constraint_weights` | `Dict[str, float]` | Per-constraint weighting factors for the Jacobian |
| `constraint_types` | `Dict[str, str]` | Per-constraint classification (e.g. `'equality'`) |

#### Strategy Functions

| Function | Symmetry | Free Variables | Target Constraints |
|----------|----------|----------------|--------------------|
| `symmetric_2d_fixed_x0(x0)` | x-axis | $[v_{y0}, T_{half}]$ | $y=0, \dot{x}=0$ |
| `symmetric_2d_fixed_t(t_half)` | x-axis | $[x_0, v_{y0}]$ | $y=0, \dot{x}=0$ |
| `symmetric_2d_fixed_y0(y0)` | y-axis | $[\dot{x}_0, T_{half}]$ | $x=0, \dot{y}=0$ |
| `symmetric_3d_fixed_x0(x0)` | x-axis | $[z_0, v_{y0}, T_{half}]$ | $y=0, \dot{x}=0, \dot{z}=0$ |
| `symmetric_xz_fixed_x0(x0)` | xz-plane | $[z_0, v_{y0}, T_{half}]$ | $y=0, \dot{x}=0, \dot{z}=0$ |
| `symmetric_xz_fixed_z0(z0)` | xz-plane | $[x_0, v_{y0}, T_{half}]$ | $y=0, \dot{x}=0, \dot{z}=0$ |
| `halo_fixed_x0(x0, libration_point)` | xz-plane | $[z_0, v_{y0}, T_{half}]$ | $y=0, \dot{x}=0, \dot{z}=0$ |
| `halo_fixed_z0(z0, libration_point)` | xz-plane | $[x_0, v_{y0}, T_{half}]$ | $y=0, \dot{x}=0, \dot{z}=0$ |

#### Usage Example

```python
from e2m2e.algorithms.strategies import CorrectionConfig, halo_fixed_z0

# Use a strategy function to generate configuration
config = halo_fixed_z0(z0=0.1, libration_point=1)
# config is an immutable CorrectionConfig instance

# Pass to DifferentialCorrection
corrector = DifferentialCorrection(dynamics)
corrector.setup_halo_orbit_fixed_z0(z0=0.1, libration_point=1)
```

---

## 3. Transfer Module

### 3.1 TransferSearch / DROTransferSearch

**File**: `e2m2e/transfer/transfer_search.py`

**Class Signature**:
```python
class TransferSearch:
    """Grid search for DRO-to-RO planar transfer trajectories"""

DROTransferSearch = TransferSearch   # alias
DROROTransferSearch = TransferSearch # alias
```

#### Design Principles

`TransferSearch` implements the grid search phase of the Cui et al. (2025) "search-then-optimize" method for two-impulse transfers from Distant Retrograde Orbit (DRO) to Resonant Orbit (RO):

1. Sample multiple departure points on the DRO
2. For each departure point, apply different tangential velocity ratios $\alpha$ and forward-integrate
3. Detect whether the trajectory intersects or reaches local minimum distance to the target RO
4. Mark feasible candidate solutions

#### Search Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha_min` | float | 0.5 | Lower bound of tangential velocity ratio |
| `alpha_max` | float | 2.5 | Upper bound of tangential velocity ratio |
| `n_alpha` | int | 101 | Grid points in $\alpha$ direction |
| `n_departure` | int | 200 | Number of departure point samples |
| `max_transfer_time` | float | 200/TU | Maximum transfer time (dimensionless) |
| `intersection_threshold` | float | 0.001 | Intersection detection threshold (DU) |
| `min_distance_threshold` | float | 100/DU | Minimum distance threshold |
| `collision_earth_radius` | float | 200/DU | Earth collision radius |
| `collision_moon_radius` | float | 100/DU | Moon collision radius |
| `integration_dt` | float | 1/(24·TU) | Integration step size |

#### Core Methods

| Method | Description |
|--------|-------------|
| `search(*, alpha_min, alpha_max, n_alpha, n_departure, max_transfer_time, departure_orbit, arrival_orbit, ...)` | Execute grid search |
| `get_feasible_results()` | Get feasible results from last search |
| `optimize(initial_guess)` | Optimize best search result using NLP |
| `set_verbose(verbose)` | Set verbosity (chainable) |
| `set_n_workers(n_workers)` | Set parallel worker count (chainable) |
| `set_parallel_backend(backend)` | Set backend: `"threads"` / `"processes"` (chainable) |

#### Usage Example

```python
from e2m2e.transfer import TransferSearch

searcher = TransferSearch(dynamics=dynamics)
results = searcher.search(
    alpha_min=0.5, alpha_max=2.5,
    n_alpha=101, n_departure=200,
    max_transfer_time=200.0,
    departure_orbit=dro_orbit,
    arrival_orbit=ro_orbit,
)
```

---

### 3.2 DROTRONLPOptimizer

**File**: `e2m2e/transfer/transfer_optimization.py`

**Class Signature**:
```python
class DROTRONLPOptimizer:
    """NLP optimizer for DRO-to-RO two-impulse transfer"""
```

#### Design Principles

Formulates the two-impulse transfer problem as a Nonlinear Programming (NLP) problem:

- **Optimization variables**: $y = \{\alpha, T, t_{ins}\}$
- **Objective function**: $J(y) = \Delta v_1 + \Delta v_2$
- **Constraints**: position continuity, velocity parallelism, collision avoidance

Uses SciPy `minimize(method="SLSQP")` by default. Optionally uses [COPT](https://www.shanshu.ai/copt) solver when `coptpy` is installed.

#### TransferType Enum

| Value | Description |
|-------|-------------|
| `DIRECT` | Direct transfer |
| `LGA` | Lunar gravity assist transfer |
| `EXTERNAL` | External transfer |

#### Core Methods

| Method | Description |
|--------|-------------|
| `optimize(initial_guess, alpha_range, t_ins_range, ...)` | Execute NLP optimization |
| `compute_departure_velocity(state, alpha, beta)` | Compute departure velocity vector |
| `forward_integrate(initial_state, t_span)` | Forward integrate trajectory |
| `set_progress_callback(callback)` | Set iteration progress callback |

#### Usage Example

```python
from e2m2e.transfer import DROTRONLPOptimizer, NLPOptimizationVariables

optimizer = DROTRONLPOptimizer(
    system=system, dynamics=dynamics,
    departure_orbit=dro_orbit, arrival_orbit=ro_orbit,
    departure_state=dro_orbit.states[0],
)
initial_vars = NLPOptimizationVariables(alpha=1.0, transfer_time=5.0, t_ins=3.0)
result = optimizer.optimize(initial_guess=initial_vars)
```

---

### 3.3 SearchConfig

**File**: `e2m2e/transfer/search_config.py`

**Class Signature**:
```python
@dataclass
class SearchConfig:
    """TransferSearch grid search configuration"""
```

`SearchConfig` centralizes search and optimization parameters into a reusable dataclass for serialization and type safety.

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `alpha_min` | `float` | Lower bound of tangential velocity ratio α |
| `alpha_max` | `float` | Upper bound of tangential velocity ratio α |
| `n_alpha` | `int` | Grid points in α direction |
| `n_departure` | `int` | Number of departure point samples |
| `max_transfer_time` | `float` | Maximum transfer time (dimensionless) |
| `intersection_threshold` | `float` | Intersection detection distance threshold |
| `min_distance_threshold` | `float` | Candidate solution minimum distance threshold |
| `collision_earth_radius` | `float` | Earth collision detection radius (dimensionless) |
| `collision_moon_radius` | `float` | Moon collision detection radius (dimensionless) |
| `integration_dt` | `float` | Integration time step (dimensionless) |
| `alpha_range` | `Tuple[float, float]` | Optimization phase α search range |
| `transfer_time_range` | `Tuple[float, float]` | Optimization phase transfer time range |
| `t_ins_range` | `Tuple[float, float]` | Optimization phase insertion time range |
| `velocity_angle_tolerance` | `float` | Velocity parallelism tolerance (radians) |

---

### 3.4 Transfer (Simplified API)

**File**: `e2m2e/transfer/transfer.py`

**Class Signature**:
```python
class Transfer:
    """Simplified DRO-RO transfer trajectory optimizer with chainable API"""
```

`Transfer` provides a high-level interface that wraps `DROTRONLPOptimizer` with a fluent API pattern.

#### Usage Example

```python
from e2m2e.transfer import Transfer

transfer = Transfer(dynamics)
result = transfer.set_orbit(start=dro_orbit, end=ro_orbit).optimize(
    initial_guess={"alpha": 1.0, "transfer_time": 15.0, "t_ins": 5.0},
    alpha_range=(0.5, 2.5),
)
```

---

### 3.5 Utility Functions

```python
from e2m2e.transfer import load_orbit_from_json, optimize_transfer, optimize_with_copt

orbit = load_orbit_from_json("path/to/orbit.json")
result = optimize_transfer(system, dynamics, departure_orbit, arrival_orbit, departure_state)
result = optimize_with_copt(optimizer, initial_guess, fallback_to_scipy=True)
```

---

## 4. Visualization Module

> `plotting.py` has been split into `config.py`, `base.py`, `family.py`, `transfer.py`, and `stability.py`. The original path still works as a re-export shim for backward compatibility.

### 4.1 PlotConfig

**File**: `e2m2e/visualization/config.py`

**Class Signature**:
```python
@dataclass
class PlotConfig:
    """Global visualization configuration"""
```

#### Key Fields

| Field | Type | Description |
|-------|------|-------------|
| `figsize` | `tuple` | Figure size (width, height) |
| `dpi` | `int` | Resolution |
| `style` | `str` | matplotlib style |
| `color_scheme` | `str` | Color scheme |
| `show_grid` | `bool` | Whether to show grid |
| `show_legend` | `bool` | Whether to show legend |
| `save_format` | `str` | Save format (png/pdf/svg) |

#### Usage Example

```python
from e2m2e.visualization import PlotConfig

config = PlotConfig(figsize=(12, 8), dpi=150, style="dark_background")
```

---

### 4.2 OrbitVisualizer & ProjectionPlane

**File**: `e2m2e/visualization/base.py`

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
| Poincaré section | `plot_poincare_section()` |
| Jacobi constant | `plot_jacobi_constant()` |
| Stability diagram | `plot_stability_diagram()` |
| Overview plot | `create_overview_plot()` |

#### Usage Example

```python
from e2m2e.visualization.base import OrbitVisualizer

viz = OrbitVisualizer(system)

# Create overview plot
fig = viz.create_overview_plot(orbit)

# Save
viz.save('orbit.png', dpi=300)
```

---

### 4.3 FamilyPlotter

**File**: `e2m2e/visualization/family.py`

**Class Signature**:
```python
class FamilyPlotter:
    """Orbit family visualizer"""
```

#### Core Methods

| Method | Description |
|--------|-------------|
| `plot_family_2d()` | Plot 2D projection of orbit family |
| `plot_family_3d()` | Plot 3D view of orbit family |
| `plot_jacobi_period_stability()` | Plot Jacobi constant–period–stability relationship |
| `plot_family_overview()` | Plot comprehensive orbit family overview |

#### Usage Example

```python
from e2m2e.visualization import FamilyPlotter

plotter = FamilyPlotter(system)
plotter.plot_family_2d(family)
plotter.plot_family_overview(family)
```

---

### 4.4 TransferPlotter

**File**: `e2m2e/visualization/transfer.py`

**Class Signature**:
```python
class TransferPlotter:
    """Transfer trajectory visualizer"""
```

#### Core Methods

| Method | Description |
|--------|-------------|
| `plot_solution_plane()` | Plot solution plane |
| `plot_transfer_orbit()` | Plot transfer trajectory |

#### Usage Example

```python
from e2m2e.visualization import TransferPlotter

plotter = TransferPlotter(system)
plotter.plot_solution_plane(search_results)
plotter.plot_transfer_orbit(transfer_result)
```

---

### 4.5 compute_stability_for_family

**File**: `e2m2e/visualization/stability.py`

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
