# e2m2e Test Implementation Plan

## Overview
Generate comprehensive unit tests for e2m2e project following module dependency order. Target: 80% coverage.

## Phase 1: core/ Module Tests (Foundational)
### 1.1 test_core_system.py
- `LibrationPoint` enum values
- `CR3BP_System.__init__`
- `CR3BP_System.from_known_system` (earth_moon, sun_earth, sun_jupiter)
- `CR3BP_System.compute_libration_points`
- `CR3BP_System.get_jacobi_constant`
- `CR3BP_System.dimensionless_to_physical` / `physical_to_dimensionless`
- `CR3BP_System.set_characteristic_scales`
- Error cases: invalid system_name

### 1.2 test_core_dynamics.py
- `CR3BP_Dynamics.__init__`
- `CR3BP_Dynamics.equations_of_motion`
- `CR3BP_Dynamics.equations_with_stm`
- `CR3BP_Dynamics.propagate` (basic, with_stm)
- `CR3BP_Dynamics.compute_jacobi_constant`
- `CR3BP_Dynamics.check_cross_section`
- STM properties (determinant = 1)

### 1.3 test_core_coordinate.py
- `ReferenceFrame` enum values
- `CoordinateTransformation.__init__`
- `CoordinateTransformation.compute_rotation_matrix`
- `CoordinateTransformation.rotating_to_inertial`
- `CoordinateTransformation.inertial_to_rotating`
- `CoordinateTransformation.barycentric_to_primary`
- `CoordinateTransformation.primary_to_barycentric`
- Transform reversibility

### 1.4 test_core_orbit.py
- `Orbit.__init__` (valid/invalid states)
- `Orbit.compute_basic_properties`
- `Orbit.get_period`
- `Orbit.get_amplitude`
- `Orbit.interpolate_at_time`
- `Orbit.save_to_file` / `load_from_file`

## Phase 2: algorithms/ Module Tests
### 2.1 test_algorithms_continuation.py
- Continuation class/methods
- Curve tracking

### 2.2 test_algorithms_differential_correction.py
- Differential correction methods
- Convergence criteria

### 2.3 test_algorithms_stability.py
- Stability index computation
- Eigenvalue analysis

## Phase 3: transfer/ Module Tests
### 3.1 test_transfer_dro_ro_nlp.py
- NLP formulation
- Constraint handling

### 3.2 test_transfer_dro_ro_search.py
- DRO-RO search algorithms
- State validation

### 3.3 test_transfer_earth_moon.py
### 3.4 test_transfer_moon_earth.py
### 3.5 test_transfer_inter_orbit.py

## Phase 4: visualization/ Module Tests
### 4.1 test_visualization_plotting.py
- Plot generation (without display)
- Figure creation

## Success Criteria
- All tests compile (no syntax errors)
- All tests pass
- Coverage report shows 80%+ coverage on core modules
- Test files follow pytest conventions

## Test File Locations
```
tests/
├── conftest.py              # Shared fixtures
├── core/
│   ├── __init__.py
│   ├── test_system.py
│   ├── test_dynamics.py
│   ├── test_coordinate.py
│   └── test_orbit.py
├── algorithms/
│   ├── __init__.py
│   ├── test_continuation.py
│   ├── test_differential_correction.py
│   └── test_stability.py
├── transfer/
│   ├── __init__.py
│   ├── test_dro_ro_nlp.py
│   ├── test_dro_ro_search.py
│   ├── test_earth_moon.py
│   ├── test_moon_earth.py
│   └── test_inter_orbit.py
└── visualization/
    ├── __init__.py
    └── test_plotting.py
```
