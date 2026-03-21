# e2m2e Test Generation Research

## Project Overview
- **Language**: Python 3.10+
- **Testing Framework**: pytest
- **Project Type**: Astrodynamics library for CR3BP orbital mechanics

## Project Structure

### Source Code (e2m2e/)
```
e2m2e/
├── core/
│   ├── system.py      - CR3BP_System class, LibrationPoint enum
│   ├── dynamics.py    - CR3BP_Dynamics class
│   ├── orbit.py       - Orbit class
│   └── coordinate.py  - CoordinateTransformation, ReferenceFrame enum
├── algorithms/
│   ├── continuation.py         - Continuation method algorithms
│   ├── differential_correction.py - Differential correction
│   └── stability.py            - Stability analysis
├── transfer/
│   ├── dro_ro_nlp.py    - NLP for DRO-RO transfers
│   ├── dro_ro_search.py - DRO-RO search algorithms
│   ├── earth_moon.py    - Earth-Moon transfer
│   ├── inter_orbit.py  - Inter-orbit transfers
│   └── moon_earth.py    - Moon-Earth transfer
└── visualization/
    └── plotting.py      - Plotting utilities
```

### Existing Tests
```
tests/
├── test_basic.py           - Basic functionality tests (functional style)
├── core/
├── algorithms/
├── transfer/
└── visualization/
```

## Dependencies
- numpy>=1.24
- scipy>=1.10
- matplotlib>=3.7
- pytest (dev)
- pytest-cov (dev)

## Key Classes to Test

### core/system.py
- `LibrationPoint` enum (L1-L5)
- `CR3BP_System` class:
  - `__init__(mu, primary, secondary)`
  - `from_known_system(system_name)` - class method
  - `compute_libration_points()`
  - `get_jacobi_constant(state)`
  - `dimensionless_to_physical(state)`
  - `physical_to_dimensionless(state)`
  - `set_characteristic_scales(distance, period)`

### core/dynamics.py
- `CR3BP_Dynamics` class:
  - `__init__(system)`
  - `equations_of_motion(t, state)`
  - `equations_with_stm(t, augmented_state)`
  - `propagate(initial_state, t_span, t_eval=None, with_stm=False)`
  - `compute_state_transition_matrix(initial_state, t)`
  - `compute_jacobi_constant(state)`
  - `check_cross_section(state, plane, value)`

### core/orbit.py
- `Orbit` class:
  - `__init__(states, times, system=None)`
  - `compute_basic_properties()`
  - `get_period()`
  - `get_amplitude(direction)`
  - `interpolate_at_time(t)`
  - `save_to_file(filename)`
  - `load_from_file(filename)`

### core/coordinate.py
- `ReferenceFrame` enum
- `CoordinateTransformation` class:
  - `__init__(system)`
  - `compute_rotation_matrix(time)`
  - `rotating_to_inertial(state, time)`
  - `inertial_to_rotating(state, time)`
  - `barycentric_to_primary(state)`
  - `primary_to_barycentric(state)`
  - `transform_velocity(state, from_frame, to_frame)`

## Testing Patterns
1. **Arrange-Act-Assert** pattern for test structure
2. Use `pytest.fixture` for shared test objects (e.g., earth_moon_system)
3. Use `pytest.mark.parametrize` for testing multiple inputs
4. Focus on:
   - Happy path tests (valid inputs → expected outputs)
   - Edge cases (boundary conditions, empty inputs)
   - Error cases (invalid inputs, exception handling)

## Build/Test Commands
- Run tests: `pytest tests/`
- Run with coverage: `pytest --cov=e2m2e tests/`
- Run specific test: `pytest tests/core/test_system.py`
