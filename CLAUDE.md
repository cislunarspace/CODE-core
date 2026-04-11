# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

e2m2e (Earth to Moon, Moon to Earth) is a Python library for designing cislunar transfer orbits, built on Circular Restricted Three-Body Problem (CR3BP) orbital dynamics. It provides system modeling, numerical algorithms, transfer trajectory design, and visualization.

v4.0.0 introduces MBSE (Model-Based Systems Engineering) infrastructure with SysML-style Protocol interfaces, Pydantic data models, requirement tracking, and automated Mermaid diagram generation.

## Commands

```bash
# Install (editable mode, required for development)
pip install -e .
pip install -e ".[dev]"          # with pytest, ruff

# Tests
pytest tests/                    # all tests
pytest tests/core/test_system.py # single test file
pytest tests/mbse/               # MBSE-specific tests

# MBSE diagram generation
python scripts/generate_mbse_diagrams.py

# Lint and format (Ruff, 100-char line length)
ruff check .
ruff check --fix .
ruff format .
```

## Architecture

Four-layer dependency graph (strictly one-directional):

```
core/           Foundation — data structures and physics models
  ↓
algorithms/     Numerical solvers — differential correction, continuation, stability, multiple shooting
  ↓
transfer/       Transfer design — grid search, NLP optimization
  ↓
visualization/  Plotting — unified PlotConfig, orbit families, transfer trajectories
```

Cross-cutting: `mbse/` — Protocol interfaces, Pydantic models, requirement/component registries, diagram generation

### Key class hierarchy

```
Dynamics (base, Template Method)
  → CR3BP_Dynamics    [Propagator, EOMProvider, SystemModel]
  → EphemerisDynamics [Propagator, EOMProvider]

CR3BP_System → CR3BP_Dynamics → DifferentialCorrection
                    ↑                    ↑
                  Orbit          Continuation, StabilityAnalysis

CorrectionConfig (frozen dataclass) ← strategy functions
  ← symmetric_2d_fixed_x0, symmetric_2d_fixed_t, ...
  ← symmetric_3d_fixed_x0, symmetric_xz_fixed_*, ...
  ← halo_fixed_z0, halo_fixed_x0

PlotConfig (Pydantic BaseModel) → OrbitVisualizer, FamilyPlotter, TransferPlotter [Visualizer Protocol]
```

### MBSE infrastructure (e2m2e/mbse/)

- **architecture/ports.py** — 7 `@runtime_checkable` Protocols: SystemModel, EOMProvider, Propagator, OrbitContainer, CorrectorStrategy, Optimizer, Visualizer
- **architecture/components.py** — Component registry with 17 registered components
- **data/core_models.py** — Pydantic models: PropagationResult (states validated to (n,6)), OrbitProperties, OrbitStability, JacobiResult
- **data/enums.py** — Shared enums: OrbitFamilyType, StabilityLabel, ConvergenceState, etc.
- **requirements/base.py** — Requirement dataclass + RequirementRegistry singleton (24 requirements, 100% coverage)
- **diagrams/generator.py** — Mermaid diagram generation (BDD, IBD, state machine, activity, sequence, requirement)

### Module responsibilities

- **core/** — `CR3BP_System` (system definition, mass parameter, libration points, Jacobi constant), `Dynamics`/`CR3BP_Dynamics` (Template Method pattern, equations of motion, STM, propagation), `Orbit`/`OrbitFamily` (composition pattern with property proxies), `CoordinateTransformation` (rotating/inertial frame conversions), `SPICEManager` (kernel management), `EphemerisDynamics` (N-body in J2000, unified (n,6) states shape)
- **algorithms/** — `DifferentialCorrection` (strategy pattern with CorrectionConfig, Richardson 3rd-order halo approximation), `Continuation` (natural and pseudo-arclength, uses CR3BP_Dynamics for physics), `StabilityAnalysis` (Floquet multipliers, bifurcation detection), `MultipleShooting` (parallel propagation via `n_workers`)
- **transfer/** — `TransferSearch`/`DROTransferSearch` (parallel grid search, SearchConfig dataclass), `DROTRONLPOptimizer` (Cui et al. 2025 two-step NLP method, NLPOptimizationVariables/Result dataclasses), `Transfer` (chainable API: `.set_orbit().optimize()`)
- **visualization/** — `PlotConfig` (Pydantic BaseModel, centralized font/size/color), `FamilyPlotter`, `TransferPlotter` (all satisfy Visualizer Protocol)
- **mbse/** — Cross-cutting MBSE infrastructure (Protocols, Pydantic models, requirement/component registries, diagram generation)

## Critical conventions

- **State vector order** is `[x, y, z, vx, vy, vz]` — changing it causes global failures
- **States shape** is `(n_points, 6)` for all Dynamics subclasses — `(6, n)` is the old convention, no longer used
- **STM shape** is `(n_points, 6, 6)` for all Dynamics subclasses
- **Numerical tolerance**: always `rtol=atol=1e-12`; never increase finite difference step sizes
- **Dimensionless units** throughout (DU, TU, VU); call `set_characteristic_scales()` before physical calculations
- **Interface stability**: public method signatures must not break backward compatibility; add new params with defaults
- **New dynamics models**: create a new subclass overriding `_get_eom_func()` and `_get_max_step()`, don't modify the base class
- **Protocol conformance**: new classes should satisfy the appropriate Protocol from `e2m2e.mbse.architecture.ports`

## Adding new modules

1. Create file in the appropriate subpackage (e.g., `e2m2e/algorithms/new_algo.py`)
2. Export in subpackage `__init__.py`
3. Register in top-level `e2m2e/__init__.py` and add to `__all__`
4. If adding dependencies, update `pyproject.toml` and re-run `pip install -e .`
5. If adding a new component, register it in `e2m2e/mbse/architecture/` and add requirements to `e2m2e/mbse/requirements/`
6. Regenerate MBSE diagrams: `python scripts/generate_mbse_diagrams.py`

## Testing

- SPICE-dependent tests are marked `@pytest.mark.spice` and auto-skip if kernels unavailable
- Kernels searched in `$SPICE_KERNEL_DIR` or `./kernels/`
- Reference epoch: `"2025-06-21T11:00:06"`
- Shared fixtures in `tests/conftest.py` provide pre-configured Earth-Moon, Sun-Earth, Sun-Jupiter systems
- Modifying `core/` requires running the full test suite; `algorithms/` or `transfer/` changes can be verified with targeted tests
- MBSE tests in `tests/mbse/` verify Protocol conformance, requirement coverage, and Pydantic model validation

## Typical workflow

```python
from e2m2e.core import CR3BP_System, CR3BP_Dynamics, Orbit
from e2m2e.algorithms import DifferentialCorrection, Continuation
from e2m2e.algorithms.strategies import halo_fixed_z0
from e2m2e.transfer import Transfer

system = CR3BP_System.from_known_system("earth_moon")
dynamics = CR3BP_Dynamics(system)

# Strategy-based correction setup
corrector = DifferentialCorrection(dynamic=dynamics)
corrector.setup_halo_orbit_fixed_z0(z0=0.1, libration_point=1)

continuation = Continuation(corrector=corrector)
transfer = Transfer(dynamics)
```
