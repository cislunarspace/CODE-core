# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

e2m2e (Earth to Moon, Moon to Earth) is a Python library for designing cislunar transfer orbits, built on Circular Restricted Three-Body Problem (CR3BP) orbital dynamics. It provides system modeling, numerical algorithms, transfer trajectory design, and visualization.

## Commands

```bash
# Install (editable mode, required for development)
pip install -e .
pip install -e ".[dev]"          # with pytest, ruff

# Tests
pytest tests/                    # all tests
pytest tests/core/test_system.py # single test file

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

### Key class hierarchy

```
CR3BP_System → CR3BP_Dynamics → DifferentialCorrection
                    ↑                    ↑
                  Orbit          Continuation, StabilityAnalysis

Dynamics (base) → CR3BP_Dynamics
                → EphemerisDynamics → HomotopyEphemerisDynamics
```

### Module responsibilities

- **core/** — `CR3BP_System` (system definition, mass parameter, libration points, Jacobi constant), `Dynamics`/`CR3BP_Dynamics` (equations of motion, STM, propagation), `Orbit`/`OrbitFamily` (trajectory containers with JSON serialization), `CoordinateTransformation` (rotating/inertial frame conversions), `SPICEManager` (kernel management), `EphemerisDynamics` (N-body in J2000), `HomotopyEphemerisDynamics` (smooth CR3BP↔ephemeris transition)
- **algorithms/** — `DifferentialCorrection` (periodic orbit correction + Richardson 3rd-order halo approximation), `Continuation` (natural and pseudo-arclength), `StabilityAnalysis` (Floquet multipliers, bifurcation detection), `MultipleShooting` (parallel propagation via `n_workers`)
- **transfer/** — `TransferSearch`/`DROTransferSearch`/`GeoTransferSearch` (parallel grid search), `DROTRONLPOptimizer` (Cui et al. 2025 two-step NLP method, optional COPT solver), `Transfer` (chainable API: `.set_orbit().optimize()`)
- **visualization/** — `PlotConfig` (centralized font/size/color), `FamilyPlotter`, `TransferPlotter`

## Critical conventions

- **State vector order** is `[x, y, z, vx, vy, vz]` — changing it causes global failures
- **Numerical tolerance**: always `rtol=atol=1e-12`; never increase finite difference step sizes
- **Dimensionless units** throughout (DU, TU, VU); call `set_characteristic_scales()` before physical calculations
- **Interface stability**: public method signatures must not break backward compatibility; add new params with defaults
- **New dynamics models**: create a new subclass, don't modify the base class

## Adding new modules

1. Create file in the appropriate subpackage (e.g., `e2m2e/algorithms/new_algo.py`)
2. Export in subpackage `__init__.py`
3. Register in top-level `e2m2e/__init__.py` and add to `__all__`
4. If adding dependencies, update `pyproject.toml` and re-run `pip install -e .`

## Testing

- SPICE-dependent tests are marked `@pytest.mark.spice` and auto-skip if kernels unavailable
- Kernels searched in `$SPICE_KERNEL_DIR` or `./kernels/`
- Reference epoch: `"2025-06-21T11:00:06"`
- Shared fixtures in `tests/conftest.py` provide pre-configured Earth-Moon, Sun-Earth, Sun-Jupiter systems
- Modifying `core/` requires running the full test suite; `algorithms/` or `transfer/` changes can be verified with targeted tests

## Typical workflow

```python
from e2m2e.core import CR3BP_System, CR3BP_Dynamics, Orbit
from e2m2e.algorithms import DifferentialCorrection, Continuation
from e2m2e.transfer import Transfer

system = CR3BP_System.from_known_system("earth_moon")
dynamics = CR3BP_Dynamics(system)
corrector = DifferentialCorrection(dynamic=dynamics)
continuation = Continuation(corrector=corrector)
transfer = Transfer(dynamics)
```
