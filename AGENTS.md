# AGENTS.md - e2m2e Repository Guidance

e2m2e (Earth to Moon, Moon to Earth) is a Python library for designing cislunar transfer orbits, built on CR3BP orbital dynamics. It provides system modeling, numerical algorithms, transfer trajectory design, and visualization.

MBSE (Model-Based Systems Engineering) infrastructure with SysML-style Protocol interfaces, Pydantic data models, requirement tracking, and automated Mermaid diagram generation.

## Commands

```bash
# Install (editable mode, required for development)
uv sync
uv sync --group dev              # with pytest, ruff, mypy

# Tests (CI requires 80% coverage)
uv run pytest tests/             # all tests
uv run pytest tests/core/test_system.py  # single test file
uv run pytest tests/mbse/        # MBSE-specific tests
uv run pytest -m spice           # only SPICE-dependent tests (require kernels)

# Lint and format (Ruff, 100-char line length)
uv run ruff check .
uv run ruff check --fix .
uv run ruff format .

# Type checking (required for CI)
uv run mypy e2m2e/ --ignore-missing-imports

# Documentation (Sphinx)
uv sync --group docs
uv run --directory docs make html

# MBSE diagram generation
uv run python scripts/generate_mbse_diagrams.py
```

## Architecture

Four-layer dependency graph (strictly one-directional):

```text
core/           Foundation — data structures and physics models
  ↓
algorithms/     Numerical solvers — differential correction, continuation, stability, multiple shooting
  ↓
transfer/       Transfer design — grid search, NLP optimization
  ↓
visualization/  Plotting — unified PlotConfig, orbit families, transfer trajectories
```

Cross-cutting: `mbse/` — Protocol interfaces, Pydantic models, requirement/component registries, diagram generation

### Key imports

```python
from e2m2e.core import CR3BP_System, CR3BP_Dynamics, Orbit, EphemerisSystem, EphemerisDynamics
from e2m2e.algorithms import DifferentialCorrection, Continuation, MultipleShooting, StabilityAnalysis
from e2m2e.algorithms.strategies import halo_fixed_z0, halo_fixed_x0  # strategy functions
from e2m2e.transfer import Transfer
from e2m2e.visualization import PlotConfig, FamilyPlotter
```

### Dynamics hierarchy

```text
Dynamics (base, Template Method)
  → CR3BP_Dynamics    [implements Propagator, EOMProvider; requires SystemModel]
  → EphemerisDynamics [implements Propagator, EOMProvider; requires SPICE kernels]

CR3BP_System → CR3BP_Dynamics → DifferentialCorrection
                    ↑                    ↑
                  Orbit          Continuation, StabilityAnalysis
```

Correction strategies in `algorithms/strategies/`:

- `symmetric_2d.py`: symmetric_2d_fixed_x0, symmetric_2d_fixed_t, symmetric_2d_fixed_y0
- `symmetric_3d.py`: symmetric_3d_fixed_x0, symmetric_xz_fixed_x0, symmetric_xz_fixed_z0
- `halo.py`: halo_fixed_z0, halo_fixed_x0

### MBSE infrastructure (e2m2e/mbse/)

- **architecture/ports.py** — 7 `@runtime_checkable` Protocols: SystemModel, EOMProvider, Propagator, OrbitContainer, CorrectorStrategy, Optimizer, Visualizer
- **architecture/components.py** — Component registry with registered components
- **data/core_models.py** — Pydantic models: PropagationResult (states validated to (n,6)), OrbitProperties, OrbitStability, JacobiResult
- **data/enums.py** — Shared enums: OrbitFamilyType, StabilityLabel, ConvergenceState, etc.
- **requirements/base.py** — Requirement dataclass + RequirementRegistry singleton
- **diagrams/generator.py** — Mermaid diagram generation (BDD, IBD, state machine, activity, sequence, requirement)

## Critical conventions

- **State vector order** is `[x, y, z, vx, vy, vz]` — changing it causes global failures
- **States shape** is `(n_points, 6)` for all Dynamics subclasses — `(6, n)` is the old convention, no longer used
- **STM shape** is `(n_points, 6, 6)` for all Dynamics subclasses
- **Numerical tolerance**: always `rtol=atol=1e-12`; never increase finite difference step sizes
- **Dimensionless units** throughout (DU, TU, VU); call `set_characteristic_scales()` before physical calculations
- **Interface stability**: public method signatures must not break backward compatibility; add new params with defaults
- **New dynamics models**: create a new subclass overriding `_get_eom_func()` and `_get_max_step()`, don't modify the base class
- **Protocol conformance**: new classes should satisfy the appropriate Protocol from `e2m2e.mbse.architecture.ports`

## Testing

- SPICE-dependent tests are marked `@pytest.mark.spice` and auto-skip if kernels unavailable
- Kernels searched in `$SPICE_KERNEL_DIR` or `./kernels/`
- Reference epoch: `"2025-06-21T11:00:06"`
- Shared fixtures in `tests/conftest.py` provide pre-configured Earth-Moon, Sun-Earth, Sun-Jupiter systems
- Modifying `core/` requires running the full test suite; `algorithms/` or `transfer/` changes can be verified with targeted tests
- MBSE tests in `tests/mbse/` verify Protocol conformance, requirement coverage, and Pydantic model validation

## Adding new modules

1. Create file in the appropriate subpackage (e.g., `e2m2e/algorithms/new_algo.py`)
2. Export in subpackage `__init__.py`
3. Register in top-level `e2m2e/__init__.py` and add to `__all__`
4. If adding dependencies, update `pyproject.toml` and re-run `uv sync`
5. If adding a new component, register it in `e2m2e/mbse/architecture/` and add requirements to `e2m2e/mbse/requirements/`
6. Regenerate MBSE diagrams: `uv run python scripts/generate_mbse_diagrams.py`

## Release workflow

- Version tags (`v*`) trigger release pipeline; tag must match `pyproject.toml` version
- CI runs lint, typecheck (mypy), and tests with 80% coverage threshold
- Multi-platform testing: Ubuntu, macOS, Windows on Python 3.10–3.13

## Optional dependencies

- **COPT** (`coptpy`): Required for `optimize_with_copt()` in transfer NLP optimization. Falls back to scipy if not installed. Check `_HAVE_COPT` flag in `e2m2e/transfer/__init__.py`.

## Agent skills

### Issue tracker

Issues live in GitHub Issues for this repo. See `docs/agents/issue-tracker.md`.

### Triage labels

Uses the five canonical labels: needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
