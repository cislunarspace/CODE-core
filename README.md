# e2m2e: Earth to Moon, Moon to Earth

**English** | [简体中文](README.zh-CN.md)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/e2m2e)](https://pypi.org/project/e2m2e/)
[![CI](https://github.com/cislunarspace/e2m2e/actions/workflows/ci.yml/badge.svg)](https://github.com/cislunarspace/e2m2e/actions/workflows/ci.yml)
[![GitHub stars](https://img.shields.io/github/stars/cislunarspace/e2m2e.svg)](https://github.com/cislunarspace/e2m2e/stargazers)
[![Rust: 1.98.0](https://img.shields.io/badge/rust-1.98.0-orange.svg)](https://www.rust-lang.org/)

e2m2e is an **algorithm toolset infrastructure** for cislunar space mission planning. In an LLM+Agent-style autonomous mission planning system, the large language model understands mission intent and decomposes/orchestrates subtasks, while e2m2e provides precise and reliable orbit computation tools: it builds dynamical models of cislunar space, generates periodic orbit families, designs transfer paths between orbits, and visualizes results for inspection.

## How to read this repository

The runtime architecture is just four pieces: `e2m2e/api/` is the sole external entry (the Facade, from which the CLI and MCP derive); `e2m2e/algorithm/` constructs problems with domain knowledge (choosing orbit families, constraints, initial guesses); `crates/` is the Rust numerical layer where heavy iterations converge; `e2m2e/data/` supplies ephemeris caches, frame data, and constant baselines. `e2m2e/tools/` is logging/visualization support, and `e2m2e/mbse/` sits outside the dependency chain.

The journey of one orbit task: `api` receives the request → `algorithm/family` picks a family and initial guess (seeds from `catalog/records` or `algorithm/normal_form`) → shooting sinks into `crates/e2m2e-integrators`, with per-step forces in `crates/e2m2e-forces` (ephemerides come from the pre-sampled cache tables of `data/frames`, never live SPICE handles; constants from `data/constants`) → the result lands in `catalog/` and is delivered through `api/cli` and `api/mcp`.

The remaining top-level directories (`tests/`, `examples/`, `docs/`, `scripts/`, `kernels/`, ...) are tests, docs, scripts, and data assets — off the runtime dependency chain. For the full design narrative see [docs/architecture/architecture.md](docs/architecture/architecture.md).

## Installation

Install with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install e2m2e
```

Release wheels cover Windows x86_64, Linux x86_64, and Linux aarch64 (arm64, e.g. Kunpeng / Phytium / Raspberry Pi); other platforms build from source.

To use it in your own project:

```bash
uv init my-project && cd my-project
uv add e2m2e
```

To develop from source (requires the [Rust 1.98.0 toolchain](https://www.rust-lang.org/tools/install); the repo pins the version via `rust-toolchain.toml`, used to build the integrator kernel):

```bash
git clone https://github.com/cislunarspace/e2m2e.git
cd e2m2e
make dev     # Single entry point: sync deps + fetch CSPICE build package & SPICE kernels + build/install the Rust extension (spice enabled by default)
```

#### Installing make on Windows

Windows does not ship with `make`. You can install it with [Scoop](https://scoop.sh/); run these commands in PowerShell:

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex
scoop install make
```

After installation, reopen PowerShell, return to the repository directory and run `make dev` (single entry point: sync deps + fetch data + build/install; see above). If you already have Scoop, just run `scoop install make`.

To run Rust tests on Windows use `make test-rust`: the test binary depends on Python's `python3.dll`, which lives in the Python installation root rather than in the venv's `Scripts/`; the Makefile auto-detects and adds it to the test process PATH. If auto-detection fails, run `make test-rust PYTHON_DLL_DIR=<directory containing python*.dll>` explicitly. When troubleshooting manually, first run `dumpbin /DEPENDENTS` or `dumpbin /IMPORTS` on the failing test EXE to identify the actually missing DLL; do not add CSPICE's `lib/` (the static library directory) to PATH.

### SPICE kernels

Ephemeris dynamics require NASA SPICE kernel files. All kernels needed by this project's tests (planetary ephemerides, Earth rotation, Moon attitude, leap seconds, and planetary constants) are packaged as `kernels-v1` in a [GitHub Release](https://github.com/cislunarspace/e2m2e/releases). Three ways to configure:

- **Automatic setup (recommended)**: `make setup` downloads them into `kernels/` (see `scripts/download_kernels.py`).
- **Manual download**: download `kernels-v1` from the Release page and extract into `kernels/`.
- **Bring your own data**: use your own kernel files placed in `kernels/`, or point `$SPICE_KERNEL_DIR` at their location.

Official source: [NASA NAIF](https://naif.jpl.nasa.gov/naif/data.html).

## Quick start

Design an Earth–Moon L2 Halo orbit (complete the SPICE kernel setup above first):

```python
from e2m2e.api import Facade

facade = Facade()

result = facade.design_orbit(
    orbit_type="Halo",
    collinear_point=2,
    amplitude=30000.0,
    epoch=[2024, 1, 1, 0, 0, 0.0],
    duration=365.25 * 86400.0,
)

print(result.orbit_type)
print(result.initial_state)
```

For parameter meanings, returned fields, and other orbit types see the [online documentation](https://cislunarspace.github.io/e2m2e/); runnable examples are in the [`examples/`](examples/) directory.

## MCP

e2m2e can act as an [MCP](https://modelcontextprotocol.io/) server exposing 13 task-level tools (orbit design, station-keeping simulation, transfer design, orbit propagation, spacetime conversion, orbit family generation, and 7 catalog tools) to LLM Agents over stdio transport without listening on any port. The tool list is derived from Facade method metadata; artifacts are automatically archived and `record_id`s chain across tools. For usage and a tool cheat sheet, see the documentation "[Using e2m2e through MCP](https://cislunarspace.github.io/e2m2e/getting-started/mcp.html)".

Install the MCP extra (in an environment that already has e2m2e):

```bash
uv pip install "e2m2e[mcp]"   # or pip install "e2m2e[mcp]"
```

Register the server in your MCP client configuration. Generic format for Claude Desktop / Cursor etc. (`command` points to the executable in the environment where e2m2e is installed):

```json
{
  "mcpServers": {
    "e2m2e": {
      "command": "/path/to/venv/bin/e2m2e",
      "args": ["mcp-serve"],
      "cwd": "/path/to/e2m2e-repo"
    }
  }
}
```

ZCode workspace configuration (`<repo>/.zcode/config.json`, nested under `mcp.servers`, connected automatically at session start):

```json
{
  "mcp": {
    "servers": {
      "e2m2e": {
        "type": "stdio",
        "command": "C:\\path\\to\\.venv\\Scripts\\e2m2e.exe",
        "args": ["mcp-serve"],
        "cwd": "C:\\path\\to\\e2m2e-repo"
      }
    }
  }
}
```

It is recommended to pin `cwd` to the directory containing `kernels/` (SPICE kernels) and `catalog/` (orbit catalog); either directory can also be given as absolute paths via the environment variables `SPICE_KERNEL_DIR` / `E2M2E_CATALOG_DIR`.

Once configured, drive it directly in natural language from your client, e.g.:

> Design an L2 southern NRHO with 3000 km perilune altitude, then run 100 Monte Carlo station-keeping simulations on it and tag the results "candidate".

## Capabilities

Completed and uncompleted parts are listed by domain. For a detailed capability matrix and API documentation see the [online documentation](https://cislunarspace.github.io/e2m2e/); version-by-version changes are in [CHANGELOG.md](CHANGELOG.md).

**Spacetime systems**

- Coordinate frame conversion: J2000 / ITRF93 (high-precision SPICE) / IAU 2006, GMAT-compatible native ITRF, dynamic frames VNB / LVLH.
- Joint spacetime transformation: TDT+GCRS ↔ TDB+EBCRS (r2s2 backend, including relativistic terms).
- SPICE ephemeris and time management: kernel loading, UTC / TDB / TAI time scales, body state and frame rotation queries.

**Integrators and dynamics**

- Rust integrator kernel: single-step RK (PD45 / PD78 / RK89), Adams multistep, Störmer–Cowell second-order integration, IAS15 (15th-order Gauss-Radau predictor-corrector with compensated summation for high-accuracy long arcs); state transition matrix (STM) propagation; event detection (terminal / direction semantics).
- Variational equations: STM of the state with respect to the initial state, plus first-order sensitivity columns with respect to force-model parameters (Cr / Cd) for orbit determination and covariance propagation.
- Dynamical models: CR3BP (fast design), ephemeris N-body (SPICE, accurate extrapolation), BCR4BP with analytic solar perturbation, plus conversions among the three.
- High-fidelity force models: point-mass and third-body gravity, spherical-harmonics gravity field (with solid tides), ECOM 9-parameter solar radiation pressure, atmospheric drag, SRP, continuous thrust; propagation accuracy aligned with GMAT and DFH to sub-100 m level.

**Mission orbit design**

- Periodic orbit families: DRO, Halo, Lyapunov, Lissajous, resonant orbits (RO), DPO, Axial, triangular-libration-point SPO / LPO, Horseshoe.
- Numerical algorithms: differential correction, multiple shooting, continuation; full-pipeline CR3BP initial guess → ephemeris correction → high-fidelity propagation.
- Nominal orbit contract (NominalOrbit): equally-spaced state table + Floquet basis + projection factor, consumed directly by station keeping.

**Transfer design**

- Impulsive transfers: Lambert solver and porkchop scans, multi-impulse optimization (Lawden primer-vector test), direct Hohmann transfers (HMN).
- Low-energy transfers: lunar gravity assist (LGA), WSB ballistic capture via solar gravity assist, invariant manifolds stitched with Poincaré sections.
- Low-thrust transfers: Q-law initial guess + shooting / collocation.
- Two-step grid search + nonlinear programming (parallelized with Rust Rayon).

**Orbit control**

- Three control laws: feature point, strict target point, relaxed target point; Monte Carlo navigation-error and thrust-error simulation.
- Angular momentum management: joint attitude/thruster control.

**Interfaces and tools**

- Facade task-level entry point providing a unified calling surface.
- MCP service wrapper: in-process `create_server` and the `e2m2e mcp-serve` subcommand (stdio transport, `[mcp]` extra); tool list derived from Facade method metadata — orbit design, station-keeping simulation, transfer design, orbit propagation, spacetime conversion, orbit family generation, and 7 catalog tools; artifacts automatically archived, `record_id`s chain across tools. For integration config and tool usage see the documentation "[Using e2m2e through MCP](https://cislunarspace.github.io/e2m2e/getting-started/mcp.html)".

## Documentation

Online docs: <https://cislunarspace.github.io/e2m2e/>

Local build:

```bash
uv sync --group docs
uv run sphinx-build -b html docs docs/_build/html
```

## Tests and code style

```bash
make test     # Rust tests + Python tests in parallel via xdist (run make setup to fetch kernels first)
make check    # cargo fmt/clippy + ruff
```

Tests are grouped into six categories by verification target, with directories mirroring source layout:

- `algorithm`: coordinate, dynamics, design, correction, transfer algorithm orchestration chains
- `api`: Facade and MCP interface validation, responses, and error translation
- `data`: data layer — kernels, reference frames, physical constants
- `mbse`: MBSE data models, requirement registration, and diagram generation
- `numerical`: integrator accuracy against analytic orbits; force-model accelerations and Jacobians
- `tools`: auxiliary utilities — logging, formatting, visualization

An additional `_meta` directory constrains the test infrastructure itself. Assertions rely mainly on analytic solutions, conserved quantities, and literature formulas; the coordinate and time chains additionally compare against GMAT reference data. Rust-side numerical methods are verified against analytic solutions in `crates/*/tests/`.

## Contributing

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

```bibtex
@software{e2m2e,
  title = {e2m2e: Earth to Moon, Moon to Earth Transfer Orbit Design Library},
  author = {ouyangjiahong},
  email = {ouyangjiahong22@nudt.edu.cn},
  url = {https://github.com/cislunarspace/e2m2e},
  version = {5.8.6},
  year = {2026},
}
```

## License

[Apache 2.0](LICENSE)
