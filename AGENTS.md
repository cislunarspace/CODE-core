# AGENTS.md - e2m2e Repository Guidance

## Development Commands

### Installation
```bash
# Install in development mode (preferred)
python -m pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests
pytest tests/

# Tests requiring SPICE kernels are marked with @pytest.mark.spice
# SPICE kernel directory: $SPICE_KERNEL_DIR or ./kernels/
```

### Code Quality
```bash
# Format and lint with Ruff (line length: 100)
ruff check .          # Check
ruff check --fix .    # Auto-fix
ruff format .         # Format
```

## Project Structure

- **Core module** (`e2m2e/core/`): CR3BP system, dynamics, orbits, coordinate transforms
- **Algorithms** (`e2m2e/algorithms/`): Differential correction, continuation, stability analysis
- **Transfer** (`e2m2e/transfer/`): Orbit transfer design and optimization
- **Visualization** (`e2m2e/visualization/`): Plotting with unified `PlotConfig`

## Key Architecture Notes

1. **CR3BP-centric**: All orbits are in circular restricted three-body problem frame
2. **Orbit data structure**: `Orbit` class stores states and times; `OrbitFamily` for families
3. **Dimensional units**: Uses characteristic scales (DU, TU, VU) - set via `set_characteristic_scales()`
4. **SPICE integration**: Ephemeris models require SPICE kernels in `$SPICE_KERNEL_DIR` or `./kernels/`

## Testing Gotchas

- SPICE-dependent tests skip if kernels not found
- Reference epoch for ephemeris tests: "2025-06-21T11:00:06"
- Test fixtures provide pre-configured Earth-Moon, Sun-Earth, Sun-Jupiter systems

## Package Configuration

- **Primary config**: `pyproject.toml` (not `setup.py`)
- **Python version**: ≥3.10
- **Core dependencies**: numpy, scipy, matplotlib, tqdm, spiceypy
- **Development dependencies**: pytest, pytest-cov, ruff

## Workflow Conventions

1. **Always use development install** (`-e .`) for code changes
2. **Run Ruff before committing** - project enforces 100-character line length
3. **Test SPICE features separately** - they require external kernel files
4. **Visualization uses `PlotConfig`** - configure fonts/sizes via this class

## Entry Points

- Main imports: `from e2m2e.core import CR3BP_System, CR3BP_Dynamics, Orbit`
- System creation: `CR3BP_System.from_known_system("earth_moon")`
- Characteristic scales must be set before physical calculations