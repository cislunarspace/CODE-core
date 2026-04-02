# e2m2e — Earth to Moon, Moon to Earth

**Earth-Moon Space Transfer Trajectory Design Library**

[![License: Apache](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

`e2m2e` is a Python library for designing **operating orbits** and **transfer orbits** in Earth-Moon space. The library uses object-oriented programming and provides a modular design.

## Installation

### Install from Source

```bash
# Clone the repository
git clone https://gitee.com/cislunarspace/e2m2e.git
cd e2m2e

# Install dependencies
python -m pip install -e .
```

## Quick Start

```python
import e2m2e
from e2m2e.core.system import CR3BP_System

# 1. Create Earth-Moon system
system = CR3BP_System.from_known_system("earth_moon")
system.set_characteristic_scales(distance=384400, period=27.32 * 86400)
system.compute_libration_points()

# Use info() method to view system information
system.info()

# 2. Get libration point information
print(f"L1 position: {system.L1}")
print(f"L2 position: {system.L2}")

# 3. Calculate Jacobi constant
state = [0.8, 0.1, 0.0, 0.0, 0.2, 0.0]
jacobi_constant = system.get_jacobi_constant(state)
print(f"Jacobi constant: {jacobi_constant:.4f}")
```

## Visualization Features

`e2m2e` provides powerful orbit visualization with a unified `PlotConfig` and specialized plotters:

```python
from e2m2e.visualization import PlotConfig, FamilyPlotter

config = PlotConfig(title=32, label=28)
config.apply_rcparams()

plotter = FamilyPlotter(system, config)
plotter.plot_family_2d(family, jacobi_values, title="DRO Family")
plotter.plot_jacobi_period_stability(jacobi_values, periods, stability_values)
```

For more visualization features and usage examples, refer to the [Visualization Module Guide](docs/guides/visualization-guide.md).

## Development and Contribution

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### Code Standards

This project uses [Ruff](https://github.com/astral-sh/ruff) for code formatting:

```bash
# Check code formatting
ruff check .

# Auto-fix code formatting
ruff check --fix .

# Format code
ruff format .
```

### Submitting Contributions

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Thanks to all researchers on the three-body problem for their pioneering work
- Thanks to the open-source community for excellent tools and libraries
