# Orbital MCP Server

Model Context Protocol server for orbital mechanics calculations, built on the e2m2e library.

## Features

- **CR3BP Dynamics**: Propagate trajectories in the Circular Restricted Three-Body Problem
- **Orbit Analysis**: Compute periods, amplitudes, stability indices, and monodromy matrices
- **State Propagation**: Forward integration with optional STM (State Transition Matrix) computation
- **Jacobi Constant**: Calculate Jacobi constants for energy-based orbit classification
- **Cross-Section Detection**: Detect Poincaré section crossings

## Installation

```bash
# Install dependencies
uv sync

# Run in development mode
uv run mcp dev orbital_mcp/server.py

# Install to Claude Desktop
uv run mcp install orbital_mcp/server.py
```

## Available Tools

| Tool | Description |
|------|-------------|
| `propagate_trajectory` | Propagate a trajectory in CR3BP |
| `compute_stm` | Compute State Transition Matrix between two times |
| `compute_jacobi` | Compute Jacobi constant for a given state |
| `get_orbit_period` | Calculate period of an orbit |
| `get_orbit_amplitude` | Calculate amplitude in x, y, or z direction |
| `check_crossing` | Check if trajectory crosses a Poincaré section |

## Configuration

The server uses stdio transport by default for local operations.

## License

MIT
