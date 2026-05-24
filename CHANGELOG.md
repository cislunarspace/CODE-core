# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [4.2.1] - 2026-05-25

### Fixed
- Import block sorting in `e2m2e/core/orbit.py` (ruff I001)

## [4.0.0] - 2026-04-14

### Added
- MBSE infrastructure with `@runtime_checkable` Protocol interfaces (SystemModel, EOMProvider, Propagator, OrbitContainer, CorrectorStrategy, Optimizer, Visualizer)
- Pydantic data models: PropagationResult, OrbitProperties, OrbitStability, JacobiResult
- Strategy-based differential correction: `halo_fixed_z0`, `halo_fixed_x0`, `symmetric_2d_fixed_x0`, etc.
- Requirement registry and automated Mermaid diagram generation
- `MultipleShooting` class with multi-process parallel propagation
- `GeoTransferSearch` for DRO to GEO parallel grid search
- `BodyName` constants class with solar system GM data
- `HomotopyEphemerisDynamics` for homotopy-based ephemeris dynamics

### Changed
- Refactored ephemeris layer for numerical safety and architecture quality
- Decomposed monolithic differential correction into strategy pattern
- `convert_to_j2000` parameter changed from `tu_seconds` to `tu_days`
- Refactored visualization to use `PlotConfig` as unified configuration

### Fixed
- EphemerisDynamics method signatures compatible with base Dynamics class
- SPICE `bodvrd` return value indexing and Pylance type errors
- Jacobi constant family plots sorted to eliminate out-of-order artifacts
- 22 core/algorithms tests and 32 visualization tests

## [3.2.0] - 2025-12-01

### Added
- Ephemeris-based dynamics (EphemerisDynamics) with SPICE kernel support
- EphemerisSystem for multi-body gravitational modeling
- SPICE kernel management utilities
- Homotopy dynamics for continuation across solution branches

### Changed
- Documentation rewritten in task-oriented style (Chinese and English)

### Fixed
- 90 broken anchor warnings in documentation

## [0.1.0] - 2025-06-21

### Added
- CR3BP system modeling (CR3BP_System, CR3BP_Dynamics)
- Orbit data structure with state vectors and period tracking
- Differential correction for 2D/3D symmetric orbits
- Natural and pseudo-arc length continuation
- Stability analysis with Floquet multipliers
- Transfer trajectory design with grid search and NLP optimization
- 2D/3D orbit visualization
