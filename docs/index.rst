.. e2m2e documentation master file

e2m2e: Cislunar Mission Planning Algorithm Toolset
==================================================

e2m2e (Earth to Moon, Moon to Earth) targets cislunar mission planning with
precise and reliable orbit computation tools: building dynamical models of
cislunar space, generating periodic orbit families, designing transfer paths
between orbits, and visualizing results for inspection. In an LLM+Agent-style
autonomous mission planning system, the large model understands intent and
orchestrates subtasks; e2m2e handles the numerical half.

How the codebase is organized: ``e2m2e/api`` is the sole entry (Facade →
CLI/MCP); ``e2m2e/algorithm`` constructs problems with domain knowledge;
``crates/`` holds the Rust numerical core; ``e2m2e/data`` supplies ephemeris
caches, frame data, and constant baselines. The journey of one orbit task
through these layers is told in the README's "How to read this repository".

.. toctree::
   :maxdepth: 2
   :caption: Quick Start

   getting-started/installation
   getting-started/quickstart
   getting-started/mcp

.. toctree::
   :maxdepth: 2
   :caption: Core Concepts

   core/system
   core/dynamics
   core/ephemeris
   core/orbit
   core/coordinate
   core/forces
   core/integrators
   core/atmosphere

.. toctree::
   :maxdepth: 2
   :caption: Periodic Orbit Design

   algorithms/differential-correction
   algorithms/strategies
   algorithms/continuation
   algorithms/orbit-family-generation
   algorithms/halo
   algorithms/halo-family
   algorithms/halo-initial-guess
   algorithms/dpo
   algorithms/axial
   algorithms/lunar-orbits
   algorithms/multiple-shooting
   algorithms/stability
   algorithms/manifolds
   algorithms/normal-form

.. toctree::
   :maxdepth: 2
   :caption: Transfer Design

   transfer/overview
   transfer/lambert
   transfer/hmn
   transfer/lga
   transfer/wsb
   transfer/low_thrust
   transfer/search
   transfer/optimization
   transfer/terminal
   transfer/propulsion

.. toctree::
   :maxdepth: 2
   :caption: Architecture

   architecture/index
   architecture/architecture
   architecture/system-dynamics-dataflow
   architecture/numerics-migration-status
   architecture/hjb-subsystem
   architecture/hjb-hamiltonian-dataflow

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/e2m2e

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/mbse/index
   reference/glossary


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
