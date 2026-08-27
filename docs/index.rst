.. e2m2e documentation master file

e2m2e: Cislunar Mission Planning Algorithm Toolset / 地月空间任务规划算法工具集
===============================================================================

**English**

e2m2e (Earth to Moon, Moon to Earth) targets cislunar mission planning with
precise and reliable orbit computation tools: building dynamical models of
cislunar space, generating periodic orbit families, designing transfer paths
between orbits, and visualizing results for inspection. In an LLM+Agent-style
autonomous mission planning system, the large model understands intent and
orchestrates subtasks; e2m2e handles the numerical half.

**简体中文**

e2m2e (Earth to Moon, Moon to Earth) 面向地月空间任务规划，提供精确可靠的
轨道计算工具：建立地月空间动力学模型，生成周期轨道族，设计轨道之间的转移
路径，并把结果画出来检查。在 LLM+Agent 式自主任务规划系统中，大模型负责
理解任务意图、分解与编排子任务，e2m2e 负责数值计算那一半。

.. toctree::
   :maxdepth: 2
   :caption: Quick Start / 快速开始

   getting-started/installation
   getting-started/quickstart
   getting-started/mcp

.. toctree::
   :maxdepth: 2
   :caption: Core Concepts / 核心概念

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
   :caption: Periodic Orbit Design / 周期轨道设计

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
   :caption: Transfer Design / 转移轨道设计

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
   :caption: Architecture / 架构

   architecture/index
   architecture/architecture
   architecture/system-dynamics-dataflow
   architecture/numerics-migration-status
   architecture/hjb-subsystem
   architecture/hjb-hamiltonian-dataflow

.. toctree::
   :maxdepth: 2
   :caption: API Reference / API 参考

   api/e2m2e

.. toctree::
   :maxdepth: 2
   :caption: Reference / 参考资料

   reference/mbse/index
   reference/glossary


Indices and tables / 索引与表格
===============================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
