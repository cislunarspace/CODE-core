.. e2m2e documentation master file

e2m2e: 地月空间任务规划算法工具集
===================================

e2m2e (Earth to Moon, Moon to Earth) 面向地月空间任务规划，提供精确可靠的
轨道计算工具：建立地月空间动力学模型，生成周期轨道族，设计轨道之间的转移
路径，并把结果画出来检查。在"LLM+Agent"式自主任务规划系统中，大模型负责
理解任务意图、分解与编排子任务，e2m2e 负责数值计算那一半。

.. toctree::
   :maxdepth: 2
   :caption: 快速开始

   getting-started/installation
   getting-started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: 核心概念

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
   :caption: 周期轨道设计

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
   :caption: 转移轨道设计

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
   :caption: 架构

   architecture/index
   architecture/architecture
   architecture/system-dynamics-dataflow
   architecture/numerics-migration-status
   architecture/hjb-subsystem
   architecture/hjb-hamiltonian-dataflow

.. toctree::
   :maxdepth: 2
   :caption: API 参考

   api/e2m2e

.. toctree::
   :maxdepth: 2
   :caption: 参考资料

   reference/mbse/index
   reference/glossary


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
