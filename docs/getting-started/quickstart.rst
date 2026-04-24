快速入门
========

e2m2e 的基本使用方法。

创建 CR3BP 系统
----------------

.. code-block:: python

   from e2m2e.core.system import CR3BP_System

   system = CR3BP_System.from_known_system("earth_moon")
   print(f"质量参数: {system.mu}")

计算 halo 轨道
---------------

.. code-block:: python

   from e2m2e.algorithms.halo import compute_halo_orbit

   # 计算 L1 halo 轨道
   orbit = compute_halo_orbit(system, "L1", amplitude_z=0.01)
