轨道
====

e2m2e 的轨道数据结构与操作。

Orbit 类
--------

:class:`~e2m2e.core.orbit.Orbit` 是最基本的轨道数据容器，用于存储、计算和持久化一条 CR3BP 轨道的全部信息。

**核心属性：**

- ``states``: 状态序列 ``[x, y, z, vx, vy, vz]``，形状 ``(n, 6)``
- ``times``: 时间序列，形状 ``(n,)``
- ``system``: 关联的 CR3BP_System 对象

**计算属性（property）：**

- ``period``: 轨道周期
- ``amplitudes``: x/y/z 方向振幅
- ``extrema``: 各分量极值
- ``center``: 轨道中心
- ``monodromy_matrix``: 单周期转移矩阵
- ``eigenvalues``: 特征值
- ``stability_index``: 稳定性指标

.. code-block:: python

   from e2m2e.core.orbit import Orbit
   from e2m2e.core.system import CR3BP_System
   import numpy as np

   system = CR3BP_System.from_known_system("earth_moon")

   # 从状态序列创建轨道
   states = np.array([...])  # (n, 6)
   times = np.linspace(0, 2 * np.pi, 100)
   orbit = Orbit(states, times, system)

   # 访问计算属性
   print(f"周期: {orbit.period}")
   print(f"x 振幅: {orbit.amplitudes['x']}")

轨道族类型
----------

支持的轨道族：

- ``halo``: Halo 轨道
- ``lyapunov``: Lyapunov 轨道
- ``vertical``: 垂直轨道
- ``axial``: 轴向轨道
- ``butterfly``: 蝴蝶轨道
- ``dragonfly``: 蜻蜓轨道

.. code-block:: python

   # 设置轨道族类型
   orbit.family_type = "halo"
   orbit.parameters = {"L": 1, "amplitude_z": 0.01}

序列化
-------

Orbit 支持 JSON 序列化/反序列化：

.. code-block:: python

   # 保存轨道
   orbit.save("my_orbit.json")

   # 加载轨道
   orbit2 = Orbit.load("my_orbit.json")

OrbitFamily 类
--------------

:class:`~e2m2e.core.orbit.OrbitFamily` 是轨道族容器，用于存储和管理多条同族轨道。

.. code-block:: python

   from e2m2e.core.orbit import OrbitFamily

   family = OrbitFamily("halo_L1", system)
   family.append(orbit1)
   family.append(orbit2)

   # 访问族内所有轨道
   for orbit in family:
       print(orbit.period)
