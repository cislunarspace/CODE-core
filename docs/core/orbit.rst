轨道
====

e2m2e 的轨道数据结构。

Orbit 类
--------

:class:`~e2m2e.core.orbit.Orbit` 是轨道数据容器，存储状态序列与时间序列。

**核心属性：**

- ``states`` — 状态序列 ``[x, y, z, vx, vy, vz]``，形状 ``(n_points, 6)``
- ``times`` — 时间序列，形状 ``(n_points,)``
- ``system`` — 关联的系统对象（``CR3BP_System`` 或 ``EphemerisSystem``）
- ``family_type`` / ``parameters`` — 轨道族类型与连续参数（由延拓等外部算法填充）
- ``metadata`` — 元数据字典（创建时间、来源、描述、标签）

.. code-block:: python

   from e2m2e.core import Orbit
   import numpy as np

   # 从状态序列创建
   orbit = Orbit(
       states=np.array([[0.8, 0, 0, 0, 0.6, 0]]),
       times=np.array([0.0]),
       system=system,
   )

**基本属性：**

``Orbit.__init__`` 末尾调用 ``compute_basic_properties()``，构造时即自动估计
``period``（x 方向零交叉检测），并计算 ``amplitudes``、``extrema``、
``mean_state``、``center``、``is_periodic``、``periodicity_error``——这些字段
在 ``__init__`` 中显式声明，经 property 代理访问。微分修正的结果写入预声明的
``correction_*`` 字段（默认 ``None``）。Jacobi 常数、稳定性仍由外部算法按需计算。

**序列化：**

.. code-block:: python

   # 保存到 JSON
   orbit.save_to_file("my_orbit.json")

   # 从 JSON 加载
   orbit2 = Orbit.load_from_file("my_orbit.json", system=system)

轨道族 (OrbitFamily)
---------------------

同类型、由连续参数（如 Jacobi 常数、振幅）索引的一组 ``Orbit`` 集合。
轨道族是延拓的结果，不是生成它的方法。

.. code-block:: python

   # 延拓返回轨道族
   from e2m2e.algorithms import Continuation

   continuation = Continuation(corrector=corrector)
   family = continuation.natural_continuation(
       seed_orbit=seed_dro,
       param_range=(0.14, 0.9),
       step_size=0.005,
   )

   # 遍历族内轨道
   for orbit in family:
       print(f"周期: {orbit.period:.6f}")

CR3BP 周期轨道族类型
--------------------

.. list-table::
   :header-rows: 1

   * - 族名
     - 相关平动点
     - 物理特征
   * - Lyapunov
     - L1, L2, L3
     - 平面周期轨道
   * - Halo
     - L1, L2
     - 三维周期轨道
   * - Vertical
     - L1–L5
     - 垂直方向振荡
   * - Butterfly
     - L1, L2
     - 连接两个共线平动点的对称轨道
   * - Dragonfly
     - L1, L2
     - 连接两个共线平动点的非对称轨道
   * - DRO
     - secondary
     - 远程逆行轨道
   * - RO
     - 全系统
     - 满足 m:n 共振比例的周期轨道
