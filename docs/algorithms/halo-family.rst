Halo 轨道族编排
===============

Halo 轨道族编排模块（``e2m2e.algorithms.halo_family``）从 ``continuation.py`` 拆出
Halo 专用逻辑：种子生成、自然参数族延拓、伪弧长延拓。

种子生成
--------

``generate_halo_seed_orbit()`` 从延拓器实例出发，生成一条 Halo 种子轨道：

.. code-block:: python

   from e2m2e.algorithms.halo_family import generate_halo_seed_orbit

   seed = generate_halo_seed_orbit(
       continuation=continuation,
       libration_point=1,
       halo_class=0,
       z_amplitude=0.01,
   )

参数说明：

- ``continuation`` — ``Continuation`` 实例
- ``libration_point`` — 平动点编号（1 或 2）
- ``halo_class`` — Halo 族分支（0=南族，1=北族）
- ``z_amplitude`` — z 方向振幅

轨道族生成
----------

``generate_halo_family()`` 是一站式入口，从种子到完整轨道族：

.. code-block:: python

   from e2m2e.algorithms.halo_family import generate_halo_family

   family = generate_halo_family(
       system=system,
       dynamics=dynamics,
       libration_point=1,
       amplitude_z_range=(0.001, 0.15),
       n_orbits=50,
   )

   print(f"生成 {len(family)} 条 Halo 轨道")

内部流程：

1. 用 Richardson 三阶近似生成初始猜测
2. 微分修正得到种子轨道
3. 自然延拓生成轨道族

伪弧长延拓
----------

对于参数-振幅关系非单调的 Halo 族（如振幅先增后减），使用伪弧长延拓：

.. code-block:: python

   from e2m2e.algorithms.halo_family import generate_halo_family_pal

   family = generate_halo_family_pal(
       system=system,
       dynamics=dynamics,
       libration_point=1,
       amplitude_z_range=(0.001, 0.15),
       n_orbits=50,
       step_size=0.002,
   )
