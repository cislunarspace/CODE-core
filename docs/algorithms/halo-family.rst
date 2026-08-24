Halo 轨道族编排
===============

Halo 轨道族编排模块（``e2m2e.algorithm.family.halo_family``）从 ``continuation.py`` 拆出
Halo 专用逻辑：种子生成、自然参数族延拓、伪弧长延拓。

种子生成
--------

``generate_halo_seed_orbit()`` 从延拓器实例出发，生成一条 Halo 种子轨道：

.. code-block:: python

   from e2m2e.algorithm.family.halo_family import generate_halo_seed_orbit

   seed = generate_halo_seed_orbit(
       continuation=continuation,
       libration_point=1,
       amplitude_z=0.001,   # 种子宜取小振幅（Richardson 近似精度高），由延拓放大
       halo_class=0,
   )

参数说明：

- ``continuation``：``Continuation`` 实例
- ``libration_point``：平动点编号（1 或 2）
- ``amplitude_z``：z 方向振幅
- ``halo_class``：Halo 族分支（0=北族，1=南族）

轨道族生成
----------

``generate_halo_family()`` 从种子轨道出发，沿 z 振幅方向自然延拓生成轨道族：

.. code-block:: python

   family = continuation.generate_halo_family(
       seed_orbit=seed,
       n_orbits=50,
       z_range=(0.001, 0.15),   # z 振幅范围
   )

   print(f"生成 {len(family)} 条 Halo 轨道")

内部流程：以上一条收敛轨道为初值，固定目标 z0 逐点微分修正，
逐步推进到 ``z_range`` 边界；修正失败时缩减步长重试，触底即终止。

伪弧长延拓
----------

对于参数-振幅关系非单调的 Halo 族（如振幅先增后减），使用伪弧长延拓：

.. code-block:: python

   from e2m2e.algorithm.family.halo_family import halo_pseudo_arclength_continuation

   family = halo_pseudo_arclength_continuation(
       continuation=continuation,
       seed_orbit=seed,
       n_orbits=50,               # 每支生成的新轨道条数
       direction="both",          # 双侧延拓
       step_size=0.0045,          # 伪弧长步长 ΔS
   )

返回 ``OrbitFamily`` 对象，包含种子轨道与各延拓支的新轨道。
