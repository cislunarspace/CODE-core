延拓法
======

延拓法用于生成周期轨道族。

基本原理
--------

从一个已知解出发，逐步改变参数，追踪解的变化轨迹。

伪弧长延拓
----------

.. code-block:: python

   from e2m2e.algorithms.continuation import pseudo_arc_length

   # 生成 halo 轨道族
   family = pseudo_arc_length(
       initial_orbit,
       parameter="amplitude_z",
       step_size=0.001,
       num_steps=100
   )
