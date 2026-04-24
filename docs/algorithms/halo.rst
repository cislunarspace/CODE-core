Halo 轨道
=========

Halo 轨道是围绕拉格朗日点的周期轨道。

特性
----

- 从地球看，轨道呈"光环"状
- 位于 L1 或 L2 点附近
- 垂直振幅可调

计算方法
--------

.. code-block:: python

   from e2m2e.algorithms.halo import compute_halo_orbit

   # L1 halo 轨道
   orbit = compute_halo_orbit(system, "L1", amplitude_z=0.01)

   # L2 halo 轨道
   orbit = compute_halo_orbit(system, "L2", amplitude_z=0.01)
