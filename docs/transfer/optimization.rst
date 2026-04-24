转移轨道优化
============

转移轨道的优化方法。

优化目标
--------

- 最小化燃料消耗 (Δv)
- 最小化转移时间
- 多目标优化

优化算法
--------

.. code-block:: python

   from e2m2e.transfer.optimization import optimize_transfer

   # 优化转移轨道
   result = optimize_transfer(
       transfer_orbit,
       system,
       objective="min_dv"
   )
