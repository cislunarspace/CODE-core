LGA 月球引力辅助转移
====================

LGA（Lunar Gravity Assist）模块实现月球引力辅助间接转移设计，
基于圆锥曲线拼接法（conic patching）计算经过月球引力辅助的转移轨道。

月球引力辅助是地月转移中的重要技术，利用月球引力改变航天器的速度大小和方向，
从而以较小的燃料消耗实现轨道转移。

设计方法
--------

LGA 转移分为三段：

1. **出发段**：从出发轨道出发的双曲线脱离段
2. **引力辅助段**：在月球影响球内利用引力改变速度矢量
3. **到达段**：到达目标轨道的双曲线插入段

使用方式
--------

.. code-block:: python

   from e2m2e.transfer.lga import LGATransfer

   transfer = LGATransfer(
       departure_orbit=departure_orbit,
       arrival_orbit=arrival_orbit,
       system=system,
   )

   result = transfer.solve(
       departure_epoch=epoch,
       tof_guess=5.0,  # 飞行时间初猜（天）
   )

   if result.success:
       print(f"总 Δv: {result.total_delta_v:.4f} km/s")
       print(f"飞行时间: {result.time_of_flight:.2f} 天")

参考文献
--------

- Cui, H. et al. (2025). Transfer orbit design for cislunar space missions.
