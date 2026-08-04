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

   from e2m2e.algorithm.transfer.lga import search_lga_trajectories, LgaSearchParams

   # departure_state / target_state：CR3BP 无量纲状态；system / dynamics：CR3BP 系统
   candidates = search_lga_trajectories(
       departure_state=departure_state,
       target_state=target_state,
       system=system,
       dynamics=dynamics,
       params=LgaSearchParams(
           tof_range=(15.0, 45.0),   # 飞行时间范围（天）
           max_total_dv=4.0,         # 总 Δv 上限（km/s）
       ),
   )

   for c in candidates:
       print(f"总 Δv: {c.total_dv:.4f} km/s, 飞行时间: {c.tof_sec / 86400:.2f} 天, "
             f"近月点高度: {c.perilune_alt_km:.1f} km")

参考文献
--------

- Cui, H. et al. (2025). Transfer orbit design for cislunar space missions.
