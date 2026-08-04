HMN 霍曼直接转移
================

HMN（Hohmann Transfer）模块实现经典的霍曼双脉冲转移设计，
适用于共面圆轨道之间的最小能量转移。

霍曼转移是航天动力学中最基本的转移轨道，由两次推力脉冲组成：
第一次在出发轨道上加速进入椭圆转移轨道，第二次在目标轨道上加速完成插入。

理论背景
--------

对于共面圆轨道间的转移，霍曼转移（两次脉冲）是最省燃料的方案。
总速度增量为：

.. math::

   \Delta v = \Delta v_1 + \Delta v_2

其中：

.. math::

   \Delta v_1 = \sqrt{\frac{\mu}{r_1}} \left(\sqrt{\frac{2r_2}{r_1+r_2}} - 1\right)

   \Delta v_2 = \sqrt{\frac{\mu}{r_2}} \left(1 - \sqrt{\frac{2r_1}{r_1+r_2}}\right)

使用方式
--------

.. code-block:: python

   from e2m2e.transfer.hmn import HohmannTransfer

   transfer = HohmannTransfer(
       r_departure=6778.0,   # 出发轨道半径（km，如 LEO）
       r_arrival=42164.0,    # 目标轨道半径（km，如 GEO）
       mu=398600.4418,       # 中心天体引力常数（km³/s²）
   )

   result = transfer.solve()

   print(f"Δv1: {result.delta_v1:.4f} km/s")
   print(f"Δv2: {result.delta_v2:.4f} km/s")
   print(f"总 Δv: {result.total_delta_v:.4f} km/s")
   print(f"转移时间: {result.transfer_time / 3600:.2f} 小时")

应用场景
--------

- LEO → GEO 转移
- 任意共面圆轨道间的最小能量转移
- 作为复杂转移设计的初猜或基准
