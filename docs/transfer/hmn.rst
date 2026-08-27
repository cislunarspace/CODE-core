HMN Hohmann Transfer / HMN 霍曼直接转移
========================================

[English](#hmn-hohmann-transfer) | [简体中文](#中文)

English
-------

The HMN module implements the classic two-impulse Hohmann transfer — minimum-
energy transfer between coplanar circular orbits: first impulse accelerates onto
the elliptical transfer orbit; second completes insertion.

Theory
~~~~~~

For coplanar circular-orbit transfers, two impulses minimize fuel:

.. math::

   \Delta v = \Delta v_1 + \Delta v_2

.. math::

   \Delta v_1 = \sqrt{\frac{\mu}{r_1}} \left(\sqrt{\frac{2r_2}{r_1+r_2}} - 1\right)

   \Delta v_2 = \sqrt{\frac{\mu}{r_2}} \left(1 - \sqrt{\frac{2r_1}{r_1+r_2}}\right)

Usage
~~~~~

.. code-block:: python

   from e2m2e.algorithm.transfer.hohmann import hohmann_delta_v, hohmann_tof

   r_departure = 6778.0    # departure radius (km, e.g. LEO)
   r_arrival = 42164.0     # target radius (km, e.g. GEO)

   dv1, dv2 = hohmann_delta_v(r_departure, r_arrival)
   tof = hohmann_tof(r_departure, r_arrival)

   print(f"Δv1: {dv1:.4f} km/s")
   print(f"Δv2: {dv2:.4f} km/s")
   print(f"Total Δv: {dv1 + dv2:.4f} km/s")
   print(f"Time of flight: {tof / 3600:.2f} hours")

Use cases
~~~~~~~~~

- LEO → GEO transfers
- Minimum-energy transfers between any coplanar circular orbits
- Initial guess or baseline for complex transfer designs

中文
----

HMN（Hohmann Transfer）模块实现经典的霍曼双脉冲转移设计，
适用于共面圆轨道之间的最小能量转移。

霍曼转移是航天动力学中最基本的转移轨道，由两次推力脉冲组成：
第一次在出发轨道上加速进入椭圆转移轨道，第二次在目标轨道上加速完成插入。

理论与公式见上方英文节；使用方式：

.. code-block:: python

   from e2m2e.algorithm.transfer.hohmann import hohmann_delta_v, hohmann_tof

   r_departure = 6778.0    # 出发轨道半径（km，如 LEO）
   r_arrival = 42164.0     # 目标轨道半径（km，如 GEO）

   dv1, dv2 = hohmann_delta_v(r_departure, r_arrival)
   tof = hohmann_tof(r_departure, r_arrival)

应用场景
~~~~~~~~

- LEO → GEO 转移
- 任意共面圆轨道间的最小能量转移
- 作为复杂转移设计的初猜或基准
