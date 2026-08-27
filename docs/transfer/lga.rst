LGA Lunar Gravity Assist / LGA 月球引力辅助转移
================================================

[English](#lga-lunar-gravity-assist) | [简体中文](#中文)

English
-------

The LGA module designs lunar-gravity-assist indirect transfers via conic
patching: lunar flybys rotate the spacecraft's velocity vector in magnitude and
direction, buying orbital changes cheaply.

Design method
~~~~~~~~~~~~~

Three legs:

1. **Departure**: hyperbolic escape from the departure orbit
2. **Flyby**: velocity-vector change inside the lunar sphere of influence
3. **Arrival**: hyperbolic insertion onto the target orbit

Usage
~~~~~

.. code-block:: python

   from e2m2e.algorithm.transfer.lga import search_lga_trajectories, LgaSearchParams
   from e2m2e.data.templates import ConvergenceState

   # departure_state / target_state: CR3BP nondimensional states; system/dynamics: CR3BP system
   candidates = search_lga_trajectories(
       departure_state=departure_state,
       target_state=target_state,
       system=system,
       dynamics=dynamics,
       params=LgaSearchParams(
           tof_range=(15.0, 45.0),   # flight-time range (days)
           max_total_dv=4.0,         # total Δv cap (km/s)
       ),
   )

   if candidates.status is not ConvergenceState.CONVERGED:
       print(f"Search incomplete: {candidates.status.value}, {candidates.cause.value}: {candidates.message}")

   for c in candidates:
       dv_km_s = c.total_dv * system.characteristic_velocity  # nondimensional → km/s
       print(f"Total Δv: {dv_km_s:.4f} km/s, TOF: {c.tof_sec / 86400:.2f} d, "
             f"perilune altitude: {c.perilune_alt_km:.1f} km")

References
~~~~~~~~~~

- Cui, H. et al. (2025). Transfer orbit design for cislunar space missions.

中文
----

LGA（Lunar Gravity Assist）模块实现月球引力辅助间接转移设计，
基于圆锥曲线拼接法（conic patching）计算经过月球引力辅助的转移轨道。

月球引力辅助是地月转移中的重要技术，利用月球引力改变航天器的速度大小和方向，
从而以较小的燃料消耗实现轨道转移。

设计方法：三段式——出发段（双曲线脱离）、引力辅助段（月球影响球内改速度矢量）、
到达段（双曲线插入）。使用方式与返回字段见上方英文节代码示例。

参考文献
~~~~~~~~

- Cui, H. et al. (2025). Transfer orbit design for cislunar space missions.
