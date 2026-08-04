WSB 太阳引力辅助低能转移
========================

WSB（Weak Stability Boundary）模块实现太阳引力辅助下的低能间接月球转移，
由 :func:`~e2m2e.algorithm.transfer.transfer_orbit` 编排器以
``"WSB"`` 路由，内部经弹道网格搜索 + ThreeBodyLambert 到达段精化闭环。

基本原理
--------

WSB 转移利用日地系统中的弱稳定边界区域：航天器先飞离月球影响球、进入
太阳摄动下的远端日心弧，在远地点附近太阳引力做功使其相对月球的速度
降到捕获阈值之下，从而以接近零的 Δv 被月球弹道捕获（Belbruno & Miller 1993）。

捕获判据采用 Belbruno 约定的二体 Kepler 能量 :math:`H_2`：

.. math::

   H_2 = \frac{|\mathbf{v}|^2}{2} - \frac{\mu_{\text{moon}}}{r}

近月点处 :math:`H_2 < 0` 表示航天器处于月球束缚态（弹道捕获）。

求解流程（两步）：

1. **弹道网格搜索**：在太阳相位角 × 出发相位角 × 飞行时间三维网格上，
   BCR4BP（含太阳摄动的双圆四体模型）前向传播，筛选近月点高度与
   :math:`H_2` 满足捕获条件的候选（``ProcessPoolExecutor`` 并行）
2. **到达段精化**：对最优候选用 ThreeBodyLambert 打靶，精化月心到达段

使用方法
--------

.. code-block:: python

   from e2m2e.algorithm.transfer import transfer_orbit
   from e2m2e.algorithm.transfer.wsb import WsbSearchParams

   result = transfer_orbit(
       "WSB",
       wsb_search_params=WsbSearchParams(
           tof_range=(90.0, 150.0),   # 飞行时间范围（天）
           max_total_dv=4.0,          # 总 Δv 上限（km/s）
       ),
       departure_state=departure_state,
       target_state=target_state,
   )

   print(f"总 Δv: {result.total_dv:.4f} km/s")
   print(f"飞行时间: {result.time_of_flight:.2f} 天")

应用场景
--------

- 低能地月转移：以极低 Δv 实现月球捕获，适用于燃料受限任务
- 月球着陆器、轨道插入器的经济转移方案
- 作为应急轨道救援的低燃料备份通道
