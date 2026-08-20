DPO 顺行轨道族
==============

DPO（Distant Prograde Orbit）是月球附近的顺行远距离轨道族，位于月球 L2 平动点附近。
与 DRO（逆行）相对，DPO 轨道在旋转系中沿月球公转方向运动。

设计方法
--------

DPO 轨道族通过 CR3BP 框架设计：

1. **种子生成**：从 L2 平动点附近的小振幅初猜出发
2. **微分修正**：使用对称策略修正周期轨道
3. **自然延拓**：沿振幅方向延拓生成轨道族

使用方式
--------

通过 ``Facade`` 一档接口设计：

.. code-block:: python

   from e2m2e.api import Facade

   facade = Facade()
   result = facade.design_orbit(
       orbit_type="DPO",
       collinear_point=2,
       amplitude=15000.0,  # km
       epoch=[2024, 1, 1, 0, 0, 0.0],
       duration=1.0,  # 年
   )

或使用底层 API：

.. code-block:: python

   from e2m2e.algorithm.design import design_orbit

   result = design_orbit(
       orbit_type="DPO",
       collinear_point=2,
       amplitude=15000.0,
       epoch=[2024, 1, 1, 0, 0, 0.0],
   )

星历修正
--------

DPO 属于不稳定轨道族，不能采用 DRO 的单圈修正后自由外推路径。
``DesignOrbitRequest`` 校验时按族分派 ``correction_method``：DPO 未显式
指定时默认即为 ``segmented``（全程分段打靶）；显式传入 ``two_level`` /
``standard`` 等冲突值时告警并改写为 ``segmented``，设计结果与响应的
``correction_method`` 字段记录实际执行的方法。圈间的准周期漂移仍由
``station_keeping`` 处理。

默认振幅 20000 km 的 DPO 周期约 23 天。为把不稳定方向的误差限制在每段
打靶的收敛范围内，生产采样使用每圈 64 个等时间 patch 点，首层最多将两圈
放入同一段，并固定节点时刻。该配置已覆盖约 30 天、1 小时输出步长的 GUI
默认量级；更长弧段的可维持性不由本设计步骤承诺。

特性
----

- 顺行轨道（prograde），与月球公转方向一致
- 位于 L2 平动点附近
- 可用于月球背面通信中继轨道设计
