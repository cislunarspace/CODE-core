轨道保持示例（main_control.py）
==============================================

设计 Halo 标称轨道后做轨道保持蒙特卡洛仿真，绘制标称与受控轨迹对比
（会合系 x-z 投影，含五个平动点标注）。样本量刻意压小（2 个控制周期、
1 个蒙特卡洛样本），是快速演示版；工程评估把样本提到惯例值 100。

前置：SPICE 内核在仓库根 ``kernels/``。

.. code-block:: bash

   uv run --no-sync python examples/main_control.py --save

.. literalinclude:: ../../examples/main_control.py
   :language: python

产出与解读
----------

- 图：``examples/main_control_halo.png``——标称轨道与受控轨道的 x-z 投影
  对比，地月天体与 L1–L5 标注同坐标系。
- 结果字段：``ctl.num_failed`` 是蒙特卡洛失败样本数；
  ``ctl.sk_statistic.rows`` 首行前两列是总 Δv 与最大单次 Δv（m/s）；
  ``ctl.controlled_ephemeris.synodic_position`` 是受控星历（全部样本失败
  时为 ``None``），画图前判空。
- ``control_orbit(result.ephemeris, control_mode=1, num_controls=2,
  num_monte_carlo=1, control_interval=10.0, ...)``：控制模式 1 = 目标点
  宽松；控制力模型与真实力模型的摄动开关可分别指定。

对照：接口层等价调用见 :doc:`/tutorials/station-keeping`。
