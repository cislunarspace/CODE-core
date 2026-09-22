轨道预报示例（main_propagate.py）
==============================================

从一条 Halo 轨道的初始状态出发，用高精度力模型外推 60 天并绘制预报轨迹。
这是最重的示例（60 天 × 小步长），适合演示“设计短弧 → 独立力模型外推”的
手动构造链路。

前置：SPICE 内核在仓库根 ``kernels/``。

.. code-block:: bash

   uv run --no-sync python examples/main_propagate.py --save

.. literalinclude:: ../../examples/main_propagate.py
   :language: python

产出与解读
----------

- 图：``examples/main_propagate_60d.png``——预报轨迹 3D（按地月尺度
  384400 km 归一化显示）。
- 脚本演示的低层链路：``SPICEManager`` → ``load_design_kernels`` →
  ``EphemerisSystem(bodies=["EARTH", "MOON", "SUN"])`` + ``CoordinateSystem``
  → ``ForceModel.from_config(result.force_config, system)``——
  ``force_config`` 直接复用设计链路回传的配置；
- ``fm.propagate(initial_state, (et0, et_end), t_eval=et_grid,
  max_steps=2_000_000)`` 返回 ``out["states"]``：逐输出时刻的 (n, 6) 状态，
  ``fm.max_step = 600.0`` 限制最大积分步长。

对照：接口层等价调用见 :doc:`/tutorials/orbit-propagation`。
