轨道设计示例（main_design.py）
==============================

端到端设计一条地月 L2 Halo 轨道：CR3BP 初猜 → 星历修正 → 加摄动高精度预报，
在会合系绘制 3D 拟周期轨迹（30 天）。

前置：SPICE 内核在仓库根 ``kernels/``（或设 ``$SPICE_KERNEL_DIR``）。

.. code-block:: bash

   uv run --no-sync python examples/main_design.py --save

.. literalinclude:: ../../examples/main_design.py
   :language: python

产出与解读
----------

- 图：``examples/main_design_halo.png``——会合系 3D 轨迹，视口对准 L2 区，
  含地月天体与 L2 平动点标注（会合系坐标从地心归一平移到质心归一，
  ``states[:, 0] -= system.mu``）。
- 控制台结果字段：``result.cr3bp_jacobi`` 是 CR3BP Jacobi 常数；
  ``result.correction.status is ConvergenceState.CONVERGED`` 与
  ``result.correction.iterations`` 报告星历修正收敛情况；
  ``result.ephemeris.synodic_position`` 是加摄动后的会合系星历
  （画图数据源）；``result.cr3bp_orbit.period`` 是 CR3BP 参考周期
  （无量纲）。
- 摄动开关字典（太阳第三体 + 地月非球形 10 阶 + 光压）即设计链路的
  力模型配置， ``--save`` 时打印每项开关。

对照：接口层等价调用见 :doc:`/tutorials/orbit-design`。
