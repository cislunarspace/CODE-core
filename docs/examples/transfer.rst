转移轨道示例（main_transfer.py）
==============================================

用二体 Lambert 求解器（Izzo 算法，``e2m2e.algorithm.transfer.solve_lambert``）
解一条地月转移：LEO（半径 6578 km）→ 月球距离（384400 km），5 天转移弧。

前置：**无 SPICE 依赖**，秒级出结果——适合作为环境冒烟测试。

.. code-block:: bash

   uv run --no-sync python examples/main_transfer.py --save

.. literalinclude:: ../../examples/main_transfer.py
   :language: python

产出与解读
----------

- 图：``examples/main_transfer_lambert.png``——转移弧 3D（轨迹为演示用
  线性插值，仅示意几何，非精确自由飞行弧）。
- 结果字段：``sol.v0``／``sol.vf`` 是出发／到达速度（km/s），
  图标题里的 Δv 取 ``‖v0‖``；``direction="short"`` 选短弧解，``revs=0``
  为零圈解。
- ``mu_earth = 398600.4415`` km³/s²（地心引力常数）为脚本内字面值。

对照：完整转移设计（HMN/LGA/WSB/低推力）见
:doc:`/tutorials/transfer-design`。
