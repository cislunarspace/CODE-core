转移轨道设计概述
================

e2m2e 实现了基于 Cui et al. (2025) 的"搜索-优化"两步法，用于 DRO（Distant Retrograde Orbit）到 RO（Regular Orbit）的转移轨道设计。

设计方法
--------

两步法流程：

1. **搜索阶段**：在参数空间中搜索可行的转移窗口
2. **优化阶段**：对搜索结果进行 NLP 精化优化

**转移类型：**

- ``DIRECT``: 直接转移
- ``LGA``: 月球引力辅助转移
- ``EXTERNAL``: 外部转移

设计变量
--------

搜索和优化的核心变量：

- **α (alpha)**: 切向速度比，控制出发速度方向
- **T**: 转移时间
- **t_ins**: 从轨道远地点到插入点的时间

目标函数
--------

最小化总脉冲：

.. math::

   J = \Delta v_1 + \Delta v_2

其中 :math:`\Delta v_1` 为出发脉冲，:math:`\Delta v_2` 为插入脉冲。

约束条件
--------

- 位置连续性：转移轨道与目标轨道在插入点位置匹配
- 速度平行性：转移轨道速度方向与目标轨道速度方向平行
- 碰撞避免：避免与地球和月球碰撞

使用示例
--------

.. code-block:: python

   from e2m2e.transfer.transfer import TransferConfig, TransferOptimizer
   from e2m2e.core.system import CR3BP_System
   from e2m2e.core.dynamics import CR3BP_Dynamics

   system = CR3BP_System.from_known_system("earth_moon")
   dynamics = CR3BP_Dynamics(system)

   # 配置转移参数
   config = TransferConfig(
       alpha_min=0.5,
       alpha_max=2.5,
       use_copt=False,  # 使用 SciPy
       fallback_to_scipy=True
   )

   # 执行优化
   optimizer = TransferOptimizer(system, dynamics, config)
   result = optimizer.optimize(departure_orbit, arrival_orbit)

   if result.success:
       print(f"总脉冲: {result.total_delta_v}")
       print(f"转移时间: {result.transfer_time}")
