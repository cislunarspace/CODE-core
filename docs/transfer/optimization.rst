转移轨道优化
============

转移轨道的 NLP（非线性规划）优化，实现搜索-优化两步法的优化阶段。

优化器
------

:class:`~e2m2e.transfer.transfer_optimization.DROTRONLPOptimizer` 实现 DRO 到 RO 的转移轨道优化。

**优化变量：** y = {α, T, t_ins}

- α: 切向速度比
- T: 转移时间
- t_ins: 插入时间

**目标函数：**

.. math::

   J(y) = \Delta v_1 + \Delta v_2

**约束条件：**

- 位置连续性约束
- 速度平行性约束
- 碰撞避免约束（地球、月球）

求解器选择
----------

支持两种求解器：

1. **SciPy** (默认): 使用 ``scipy.optimize.minimize``
2. **COPT**: 商业优化器，性能更优 (需单独安装)

.. code-block:: python

   from e2m2e.transfer.transfer import TransferConfig

   # 使用 SciPy
   config = TransferConfig(use_copt=False)

   # 使用 COPT（需安装 coptpy）
   config = TransferConfig(use_copt=True, fallback_to_scipy=True)

配置参数
--------

:class:`~e2m2e.transfer.transfer.TransferConfig` 控制优化行为：

.. code-block:: python

   config = TransferConfig(
       alpha_min=0.5,           # α 下界
       alpha_max=2.5,           # α 上界
       earth_radius=200.0/DU,   # 地球碰撞半径（无量纲）
       moon_radius=100.0/DU,    # 月球碰撞半径（无量纲）
       use_relaxed_velocity=True,  # 松弛速度约束
       velocity_angle_tol=0.05,    # 速度角度容差（弧度）
       use_copt=False,
       fallback_to_scipy=True
   )

结果分析
--------

:class:`~e2m2e.transfer.transfer.TransferOptimizationResult` 包含优化结果：

.. code-block:: python

   result = optimizer.optimize(departure_orbit, arrival_orbit)

   if result.success:
       print(f"总脉冲: {result.total_delta_v:.6f}")
       print(f"出发脉冲: {result.delta_v1:.6f}")
       print(f"插入脉冲: {result.delta_v2:.6f}")
       print(f"转移时间: {result.transfer_time:.2f}")

       # 转移轨迹
       trajectory = result.transfer_trajectory  # (n_steps, 6)
       times = result.transfer_trajectory_times  # (n_steps,)
