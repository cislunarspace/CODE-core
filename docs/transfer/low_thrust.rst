小推力转移
==========

基于 7D 增广状态 ``[r, v, m]`` 的小推力最优控制转移设计。通过
:func:`~e2m2e.algorithm.transfer.transfer_orbit` 编排器路由，内部调用
:class:`~e2m2e.algorithm.transfer.lowthrust_shooting.LowThrustShooting` 或
:class:`~e2m2e.algorithm.transfer.lowthrust_collocation.LowThrustCollocation`
完成 Q-law 初猜 + SLSQP 打磨闭环。

基本原理
--------

小推力转移在 7D 增广状态空间中建模：

.. math::

   \dot{\mathbf{r}} = \mathbf{v}, \quad
   \dot{\mathbf{v}} = \mathbf{a}_{\text{grav}} + \frac{T \cdot \delta}{m} \hat{\mathbf{u}}, \quad
   \dot{m} = -\frac{T \cdot \delta}{I_{sp} \, g_0}

其中：

- :math:`T` 为最大推力（N），:math:`\delta \in [0, 1]` 为节流系数
- :math:`\hat{\mathbf{u}}` 为推力方向，用球面角度参数化 :math:`\alpha(\theta_1, \theta_2)`
- :math:`I_{sp}` 为比冲（s），:math:`g_0 = 9.81 \, \text{m/s}^2`

求解流程（两级）：

1. **Q-law 初猜**：Lyapunov 反馈律前向积分，生成次优控制历史
2. **SLSQP 打磨**：min-fuel NLP，决策变量为各段常量控制 ``(throttle, θ₁, θ₂)``，
   约束为终端位置速度匹配目标

使用方法
--------

.. code-block:: python

   from e2m2e.algorithm.transfer import transfer_orbit, EngineConfig
   from e2m2e.data.templates import ConvergenceState

   result = transfer_orbit(
       "low_thrust",
       engine_config=EngineConfig(t_max=0.5, isp=3000.0),
       initial_mass=1000.0,
       n_segments=10,
       target_oe=(7200.0, 0.0, 0.0),
       solver_method="shooting",
       duration_days=30.0,
       departure_state=departure_6d,
       target_state=target_6d,
   )

   # 结果访问
   print(f"等效 Δv = {result.details.equivalent_delta_v:.4f} km/s")
   print(f"燃料消耗 = {result.details.fuel_consumed:.2f} kg")
   print(f"收敛 = {result.details.status == ConvergenceState.CONVERGED}")

配点法求解器（大规模更鲁棒）：

.. code-block:: python

   result = transfer_orbit(
       "low_thrust",
       engine_config=EngineConfig(t_max=0.5, isp=3000.0),
       initial_mass=1000.0,
       n_segments=20,
       target_oe=(42164.0, 0.0, 0.0),
       solver_method="collocation",
       duration_days=200.0,
       departure_state=departure_6d,
       target_state=target_6d,
   )

与脉冲转移的 Δv 对比
---------------------

脉冲转移和小推力转移的 Δv 口径不同：

- **脉冲 Δv**：:math:`\Delta v = \sum |\Delta v_i|`，瞬时速度增量之和
- **小推力等效 Δv**：:math:`\Delta v_{\text{eq}} = I_{sp} \cdot g_0 \cdot \ln(m_0 / m_f) / 1000`，
  Tsiolkovsky 方程反算（km/s）

小推力因持续推力（gravity loss），等效 Δv 通常 ≥ 脉冲 Δv。在短弧小轨道改变
场景下差异较小；在大能量转移（如 LEO→GEO）中差异显著。

.. code-block:: python

   from e2m2e.algorithm.transfer import transfer_orbit, EngineConfig
   from e2m2e.algorithm.transfer.hohmann import hohmann_delta_v

   # 脉冲 Δv
   dv1, dv2 = hohmann_delta_v(7000.0, 42164.0)
   dv_impulsive = dv1 + dv2

   # 小推力等效 Δv（从 result.details.equivalent_delta_v 获取）

推进参数说明
------------

:class:`~e2m2e.algorithm.transfer.lowthrust_shooting.EngineConfig` 字段：

- ``t_max``: 最大推力（N）。典型值域：

  - 电推进（Hall thruster）: 0.01~1.0 N
  - 电推进（ion thruster）: 0.0001~0.1 N
  - 化学低推力: 1~100 N

- ``isp``: 比冲（s）。典型值域：

  - 电推进（Hall thruster）: 1500~3000 s
  - 电推进（ion thruster）: 2000~5000 s
  - 化学推进: 300~450 s

其他关键参数：

- ``initial_mass``: 航天器初始质量（kg）
- ``n_segments``: 求解器段数。越多精度越高但计算越慢。典型值 5~50
- ``solver_method``: ``"shooting"`` （直接打靶，解析雅可比快 5-24x）或
  ``"collocation"`` （Hermite-Simpson 配点，大规模更鲁棒）
- ``duration_days``: 飞行时间（天）。LEO→GEO 约 100~300 天；LEO→月球约 3~180 天
- ``target_oe``: Q-law 目标 ``(a_T, e_T, i_T)`` （km, 无量纲, 无量纲）

质量演化与推力历史
------------------

求解结果包含完整的 7D 状态历史和各段控制参数：

.. code-block:: python

   import numpy as np

   # 质量剖面
   masses = result.details.states_7d[:, 6]
   print(f"初始质量: {masses[0]:.2f} kg")
   print(f"末态质量: {masses[-1]:.2f} kg")

   # 各段控制
   for i, seg in enumerate(result.details.segments):
       print(f"段 {i}: throttle={seg.throttle:.3f}, "
             f"方向={seg.direction}")
