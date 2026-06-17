转移轨道优化
============

转移轨道的 NLP（非线性规划）优化，实现"搜索-优化"两步法的优化阶段。
对搜索阶段得到的可行候选解进行精化，求解满足约束的最小脉冲转移轨道。

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

- 位置连续性约束：转移轨迹末端与目标轨道插入点位置重合
- 速度平行性约束：转移轨迹末端速度与目标轨道插入点速度平行
- 碰撞避免约束：轨迹不与地球或月球相交

求解器选择
----------

支持两种求解器：

1. **SciPy SLSQP** (默认): 使用 ``scipy.optimize.minimize`` 的 SLSQP 方法
2. **COPT**: 杉数科技商业优化器，性能更优（需单独安装 ``coptpy``）

求解器通过 :class:`~e2m2e.transfer.transfer.TransferConfig` 的 ``use_copt`` 字段控制，
底层由 :class:`~e2m2e.transfer.optimizers.SciPyTransferOptimizer` 和
:class:`~e2m2e.transfer.optimizers.COPTTransferOptimizer` 统一适配。

.. code-block:: python

   from e2m2e.transfer import TransferConfig

   # 使用 SciPy（默认）
   config = TransferConfig(use_copt=False)

   # 使用 COPT（需安装 coptpy），失败时回退 SciPy
   config = TransferConfig(use_copt=True, fallback_to_scipy=True)

配置参数
--------

:class:`~e2m2e.transfer.transfer.TransferConfig` 控制优化行为：

.. code-block:: python

   from e2m2e.transfer import TransferConfig

   DU = 3.84405000e5  # 地月距离 (km)

   config = TransferConfig(
       alpha_min=0.5,              # α 下界
       alpha_max=2.5,              # α 上界
       earth_radius=200.0 / DU,    # 地球碰撞半径（无量纲）
       moon_radius=100.0 / DU,     # 月球碰撞半径（无量纲）
       use_relaxed_velocity=True,  # 使用松弛速度约束
       velocity_angle_tol=0.05,    # 速度角度容差（弧度）
       use_copt=False,             # 求解器选择
       fallback_to_scipy=True,   # COPT 失败时回退 SciPy
       verbose=False,              # 是否打印迭代信息
   )

优化变量与约束详解
------------------

**优化变量范围：**

- α ∈ [alpha_min, alpha_max]，典型值 (0.5, 2.5)
- T ∈ [transfer_time_range]，典型值 (1.0, 30.0) TU
- t_ins ∈ [t_ins_range]，典型值 (0.0, 10.0) TU

**约束类型：**

1. **位置连续性（等式约束）**

   .. math::

      c_{\text{pos}}(y) = (x_f - x_{\text{ins}})^2 + (y_f - y_{\text{ins}})^2 + (z_f - z_{\text{ins}})^2 = 0

2. **速度平行性（等式约束，默认）**

   .. math::

      c_{\text{vel}}(y) = \frac{\mathbf{v}_f \cdot \mathbf{v}_{\text{ins}}}{\|\mathbf{v}_f\| \|\mathbf{v}_{\text{ins}}\|} - 1 = 0

3. **松弛速度约束（不等式约束，可选）**

   当 ``use_relaxed_velocity=True`` 时，将等式约束松弛为不等式：

   .. math::

      \cos(\theta_{\text{max}}) - \frac{\mathbf{v}_f \cdot \mathbf{v}_{\text{ins}}}{\|\mathbf{v}_f\| \|\mathbf{v}_{\text{ins}}\|} \geq 0

   其中 :math:`\theta_{\text{max}} = \text{velocity_angle_tol}`。

高层接口：Transfer
------------------

:class:`~e2m2e.transfer.transfer.Transfer` 提供简化的链式调用接口，封装了
优化器构造、配置管理和结果提取的细节。

.. code-block:: python

   from e2m2e.transfer import Transfer, TransferConfig
   from e2m2e.core.system import CR3BP_System
   from e2m2e.core.dynamics import CR3BP_Dynamics

   system = CR3BP_System.from_known_system("earth_moon")
   dynamics = CR3BP_Dynamics(system)

   transfer = Transfer(dynamics)
   transfer.set_orbit(start=dro_orbit, end=ro_orbit)

   result = transfer.optimize(
       initial_guess={"alpha": 1.0, "transfer_time": 15.0, "t_ins": 5.0},
       alpha_range=(0.5, 2.5),
       use_relaxed_velocity=True,
       velocity_angle_tol=0.05,
   )

   if result.success:
       print(f"总脉冲: {result.total_delta_v:.6f} DU")
       print(f"转移时间: {result.transfer_time:.2f} TU")

底层接口：DROTRONLPOptimizer
-----------------------------

直接使用 :class:`~e2m2e.transfer.transfer_optimization.DROTRONLPOptimizer`
可获得更精细的控制，包括缓存、进度回调和约束定制。

.. code-block:: python

   from e2m2e.transfer import (
       DROTRONLPOptimizer,
       NLPOptimizationVariables,
       TransferConfig,
   )

   # 构造优化器
   optimizer = DROTRONLPOptimizer(
       system=system,
       dynamics=dynamics,
       departure_orbit=dro_orbit,
       arrival_orbit=ro_orbit,
       departure_state=dro_orbit.states[0],
       config=TransferConfig(
           alpha_min=0.5,
           alpha_max=2.5,
           use_relaxed_velocity=True,
           velocity_angle_tol=0.05,
       ),
   )

   # 启用缓存（避免同一点重复积分）
   optimizer.enable_cache(True)

   # 设置进度回调（可选）
   def on_progress(iteration, obj, alpha, T, t_ins):
       print(f"  iter={iteration:3d}  J={obj:.6f}  α={alpha:.4f}  T={T:.2f}  t_ins={t_ins:.2f}")

   optimizer.set_progress_callback(on_progress)

   # 执行优化
   result = optimizer.optimize(
       initial_guess=NLPOptimizationVariables(
           alpha=1.0,
           transfer_time=15.0,
           t_ins=5.0,
       ),
       alpha_range=(0.5, 2.5),
       transfer_time_range=(1.0, 30.0),
       t_ins_range=(0.0, 10.0),
       use_relaxed_velocity_constraint=True,
       velocity_angle_constraint=0.05,
       verbose=True,
   )

便捷函数
--------

:func:`~e2m2e.transfer.transfer_optimization.optimize_transfer` 提供一步到位的优化：

.. code-block:: python

   from e2m2e.transfer.transfer_optimization import optimize_transfer, NLPOptimizationVariables

   result = optimize_transfer(
       system=system,
       dynamics=dynamics,
       departure_orbit=dro_orbit,
       arrival_orbit=ro_orbit,
       departure_state=dro_orbit.states[0],
       initial_guess=NLPOptimizationVariables(alpha=1.0, transfer_time=15.0, t_ins=5.0),
   )

COPT 求解
---------

:func:`~e2m2e.transfer.transfer_optimization.optimize_with_copt` 使用 COPT 求解 NLP，
失败时自动回退 SciPy：

.. code-block:: python

   from e2m2e.transfer.transfer_optimization import optimize_with_copt

   result = optimize_with_copt(
       optimizer,
       initial_guess=NLPOptimizationVariables(alpha=1.0, transfer_time=15.0, t_ins=5.0),
       fallback_to_scipy=True,
       max_iter=1000,
       threads=1,
       time_limit=300.0,
   )

结果分析
--------

:class:`~e2m2e.transfer.transfer.TransferOptimizationResult` 包含优化结果：

.. code-block:: python

   result = optimizer.optimize(...)

   if result.success:
       print(f"总脉冲: {result.total_delta_v:.6f} DU")
       print(f"出发脉冲: {result.delta_v1:.6f} DU")
       print(f"插入脉冲: {result.delta_v2:.6f} DU")
       print(f"转移时间: {result.transfer_time:.2f} TU")
       print(f"插入时间: {result.t_ins:.2f} TU")
       print(f"约束违反: {result.constraints_violation:.2e}")

       # 转移轨迹
       trajectory = result.transfer_trajectory      # (n_steps, 6)
       times = result.transfer_trajectory_times     # (n_steps,)

       # 可视化
       import matplotlib.pyplot as plt
       fig, ax = plt.subplots(figsize=(8, 6))
       ax.plot(trajectory[:, 0], trajectory[:, 1], label="Transfer", color="blue")
       ax.set_xlabel("x (DU)")
       ax.set_ylabel("y (DU)")
       ax.legend()
       ax.set_title(f"Optimized Transfer (Δv={result.total_delta_v:.4f})")
       plt.tight_layout()
       plt.show()

端到端示例：搜索 → 优化 → 成本分析
------------------------------------

以下示例展示从搜索结果出发，经 NLP 优化，到成本分解和可视化的流程：

.. code-block:: python

   from e2m2e.core.system import CR3BP_System
   from e2m2e.core.dynamics import CR3BP_Dynamics
   from e2m2e.transfer import (
       TransferSearch,
       DROTRONLPOptimizer,
       NLPOptimizationVariables,
       TransferConfig,
       compute_transfer_cost,
       load_orbit_from_json,
   )

   # 建立系统
   system = CR3BP_System.from_known_system("earth_moon")
   dynamics = CR3BP_Dynamics(system)

   # 加载轨道
   dro_orbit = load_orbit_from_json("data/dro_orbit.json")
   ro_orbit = load_orbit_from_json("data/ro_orbit.json")

   # 1. 搜索阶段
   searcher = TransferSearch(dynamics)
   results = searcher.search(
       alpha_min=0.5, alpha_max=2.5, n_alpha=101, n_departure=200,
       max_transfer_time=30.0, intersection_threshold=1e-3,
       min_distance_threshold=1e-3,
       collision_earth_radius=200.0 / 384405.0,
       collision_moon_radius=100.0 / 384405.0,
       integration_dt=0.01,
       departure_orbit=dro_orbit, arrival_orbit=ro_orbit,
       verbose=True,
   )

   feasible = searcher.get_feasible_results()
   best = min(feasible, key=lambda r: r["dv_departure"] + r["dv_insertion"])

   # 2. 优化阶段
   optimizer = DROTRONLPOptimizer(
       system=system, dynamics=dynamics,
       departure_orbit=dro_orbit, arrival_orbit=ro_orbit,
       departure_state=best["departure_state"],
       config=TransferConfig(
           alpha_min=0.5, alpha_max=2.5,
           use_relaxed_velocity=True, velocity_angle_tol=0.05,
       ),
   )

   nlp_result = optimizer.optimize(
       initial_guess=NLPOptimizationVariables(
           alpha=best["alpha"],
           transfer_time=best["transfer_time"],
           t_ins=best.get("t_ins", 5.0),
       ),
       alpha_range=(0.5, 2.5),
       transfer_time_range=(1.0, 30.0),
       t_ins_range=(0.0, 10.0),
       use_relaxed_velocity_constraint=True,
       velocity_angle_constraint=0.05,
   )

   # 3. 成本分析
   if nlp_result.success:
       cost = compute_transfer_cost(
           departure_state=nlp_result.departure_state,
           initial_velocity=nlp_result.departure_state[3:] * best["alpha"],
           final_velocity=nlp_result.final_state[3:],
           insertion_velocity=nlp_result.insertion_state[3:],
       )
       print(f"成本分解: Δv1={cost.dv1:.6f}, Δv2={cost.dv2:.6f}, 总计={cost.total:.6f}")

       # 4. 可视化
       import matplotlib.pyplot as plt
       traj = nlp_result.transfer_trajectory
       fig, ax = plt.subplots(figsize=(8, 6))
       ax.plot(traj[:, 0], traj[:, 1], label="Transfer", color="blue", linewidth=1.5)
       ax.scatter(dro_orbit.states[:, 0], dro_orbit.states[:, 1],
                  s=1, label="DRO", color="green", alpha=0.5)
       ax.scatter(ro_orbit.states[:, 0], ro_orbit.states[:, 1],
                  s=1, label="RO", color="red", alpha=0.5)
       ax.set_xlabel("x (DU)")
       ax.set_ylabel("y (DU)")
       ax.legend()
       ax.set_title(f"Optimized DRO-RO Transfer (Δv={cost.total:.4f})")
       plt.tight_layout()
       plt.show()
