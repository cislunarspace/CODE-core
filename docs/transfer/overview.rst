转移轨道设计概述
================

e2m2e 提供多种转移轨道设计方法，覆盖从简单共面转移（霍曼转移）到复杂引力辅助转移（LGA）的各类场景。

转移设计模块
------------

- :doc:`lambert` — Lambert 二体求解器，支持短程/长程与多圈解
- :doc:`hmn` — 霍曼直接转移（HMN），适用于共面圆轨道间的最小能量转移
- :doc:`lga` — 月球引力辅助间接转移（LGA），基于圆锥曲线拼接法
- :doc:`wsb` — 太阳引力辅助低能间接转移（WSB），H₂<0 弹道捕获
- :doc:`low_thrust` — 小推力转移，Q-law 初猜 + 打靶/配点
- :doc:`search` — 网格搜索，参数空间扫描可行转移窗口
- :doc:`optimization` — NLP 优化，以搜索结果为初值精化转移轨道
- :doc:`terminal` — 终端条件抽象
- :doc:`propulsion` — 推进系统建模

搜索-优化两步法（DRO-RO）
--------------------------

基于 Cui et al. (2025) 的"搜索-优化"两步法，用于 DRO（Distant Retrograde Orbit）到 RO（共振轨道）的转移轨道设计。

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

端到端示例
----------

以下示例展示从搜索到优化再到成本分析的 DRO-RO 转移设计流程：

.. code-block:: python

   from e2m2e.algorithm.dynamics.system import CR3BP_System
   from e2m2e.algorithm.dynamics.dynamics import CR3BP_Dynamics
   from e2m2e.data.constants import Datum
   from e2m2e.algorithm.transfer import (
       Transfer,
       TransferSearch,
       TransferConfig,
       DROTRONLPOptimizer,
       NLPOptimizationVariables,
       load_orbit_from_json,
   )

   # 1. 建立动力学（μ 取 DE421 基准，ADR 0022）
   system = CR3BP_System(
       mu=Datum.DE421.mu, primary="Earth", secondary="Moon"
   )._with_default_scales()
   system.compute_libration_points()
   dynamics = CR3BP_Dynamics(system)

   # 2. 加载出发轨道（DRO）和目标轨道（RO）
   dro_orbit = load_orbit_from_json("data/dro_orbit.json")
   ro_orbit = load_orbit_from_json("data/ro_orbit.json")

   # ========== 路径 A：高层接口（搜索 + 优化一键完成）==========
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

   # ========== 路径 B：底层接口（搜索与优化分步控制）==========
   # 2.1 搜索阶段：网格搜索可行转移窗口
   searcher = TransferSearch(dynamics)

   results = searcher.search(
       alpha_min=0.5,
       alpha_max=2.5,
       n_alpha=101,
       n_departure=200,
       max_transfer_time=30.0,
       intersection_threshold=1e-3,
       min_distance_threshold=1e-3,
       collision_earth_radius=200.0 / 384405.0,
       collision_moon_radius=100.0 / 384405.0,
       integration_dt=0.01,
       departure_orbit=dro_orbit,
       arrival_orbit=ro_orbit,
       verbose=True,
   )

   # 筛选可行解
   feasible = searcher.get_feasible_results()
   print(f"可行解数量: {len(feasible)}")

   # 2.2 优化阶段：以最佳可行解为初值进行 NLP 精化
   best = min(feasible, key=lambda r: r["dv_departure"] + r["dv_insertion"])

   optimizer = DROTRONLPOptimizer(
       system=system,
       dynamics=dynamics,
       departure_orbit=dro_orbit,
       arrival_orbit=ro_orbit,
       departure_state=best["departure_state"],
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

   if nlp_result.success:
       print(f"优化后总脉冲: {nlp_result.total_delta_v:.6f} DU")
       print(f"优化后转移时间: {nlp_result.transfer_time:.2f} TU")

   # 2.3 成本分析
   from e2m2e.algorithm.transfer.cost import compute_transfer_cost

   cost = compute_transfer_cost(
       departure_state=nlp_result.departure_state,
       initial_velocity=nlp_result.departure_state[3:] * best["alpha"],
       final_velocity=nlp_result.final_state[3:],
       insertion_velocity=nlp_result.insertion_state[3:],
   )
   print(f"成本分解: Δv1={cost.dv1:.6f}, Δv2={cost.dv2:.6f}, 总计={cost.total:.6f}")

   # 2.4 可视化转移轨迹
   import matplotlib.pyplot as plt

   traj = nlp_result.transfer_trajectory
   fig, ax = plt.subplots(figsize=(8, 6))
   ax.plot(traj[:, 0], traj[:, 1], label="Transfer", color="blue")
   ax.scatter(dro_orbit.states[:, 0], dro_orbit.states[:, 1],
              s=1, label="DRO", color="green", alpha=0.5)
   ax.scatter(ro_orbit.states[:, 0], ro_orbit.states[:, 1],
              s=1, label="RO", color="red", alpha=0.5)
   ax.set_xlabel("x (DU)")
   ax.set_ylabel("y (DU)")
   ax.legend()
   ax.set_title("DRO-RO Transfer Trajectory")
   plt.tight_layout()
   plt.show()
