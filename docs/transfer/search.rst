转移窗口搜索
============

:class:`~e2m2e.algorithm.transfer.transfer_search.TransferSearch` 实现转移轨道的参数空间搜索，是"搜索-优化"两步法中的第一步。

搜索算法
--------

采用网格搜索策略，在指定的参数范围内均匀采样：

1. 设定出发轨道和到达轨道
2. 从出发轨道等时间间隔采样出发点
3. 对每个出发点，在 α 范围内均匀采样
4. 对每个 α 值计算出发注入速度、前向积分转移轨道
5. 检查碰撞、相交和距离约束，记录可行解

搜索变量
--------

- **α (alpha)**: 切向速度比，范围 ``[alpha_min, alpha_max]``
- **出发点位置**: 在出发轨道上的位置，由 ``n_departure`` 控制采样数量

参数配置
--------

搜索参数通过 :class:`~e2m2e.algorithm.transfer.config.TransferConfig` 的 ``search_*`` 字段集中管理，
同时 ``TransferSearch`` 提供向后兼容的属性代理访问。

.. code-block:: python

   from e2m2e.algorithm.transfer import TransferSearch, TransferConfig
   from e2m2e.algorithm.dynamics.dynamics import CR3BP_Dynamics

   # 方式一：通过 TransferConfig 的 search_* 字段配置
   cfg = TransferConfig(
       search_alpha_min=0.5,
       search_alpha_max=2.5,
       search_n_alpha=101,
       search_n_departure=200,
       search_max_transfer_time=30.0,
       search_intersection_threshold=1e-3,
       search_min_distance_threshold=1e-3,
       search_collision_earth_radius=200.0 / 384405.0,
       search_collision_moon_radius=100.0 / 384405.0,
       search_integration_dt=0.01,
   )
   searcher = TransferSearch(dynamics, config=cfg)

   # 方式二：直接设置属性（向后兼容）
   searcher = TransferSearch(dynamics)
   searcher.alpha_min = 0.5
   searcher.alpha_max = 2.5
   searcher.n_alpha = 101

执行搜索
--------

``search()`` 方法一次性传入所有搜索参数，执行网格搜索：

.. code-block:: python

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

   print(f"总候选解: {len(results)}")

并行搜索
--------

支持多进程和多线程并行搜索以加速计算。多进程（默认）绕过 GIL，适合 CPU 密集的积分运算；
多线程保留细粒度 tqdm 进度条，适合需要实时进度的场景。

.. code-block:: python

   # 使用所有 CPU 核心（多进程，默认）
   results = searcher.search(
       ...,  # 其他参数同上
       n_workers=None,
       parallel_backend="processes",
   )

   # 指定线程数（多线程，适合 I/O 或需要实时进度的场景）
   results = searcher.search(
       ...,  # 其他参数同上
       n_workers=4,
       parallel_backend="threads",
   )

结果筛选
--------

搜索结果字典包含丰富的字段，用于筛选可行解：

.. code-block:: python

   # 获取所有可行解（无碰撞，且相交或距离足够近）
   feasible = searcher.get_feasible_results()

   # 按总脉冲排序
   best = min(feasible, key=lambda r: r["dv_departure"] + r["dv_insertion"])

   # 手动筛选：转移时间 < 20 TU 且未碰撞
   filtered = [
       r for r in results
       if r["transfer_time"] < 20.0 and not r["collision_found"]
   ]

搜索结果字典关键字段
--------------------

每个搜索结果是一个字典，包含以下关键字段：

- ``success`` (bool): 积分是否成功
- ``alpha`` (float): 该解使用的切向速度比
- ``departure_state`` (ndarray): 出发点六维状态
- ``departure_time`` (float): 出发点时间
- ``transfer_time`` (float): 转移时间
- ``transfer_trajectory`` (ndarray): 转移轨迹 ``(n_steps, 6)``
- ``transfer_times`` (ndarray): 轨迹时间序列
- ``min_distance`` (float): 到目标轨道的最小距离
- ``min_distance_idx`` (int): 最小距离对应的轨迹步索引
- ``min_distance_orbit_idx`` (int): 最小距离对应的目标轨道步索引
- ``dv_departure`` (float): 出发脉冲大小
- ``dv_insertion`` (float): 插入脉冲大小（粗估）
- ``intersection_found`` (bool): 是否与目标轨道相交
- ``first_intersection_idx`` (int | None): 首次相交的轨迹步索引
- ``first_intersection_time`` (float | None): 首次相交的时间
- ``local_minimum_found`` (bool): 是否发现局部最小距离
- ``local_minimum_distance`` (float): 局部最小距离值
- ``collision_found`` (bool): 是否碰撞
- ``collision_body`` (str | None): 碰撞天体（"earth" 或 "moon"）
- ``status`` (str): 状态描述（"success" / "collision" / "no_intersection" / "integration_failed"）

注意：``status="integration_failed"`` 的结果是稀疏字典，不含 ``transfer_trajectory``、
``transfer_time``、``min_distance``、``collision_found`` 等键，且 ``dv_insertion=None``。
消费结果前应先检查 ``success`` 或 ``status``，避免对缺失键取值。

另外，``TransferSearch.configure_search(**kwargs)`` 提供批量设置搜索属性的便捷方法，
与逐个赋值属性等价。

搜索后自动优化
--------------

``TransferSearch`` 提供 ``optimize()`` 方法，可直接以搜索结果为初值执行 NLP 优化：

.. code-block:: python

   # 先执行搜索
   searcher.search(...)

   # 再以最佳可行解为初值自动优化
   nlp_result = searcher.optimize()

   if nlp_result.success:
       print(f"优化后总脉冲: {nlp_result.total_delta_v:.6f}")

示例
----

以下示例展示从轨道加载到搜索完成的流程：

.. code-block:: python

   from e2m2e.algorithm.dynamics.system import CR3BP_System
   from e2m2e.algorithm.dynamics.dynamics import CR3BP_Dynamics
   from e2m2e.algorithm.transfer import TransferSearch, TransferConfig, load_orbit_from_json

   # 建立系统
   system = CR3BP_System(
       mu=0.0121506683, primary="Earth", secondary="Moon"
   )._with_default_scales()
   system.compute_libration_points()
   dynamics = CR3BP_Dynamics(system)

   # 加载轨道
   dro_orbit = load_orbit_from_json("data/dro_orbit.json")
   ro_orbit = load_orbit_from_json("data/ro_orbit.json")

   # 配置搜索
   cfg = TransferConfig(
       search_alpha_min=0.5,
       search_alpha_max=2.5,
       search_n_alpha=101,
       search_n_departure=200,
       search_max_transfer_time=30.0,
       search_intersection_threshold=1e-3,
       search_min_distance_threshold=1e-3,
       search_collision_earth_radius=200.0 / 384405.0,
       search_collision_moon_radius=100.0 / 384405.0,
       search_integration_dt=0.01,
   )
   searcher = TransferSearch(dynamics, config=cfg)

   # 执行搜索（多进程并行）；搜索参数已随 config 挂在 searcher 上
   results = searcher.search(
       departure_orbit=dro_orbit,
       arrival_orbit=ro_orbit,
       verbose=True,
       n_workers=None,
       parallel_backend="processes",
   )

   # 分析结果
   feasible = searcher.get_feasible_results()
   print(f"总候选: {len(results)}, 可行解: {len(feasible)}")

   if feasible:
       best = min(feasible, key=lambda r: r["dv_departure"] + r["dv_insertion"])
       print(f"最佳候选: α={best['alpha']:.4f}, "
             f"Δv={best['dv_departure']+best['dv_insertion']:.6f}, "
             f"T={best['transfer_time']:.2f}")
