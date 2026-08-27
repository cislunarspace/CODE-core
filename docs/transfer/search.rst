Transfer Window Search / 转移窗口搜索
=====================================

[English](#transfer-window-search) | [简体中文](#中文)

English
-------

:class:`~e2m2e.algorithm.transfer.transfer_search.TransferSearch` scans transfer
parameter space — step one of the search-optimize two-step method.

Algorithm
~~~~~~~~~

Grid search: uniform sampling over parameter ranges:

1. Set departure & arrival orbits
2. Sample departure points uniformly in time along the departure orbit
3. For each departure point, sample α uniformly
4. For each α, compute injection velocity and forward-propagate the transfer arc
5. Screen collision/intersection/distance constraints, recording feasible solutions

Search variables
~~~~~~~~~~~~~~~~

- **α (alpha)**: tangential velocity ratio in ``[alpha_min, alpha_max]``
- **Departure point**: position on the departure orbit; count set by ``n_departure``

Configuration
~~~~~~~~~~~~~

Parameters live on :class:`~e2m2e.algorithm.transfer.config.TransferConfig`'s
``search_*`` fields, with backward-compatible property proxies on
``TransferSearch``:

.. code-block:: python

   from e2m2e.algorithm.transfer import TransferSearch, TransferConfig
   from e2m2e.algorithm.dynamics.dynamics import CR3BP_Dynamics

   # Option 1: via TransferConfig's search_* fields
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

   # Option 2: attribute assignment (backward compatible)
   searcher = TransferSearch(dynamics)
   searcher.alpha_min = 0.5
   searcher.alpha_max = 2.5
   searcher.n_alpha = 101

Running a search
~~~~~~~~~~~~~~~~

``search()`` takes all parameters at once:

.. code-block:: python

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

   print(f"Total candidates: {len(results)}")

中文
----

:class:`~e2m2e.algorithm.transfer.transfer_search.TransferSearch` 实现转移轨道的参数空间搜索，是搜索-优化两步法中的第一步。

搜索算法采用网格策略：设定出发/到达轨道 → 出发轨道等时间采样出发点 → 每个
出发点在 α 范围内均匀采样 → 计算注入速度并前向积分 → 碰撞/相交/距离约束筛选，
记录可行解。搜索变量为 α（切向速度比）与出发点位置（``n_departure`` 控制）。

参数经 :class:`~e2m2e.algorithm.transfer.config.TransferConfig` 的 ``search_*``
字段集中管理（属性代理向后兼容）；``search()`` 一次传入全部参数执行。
配置与执行代码见上方英文节；数值内核已下沉 Rust（ADR 0017），
默认后端 ``rust`` 。
