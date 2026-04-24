转移窗口搜索
============

:class:`~e2m2e.transfer.transfer_search.TransferSearch` 实现转移轨道的参数空间搜索。

搜索算法
--------

采用网格搜索策略，在指定的参数范围内均匀采样：

1. 设定出发轨道和到达轨道
2. 在 α 范围内均匀采样
3. 对每个 α 值，积分转移轨道
4. 检查碰撞和约束满足情况
5. 记录可行解

搜索变量
--------

- **α (alpha)**: 切向速度比，范围 ``[alpha_min, alpha_max]``
- **出发点位置**: 在出发轨道上的位置

参数配置
--------

.. code-block:: python

   from e2m2e.transfer.transfer_search import TransferSearch

   search = TransferSearch(system, dynamics)

   # 设置轨道
   search.set_departure_orbit(departure_orbit)
   search.set_arrival_orbit(arrival_orbit)

   # 设置搜索参数
   search.alpha_min = 0.5
   search.alpha_max = 2.5
   search.n_alpha = 101

   # 执行搜索
   results = search.search()

并行搜索
--------

支持多进程并行搜索以加速计算：

.. code-block:: python

   # 自动使用所有可用核心
   results = search.search(parallel=True)

   # 指定进程数
   results = search.search(parallel=True, n_workers=4)

结果筛选
--------

搜索结果可通过以下条件筛选：

- 最大 Δv 阈值
- 转移时间范围
- 碰撞约束满足情况
