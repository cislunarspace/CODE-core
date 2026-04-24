转移窗口搜索
============

转移窗口搜索算法。

搜索方法
--------

- **网格搜索**: 在参数空间均匀采样
- **随机搜索**: Monte Carlo 采样
- **梯度搜索**: 基于梯度的优化搜索

实现
----

.. code-block:: python

   from e2m2e.transfer.search import search_transfer_window

   # 搜索转移窗口
   windows = search_transfer_window(
       initial_orbit,
       target_orbit,
       system,
       search_params
   )
