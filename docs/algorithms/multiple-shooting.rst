多重打靶法
==========

多重打靶法用于求解复杂轨道问题。

基本原理
--------

将轨道分成多段，每段独立积分，通过匹配条件连接。

.. math::

   x_i(t_{i+1}) = x_{i+1}(t_{i+1})

实现
----

.. code-block:: python

   from e2m2e.algorithms.multiple_shooting import multiple_shooting

   # 求解转移轨道
   result = multiple_shooting(
       initial_state,
       target_state,
       system,
       num_segments=5
   )
