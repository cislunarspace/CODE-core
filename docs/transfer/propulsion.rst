脉冲推进模型
============

:class:`~e2m2e.algorithm.transfer.propulsion.ImpulsivePropulsion` 用于计算转移轨道的出发注入速度与代价。

基本原理
--------

脉冲推进将出发速度分解为切向与法向分量：

.. math::

   \mathbf{v} = \alpha \, |\mathbf{v}| \, \hat{\mathbf{t}} + \beta \, |\mathbf{v}| \, \hat{\mathbf{n}}

其中：

- α（alpha）为切向速度比，控制出发速度沿原始速度方向的分量
- β（beta）为法向速度比（默认 0.0，即纯切向）
- :math:`\hat{\mathbf{t}}` 为原始速度方向单位向量
- :math:`\hat{\mathbf{n}}` 为轨道面法向单位向量

使用方法
--------

.. code-block:: python

   from e2m2e.algorithm.transfer.propulsion import ImpulsivePropulsion

   propulsion = ImpulsivePropulsion()

   # 计算出发注入速度
   v_inj = propulsion.compute_departure_velocity(
       dro_orbit.states[0],
       alpha=1.2,
       beta=0.0,
   )

   # 转移代价（Δv 分解）由 compute_cost 计算，见下文示例

参数说明
--------

- ``normal``：轨道面法向量，默认 [0, 0, 1]（z 轴）

与转移代价的配合
----------------

``ImpulsivePropulsion`` 内部调用 :func:`~e2m2e.algorithm.transfer.cost.compute_transfer_cost`
计算 Δv 分解：

.. code-block:: python

   from e2m2e.algorithm.transfer.cost import compute_transfer_cost

   cost = compute_transfer_cost(
       departure_state=departure_state,
       initial_velocity=initial_velocity,
       final_velocity=final_velocity,
       insertion_velocity=insertion_velocity,
   )
   print(f"Δv1 = {cost.dv1:.6f}, Δv2 = {cost.dv2:.6f}, 总计 = {cost.total:.6f}")
