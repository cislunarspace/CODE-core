Impulsive Propulsion Model / 脉冲推进模型
==========================================

[English](#impulsive-propulsion-model) | [简体中文](#中文)

English
-------

:class:`~e2m2e.algorithm.transfer.propulsion.ImpulsivePropulsion` computes
departure injection velocities and costs for transfers.

Principle
~~~~~~~~~

Departure velocity decomposes into tangential & normal components:

.. math::

   \mathbf{v} = \alpha \, |\mathbf{v}| \, \hat{\mathbf{t}} + \beta \, |\mathbf{v}| \, \hat{\mathbf{n}}

with:

- α: tangential velocity ratio along original velocity direction
- β: normal ratio (default 0.0 — purely tangential)
- :math:`\hat{\mathbf{t}}`: original-velocity unit vector
- :math:`\hat{\mathbf{n}}`: orbit-normal unit vector

Usage
~~~~~

.. code-block:: python

   from e2m2e.algorithm.transfer.propulsion import ImpulsivePropulsion

   propulsion = ImpulsivePropulsion()

   # Departure injection velocity
   v_inj = propulsion.compute_departure_velocity(
       dro_orbit.states[0],
       alpha=1.2,
       beta=0.0,
   )

Parameters:

- ``normal``: orbit normal, default [0, 0, 1] (z axis)

Cost decomposition
~~~~~~~~~~~~~~~~~~

Internally delegates to
:func:`~e2m2e.algorithm.transfer.cost.compute_transfer_cost`:

.. code-block:: python

   from e2m2e.algorithm.transfer.cost import compute_transfer_cost

   cost = compute_transfer_cost(
       departure_state=departure_state,
       initial_velocity=initial_velocity,
       final_velocity=final_velocity,
       insertion_velocity=insertion_velocity,
   )
   print(f"Δv1 = {cost.dv1:.6f}, Δv2 = {cost.dv2:.6f}, total = {cost.total:.6f}")

中文
----

:class:`~e2m2e.algorithm.transfer.propulsion.ImpulsivePropulsion` 用于计算转移轨道的出发注入速度与代价。

基本原理：出发速度分解为切向与法向分量（α 切向速度比 + β 法向速度比，
β 默认 0）。参数 ``normal`` 为轨道面法向量，默认 [0, 0, 1]。
Δv 分解由 :func:`~e2m2e.algorithm.transfer.cost.compute_transfer_cost`
完成。接口与示例见上方英文节。
