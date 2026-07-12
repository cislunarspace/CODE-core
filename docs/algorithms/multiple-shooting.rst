多重打靶法
==========

多重打靶法将长轨迹拆分为多段，在每段端点施加连续性约束，从而降低积分敏感性、提高收敛性。

基本原理
--------

将轨道分成 N 段（patch points），每段独立积分，在相邻段的连接点施加位置连续约束：

.. math::

   \mathbf{x}_i(t_{i+1}) = \mathbf{x}_{i+1}(t_{i+1})

通过调整各段初始状态和（可选的）时间节点，使所有连续性约束同时满足。

标准多重打靶
------------

:class:`~e2m2e.algorithms.multiple_shooting.MultipleShooting` 是标准实现。

.. code-block:: python

   from e2m2e.algorithms import MultipleShooting, sample_patch_points

   ms = MultipleShooting(dynamics=dynamics)

   # 从种子轨道采样 patch points
   t_patch, state_patch = sample_patch_points(seed_orbit, n_points=5)

   # 执行修正
   result = ms.correct(
       t_patch=t_patch,
       state_patch=state_patch,
       max_iter=50,
       tolerance=1e-10,
       var_time=True,    # 允许调整时间节点
   )

   if result.converged:
       print(f"收敛，最大残差 {result.max_residual:.2e}")
       print(f"修正后 patch points: {result.state_patch}")

参数说明：

- ``t_patch`` — 时间节点数组，形状 ``(N,)``
- ``state_patch`` — 状态量数组，形状 ``(N, 6)``
- ``max_iter`` — 最大迭代次数
- ``tolerance`` — 位置残差收敛容差
- ``var_time`` — 是否允许调整时间节点（``True`` 时自由度更大）

两层多重打靶
------------

:class:`~e2m2e.algorithms.two_level_multiple_shooting.TwoLevelMultipleShooting`
将问题分解为两层交替求解：

- **Level 1（局部问题）**：逐段调整出发速度使位置连续
- **Level 2（全局问题）**：联合调整内部节点的位置和时间使速度连续

这种分解将高维耦合问题拆解为交替求解的低维子问题，适用于自由时间多段轨道设计
（如多圈共振轨道、星际转移等）。

.. code-block:: python

   from e2m2e.algorithms.two_level_multiple_shooting import TwoLevelMultipleShooting

   tms = TwoLevelMultipleShooting(dynamics=dynamics)

   result = tms.correct(
       t_patch=t_patch,
       state_patch=state_patch,
       max_iter=20,
       tolerance=1e-8,
       velocity_tolerance=1e-6,
   )

   if result.converged:
       print(f"外层迭代: {result.outer_iterations}")
       print(f"位置残差: {result.max_residual:.2e}")
       print(f"速度残差: {result.velocity_residual:.2e}")

星历修正
--------

多重打靶在星历模型中的应用，通过 :func:`~e2m2e.algorithms.ephemeris_correction.correct_ephemeris_patch_points`
统一调度：

.. code-block:: python

   from e2m2e.algorithms.ephemeris_correction import correct_ephemeris_patch_points

   result = correct_ephemeris_patch_points(
       method="standard",      # 或 "two_level"、"homotopy"
       dynamics=dynamics,
       t_patch=t_patch,
       state_patch=state_patch,
       tolerance=1e-3,
       max_iter=10,
       verbose=False,
       n_workers=1,
       kernel_dir="/path/to/kernels",
   )

支持三种修正方法：

- ``"standard"`` — 标准多重打靶
- ``"two_level"`` — 两层多重打靶
- ``"homotopy"`` — 同伦过渡修正（详见 :doc:`homotopy-correction`）

参考
----

- Montenbruck O, Gill E. *Satellite Orbits*, Chapter 7.
- 轨道力学中的多重打靶法：将长弧段分解为短弧段序列，降低对初始猜测的敏感性。
