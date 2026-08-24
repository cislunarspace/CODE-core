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

:class:`~e2m2e.algorithm.solver.multiple_shooting.MultipleShooting` 是标准实现。

.. code-block:: python

   from e2m2e.algorithm.solver import MultipleShooting, sample_patch_points
   from e2m2e.data.templates import ConvergenceState

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

   if result.status == ConvergenceState.CONVERGED:
       print(f"收敛，最大残差 {result.max_residual:.2e}")
       print(f"修正后 patch points: {result.state_patch}")

参数说明：

- ``t_patch``：时间节点数组，形状 ``(N,)``
- ``state_patch``：状态量数组，形状 ``(N, 6)``
- ``max_iter``：最大迭代次数
- ``tolerance``：位置残差收敛容差
- ``var_time``：是否允许调整时间节点（``True`` 时自由度更大）

近月点相关采样
--------------

近月点速度大、STM 条件数高时，等时间采样可能不够。``design_orbit`` 按族
选用不同策略（不暴露到请求模型）：

- **Halo**：近月点加密（
  :func:`~e2m2e.algorithm.solver.multiple_shooting.sample_patch_points_perilune_clustered`）
- **NRHO**：等时间（#473；第 1 步 ``revs_per_group=1`` ）。删近月点附近节点
  （:func:`~e2m2e.algorithm.solver.multiple_shooting.sample_patch_points_drop_near_perilune`）
  保留为工具函数，强制含历元 ``t=0`` ；不再作生产默认

近月点加密：先积分一圈定位近月点，在其两侧窗口内加密节点：

.. code-block:: python

   from e2m2e.algorithm.solver import sample_patch_points_perilune_clustered

   t_patch, state_patch = sample_patch_points_perilune_clustered(
       orbit,
       dynamics,
       n_base=8,              # 窗口外等时间间隔节点数
       n_perilune=5,          # 近月点窗口内加密节点数（含近月点本身）
       perilune_window=0.15,  # 加密窗口半宽，占周期比例
   )

删近月点附近节点：非历元节点落在近月点禁区之外的互补弧上，并强制包含
历元 ``t=0`` （避免 segmented 星历网格前缀空洞）：

.. code-block:: python

   from e2m2e.algorithm.solver import sample_patch_points_drop_near_perilune

   t_patch, state_patch = sample_patch_points_drop_near_perilune(
       orbit,
       dynamics,
       n_points=8,
       drop_window=0.12,  # 近月点禁区半宽，占周期比例
   )

二者均返回按时间升序的 ``(t_patch, states)`` 。非 CR3BP 动力学（无 ``mu``
属性）时退化为等时间间隔采样；``drop_near`` 去重后点数不足时同样回退等时间。

星历修正
--------

星历模型下的 patch points 修正不再走 Python ``MultipleShooting`` ：设计链路
统一走 Rust 多重打靶 ``e2m2e.integrators.multiple_shooting_correct_py``
（segmented 与稳定轨道默认路径），速度残差经 ``vel_weight`` 加权与位置
残差同尺度收敛。旧的 Python 分发器（``ephemeris_correction`` 包）与
``TwoLevelMultipleShooting`` 已删除。

参考
----

- Montenbruck O, Gill E. *Satellite Orbits*, Chapter 7.
- 轨道力学中的多重打靶法：将长弧段分解为短弧段序列，降低对初始猜测的敏感性。

