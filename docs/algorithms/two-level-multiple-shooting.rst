两层多重打靶
============

:class:`~e2m2e.algorithm.solver.two_level_multiple_shooting.TwoLevelMultipleShooting`
将多重打靶问题分解为两层交替求解，适用于自由时间多段轨道设计。

基本原理
--------

标准多重打靶同时修正所有节点的状态和时间，变量维度为 ``6N + N``。
两层方法将其分解为两个低维子问题：

- **Level 1（局部问题）**：逐段调整出发速度，使相邻段在连接点处位置连续。
  每段独立求解，变量维度为 3（出发速度的三个分量）。
- **Level 2（全局问题）**：联合调整内部节点的位置和时间，使速度连续。
  变量维度为 4\*(N-2)（内部节点的位置 + 时间）。

两层交替执行直到收敛。

使用方法
--------

.. code-block:: python

   from e2m2e.algorithm.solver.two_level_multiple_shooting import TwoLevelMultipleShooting

   tms = TwoLevelMultipleShooting(dynamics=dynamics)

   result = tms.correct(
       t_patch=t_patch,
       state_patch=state_patch,
       max_outer_iterations=20,
       position_tolerance=1e-8,
       velocity_tolerance=1e-6,
   )

   if result.converged:
       print(f"外层迭代: {result.outer_iterations}")
       print(f"位置残差: {result.final_position_residual:.2e}")
       print(f"速度残差: {result.final_velocity_residual:.2e}")

参数说明
--------

- ``t_patch`` — 时间节点数组，形状 ``(N,)``
- ``state_patch`` — 状态量数组，形状 ``(N, 6)``
- ``max_outer_iterations`` — 外层（Level 1 + Level 2）最大迭代次数（默认 10）
- ``max_level1_iterations`` — Level 1 每段最大迭代次数（默认 20）
- ``position_tolerance`` — 位置残差收敛容差
- ``velocity_tolerance`` — 速度残差收敛容差
- ``level1_position_tolerance`` — Level 1 内部容差（默认 ``None``，即取 ``position_tolerance``）

结果字段
--------

``TwoLevelMultipleShootingResult``：

- ``t_patch`` / ``state_patch`` — 修正后的 patch points
- ``converged`` — 是否收敛
- ``status`` — 终止原因
- ``outer_iterations`` — 外层迭代次数
- ``level1_iterations`` — 每段 Level 1 迭代次数（外层迭代 × 弧段）
- ``final_position_residual`` — 最终最大位置残差
- ``final_velocity_residual`` — 最终最大速度残差
- ``per_patch_position_residual`` — 各段位置残差
- ``per_patch_velocity_residual`` — 各段速度残差
- ``residual_history`` — 每次外层迭代的（最大位置残差, 最大速度残差）记录

与标准多重打靶的区别
--------------------

.. list-table::
   :header-rows: 1

   * - 特性
     - 标准多重打靶
     - 两层多重打靶
   * - 求解方式
     - 同时修正所有变量
     - 交替求解两个子问题
   * - 变量维度
     - ``6N + N``
     - Level 1: ``3``，Level 2: ``4(N-2)``
   * - 适用场景
     - 通用
     - 自由时间多段轨道、高维问题
   * - 收敛速度
     - 牛顿法二次收敛
     - 交替收敛，每步更便宜
