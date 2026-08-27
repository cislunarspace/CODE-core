Continuation / 延拓法
=====================

[English](#continuation) | [简体中文](#中文)

English
-------

Continuation starts from one known orbit and steps along a family parameter to
generate neighbors.

Basic principle
~~~~~~~~~~~~~~~

Given a converged periodic orbit (the seed), continuation nudges initial
conditions stepwise along a parameter (Jacobi constant, amplitude…),
re-converging each step via differential correction, producing a family.

Natural continuation
~~~~~~~~~~~~~~~~~~~~

Natural continuation advances along a single parameter, seeding each step with
the current solution:

.. code-block:: python

   from e2m2e.algorithm.solver import DifferentialCorrection, Continuation

   corrector = DifferentialCorrection(dynamics)
   corrector.setup_2D_symmetric_x_fixed_x0(x0=initial_state[0])

   continuation = Continuation(corrector=corrector)
   result = continuation.natural_continuation(
       seed_orbit=seed_dro,       # seed orbit
       param_range=(0.14, 0.9),   # parameter sweep range
       step_size=0.005,           # step size
   )

   print(f"Generated {len(result.family.orbits)} orbits")

Parameters:

- ``seed_orbit``: a converged periodic orbit
- ``param_range``: sweep range (e.g., x0 range)
- ``step_size``: per-step increment

Pseudo-arclength continuation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adds an arclength constraint atop natural continuation, handling turning
points — suited to families with non-monotonic amplitude–parameter relations
(e.g., Halo's amplitude-period curve).

.. code-block:: python

   family = continuation.pseudo_arclength_continuation(
       seed_orbit=seed_orbit,
       n_orbits=50,              # new members on this branch
       step_size=0.005,          # pseudo-arclength step ΔS
       direction="positive",     # direction; call twice for both sides
   )

Halo continuation
~~~~~~~~~~~~~~~~~

Halos have dedicated orchestration combining Richardson's analytic approximation
with continuation:

.. code-block:: python

   # 1. Seed (small amplitude suits Richardson; continuation amplifies)
   seed_orbit = continuation.generate_halo_seed_orbit(
       libration_point=1,
       amplitude_z=0.001,
       halo_class=0,        # 0=northern, 1=southern
   )

   # 2. Natural-parameter continuation from the seed
   family = continuation.generate_halo_family(
       seed_orbit,
       n_orbits=50,
       z_range=(0.001, 0.15),   # z-amplitude range
   )

Failure handling
~~~~~~~~~~~~~~~~

When a step's correction fails to converge, the continuer shrinks the step and
retries; once the floor (``min_step_size``) is hit without convergence, that
direction terminates. Only converged orbits join the family.

After continuation, diagnose via the ``Continuation`` instance's
``continuation_stats`` attribute: dict with ``total_steps``,
``successful_steps``, ``failed_steps``.

中文
----

延拓法从一条已知轨道出发，沿某个轨道族参数方向逐步生成相邻轨道。

基本原理
~~~~~~~~

给定一条已收敛的周期轨道（种子），延拓法沿某个参数方向（如 Jacobi 常数、振幅）
逐步微调初始条件，每步用微分修正重新收敛，生成一族连续变化的轨道。

自然延拓
~~~~~~~~

自然延拓沿单一参数方向逐步推进，每步以当前解为初值：

.. code-block:: python

   from e2m2e.algorithm.solver import DifferentialCorrection, Continuation

   corrector = DifferentialCorrection(dynamics)
   corrector.setup_2D_symmetric_x_fixed_x0(x0=initial_state[0])

   continuation = Continuation(corrector=corrector)
   result = continuation.natural_continuation(
       seed_orbit=seed_dro,       # 种子轨道
       param_range=(0.14, 0.9),   # 参数扫描范围
       step_size=0.005,           # 步长
   )

   print(f"生成 {len(result.family.orbits)} 条轨道")

参数说明：

- ``seed_orbit`` ：种子轨道（已收敛的周期轨道）
- ``param_range`` ：参数扫描范围（如 x0 范围）
- ``step_size`` ：每步步长

伪弧长延拓
~~~~~~~~~~

伪弧长延拓在自然延拓的基础上增加弧长约束，能处理转向点（turning point），
适合参数-振幅关系非单调的轨道族（如 Halo 轨道的振幅-周期曲线）。

.. code-block:: python

   family = continuation.pseudo_arclength_continuation(
       seed_orbit=seed_orbit,
       n_orbits=50,              # 本支生成的新轨道条数
       step_size=0.005,          # 伪弧长步长 ΔS
       direction="positive",     # 延拓方向；双侧延拓调用两次
   )

Halo 轨道延拓
~~~~~~~~~~~~~~

Halo 轨道有专用的延拓编排，结合 Richardson 三阶解析近似和延拓法：

.. code-block:: python

   # 1. 先生成种子轨道（Richardson 近似在小振幅下精度高，种子宜取小振幅，
   #    再由延拓逐步放大）
   seed_orbit = continuation.generate_halo_seed_orbit(
       libration_point=1,
       amplitude_z=0.001,
       halo_class=0,        # 0=北族，1=南族
   )

   # 2. 从种子出发做自然参数延拓，返回 Orbit 列表
   family = continuation.generate_halo_family(
       seed_orbit,
       n_orbits=50,
       z_range=(0.001, 0.15),   # z 振幅范围
   )

延拓失败处理
~~~~~~~~~~~~

延拓过程中，某一步微分修正不收敛时，延拓器会缩减步长重试；
步长缩减到下限（``min_step_size`` ）仍不收敛时，终止该方向的延拓。
轨道族中只追加收敛的轨道，失败步不会进入结果。

延拓结束后，可通过 ``Continuation`` 实例的 ``continuation_stats``
属性诊断：字典含 ``total_steps`` 、``successful_steps`` 、``failed_steps``
三个键，分别记录总步数与成功/失败步数。
