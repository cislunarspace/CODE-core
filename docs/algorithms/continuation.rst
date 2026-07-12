延拓法
======

延拓法从一条已知轨道出发，沿某个轨道族参数方向逐步生成相邻轨道。

基本原理
--------

给定一条已收敛的周期轨道（种子），延拓法沿某个参数方向（如 Jacobi 常数、振幅）
逐步微调初始条件，每步用微分修正重新收敛，生成一族连续变化的轨道。

自然延拓
--------

自然延拓沿单一参数方向逐步推进，每步以当前解为初值：

.. code-block:: python

   from e2m2e.algorithms import DifferentialCorrection, Continuation

   corrector = DifferentialCorrection(dynamics)
   corrector.setup_2D_symmetric_x_fixed_x0(x0=initial_state[0])

   continuation = Continuation(corrector=corrector)
   family = continuation.natural_continuation(
       seed_orbit=seed_dro,       # 种子轨道
       param_range=(0.14, 0.9),   # 参数扫描范围
       step_size=0.005,           # 步长
   )

   print(f"生成 {len(family)} 条轨道")

参数说明：

- ``seed_orbit`` — 种子轨道（已收敛的周期轨道）
- ``param_range`` — 参数扫描范围（如 x0 范围）
- ``step_size`` — 每步步长

伪弧长延拓
----------

伪弧长延拓在自然延拓的基础上增加弧长约束，能处理转向点（turning point），
适合参数-振幅关系非单调的轨道族（如 Halo 轨道的振幅-周期曲线）。

.. code-block:: python

   family = continuation.pseudo_arc_length_continuation(
       seed_orbit=seed_orbit,
       param_range=(0.005, 0.15),
       step_size=0.002,
       max_arc_length=1.0,
   )

Halo 轨道延拓
--------------

Halo 轨道有专用的延拓编排，结合 Richardson 三阶解析近似和延拓法：

.. code-block:: python

   from e2m2e.algorithms.halo_family import generate_halo_family

   family = generate_halo_family(
       system=system,
       dynamics=dynamics,
       libration_point=1,
       amplitude_z_range=(0.001, 0.15),
       n_orbits=50,
   )

延拓失败处理
------------

延拓过程中，某些步可能因微分修正不收敛而失败。默认行为是记录失败并继续：

.. code-block:: python

   # 延拓结果包含成功与失败的轨道
   family = continuation.natural_continuation(...)

   # 检查每条轨道的状态
   for orbit in family:
       if hasattr(orbit, 'converged') and not orbit.converged:
           print(f"步 {i} 未收敛")
