Halo Initial Guess / Halo 初始猜测
==================================

[English](#halo-initial-guess) | [简体中文](#中文)

English
-------

:mod:`e2m2e.algorithm.family.halo_initial_guess` provides Richardson's
third-order analytic approximation for Halo initial-guess parameters.

Richardson third-order approximation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Richardson (1980) gives a third-order analytic solution for Halos near collinear
points:

.. code-block:: python

   from e2m2e.algorithm.family.halo_initial_guess import compute_halo_initial_guess

   guess = compute_halo_initial_guess(mu=0.01215, z_amplitude=0.001, L=1, halo_class=0)

   print(f"x0 = {guess['x0']}")      # initial x position
   print(f"vy0 = {guess['vy0']}")    # initial y velocity
   print(f"T_half = {guess['T_half']}")  # half-period

Parameters:

- ``mu``: CR3BP mass parameter
- ``z_amplitude``: z amplitude (nondimensional)
- ``L``: libration point (1=L1, 2=L2)
- ``halo_class``: branch (0=northern, 1=southern)

Internal solve steps
~~~~~~~~~~~~~~~~~~~~

1. Solve γ — libration-point-to-secondary distance parameter (quintic via Brent)
2. Compute planar oscillation frequency ω_p
3. Compute Richardson third-order coefficients (A, B, C, D…)
4. Assemble the analytic solution into initial-state parameters

Pairing with differential correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The analytic guess feeds differential correction:

.. code-block:: python

   from e2m2e.algorithm.family.halo_initial_guess import compute_halo_initial_guess
   from e2m2e.algorithm.solver import DifferentialCorrection
   from e2m2e.data.types.orbit import Orbit
   import numpy as np

   # 1. Analytic approximation (small-amplitude seeds suit its accuracy)
   guess = compute_halo_initial_guess(mu=system.mu, z_amplitude=0.001, L=1, halo_class=0)

   # 2. Assemble initial state
   initial_state = np.array([
       guess["x0"], 0.0, 0.001,
       0.0, guess["vy0"], 0.0,
   ])

   # 3. Differential correction
   corrector = DifferentialCorrection(dynamics)
   corrector.setup_halo_orbit_fixed_z0(z0=0.001, libration_point=1)

   initial_guess = Orbit(
       states=initial_state.reshape(1, -1),
       times=np.array([0.0]),
       system=system,
   )
   initial_guess.period = guess["T_half"] * 2

   halo_result = corrector.iterate_correction(initial_guess=initial_guess)
   halo = halo_result.orbit  # corrected orbit (None = failure)

Reference
~~~~~~~~~

- Richardson D L. Analytical construction of a periodic solution about the collinear points[J]. *Celestial Mechanics*, 1980, 22(3): 303-320.

中文
----

:mod:`e2m2e.algorithm.family.halo_initial_guess` 模块提供 Richardson 三阶解析近似，
用于生成 Halo 轨道的初始猜测参数。

Richardson 三阶近似
~~~~~~~~~~~~~~~~~~~~

Richardson (1980) 给出共线平动点附近 Halo 轨道的三阶解析解：

.. code-block:: python

   from e2m2e.algorithm.family.halo_initial_guess import compute_halo_initial_guess

   guess = compute_halo_initial_guess(mu=0.01215, z_amplitude=0.001, L=1, halo_class=0)

   print(f"x0 = {guess['x0']}")      # x 方向初始位置
   print(f"vy0 = {guess['vy0']}")    # y 方向初始速度
   print(f"T_half = {guess['T_half']}")  # 半周期

参数说明：

- ``mu`` ：CR3BP 质量参数
- ``z_amplitude`` ：z 方向振幅（无量纲）
- ``L`` ：平动点编号（1=L1，2=L2）
- ``halo_class`` ：Halo 族分支（0=北族，1=南族）

内部求解过程
~~~~~~~~~~~~

1. 求解平动点到次天体的距离参数 γ（五次方程，Brent 方法）
2. 计算面内振荡频率 ω_p
3. 计算 Richardson 三阶系数（A、B、C、D 等）
4. 组装三阶解析解，返回初始状态参数

与微分修正的配合
~~~~~~~~~~~~~~~~

解析近似作为微分修正的初始猜测：

.. code-block:: python

   from e2m2e.algorithm.family.halo_initial_guess import compute_halo_initial_guess
   from e2m2e.algorithm.solver import DifferentialCorrection
   from e2m2e.data.types.orbit import Orbit
   import numpy as np

   # 1. 解析近似（小振幅种子，Richardson 近似精度高）
   guess = compute_halo_initial_guess(mu=system.mu, z_amplitude=0.001, L=1, halo_class=0)

   # 2. 组装初始状态
   initial_state = np.array([
       guess["x0"], 0.0, 0.001,
       0.0, guess["vy0"], 0.0,
   ])

   # 3. 微分修正
   corrector = DifferentialCorrection(dynamics)
   corrector.setup_halo_orbit_fixed_z0(z0=0.001, libration_point=1)

   initial_guess = Orbit(
       states=initial_state.reshape(1, -1),
       times=np.array([0.0]),
       system=system,
   )
   initial_guess.period = guess["T_half"] * 2

   halo_result = corrector.iterate_correction(initial_guess=initial_guess)
   halo = halo_result.orbit  # 修正后的轨道（None 表示失败）

参考
~~~~

- Richardson D L. Analytical construction of a periodic solution about the collinear points[J]. *Celestial Mechanics*, 1980, 22(3): 303-320.
