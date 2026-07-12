Halo 轨道
=========

Halo 轨道是围绕共线平动点（L1 或 L2）的三维周期轨道，从地球看呈"光环"状。

初始猜测
--------

Richardson 三阶解析近似为 Halo 轨道提供初始猜测：

.. code-block:: python

   from e2m2e.algorithms.halo_initial_guess import compute_halo_initial_guess

   # L1 Halo，z 方向振幅 0.01
   guess = compute_halo_initial_guess(mu=system.mu, z_A=0.01, L=1, halo_class=0)

   print(f"x0 = {guess['x0']}")
   print(f"vy0 = {guess['vy0']}")
   print(f"半周期 = {guess['T_half']}")

``halo_class`` 区分 Halo 族的分支：

- ``0``：南族 Halo（z0 < 0）
- ``1``：北族 Halo（z0 > 0）

``L`` 选择平动点：``1`` 为 L1，``2`` 为 L2。

微分修正
--------

用 Halo 专用策略将解析近似精化为精确周期轨道：

.. code-block:: python

   from e2m2e.algorithms import DifferentialCorrection
   from e2m2e.core import Orbit
   import numpy as np

   # 1. 组装初始状态
   initial_state = np.array([
       guess["x0"], 0.0, 0.01,
       0.0, guess["vy0"], 0.0,
   ])

   # 2. Halo 微分修正策略
   corrector = DifferentialCorrection(dynamics)
   corrector.setup_halo_orbit_fixed_z0(z0=0.01, libration_point=1)

   initial_guess = Orbit(
       states=initial_state.reshape(1, -1),
       times=np.array([0.0]),
       system=system,
   )
   initial_guess.period = guess["T_half"] * 2

   # 3. 执行修正
   halo = corrector.iterate_correction(initial_guess=initial_guess)
   if halo is not None:
       print(f"Halo 周期: {halo.period:.6f}")

轨道族生成
----------

从修正后的种子 Halo 出发，用延拓法生成一族 Halo 轨道：

.. code-block:: python

   from e2m2e.algorithms.halo_family import generate_halo_family

   family = generate_halo_family(
       system=system,
       dynamics=dynamics,
       libration_point=1,
       amplitude_z_range=(0.001, 0.15),
       n_orbits=50,
   )

   print(f"生成 {len(family)} 条 Halo 轨道")

NRHO（近直线晕轨道）
--------------------

高振幅 Halo 轨道呈现近直线形态（NRHO），是月球 Gateway 空间站的典型轨道构型。
通过增大 z 振幅（如 ``amplitude_z > 0.1``）即可生成。

参考
----

- Richardson D L. Analytical construction of a periodic solution about the collinear points[J]. *Celestial Mechanics*, 1980, 22(3): 303-320.
- Howell K C. Three-dimensional, periodic, 'halo' orbits in the restricted three-body problem[D]. Stanford University, 1983.
