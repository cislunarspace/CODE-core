快速入门
========

从零开始生成第一条周期轨道。

创建 CR3BP 系统
----------------

.. code-block:: python

   from e2m2e.algorithm.dynamics import CR3BP_System

   # 创建地月 CR3BP 系统
   system = CR3BP_System(
       mu=0.0121506683,
       primary="Earth",
       secondary="Moon",
   )._with_default_scales()

   # 计算平动点
   system.compute_libration_points()
   print(f"L1 = {system.L1}")
   print(f"L2 = {system.L2}")

   # 查看系统信息
   system.info()

生成 DRO 轨道族
-----------------

以下示例从种子轨道出发，通过微分修正 + 延拓生成 DRO 轨道族：

.. code-block:: python

   from e2m2e.algorithm.dynamics import CR3BP_System, CR3BP_Dynamics
   from e2m2e.data.types.orbit import Orbit
   from e2m2e.algorithm.solver import DifferentialCorrection, Continuation
   import numpy as np

   # 1. 创建系统与动力学
   system = CR3BP_System(
       mu=0.0121506683, primary="Earth", secondary="Moon"
   )._with_default_scales()
   system.compute_libration_points()
   dynamics = CR3BP_Dynamics(system)

   # 2. 种子轨道（DRO 初始猜测）
   initial_state = [0.79188556619742, 0.0, 0.0, 0.0, 0.53682, 0.0]
   seed_orbit = Orbit(states=[initial_state], times=[0], system=system)

   # 3. 微分修正：固定 x0 的 2D 对称策略
   corrector = DifferentialCorrection(dynamics)
   corrector.setup_2D_symmetric_x_fixed_x0(x0=initial_state[0])
   seed_dro = corrector.iterate_correction(initial_guess=seed_orbit)

   if seed_dro is not None:
       print(f"修正成功，周期 = {seed_dro.period:.6f}")

   # 4. 延拓生成轨道族
   continuation = Continuation(corrector=corrector)
   family = continuation.natural_continuation(
       seed_orbit=seed_dro,
       param_range=(0.14, 0.9),
       step_size=0.005,
   )
   print(f"轨道族包含 {len(family)} 条轨道")

生成 Halo 轨道
---------------

.. code-block:: python

   from e2m2e.algorithm.family.halo_initial_guess import compute_halo_initial_guess

   # Richardson 三阶解析近似生成初始猜测
   z0 = 0.01  # z 方向振幅
   guess = compute_halo_initial_guess(system.mu, z0, L=1, halo_class=0)

   initial_state = np.array([
       guess["x0"], 0.0, z0,
       0.0, guess["vy0"], 0.0,
   ])

   # Halo 微分修正策略
   corrector = DifferentialCorrection(dynamics)
   corrector.setup_halo_orbit_fixed_z0(z0=z0, libration_point=1)

   initial_guess = Orbit(
       states=initial_state.reshape(1, -1),
       times=np.array([0.0]),
       system=system,
   )
   initial_guess.period = guess["T_half"] * 2

   halo = corrector.iterate_correction(initial_guess=initial_guess)
   if halo is not None:
       print(f"Halo 周期: {halo.period:.6f}")

多重打靶
--------

沿用上一节微分修正得到的 ``seed_dro`` 作为打靶初值，因此以下示例
以 ``seed_dro`` 修正成功（不为 ``None``）为前提：

.. code-block:: python

   from e2m2e.algorithm.solver import MultipleShooting, sample_patch_points

   ms = MultipleShooting(dynamics=dynamics)
   t_patch, state_patch = sample_patch_points(seed_dro, n_points=5)

   result = ms.correct(
       t_patch=t_patch,
       state_patch=state_patch,
       max_iter=50,
       tolerance=1e-10,
       var_time=True,
   )

   if result.converged:
       print(f"收敛，最大残差 {result.max_residual:.2e}")

转移轨道设计
------------

.. code-block:: python

   from e2m2e.algorithm.transfer import Transfer

   transfer = Transfer(dynamics)
   result = transfer.set_orbit(start=dro_orbit, end=ro_orbit).optimize(
       initial_guess={"alpha": 1.0, "transfer_time": 15.0, "t_ins": 5.0},
       alpha_range=(0.5, 2.5),
   )

力模型组合（星历传播）
-----------------------

.. code-block:: python

   from e2m2e.algorithm.coordinate import (
       CelestialBodyOrigin,
       CoordinateSystem,
       ICRSAxes,
   )
   from e2m2e.algorithm.dynamics import EphemerisSystem
   from e2m2e.data.kernels.manager import SPICEManager
   from e2m2e.algorithm.forces import ForceModel, GravityField, DragModel
   from e2m2e.algorithm.forces.atmosphere import ExponentialAtmosphere

   # 加载 SPICE 内核
   spice = SPICEManager()
   spice.load_kernel("kernels/de440s.bsp")

   # 创建星历系统（frame 默认为 J2000）
   eph_system = EphemerisSystem(
       bodies=["EARTH", "MOON", "SUN"],
       spice=spice, origin="EARTH",
   )

   # ForceModel 要求系统已设置坐标系
   eph_system.coordinate_system = CoordinateSystem(
       axes=ICRSAxes(), origin=CelestialBodyOrigin(body="EARTH", spice=spice)
   )

   # 组合力模型
   fm = ForceModel(eph_system)
   fm.add_force(GravityField("EARTH", degree=2, order=0), name="j2")
   fm.add_force(
       DragModel(atmosphere=ExponentialAtmosphere(), area=10.0, mass=1000.0),
       name="drag",
   )

   # 传播 LEO 轨道
   r = 6378.137 + 400.0
   v = np.sqrt(398600.4415 / r)
   state0 = np.array([r, 0.0, 0.0, 0.0, v, 0.0])

   et0 = spice.utc_to_et("2025-06-21T11:00:06")
   result = fm.propagate(state0, (et0, et0 + 86400.0))

可视化
-------

.. code-block:: python

   from e2m2e.tools.viz import PlotConfig, FamilyPlotter

   config = PlotConfig(title=32, label=28)
   config.apply_rcparams()

   plotter = FamilyPlotter(system, config)
   plotter.plot_family_2d(family, jacobi_values, title="DRO Family")

下一步
------

- :doc:`../core/system` — 系统与平动点详解
- :doc:`../core/dynamics` — 动力学与传播
- :doc:`../core/forces` — 力模型组合
- :doc:`../algorithms/differential-correction` — 微分修正策略
- :doc:`../transfer/overview` — 转移轨道设计

