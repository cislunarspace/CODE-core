Quick Start / 快速入门
======================

[English](#quick-start) | [简体中文](#简体中文)

English
-------

Generate your first periodic orbit from scratch.

Create a CR3BP system
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from e2m2e.algorithm.dynamics import CR3BP_System
   from e2m2e.data.constants import Datum

   # Create the Earth-Moon CR3BP system (μ from the DE421 datum, ADR 0022)
   system = CR3BP_System(
       mu=Datum.DE421.mu,
       primary="Earth",
       secondary="Moon",
   )._with_default_scales()

   # Compute libration points
   system.compute_libration_points()
   print(f"L1 = {system.L1}")
   print(f"L2 = {system.L2}")

   # Print system info
   system.info()

Generate a DRO family
~~~~~~~~~~~~~~~~~~~~~

This example grows a DRO family from a seed orbit via differential correction +
continuation:

.. code-block:: python

   from e2m2e.algorithm.dynamics import CR3BP_System, CR3BP_Dynamics
   from e2m2e.data.constants import Datum
   from e2m2e.data.types.orbit import Orbit
   from e2m2e.algorithm.solver import DifferentialCorrection, Continuation
   import numpy as np

   # 1. Create system and dynamics (μ from DE421, ADR 0022)
   system = CR3BP_System(
       mu=Datum.DE421.mu, primary="Earth", secondary="Moon"
   )._with_default_scales()
   system.compute_libration_points()
   dynamics = CR3BP_Dynamics(system)

   # 2. Seed orbit (DRO initial guess)
   initial_state = [0.79188556619742, 0.0, 0.0, 0.0, 0.53682, 0.0]
   seed_orbit = Orbit(states=[initial_state], times=[0], system=system)
   seed_orbit.period = 6.307  # Period guess (TU), refined during correction

   # 3. Differential correction: 2D symmetric strategy with x0 fixed
   corrector = DifferentialCorrection(dynamics)
   corrector.setup_2D_symmetric_x_fixed_x0(x0=initial_state[0])
   result = corrector.iterate_correction(initial_guess=seed_orbit)
   seed_dro = result.orbit  # Corrected orbit (None means failure)

   if seed_dro is not None:
       print(f"Correction succeeded, period = {seed_dro.period:.6f}")

   # 4. Continue into a family
   continuation = Continuation(corrector=corrector)
   cont_result = continuation.natural_continuation(
       seed_orbit=seed_dro,
       param_range=(0.14, 0.9),
       step_size=0.005,
   )
   print(f"Family contains {len(cont_result.family.orbits)} orbits")

Generate a Halo orbit
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from e2m2e.algorithm.family.halo_initial_guess import compute_halo_initial_guess

   # Richardson third-order analytic approximation for the initial guess
   z0 = 0.001  # z amplitude (small-amplitude seeds suit Richardson's accuracy)
   guess = compute_halo_initial_guess(system.mu, z0, L=1, halo_class=0)

   initial_state = np.array([
       guess["x0"], 0.0, z0,
       0.0, guess["vy0"], 0.0,
   ])

   # Halo differential-correction strategy
   corrector = DifferentialCorrection(dynamics)
   corrector.setup_halo_orbit_fixed_z0(z0=z0, libration_point=1)

   initial_guess = Orbit(
       states=initial_state.reshape(1, -1),
       times=np.array([0.0]),
       system=system,
   )
   initial_guess.period = guess["T_half"] * 2

   halo_result = corrector.iterate_correction(initial_guess=initial_guess)
   halo = halo_result.orbit
   if halo is not None:
       print(f"Halo period: {halo.period:.6f}")

Multiple shooting
~~~~~~~~~~~~~~~~~

Reuses ``seed_dro`` — corrected in the previous section — as the shooting
initial value; these examples assume `seed_dro` corrected successfully (not
``None``):

.. code-block:: python

   from e2m2e.algorithm.solver import MultipleShooting, sample_patch_points
   from e2m2e.data.templates import ConvergenceState

   ms = MultipleShooting(dynamics=dynamics)
   t_patch, state_patch = sample_patch_points(seed_dro, n_points=5)

   result = ms.correct(
       t_patch=t_patch,
       state_patch=state_patch,
       max_iter=50,
       tolerance=1e-10,
       var_time=True,
   )

   if result.status == ConvergenceState.CONVERGED:
       print(f"Converged, max residual {result.max_residual:.2e}")

Transfer design
~~~~~~~~~~~~~~~

.. code-block:: python

   from e2m2e.algorithm.transfer import Transfer

   transfer = Transfer(dynamics)
   # Departure orbit: seed_dro from above; arrival: last family member as the target initial value
   result = transfer.set_orbit(
       start=seed_dro, end=cont_result.family.orbits[-1]
   ).optimize(
       initial_guess={"alpha": 1.0, "transfer_time": 15.0, "t_ins": 5.0},
       alpha_range=(0.5, 2.5),
   )

Force models (ephemeris propagation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

   # Load SPICE kernels
   spice = SPICEManager()
   spice.load_kernel("kernels/de440s.bsp")

   # Create an ephemeris system (frame defaults to J2000)
   eph_system = EphemerisSystem(
       bodies=["EARTH", "MOON", "SUN"],
       spice=spice, origin="EARTH",
   )

   # ForceModel requires the coordinate system to be set on the system
   eph_system.coordinate_system = CoordinateSystem(
       axes=ICRSAxes(), origin=CelestialBodyOrigin(body="EARTH", spice=spice)
   )

   # Assemble force models
   fm = ForceModel(eph_system)
   fm.add_force(GravityField("EARTH", degree=2, order=0), name="j2")
   fm.add_force(
       DragModel(atmosphere=ExponentialAtmosphere(), area=10.0, mass=1000.0),
       name="drag",
   )

   # Propagate a LEO orbit
   r = 6378.137 + 400.0
   v = np.sqrt(398600.4415 / r)
   state0 = np.array([r, 0.0, 0.0, 0.0, v, 0.0])

   et0 = spice.utc_to_et("2025-06-21T11:00:06")
   result = fm.propagate(state0, (et0, et0 + 86400.0))

Next steps
~~~~~~~~~~

- :doc:`../core/system`: systems & libration points in depth
- :doc:`../core/dynamics`: dynamics & propagation
- :doc:`../core/forces`: combining force models
- :doc:`../algorithms/differential-correction`: correction strategies
- :doc:`../transfer/overview`: transfer design

简体中文
--------

从零开始生成第一条周期轨道。

创建 CR3BP 系统
~~~~~~~~~~~~~~~

.. code-block:: python

   from e2m2e.algorithm.dynamics import CR3BP_System
   from e2m2e.data.constants import Datum

   # 创建地月 CR3BP 系统（μ 取 DE421 基准，ADR 0022）
   system = CR3BP_System(
       mu=Datum.DE421.mu,
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
~~~~~~~~~~~~~~~

以下示例从种子轨道出发，通过微分修正 + 延拓生成 DRO 轨道族：

.. code-block:: python

   from e2m2e.algorithm.dynamics import CR3BP_System, CR3BP_Dynamics
   from e2m2e.data.constants import Datum
   from e2m2e.data.types.orbit import Orbit
   from e2m2e.algorithm.solver import DifferentialCorrection, Continuation
   import numpy as np

   # 1. 创建系统与动力学（μ 取 DE421 基准，ADR 0022）
   system = CR3BP_System(
       mu=Datum.DE421.mu, primary="Earth", secondary="Moon"
   )._with_default_scales()
   system.compute_libration_points()
   dynamics = CR3BP_Dynamics(system)

   # 2. 种子轨道（DRO 初始猜测）
   initial_state = [0.79188556619742, 0.0, 0.0, 0.0, 0.53682, 0.0]
   seed_orbit = Orbit(states=[initial_state], times=[0], system=system)
   seed_orbit.period = 6.307  # 周期初猜（TU），修正中迭代更新

   # 3. 微分修正：固定 x0 的 2D 对称策略
   corrector = DifferentialCorrection(dynamics)
   corrector.setup_2D_symmetric_x_fixed_x0(x0=initial_state[0])
   result = corrector.iterate_correction(initial_guess=seed_orbit)
   seed_dro = result.orbit  # 修正后的轨道（None 表示失败）

   if seed_dro is not None:
       print(f"修正成功，周期 = {seed_dro.period:.6f}")

   # 4. 延拓生成轨道族
   continuation = Continuation(corrector=corrector)
   cont_result = continuation.natural_continuation(
       seed_orbit=seed_dro,
       param_range=(0.14, 0.9),
       step_size=0.005,
   )
   print(f"轨道族包含 {len(cont_result.family.orbits)} 条轨道")

生成 Halo 轨道
~~~~~~~~~~~~~~~

.. code-block:: python

   from e2m2e.algorithm.family.halo_initial_guess import compute_halo_initial_guess

   # Richardson 三阶解析近似生成初始猜测
   z0 = 0.001  # z 方向振幅（小振幅种子，Richardson 近似精度高）
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

   halo_result = corrector.iterate_correction(initial_guess=initial_guess)
   halo = halo_result.orbit
   if halo is not None:
       print(f"Halo 周期: {halo.period:.6f}")

多重打靶
~~~~~~~~

沿用上一节微分修正得到的 ``seed_dro`` 作为打靶初值，因此以下示例
以 ``seed_dro`` 修正成功（不为 ``None`` ）为前提：

.. code-block:: python

   from e2m2e.algorithm.solver import MultipleShooting, sample_patch_points
   from e2m2e.data.templates import ConvergenceState

   ms = MultipleShooting(dynamics=dynamics)
   t_patch, state_patch = sample_patch_points(seed_dro, n_points=5)

   result = ms.correct(
       t_patch=t_patch,
       state_patch=state_patch,
       max_iter=50,
       tolerance=1e-10,
       var_time=True,
   )

   if result.status == ConvergenceState.CONVERGED:
       print(f"收敛，最大残差 {result.max_residual:.2e}")

转移轨道设计
~~~~~~~~~~~~

.. code-block:: python

   from e2m2e.algorithm.transfer import Transfer

   transfer = Transfer(dynamics)
   # 出发轨道用前文微分修正得到的 seed_dro，到达轨道取延拓族末端成员作初值
   result = transfer.set_orbit(
       start=seed_dro, end=cont_result.family.orbits[-1]
   ).optimize(
       initial_guess={"alpha": 1.0, "transfer_time": 15.0, "t_ins": 5.0},
       alpha_range=(0.5, 2.5),
   )

力模型组合（星历传播）
~~~~~~~~~~~~~~~~~~~~~~~

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

下一步
~~~~~~

- :doc:`../core/system`：系统与平动点详解
- :doc:`../core/dynamics`：动力学与传播
- :doc:`../core/forces`：力模型组合
- :doc:`../algorithms/differential-correction`：微分修正策略
- :doc:`../transfer/overview`：转移轨道设计
