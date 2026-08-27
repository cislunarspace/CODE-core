Differential Correction Strategies / 微分修正策略
=================================================

[English](#differential-correction-strategies) | [简体中文](#中文)

English
-------

Differential Correction iteratively adjusts initial conditions until the orbit
satisfies periodicity constraints. The strategy layer separates configuration
from the iterative solver: each strategy only produces an immutable
``CorrectionConfig``; ``DifferentialCorrection`` hands the config to the Rust
CR3BP correction kernel for STM Newton iterations, orchestrating result orbits
on the Python side.

Strategy overview
~~~~~~~~~~~~~~~~~

.. list-table:: Strategy comparison
   :header-rows: 1
   :widths: 20 15 20 25 20

   * - Strategy
     - Orbits
     - Symmetry
     - Fixed params
     - Free variables
   * - ``symmetric_2d_fixed_x0``
     - Lyapunov, DRO
     - x axis
     - x0
     - y_dot0, T_half
   * - ``symmetric_2d_fixed_t``
     - Fixed-period orbits
     - x axis
     - T_half
     - x0, y_dot0
   * - ``symmetric_2d_fixed_y0``
     - Resonant orbits (RO)
     - y axis
     - y0
     - x_dot0, T_half
   * - ``symmetric_3d_fixed_x0``
     - Spatial periodic orbits
     - x axis
     - x0
     - z0, y_dot0, T_half
   * - ``symmetric_xz_fixed_x0``
     - XZ-symmetric orbits
     - XZ plane
     - x0
     - z0, y_dot0, T_half
   * - ``symmetric_xz_fixed_z0``
     - XZ-symmetric orbits
     - XZ plane
     - z0
     - x0, y_dot0, T_half
   * - ``halo_fixed_z0``
     - Halos (N/S branches)
     - XZ plane
     - z0, libration point
     - x0, y_dot0, T_half
   * - ``halo_fixed_x0``
     - Halos (N/S branches)
     - XZ plane
     - x0, libration point
     - z0, y_dot0, T_half
   * - ``axial_fixed_vz0``
     - Axial (Gómez Type B)
     - x axis
     - vz0, libration point
     - x0, y_dot0, T_half
   * - ``spo_fixed_x0``
     - L4/L5 SPO
     - none (full-period closure)
     - x0
     - y0, vx0, vy0, T_full
   * - ``lpo_fixed_x0``
     - L4/L5 LPO
     - none (full-period closure)
     - x0
     - y0, vx0, vy0, T_full

Selection guide
~~~~~~~~~~~~~~~

- Planar Lyapunov (near L1/L2/L3): ``symmetric_2d_fixed_x0``, adjusting y_dot0 & half-period.
- Fixed period (e.g., DRO): ``symmetric_2d_fixed_t``, adjusting x0 & y_dot0.
- Resonant (departing the y axis): ``symmetric_2d_fixed_y0``, adjusting x_dot0 & half-period.
- Halo (3-D, XZ-symmetric): ``halo_fixed_z0`` or ``halo_fixed_x0``.
- Axial (L1/L2, Type B, departing xy plane): ``axial_fixed_vz0``, adjusting x0/y_dot0/half-period.
- L4/L5 SPO/LPO (planar, no symmetry): ``spo_fixed_x0`` / ``lpo_fixed_x0``, full-period closure.
- General spatial orbits: ``symmetric_3d_fixed_x0`` or ``symmetric_xz_fixed_*``.

Symmetry principles
~~~~~~~~~~~~~~~~~~~

CR3BP dynamics carries these symmetries in the rotating frame:

- **x-axis symmetry**: if (x, y, z, x_dot, y_dot, z_dot) is a solution, so is
  (x, -y, -z, -x_dot, y_dot, -z_dot). Periodic orbits can depart perpendicular
  to x (y=0, x_dot=0), recrossing perpendicular after half a period.

- **XZ-plane symmetry**: mirror across y → (x, -y, z, -x_dot, y_dot, -z_dot).
  Halos depart on the XZ plane (y=0), returning to it half a period later.

- **y-axis symmetry**: (-x, y, -z, x_dot, -y_dot, z_dot). Resonants depart the y
  axis (x=0, x_dot=0), recrossing it after half a period.

Code examples
~~~~~~~~~~~~~

Symmetric 2D (planar Lyapunov)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from e2m2e.algorithm.dynamics import CR3BP_System
   from e2m2e.algorithm.dynamics.dynamics import CR3BP_Dynamics
   from e2m2e.data.constants import Datum
   from e2m2e.data.types.orbit import Orbit
   from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection
   import numpy as np

   # 1. Earth-Moon system (μ from DE421, ADR 0022)
   system = CR3BP_System(
       mu=Datum.DE421.mu, primary="Earth", secondary="Moon"
   )._with_default_scales()
   system.set_characteristic_scales(384400, 27.32 * 86400)
   system.compute_libration_points()

   # 2. Dynamics object
   dynamics = CR3BP_Dynamics(system)

   # 3. Strategy: 2D symmetric with x0 fixed (Lyapunov)
   x0 = system.L1[0] + 0.01  # near L1
   corrector = DifferentialCorrection(dynamics)
   corrector.setup_2D_symmetric_x_fixed_x0(x0=x0)

   # 4. Initial guess (perpendicular departure from the x axis)
   initial_state = np.array([x0, 0.0, 0.0, 0.0, 0.15, 0.0])
   initial_guess = Orbit(
       states=initial_state.reshape(1, -1),
       times=np.array([0.0]),
       system=system,
   )
   initial_guess.period = 3.0  # period guess (nondimensional)

   # 5. Correct
   result = corrector.iterate_correction(
       initial_guess=initial_guess, verbose=False
   )
   orbit = result.orbit  # corrected orbit (None = failure)

   if orbit is not None:
       print(f"Converged: period={orbit.period:.4f}, family={orbit.family_type}")
       print(f"Initial state: {orbit.states[0]}")
   else:
       print(f"Correction failed: {result.message}")

Symmetric 2D (fixed-period DRO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Fix half-period; adjust x0 and y_dot0
   target_period = 2.8
   corrector = DifferentialCorrection(dynamics)
   corrector.setup_2D_symmetric_x_fixed_t(t_half=target_period / 2)

   initial_state = np.array([0.5, 0.0, 0.0, 0.0, 0.1, 0.0])
   initial_guess = Orbit(
       states=initial_state.reshape(1, -1),
       times=np.array([0.0]),
       system=system,
   )
   initial_guess.period = target_period

   result = corrector.iterate_correction(initial_guess=initial_guess)
   orbit = result.orbit
   if orbit is not None:
       print(f"DRO period: {orbit.period:.4f} (target: {target_period})")

Symmetric 3D (spatial periodic)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Fix x0; adjust z0, y_dot0, T_half
   x0 = system.L1[0] + 0.01
   corrector = DifferentialCorrection(dynamics)
   corrector.setup_3D_symmetric_x_fixed_x0(x0=x0)

   initial_state = np.array([x0, 0.0, 0.005, 0.0, 0.05, 0.0])
   initial_guess = Orbit(
       states=initial_state.reshape(1, -1),
       times=np.array([0.0]),
       system=system,
   )
   initial_guess.period = 3.0

   result = corrector.iterate_correction(initial_guess=initial_guess)
   orbit = result.orbit
   if orbit is not None:
       print(f"3D orbit period: {orbit.period:.4f}")

Halo (fixed z amplitude)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from e2m2e.algorithm.family.halo_initial_guess import compute_halo_initial_guess

   # 1. Richardson third-order guess
   mu = system.mu
   z0 = 0.001  # small-amplitude seed suits Richardson's accuracy
   guess = compute_halo_initial_guess(mu, z0, L=1, halo_class=0)

   # 2. Initial state (northern Halo: z0 > 0)
   initial_state = np.array([
       guess["x0"], 0.0, z0,
       0.0, guess["vy0"], 0.0,
   ])

   # 3. Halo strategy (fixed z0)
   corrector = DifferentialCorrection(dynamics)
   corrector.setup_halo_orbit_fixed_z0(z0=z0, libration_point=1)

   initial_guess = Orbit(
       states=initial_state.reshape(1, -1),
       times=np.array([0.0]),
       system=system,
   )
   initial_guess.period = guess["T_half"] * 2

   # 4. Correct
   result = corrector.iterate_correction(initial_guess=initial_guess)
   orbit = result.orbit
   if orbit is not None:
       print(f"Halo period: {orbit.period:.4f}")
       print(f"Jacobi constant: {system.get_jacobi_constant(orbit.states[0]):.6f}")

How strategies meet correctors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   ┌─────────────────┐     CorrectionConfig      ┌─────────────────────┐
   │  Strategy Func  │ ──────────────────────────> │ DifferentialCorrection│
   │ (e.g. halo_*)   │    (immutable config)       │   ._apply_config()    │
   └─────────────────┘                             └─────────────────────┘
                                                          │
                                                          │ iterate_correction()
                                                          ▼
                                                   ┌──────────────┐
                                                   │   Orbit      │
                                                   │ (periodic)   │
                                                   └──────────────┘

Strategy functions only produce configs, never touching states;
``DifferentialCorrection`` forwards STM Newton iteration to the Rust CR3BP
kernel, keeping problem construction / results / ``Orbit`` orchestration in
Python. This separation means:

- New strategies need no solver changes
- One strategy serves different correctors
- Configs serialize, compare, test

Calling strategies directly (advanced)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For custom flows, call strategy functions for a ``CorrectionConfig``:

.. code-block:: python

   from e2m2e.algorithm.family.strategies import (
       symmetric_2d_fixed_x0,
       symmetric_3d_fixed_x0,
       halo_fixed_z0,
   )

   config = halo_fixed_z0(z0=0.001, libration_point=1)
   print(config.setup_type)          # "halo_orbit_fixed_z0"
   print(config.free_variables)      # ["x0", "y_dot0", "T_half"]
   print(config.target_conditions)   # {"y": 0.0, "x_dot": 0.0, "z_dot": 0.0}

   # Apply manually
   corrector = DifferentialCorrection(dynamics)
   corrector._apply_config(config)

``CorrectionConfig`` is frozen (`frozen=True`): all fields immutable past
creation — well-suited as contract objects between functions.

Convergence diagnostics
~~~~~~~~~~~~~~~~~~~~~~~

After failures, diagnose via:

.. code-block:: python

   corrector.iterate_correction(initial_guess=initial_guess)

   history = corrector.get_convergence_history()
   print(history["errors"])           # per-iteration residual norms
   print(history["iterations"])       # total iterations
   print(history["status"])           # final status
   print(history["cause"])            # termination reason code
   print(history["message"])           # human-readable explanation

Common causes: normal convergence / corrections below machine precision /
divergence (poor guess) / singular Jacobian (constraints linearly dependent —
change strategy) / stagnation (local minimum — change guess) / convergence to a
parasitic root (T≈0 — adjust guess).

References
~~~~~~~~~~

- Broucke R A. Periodic orbits in the restricted three body problem with Earth-moon masses[R]. 1968.
- Richardson D L. Analytical construction of a periodic solution about the collinear points[J]. Celestial Mechanics, 1980.
- Howell K C. Three-dimensional, periodic, 'halo' orbits in the restricted three-body problem[D]. Stanford University, 1983.

中文
----

微分修正（Differential Correction）通过迭代调整初始条件，使轨道满足周期性约束。
策略层将配置逻辑与迭代求解器分离：每种策略只负责生成不可变的 ``CorrectionConfig`` ，
``DifferentialCorrection`` 负责将配置交给 Rust CR3BP 微分修正内核执行 STM
Newton 迭代，并在 Python 侧编排结果轨道。

策略概览
~~~~~~~~

.. list-table:: 策略对照表
   :header-rows: 1
   :widths: 20 15 20 25 20

   * - 策略
     - 适用轨道
     - 对称性
     - 固定参数
     - 自由变量
   * - ``symmetric_2d_fixed_x0``
     - Lyapunov、DRO
     - x 轴
     - x0
     - y_dot0, T_half
   * - ``symmetric_2d_fixed_t``
     - 固定周期轨道
     - x 轴
     - T_half
     - x0, y_dot0
   * - ``symmetric_2d_fixed_y0``
     - 共振轨道（RO）
     - y 轴
     - y0
     - x_dot0, T_half
   * - ``symmetric_3d_fixed_x0``
     - 空间周期轨道
     - x 轴
     - x0
     - z0, y_dot0, T_half
   * - ``symmetric_xz_fixed_x0``
     - XZ 对称轨道
     - XZ 平面
     - x0
     - z0, y_dot0, T_half
   * - ``symmetric_xz_fixed_z0``
     - XZ 对称轨道
     - XZ 平面
     - z0
     - x0, y_dot0, T_half
   * - ``halo_fixed_z0``
     - Halo 轨道（北/南族）
     - XZ 平面
     - z0, 平动点
     - x0, y_dot0, T_half
   * - ``halo_fixed_x0``
     - Halo 轨道（北/南族）
     - XZ 平面
     - x0, 平动点
     - z0, y_dot0, T_half
   * - ``axial_fixed_vz0``
     - Axial（Gómez Type B 分岔）
     - x 轴
     - vz0, 平动点
     - x0, y_dot0, T_half
   * - ``spo_fixed_x0``
     - L4/L5 短周期（SPO）
     - 无（全周期闭合）
     - x0
     - y0, vx0, vy0, T_full
   * - ``lpo_fixed_x0``
     - L4/L5 长周期（LPO）
     - 无（全周期闭合）
     - x0
     - y0, vx0, vy0, T_full

策略选择指南
~~~~~~~~~~~~

根据轨道类型选择策略：

- 平面 Lyapunov 轨道（L1/L2/L3 附近）：使用 ``symmetric_2d_fixed_x0`` 固定 x0，调整 y_dot0 和半周期。
- 固定周期轨道（如 DRO）：使用 ``symmetric_2d_fixed_t`` 固定半周期，调整 x0 和 y_dot0。
- 共振轨道（从 y 轴出发）：使用 ``symmetric_2d_fixed_y0`` 固定 y0，调整 x_dot0 和半周期。
- Halo 轨道（三维，XZ 平面对称）：使用 ``halo_fixed_z0`` 固定 z 振幅，或 ``halo_fixed_x0`` 固定 x 坐标。
- Axial 轨道（L1/L2，Gómez Type B 分岔，xy 平面出发）：使用 ``axial_fixed_vz0`` 固定 vz0，调整 x0、y_dot0 和半周期。
- L4/L5 短周期 / 长周期（SPO/LPO，平面无对称）：使用 ``spo_fixed_x0`` / ``lpo_fixed_x0`` 固定 x0，全周期闭合。
- 一般空间周期轨道：使用 ``symmetric_3d_fixed_x0`` 或 ``symmetric_xz_fixed_*`` 变体。

对称性原理
~~~~~~~~~~

CR3BP 动力学在旋转坐标系中具有以下对称性：

- **x 轴对称**：若 (x, y, z, x_dot, y_dot, z_dot) 是解，则 (x, -y, -z, -x_dot, y_dot, -z_dot) 也是解。
  利用此对称性，周期轨道可从 x 轴垂直出发（y=0, x_dot=0），半周期后再次垂直穿越 x 轴。

- **XZ 平面对称**：若 (x, y, z, x_dot, y_dot, z_dot) 是解，则 (x, -y, z, -x_dot, y_dot, -z_dot) 也是解。
  Halo 轨道利用此对称性，从 XZ 平面出发（y=0），半周期后再次到达 XZ 平面。

- **y 轴对称**：若 (x, y, z, x_dot, y_dot, z_dot) 是解，则 (-x, y, -z, x_dot, -y_dot, z_dot) 也是解。
  共振轨道利用此对称性，从 y 轴出发（x=0, x_dot=0），半周期后再次穿越 y 轴。

代码示例
~~~~~~~~

对称 2D（平面 Lyapunov 轨道）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

（见上方英文节示例：固定 x0 的 2D 对称修正。）

对称 2D（固定周期 DRO）
^^^^^^^^^^^^^^^^^^^^^^^^

固定半周期，调整 x0 和 y_dot0；对称 3D 与 Halo 示例同样见英文节。

策略与修正器的协作关系
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   ┌─────────────────┐     CorrectionConfig      ┌─────────────────────┐
   │  Strategy Func  │ ──────────────────────────> │ DifferentialCorrection│
   │ (e.g. halo_*)   │    (immutable config)       │   ._apply_config()    │
   └─────────────────┘                             └─────────────────────┘
                                                          │
                                                          │ iterate_correction()
                                                          ▼
                                                   ┌──────────────┐
                                                   │   Orbit      │
                                                   │ (periodic)   │
                                                   └──────────────┘

策略函数（如 ``halo_fixed_z0`` ）只负责生成配置，不直接操作状态；
``DifferentialCorrection`` 接收配置后，将 STM Newton 迭代交给 Rust CR3BP
内核执行，Python 侧只保留问题构造、结果和 ``Orbit`` 编排。这种分离使得：

- 新增策略无需修改迭代器代码
- 同一策略可被不同修正器复用
- 配置可序列化、可对比、可测试

直接调用策略函数（高级用法）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

若需要自定义修正流程，可直接调用策略函数获取 ``CorrectionConfig`` ：

.. code-block:: python

   from e2m2e.algorithm.family.strategies import (
       symmetric_2d_fixed_x0,
       symmetric_3d_fixed_x0,
       halo_fixed_z0,
   )

   # 获取配置对象
   config = halo_fixed_z0(z0=0.001, libration_point=1)
   print(config.setup_type)          # "halo_orbit_fixed_z0"
   print(config.free_variables)      # ["x0", "y_dot0", "T_half"]
   print(config.target_conditions)   # {"y": 0.0, "x_dot": 0.0, "z_dot": 0.0}

   # 手动应用到修正器
   corrector = DifferentialCorrection(dynamics)
   corrector._apply_config(config)

``CorrectionConfig`` 是一个 ``frozen=True`` 的数据类，所有字段在创建后不可变，
适合作为函数间传递的契约对象。

收敛诊断
~~~~~~~~~

修正失败后，可通过以下属性诊断：

.. code-block:: python

   corrector.iterate_correction(initial_guess=initial_guess)

   # 收敛历史
   history = corrector.get_convergence_history()
   print(history["errors"])           # 各迭代步的残差范数
   print(history["iterations"])       # 总迭代次数
   print(history["status"])           # 最终状态
   print(history["cause"])            # 终止原因码
   print(history["message"])           # 人可读说明

常见终止原因：

- ``收敛成功：误差小于容差`` ：正常收敛
- ``收敛成功：修正量过小但误差足够小`` ：接近机器精度
- ``发散：误差超过限制`` ：初始猜测质量差，需调整
- ``雅可比矩阵奇异`` ：约束与自由变量线性相关，需更换策略
- ``停滞：修正量过小`` ：陷入局部极小，需更换初始猜测
- ``收敛但周期无效`` ：收敛到寄生根（T≈0），需调整初始猜测

参考
~~~~

- Broucke R A. Periodic orbits in the restricted three body problem with Earth-moon masses[R]. 1968.
- Richardson D L. Analytical construction of a periodic solution about the collinear points[J]. Celestial Mechanics, 1980.
- Howell K C. Three-dimensional, periodic, 'halo' orbits in the restricted three-body problem[D]. Stanford University, 1983.
