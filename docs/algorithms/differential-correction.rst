微分修正策略
============

微分修正（Differential Correction）通过迭代调整初始条件，使轨道满足周期性约束。
策略层将"配置逻辑"与"迭代求解器"分离：每种策略只负责生成不可变的 ``CorrectionConfig``，
``DifferentialCorrection`` 负责执行牛顿迭代。

策略概览
--------

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
------------

根据轨道类型选择策略：

- 平面 Lyapunov 轨道（L1/L2/L3 附近）：使用 ``symmetric_2d_fixed_x0`` 固定 x0，调整 y_dot0 和半周期。
- 固定周期轨道（如 DRO）：使用 ``symmetric_2d_fixed_t`` 固定半周期，调整 x0 和 y_dot0。
- 共振轨道（从 y 轴出发）：使用 ``symmetric_2d_fixed_y0`` 固定 y0，调整 x_dot0 和半周期。
- Halo 轨道（三维，XZ 平面对称）：使用 ``halo_fixed_z0`` 固定 z 振幅，或 ``halo_fixed_x0`` 固定 x 坐标。
- Axial 轨道（L1/L2，Gómez Type B 分岔，xy 平面出发）：使用 ``axial_fixed_vz0`` 固定 vz0，调整 x0、y_dot0 和半周期。
- L4/L5 短周期 / 长周期（SPO/LPO，平面无对称）：使用 ``spo_fixed_x0`` / ``lpo_fixed_x0`` 固定 x0，全周期闭合。
- 一般空间周期轨道：使用 ``symmetric_3d_fixed_x0`` 或 ``symmetric_xz_fixed_*`` 变体。

对称性原理
----------

CR3BP 动力学在旋转坐标系中具有以下对称性：

- **x 轴对称**：若 (x, y, z, x_dot, y_dot, z_dot) 是解，则 (x, -y, -z, -x_dot, y_dot, -z_dot) 也是解。
  利用此对称性，周期轨道可从 x 轴垂直出发（y=0, x_dot=0），半周期后再次垂直穿越 x 轴。

- **XZ 平面对称**：若 (x, y, z, x_dot, y_dot, z_dot) 是解，则 (x, -y, z, -x_dot, y_dot, -z_dot) 也是解。
  Halo 轨道利用此对称性，从 XZ 平面出发（y=0），半周期后再次到达 XZ 平面。

- **y 轴对称**：若 (x, y, z, x_dot, y_dot, z_dot) 是解，则 (-x, y, -z, x_dot, -y_dot, z_dot) 也是解。
  共振轨道利用此对称性，从 y 轴出发（x=0, x_dot=0），半周期后再次穿越 y 轴。

代码示例
--------

对称 2D（平面 Lyapunov 轨道）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from e2m2e.algorithm.dynamics import CR3BP_System
   from e2m2e.algorithm.dynamics.dynamics import CR3BP_Dynamics
   from e2m2e.data.constants import Datum
   from e2m2e.data.types.orbit import Orbit
   from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection
   import numpy as np

   # 1. 创建地月系统（μ 取 DE421 基准，ADR 0022）
   system = CR3BP_System(
       mu=Datum.DE421.mu, primary="Earth", secondary="Moon"
   )._with_default_scales()
   system.set_characteristic_scales(384400, 27.32 * 86400)
   system.compute_libration_points()

   # 2. 创建动力学对象
   dynamics = CR3BP_Dynamics(system)

   # 3. 选择策略：固定 x0 的 2D 对称修正（Lyapunov 轨道）
   x0 = system.L1[0] + 0.01  # L1 附近
   corrector = DifferentialCorrection(dynamics)
   corrector.setup_2D_symmetric_x_fixed_x0(x0=x0)

   # 4. 提供初始猜测（从 x 轴垂直出发）
   initial_state = np.array([x0, 0.0, 0.0, 0.0, 0.15, 0.0])
   initial_guess = Orbit(
       states=initial_state.reshape(1, -1),
       times=np.array([0.0]),
       system=system,
   )
   initial_guess.period = 3.0  # 周期猜测（无量纲）

   # 5. 执行微分修正
   result = corrector.iterate_correction(
       initial_guess=initial_guess, verbose=False
   )
   orbit = result.orbit  # 修正后的轨道（None 表示失败）

   if orbit is not None:
       print(f"收敛: 周期={orbit.period:.4f}, 族={orbit.family_type}")
       print(f"初始状态: {orbit.states[0]}")
   else:
       print(f"修正失败: {result.message}")

对称 2D（固定周期 DRO）
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 固定半周期，调整 x0 和 y_dot0
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
       print(f"DRO 周期: {orbit.period:.4f} (目标: {target_period})")

对称 3D（空间周期轨道）
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 固定 x0，调整 z0、y_dot0 和 T_half
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
       print(f"3D 轨道周期: {orbit.period:.4f}")

Halo 轨道（固定 z 振幅）
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from e2m2e.algorithm.family.halo_initial_guess import compute_halo_initial_guess

   # 1. 使用 Richardson 三阶近似生成初始猜测
   mu = system.mu
   z0 = 0.001  # z 方向振幅（小振幅种子，Richardson 近似精度高）
   guess = compute_halo_initial_guess(mu, z0, L=1, halo_class=0)

   # 2. 组装初始状态（北 Halo：z0 > 0）
   initial_state = np.array([
       guess["x0"], 0.0, z0,
       0.0, guess["vy0"], 0.0,
   ])

   # 3. 选择 Halo 策略（固定 z0）
   corrector = DifferentialCorrection(dynamics)
   corrector.setup_halo_orbit_fixed_z0(z0=z0, libration_point=1)

   initial_guess = Orbit(
       states=initial_state.reshape(1, -1),
       times=np.array([0.0]),
       system=system,
   )
   initial_guess.period = guess["T_half"] * 2

   # 4. 执行修正
   result = corrector.iterate_correction(initial_guess=initial_guess)
   orbit = result.orbit
   if orbit is not None:
       print(f"Halo 轨道周期: {orbit.period:.4f}")
       print(f"Jacobi 常数: {system.get_jacobi_constant(orbit.states[0]):.6f}")

策略与修正器的协作关系
----------------------

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

策略函数（如 ``halo_fixed_z0``）只负责生成配置，不直接操作状态；
``DifferentialCorrection`` 接收配置后，使用 STM 牛顿法迭代求解。
这种分离使得：

- 新增策略无需修改迭代器代码
- 同一策略可被不同修正器复用
- 配置可序列化、可对比、可测试

直接调用策略函数（高级用法）
------------------------------

若需要自定义修正流程，可直接调用策略函数获取 ``CorrectionConfig``：

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
--------

修正失败后，可通过以下属性诊断：

.. code-block:: python

   corrector.iterate_correction(initial_guess=initial_guess)

   # 收敛历史
   history = corrector.get_convergence_history()
   print(history["errors"])           # 各迭代步的残差范数
   print(history["iterations"])       # 总迭代次数
   print(history["termination_reason"])  # 终止原因

常见终止原因：

- ``收敛成功：误差小于容差`` — 正常收敛
- ``收敛成功：修正量过小但误差足够小`` — 接近机器精度
- ``发散：误差超过限制`` — 初始猜测质量差，需调整
- ``雅可比矩阵奇异`` — 约束与自由变量线性相关，需更换策略
- ``停滞：修正量过小`` — 陷入局部极小，需更换初始猜测
- ``收敛但周期无效`` — 收敛到寄生根（T≈0），需调整初始猜测

参考
----

- Broucke R A. Periodic orbits in the restricted three body problem with Earth-moon masses[R]. 1968.
- Richardson D L. Analytical construction of a periodic solution about the collinear points[J]. Celestial Mechanics, 1980.
- Howell K C. Three-dimensional, periodic, 'halo' orbits in the restricted three-body problem[D]. Stanford University, 1983.

