微分修正策略
============

微分修正策略函数将配置逻辑与迭代求解器分离。每个策略返回一个不可变的
``CorrectionConfig``，``DifferentialCorrection`` 接收配置后执行牛顿迭代。

策略函数
--------

e2m2e 提供以下策略函数：

**平面策略（2D 对称）：**

- ``symmetric_2d_fixed_x0``：固定 x0，调整 y_dot0 和半周期。适用于 Lyapunov、DRO。
- ``symmetric_2d_fixed_t``：固定半周期，调整 x0 和 y_dot0。适用于固定周期轨道。
- ``symmetric_2d_fixed_y0``：固定 y0，调整 x_dot0 和半周期。适用于共振轨道（RO）。

**空间策略（3D 对称）：**

- ``symmetric_3d_fixed_x0``：固定 x0，调整 z0、y_dot0 和半周期。
- ``symmetric_xz_fixed_x0``：XZ 平面对称，固定 x0。
- ``symmetric_xz_fixed_z0``：XZ 平面对称，固定 z0。
- ``axial_fixed_vz0``：Axial 轨道（Gómez Type B 分岔），x 轴对称，固定 vz0，调整 x0、y_dot0 和半周期。

**Halo 专用策略：**

- ``halo_fixed_z0``：固定 z 振幅和平动点，调整 x0、y_dot0 和半周期。
- ``halo_fixed_x0``：固定 x 坐标和平动点。

**L4/L5 三角平动点策略（平面，无对称，全周期闭合）：**

- ``spo_fixed_x0``：L4/L5 短周期（SPO），固定 x0，调整 y0、vx0、vy0 和全周期。
- ``lpo_fixed_x0``：L4/L5 长周期（LPO），与 SPO 同框架（大振幅成员呈马蹄形）。

CorrectionConfig
----------------

``CorrectionConfig`` 是 ``frozen=True`` 的数据类：

.. code-block:: python

   from e2m2e.algorithm.family.strategies import halo_fixed_z0

   config = halo_fixed_z0(z0=0.01, libration_point=1)
   print(config.setup_type)          # "halo_orbit_fixed_z0"
   print(config.free_variables)      # ["x0", "y_dot0", "T_half"]
   print(config.target_conditions)   # {"y": 0.0, "x_dot": 0.0, "z_dot": 0.0}

策略与修正器的协作
------------------

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

新增策略无需修改迭代器代码，配置可序列化、可对比、可测试。
