动力学
======

e2m2e 的动力学层负责在指定系统上积分运动方程、得到状态历史。

Dynamics 基类
-------------

:class:`~e2m2e.core.dynamics.Dynamics` 采用模板方法模式：

- ``propagate()`` — 编排整条轨迹的积分（算法骨架）
- ``_get_eom_func()`` — 钩子方法，子类提供具体的 ODE 右端函数
- ``_get_max_step()`` — 钩子方法，子类提供最大步长

**传播结果：**

- ``states`` — 状态序列，形状 ``(n_points, 6)``
- ``stm``（可选）— 状态转移矩阵，形状 ``(n_points, 6, 6)``

CR3BP 动力学
-------------

:class:`~e2m2e.core.dynamics.CR3BP_Dynamics` 在旋转坐标系中积分 CR3BP 运动方程。

**运动方程：**

.. math::

   \ddot{x} - 2\dot{y} = \frac{\partial \Omega}{\partial x}

   \ddot{y} + 2\dot{x} = \frac{\partial \Omega}{\partial y}

   \ddot{z} = \frac{\partial \Omega}{\partial z}

其中伪势能 Ω 由引力势与离心势组成：

.. math::

   \Omega = \frac{1}{2}(x^2 + y^2) + \frac{1-\mu}{r_1} + \frac{\mu}{r_2}

   r_1 = \sqrt{(x+\mu)^2 + y^2 + z^2}, \quad r_2 = \sqrt{(x-1+\mu)^2 + y^2 + z^2}

**使用示例：**

.. code-block:: python

   from e2m2e.core import CR3BP_System, CR3BP_Dynamics, Orbit
   import numpy as np

   system = CR3BP_System(
       mu=0.0121506683, primary="Earth", secondary="Moon"
   )._with_default_scales()
   system.compute_libration_points()

   dynamics = CR3BP_Dynamics(system)

   # 传播一条轨道
   initial_state = np.array([0.8, 0, 0, 0, 0.6, 0])
   orbit = Orbit(
       states=initial_state.reshape(1, -1),
       times=np.array([0.0]),
       system=system,
   )
   orbit.period = 3.0

   result = dynamics.propagate(
       initial_state, t_span=(0, orbit.period), max_steps=10000
   )

   print(f"状态形状: {result.states.shape}")
   print(f"末状态: {result.states[-1]}")

状态转移矩阵 (STM)
-------------------

STM 描述初始状态微小偏差的线性演化：

.. math::

   \delta \mathbf{x}(t) = \boldsymbol{\Phi}(t, t_0) \, \delta \mathbf{x}(t_0)

.. code-block:: python

   result = dynamics.propagate(
       initial_state, t_span=(0, 3.0), with_stm=True
   )
   print(f"STM 形状: {result.stm.shape}")  # (n_points, 6, 6)

星历动力学
----------

:class:`~e2m2e.core.ephemeris_dynamics.EphemerisDynamics` 基于 SPICE 星历计算 N 体引力。
详见 :doc:`ephemeris`。

力模型传播
----------

:class:`~e2m2e.core.forces.force_model.ForceModel` 用 Rust 积分器实现自适应传播，
不继承 ``Dynamics``。详见 :doc:`forces`。

.. code-block:: python

   from e2m2e.core.forces import ForceModel, GravityField

   fm = ForceModel(eph_system)
   fm.add_force(GravityField("EARTH", degree=2, order=0))

   result = fm.propagate(state0, t_span, t_eval=t_eval)
