动力学
======

e2m2e 的动力学层负责在指定系统上积分运动方程、得到状态历史。

Dynamics 基类
-------------

:class:`~e2m2e.algorithm.dynamics.dynamics.Dynamics` 采用模板方法模式：

- ``propagate()`` — 编排整条轨迹的积分（算法骨架）
- ``_get_eom_func()`` — 钩子方法，子类提供具体的 ODE 右端函数
- ``_get_max_step()`` — 钩子方法，子类提供最大步长

**传播结果：**

- ``states`` — 状态序列，形状 ``(n_points, 6)``
- ``stm``（可选）— 状态转移矩阵，形状 ``(n_points, 6, 6)``

CR3BP 动力学
-------------

:class:`~e2m2e.algorithm.dynamics.dynamics.CR3BP_Dynamics` 在旋转坐标系中积分 CR3BP 运动方程。

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

   from e2m2e.algorithm.dynamics import CR3BP_System, CR3BP_Dynamics, Orbit
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
       initial_state, t_span=(0, orbit.period)
   )  # 需要控制步长时设置 dynamics.max_step 属性

   print(f"状态形状: {result['states'].shape}")
   print(f"末状态: {result['states'][-1]}")

状态转移矩阵 (STM)
-------------------

STM 描述初始状态微小偏差的线性演化：

.. math::

   \delta \mathbf{x}(t) = \boldsymbol{\Phi}(t, t_0) \, \delta \mathbf{x}(t_0)

.. code-block:: python

   result = dynamics.propagate(
       initial_state, t_span=(0, 3.0), with_stm=True
   )
   print(f"STM 形状: {result['stm'].shape}")  # (n_points, 6, 6)

BCR4BP 动力学
-------------

:class:`~e2m2e.algorithm.dynamics.BCR4BP_Dynamics` 在 CR3BP 运动方程上叠加
太阳质点摄动（直接项与间接项），太阳位置由
:class:`~e2m2e.algorithm.dynamics.BCR4BPSystem` 解析给出，方程显式含时
（时间周期系统，周期约一个会合月）。``propagate`` 接口语义与
``CR3BP_Dynamics`` 一致，支持 ``with_stm=True``。

.. code-block:: python

   from e2m2e.algorithm.dynamics import BCR4BP_Dynamics, BCR4BPSystem

   system = BCR4BPSystem.earth_moon(sun_phase0=0.0)
   dynamics = BCR4BP_Dynamics(system)

   result = dynamics.propagate(state, t_span=(0, 3.0), with_stm=True)

BCR4BP 无 Jacobi 积分（太阳项显式含时），``with_jacobi=True`` 抛
``NotImplementedError``。双圆近似与星历（地+月+日点质量 ForceModel）
对比，1 天外推位置误差在 1e3 km 量级，主误差来自月球圆轨道近似；
2 天时 BCR4BP 比 CR3BP 更接近含太阳的星历。系统定义与太阳参数见
:doc:`system` 的「BCR4BP 系统」一节。

星历动力学
----------

:class:`~e2m2e.algorithm.dynamics.ephemeris_dynamics.EphemerisDynamics` 基于 SPICE 星历计算 N 体引力。
详见 :doc:`ephemeris`。

力模型传播
----------

:class:`~e2m2e.algorithm.forces.force_model.ForceModel` 用 Rust 积分器实现自适应传播，
不继承 ``Dynamics``。详见 :doc:`forces`。

.. code-block:: python

   from e2m2e.algorithm.dynamics import CelestialBodyOrigin, CoordinateSystem, ICRSAxes
   from e2m2e.algorithm.forces import ForceModel, GravityField

   # ForceModel 要求系统持有坐标系（球谐引力等力需要坐标变换）
   eph_system.coordinate_system = CoordinateSystem(
       axes=ICRSAxes(),
       origin=CelestialBodyOrigin(body="EARTH", spice=spice),
   )

   fm = ForceModel(eph_system)
   fm.add_force(GravityField("EARTH", degree=2, order=0))

   result = fm.propagate(state0, t_span, t_eval=t_eval)

``propagate`` 支持 ``with_stm=True`` 同时传播状态转移矩阵，详见 :doc:`forces`。

事件检测
--------

``Dynamics.propagate`` 接受 ``events`` 参数，语义与 scipy ``solve_ivp``
一致：事件函数 ``g(t, state) -> float`` 的零点即事件面，函数对象可携带
``terminal``（True 时首次触发即终止积分，轨迹末点即事件点）与
``direction``（> 0 只记上行穿越、< 0 只记下行、0 双向）属性。
``with_stm=True`` 时事件函数接收 42 维增广状态。

:class:`~e2m2e.algorithm.manifold.sections.PoincareSection` 的
:meth:`~e2m2e.algorithm.manifold.sections.PoincareSection.event` 方法直接生成
scipy 语义的事件函数（截面函数只依赖前 6 维，增广传播时自动截取）：

.. code-block:: python

   from e2m2e.algorithm.manifold import PoincareSection

   section = PoincareSection.plane(axis=1, value=0.0)   # xz 平面 y=0
   event = section.event(direction=-1, terminal=True)   # 首次下行穿越即停

   result = dynamics.propagate(y0, (0.0, 10.0), events=[event])
   t_hit = result["t_events"][0]    # 触发时刻
   y_hit = result["y_events"][0]    # 触发状态（terminal 时即轨迹末点）

传入 ``events`` 时返回字典新增 ``t_events`` 与 ``y_events`` 键（逐事件的
触发时刻与状态数组）；不传则不含这两个键。与事后检测（密采样 + Brent
插值，见 :doc:`../algorithms/manifolds`）相比，积分中检测不依赖采样密度，
穿越残差由积分器求精保证。

``ForceModel.propagate(events=...)`` 走 Rust 快速路径：事件检测由
``e2m2e.integrators.solve_ivp_events``（Rust ``solve_ivp_events_py``
的薄封装）在积分内循环完成，每个接受步端点评估事件函数，符号变化时
步内二分求精，末点为求精后的事件点而非触发步终点；返回字典新增
``t_events``/``y_events``/``n_steps``。求精精度受步内线性插值误差限制，
需要更紧的事件时刻时减小 ``max_step``。
