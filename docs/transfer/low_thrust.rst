Low-Thrust Transfers / 小推力转移
=================================

[English](#low-thrust-transfers) | [简体中文](#中文)

English
-------

Optimal-control low-thrust transfer design over the 7-D augmented state
``[r, v, m]``, routed via :func:`~e2m2e.algorithm.transfer.transfer_orbit`,
internally calling :class:`~e2m2e.algorithm.transfer.lowthrust_shooting.LowThrustShooting`
or :class:`~e2m2e.algorithm.transfer.lowthrust_collocation.LowThrustCollocation`
to close the Q-law guess + SLSQP polish loop.

Principle
~~~~~~~~~

Low-thrust transfers model in 7-D augmented space:

.. math::

   \dot{\mathbf{r}} = \mathbf{v}, \quad
   \dot{\mathbf{v}} = \mathbf{a}_{\text{grav}} + \frac{T \cdot \delta}{m} \hat{\mathbf{u}}, \quad
   \dot{m} = -\frac{T \cdot \delta}{I_{sp} \, g_0}

with:

- :math:`T` max thrust (N); throttle :math:`\delta \in [0, 1]`
- thrust direction :math:`\hat{\mathbf{u}}` parameterized by spherical angles
  :math:`\alpha(\theta_1, \theta_2)`
- specific impulse :math:`I_{sp}` (s), :math:`g_0 = 9.81 \, \text{m/s}^2`

Two-level solve:

1. **Q-law guess**: Lyapunov feedback-law forward integration produces a
   suboptimal control history
2. **SLSQP polish**: min-fuel NLP over per-segment constant controls
   ``(throttle, θ₁, θ₂)``, terminal position/velocity matched to target

Usage
~~~~~

.. code-block:: python

   from e2m2e.algorithm.transfer import transfer_orbit, EngineConfig
   from e2m2e.data.templates import ConvergenceState

   result = transfer_orbit(
       "low_thrust",
       engine_config=EngineConfig(t_max=0.5, isp=3000.0),
       initial_mass=1000.0,
       n_segments=10,
       target_oe=(7200.0, 0.0, 0.0),
       solver_method="shooting",
       duration_days=30.0,
       departure_state=departure_6d,
       target_state=target_6d,
   )

   print(f"Equivalent Δv = {result.details.equivalent_delta_v:.4f} km/s")
   print(f"Fuel consumed = {result.details.fuel_consumed:.2f} kg")
   print(f"Converged = {result.details.status == ConvergenceState.CONVERGED}")

Collocation (more robust at scale):

.. code-block:: python

   result = transfer_orbit(
       "low_thrust",
       engine_config=EngineConfig(t_max=0.5, isp=3000.0),
       initial_mass=1000.0,
       n_segments=20,
       target_oe=(42164.0, 0.0, 0.0),
       solver_method="collocation",
       duration_days=200.0,
       departure_state=departure_6d,
       target_state=target_6d,
   )

Δv semantics vs impulsive
~~~~~~~~~~~~~~~~~~~~~~~~~

The two Δv notions differ:

- **Impulsive Δv**: :math:`\Delta v = \sum |\Delta v_i|` — sum of instantaneous jumps
- **Low-thrust equivalent Δv**:
  :math:`\Delta v_{\text{eq}} = I_{sp} \cdot g_0 \cdot \ln(m_0 / m_f) / 1000` —
  Tsiolkovsky inversion (km/s)

Continuous thrusting (gravity losses) makes equivalent Δv ≥ impulsive ΔV.
Differences are small for short arcs & small changes; significant for large-
energy transfers (LEO→GEO).

Engine parameters
~~~~~~~~~~~~~~~~~

:class:`~e2m2e.algorithm.transfer.lowthrust_shooting.EngineConfig` fields:

- ``t_max``: max thrust (N). Typical: Hall thrusters 0.01–1.0 N; ion thrusters
  0.0001–0.1 N; chemical low-thrust 1–100 N
- ``isp``: specific impulse (s). Typical: Hall 1500–3000 s; ion 2000–5000 s;
  chemical 300–450 s

Other keys:

- ``initial_mass``: spacecraft wet mass (kg)
- ``n_segments``: solver segments — more accuracy, slower; typically 5–50
- ``solver_method``: ``"shooting"`` (analytic Jacobians, 5–24× faster) or
  ``"collocation"`` (Hermite-Simpson, robust at scale)
- ``duration_days``: flight time (days). LEO→GEO ≈ 100–300 d; LEO→Moon ≈ 3–180 d

中文
----

基于 7D 增广状态 ``[r, v, m]`` 的小推力最优控制转移设计。通过
:func:`~e2m2e.algorithm.transfer.transfer_orbit` 编排器路由，内部调用
:class:`~e2m2e.algorithm.transfer.lowthrust_shooting.LowThrustShooting` 或
:class:`~e2m2e.algorithm.transfer.lowthrust_collocation.LowThrustCollocation`
完成 Q-law 初猜 + SLSQP 打磨闭环。

基本原理与两级求解流程（Q-law 初猜 → SLSQP 打磨）见上方英文节公式；
使用方法、配点求解器调用与脉冲 Δv 对比口径同样见英文节代码示例。

推进参数说明：

- ``t_max`` ：最大推力（N）。电推进 Hall 0.01~1.0 N；ion 0.0001~0.1 N；化学低推力 1~100 N
- ``isp`` ：比冲（s）。Hall 1500~3000 s；ion 2000~5000 s；化学 300~450 s
- ``initial_mass`` ：初始质量（kg）；``n_segments`` 段数（典型 5~50）
- ``solver_method`` ：``"shooting"`` （解析雅可比快 5-24x）或 ``"collocation"`` （大规模更鲁棒）
- ``duration_days`` ：飞行时间（天）。LEO→GEO 约 100~300 天；LEO→月球约 3~180 天
