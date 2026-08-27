Transfer Optimization / 转移轨道优化
=====================================

[English](#transfer-optimization) | [简体中文](#中文)

English
-------

NLP (nonlinear programming) optimization of transfer orbits — the optimize stage
of the search-optimize two-step method, refining feasible candidates into
minimum-impulse transfers satisfying constraints.

Optimizer
~~~~~~~~~

:class:`~e2m2e.algorithm.transfer.transfer_optimization.DROTRONLPOptimizer`
optimizes DRO→RO transfers.

**Decision variables:** y = {α, T, t_ins}

- α: tangential velocity ratio
- T: transfer time
- t_ins: insertion time

**Objective:**

.. math::

   J(y) = \Delta v_1 + \Delta v_2

**Constraints:**

- Position continuity at insertion
- Velocity parallelism with the target's insertion velocity
- Collision avoidance vs Earth & Moon

Solver choice
~~~~~~~~~~~~~

Two solvers:

1. **SciPy SLSQP** (default): via ``scipy.optimize.minimize``
2. **COPT**: Cardinal commercial optimizer (needs ``coptpy``)

Controlled by :class:`~e2m2e.algorithm.transfer.config.TransferConfig`'s
``nlp_use_copt``, adapted uniformly inside
:class:`~e2m2e.algorithm.transfer.transfer_optimization.DROTRONLPOptimizer`:

.. code-block:: python

   from e2m2e.algorithm.transfer import TransferConfig

   # SciPy (default)
   config = TransferConfig(nlp_use_copt=False)

   # COPT (needs coptpy), falling back to SciPy on failure
   config = TransferConfig(nlp_use_copt=True, nlp_fallback_to_scipy=True)

Configuration parameters
~~~~~~~~~~~~~~~~~~~~~~~~

Typical full configuration:

.. code-block:: python

   DU = 3.84405000e5  # Earth-Moon distance (km)

   config = TransferConfig(
       nlp_alpha_min=0.5,
       nlp_alpha_max=2.5,
       nlp_earth_radius=200.0 / DU,    # nondimensional collision radii
       nlp_moon_radius=100.0 / DU,
       nlp_use_relaxed_velocity=True,
       nlp_velocity_angle_tol=0.05,
       nlp_use_copt=False,
       nlp_fallback_to_scipy=True,
       nlp_verbose=False,
   )

Variables & constraints in detail
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Ranges:** α ∈ [alpha_min, alpha_max] (typical 0.5–2.5);
T ∈ [transfer_time_range] (typical 1.0–30.0 TU); t_ins ∈ [t_ins_range]
(typical 0.0–10.0 TU).

**Constraint types:**

1. **Position continuity (equality)**

   .. math::

      c_{\text{pos}}(y) = (x_f - x_{\text{ins}})^2 + (y_f - y_{\text{ins}})^2 + (z_f - z_{\text{ins}})^2 = 0

2. **Velocity parallelism (equality)**

   .. math::

      c_{\text{vel}}(y) = \frac{\mathbf{v}_f \cdot \mathbf{v}_{\text{ins}}}{\|\mathbf{v}_f\| \|\mathbf{v}_{\text{ins}}\|} - 1 = 0

   Default depends on path: direct :class:`~e2m2e.algorithm.transfer.transfer_optimization.DROTRONLPOptimizer`
   use defaults to equality; through :class:`~e2m2e.algorithm.transfer.transfer.Transfer`
   with ``TransferConfig``, ``nlp_use_relaxed_velocity`` defaults True — relaxed to
   inequality.

3. **Relaxed velocity constraint (optional inequality)**

   When ``nlp_use_relaxed_velocity=True``:

   .. math::

      \cos(\theta_{\text{max}}) - \frac{\mathbf{v}_f \cdot \mathbf{v}_{\text{ins}}}{\|\mathbf{v}_f\| \|\mathbf{v}_{\text{ins}}\|} \geq 0

   with :math:`\theta_{\text{max}} =` ``nlp_velocity_angle_tol``.

High-level interface: Transfer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~e2m2e.algorithm.transfer.transfer.Transfer` provides simplified chained
calls encapsulating optimizer construction, config management, and result
extraction:

.. code-block:: python

   from e2m2e.algorithm.transfer import Transfer, TransferConfig
   from e2m2e.algorithm.dynamics.system import CR3BP_System
   from e2m2e.algorithm.dynamics.dynamics import CR3BP_Dynamics
   from e2m2e.data.constants import Datum

   system = CR3BP_System(
       mu=Datum.DE421.mu, primary="Earth", secondary="Moon"
   )._with_default_scales()
   system.compute_libration_points()
   dynamics = CR3BP_Dynamics(system)

   transfer = Transfer(dynamics)
   transfer.set_orbit(start=dro_orbit, end=ro_orbit)

   result = transfer.optimize(
       initial_guess={"alpha": 1.0, "transfer_time": 15.0, "t_ins": 5.0},
       alpha_range=(0.5, 2.5),
       use_relaxed_velocity=True,
       velocity_angle_tol=0.05,
   )

中文
----

转移轨道的 NLP（非线性规划）优化，实现搜索-优化两步法的优化阶段。
对搜索阶段得到的可行候选解进行精化，求解满足约束的最小脉冲转移轨道。

**优化变量** y = {α, T, t_ins}；目标最小化总脉冲 Δv₁+Δv₂；约束含位置连续、
速度平行（可松弛为角度不等式，``nlp_use_relaxed_velocity`` 默认经 ``Transfer``
路径启用）、碰撞避免。

求解器两种：SciPy SLSQP（默认）与 COPT 商业优化器（需 ``coptpy`` ），
由 ``nlp_use_copt`` 控制、失败回退由 ``nlp_fallback_to_scipy`` 控制。
配置字段与高层链式接口示例见上方英文节；公式细节（位置连续、速度平行/
松弛不等式）同样见英文节。
