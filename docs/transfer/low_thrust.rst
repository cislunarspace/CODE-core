Low-Thrust Transfers
====================

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

Discrete gears and variable-duration arcs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~e2m2e.algorithm.transfer.lowthrust_collocation.LowThrustCollocation`
adds two collocation modes for engines that only fire at discrete throttle
gears, alongside the continuous-throttle solve:

- ``solve_discrete(levels)``: each time interval's throttle is pinned to one of
  the gears ``{0, 60, 100}`` (%). The time grid is uniform, each segment's
  midpoint takes its interval gear (no interpolated throttle), and every arc
  must last at least 1 h (``(tf - t0) / n_segments >= 3600`` s). NLP variables
  are node states and thrust directions only.
- ``solve_variable_time(levels)``: same gear set, but interior node times
  become NLP decision variables (``t0``/``tf`` pinned; per-node decision vector
  ``[state (7), control (3), time (1)]``), with the 1 h minimum arc duration
  enforced as an inequality constraint. Suited to coast/burn arcs of unequal
  length.

Both maximize final mass (objective ``-m_N``) and return the same
:class:`~e2m2e.algorithm.transfer.lowthrust_shooting.LowThrustShootingSolution`
container as the shooting solver, sharing
:class:`~e2m2e.algorithm.transfer.lowthrust_shooting.EngineConfig`:

.. code-block:: python

   from e2m2e.algorithm.transfer.lowthrust_collocation import LowThrustCollocation

   solver = LowThrustCollocation(
       system, forces, engine, initial_state, initial_mass,
       target_state, t0, tf,
   )
   sol = solver.solve_discrete([0, 60, 100, 100, 60, 0])   # one gear per arc
   sol = solver.solve_variable_time([0, 60, 100, 100, 0])  # node times adapt
   print(f"Final mass = {sol.final_mass:.2f} kg, status = {sol.status}")

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
