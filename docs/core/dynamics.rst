Dynamics
========

The dynamics layer integrates equations of motion on a given system to produce
state histories.

The Dynamics base class
~~~~~~~~~~~~~~~~~~~~~~~

:class:`~e2m2e.algorithm.dynamics.dynamics.Dynamics` follows the template-method
pattern:

- ``propagate()``: orchestrates integrating the whole trajectory (algorithm skeleton)
- ``_get_eom_func()``: hook — subclasses supply the concrete ODE right-hand side
- ``_get_max_step()``: hook — subclasses supply the maximum step size

**Propagation results:**

- ``states``: state series, shape ``(n_points, 6)``
- ``stm`` (optional): state transition matrices, shape ``(n_points, 6, 6)``

CR3BP dynamics
~~~~~~~~~~~~~~

:class:`~e2m2e.algorithm.dynamics.dynamics.CR3BP_Dynamics` integrates the CR3BP
equations of motion in the rotating frame.

**Equations of motion:**

.. math::

   \ddot{x} - 2\dot{y} = \frac{\partial \Omega}{\partial x}

   \ddot{y} + 2\dot{x} = \frac{\partial \Omega}{\partial y}

   \ddot{z} = \frac{\partial \Omega}{\partial z}

with pseudo-potential Ω combining gravitational and centrifugal potentials:

.. math::

   \Omega = \frac{1}{2}(x^2 + y^2) + \frac{1-\mu}{r_1} + \frac{\mu}{r_2}

   r_1 = \sqrt{(x+\mu)^2 + y^2 + z^2}, \quad r_2 = \sqrt{(x-1+\mu)^2 + y^2 + z^2}

**Example:**

.. code-block:: python

   from e2m2e.algorithm.dynamics import CR3BP_System, CR3BP_Dynamics
   from e2m2e.data.constants import Datum
   from e2m2e.data.types.orbit import Orbit
   import numpy as np

   system = CR3BP_System(
       mu=Datum.DE421.mu, primary="Earth", secondary="Moon"
   )._with_default_scales()
   system.compute_libration_points()

   dynamics = CR3BP_Dynamics(system)

   # Propagate one orbit
   initial_state = np.array([0.8, 0, 0, 0, 0.6, 0])
   orbit = Orbit(
       states=initial_state.reshape(1, -1),
       times=np.array([0.0]),
       system=system,
   )
   orbit.period = 3.0

   result = dynamics.propagate(
       initial_state, t_span=(0, orbit.period)
   )  # Set dynamics.max_step when you need to control step sizes

   print(f"State shape: {result['states'].shape}")
   print(f"Final state: {result['states'][-1]}")

State transition matrix (STM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The STM describes linear evolution of small deviations from the initial state:

.. math::

   \delta \mathbf{x}(t) = \boldsymbol{\Phi}(t, t_0) \, \delta \mathbf{x}(t_0)

.. code-block:: python

   result = dynamics.propagate(
       initial_state, t_span=(0, 3.0), with_stm=True
   )
   print(f"STM shape: {result['stm'].shape}")  # (n_points, 6, 6)

BCR4BP dynamics
~~~~~~~~~~~~~~~

:class:`~e2m2e.algorithm.dynamics.BCR4BP_Dynamics` adds solar point-mass
perturbation (direct + indirect terms) atop the CR3BP equations; sun position is
analytic from :class:`~e2m2e.algorithm.dynamics.BCR4BPSystem`, so the equations
are explicitly time-dependent (time-periodic system, period ≈ one synodic
month). ``propagate`` semantics match ``CR3BP_Dynamics``, supporting
``with_stm=True``.

.. code-block:: python

   from e2m2e.algorithm.dynamics import BCR4BP_Dynamics, BCR4BPSystem

   system = BCR4BPSystem.earth_moon(sun_phase0=0.0)
   dynamics = BCR4BP_Dynamics(system)

   result = dynamics.propagate(state, t_span=(0, 3.0), with_stm=True)

BCR4BP has no Jacobi integral (explicitly time-dependent solar terms);
``with_jacobi=True`` raises ``NotImplementedError``. Against an ephemeris
ForceModel (Earth+Moon+Sun point masses), one-day extrapolation diverges ~1e3 km,
dominated by lunar circular-orbit approximation; at two days BCR4BP tracks the
sun-inclusive ephemeris better than CR3BP. System definition & sun parameters:
see :doc:`system`.

Ephemeris dynamics
~~~~~~~~~~~~~~~~~~

:class:`~e2m2e.algorithm.dynamics.ephemeris_dynamics.EphemerisDynamics` computes
N-body gravity from SPICE ephemerides. See :doc:`ephemeris`.

Force-model propagation
~~~~~~~~~~~~~~~~~~~~~~~

:class:`~e2m2e.algorithm.forces.force_model.ForceModel` implements adaptive
propagation with Rust integrators and does *not* inherit ``Dynamics``. See
:doc:`forces`.

.. code-block:: python

   from e2m2e.algorithm.coordinate import CelestialBodyOrigin, CoordinateSystem, ICRSAxes
   from e2m2e.algorithm.forces import ForceModel, GravityField

   # ForceModel requires the system to hold a coordinate system (forces like
   # spherical harmonics need conversions)
   eph_system.coordinate_system = CoordinateSystem(
       axes=ICRSAxes(),
       origin=CelestialBodyOrigin(body="EARTH", spice=spice),
   )

   fm = ForceModel(eph_system)
   fm.add_force(GravityField("EARTH", degree=2, order=0))

   result = fm.propagate(state0, t_span, t_eval=t_eval)

``propagate`` supports ``with_stm=True`` for simultaneous STM propagation — see
:doc:`forces`.

Event detection
~~~~~~~~~~~~~~~

``Dynamics.propagate`` accepts an ``events`` argument with scipy
``solve_ivp`` semantics: zeros of event functions ``g(t, state) -> float`` are
event surfaces; function objects may carry ``terminal`` (True → stop at first
trigger; trajectory endpoint becomes the event point) and ``direction``
(> 0 rising crossings only, < 0 falling only, 0 both). With ``with_stm=True``
events receive the 42-dim augmented state.

:class:`~e2m2e.algorithm.manifold.sections.PoincareSection`'s
:meth:`~e2m2e.algorithm.manifold.sections.PoincareSection.event` builds
scipy-semantic events directly (section functions use the first 6 dims;
augmented propagation auto-truncates):

.. code-block:: python

   from e2m2e.algorithm.manifold import PoincareSection

   section = PoincareSection.plane(axis=1, value=0.0)   # xz plane at y=0
   event = section.event(direction=-1, terminal=True)   # stop at first descending crossing

   result = dynamics.propagate(y0, (0.0, 10.0), events=[event])
   t_hit = result["t_events"][0]    # trigger times
   y_hit = result["y_events"][0]    # trigger states (endpoint when terminal)

With ``events``, the return dict gains ``t_events``/``y_events`` keys
(per-event arrays); without them those keys are absent. Versus post-hoc
detection (dense sampling + Brent interpolation, see
:doc:`../algorithms/manifolds`), in-integration detection doesn't depend on
sampling density — crossing residuals rest on integrator refinement guarantees.

``ForceModel.propagate`` does not support ``events``: passing non-None events
raises ``NotImplementedError`` immediately (event detection must share the Rust
inner loop with force evaluation; compiled-forces' Rust API doesn't offer it yet,
and Python-RHS fallbacks are forbidden). For events use ``Dynamics.propagate``
above or post-hoc detection on results (dense sampling + Brent interpolation,
see :doc:`../algorithms/manifolds`).
