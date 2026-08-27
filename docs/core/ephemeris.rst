Ephemeris System & Dynamics
===========================

e2m2e provides SPICE-based ephemeris systems and N-body dynamics for
high-fidelity orbit design.

SPICE manager
~~~~~~~~~~~~~

:class:`~e2m2e.data.kernels.manager.SPICEManager` wraps the NASA SPICE toolkit,
exposing ephemeris queries, time conversion, and kernel management.

**Kernel types:**

- **Leap-second kernels** (``.tls``): UTC ↔ ET conversion; loaded automatically
- **Ephemeris kernels** (``.bsp``): body position/velocity data; loaded manually

**Recommended ephemeris kernels:**

============  ====================================
File          Description
============  ====================================
de440.bsp     JPL DE440 (recommended, 1550–2650)
de440s.bsp    JPL DE440 short (1849–2150)
de435.bsp     JPL DE435 (1550–2650)
de438.bsp     JPL DE438 (long-term)
============  ====================================

The project's ``kernels-v1`` release asset (see
:doc:`../getting-started/installation`) ships de430.bsp and de440s.bsp. Note
that ``find_ephemeris_kernel``'s priority list excludes de430.bsp — using de430
requires a manual ``load_kernel``.

Kernel files should come from the project's
`GitHub Release <https://github.com/cislunarspace/e2m2e/releases>`_
(``kernels-v1`` carries de430.bsp, de440s.bsp, and every other required kernel);
the official source (when reachable) is
`NASA NAIF <https://naif.jpl.nasa.gov/naif/data.html>`_.

.. code-block:: python

   from e2m2e.data.kernels.manager import SPICEManager

   mgr = SPICEManager()
   mgr.load_kernel("path/to/de440.bsp")

   # Time conversion
   et = mgr.utc_to_et("2024-01-01T00:00:00")

   # Lunar state
   state = mgr.get_body_state("MOON", et, "J2000", "EARTH")

   mgr.unload_kernel("path/to/de440.bsp")

Ephemeris cache
^^^^^^^^^^^^^^^

Call ``enable_ephem_cache`` once before integration to batch pre-sample the
listed bodies' (position, velocity) on a uniform grid and build CubicSplines;
afterwards ``get_body_position`` / ``get_body_state`` answer in-range queries by
interpolation without crossing the SPICE boundary. With a fully-loaded
ForceModel this typically accelerates propagation >10×.

The cache binds to currently loaded kernels; ``unload_kernel`` invalidates it
automatically.

.. code-block:: python

   mgr.enable_ephem_cache(
       ["MOON", "SUN"], et_start, et_end, dt=3600.0
   )

   # ... ForceModel propagation ...

   mgr.disable_ephem_cache()  # Turn off; fall back to per-step SPICE queries

Full signature:
``enable_ephem_cache(bodies, et_start, et_end, *, dt=3600.0, frame="J2000", observer="EARTH")``:
``bodies`` lists body names to cache; ``et_start``/``et_end`` bound the interval
in ET seconds; ``dt`` is the grid step in seconds.

Ephemeris system
~~~~~~~~~~~~~~~~

:class:`~e2m2e.algorithm.dynamics.ephemeris_system.EphemerisSystem` manages
queries over a set of bodies behind one unified data-access layer.

.. code-block:: python

   from e2m2e.data.templates.enums import ReferenceFrame
   from e2m2e.data.kernels.manager import SPICEManager
   from e2m2e.algorithm.dynamics.ephemeris_system import EphemerisSystem

   mgr = SPICEManager()
   mgr.load_kernel("path/to/de440.bsp")

   system = EphemerisSystem(
       bodies=["EARTH", "MOON", "SUN"],
       spice=mgr,
       origin="EARTH",
       frame=ReferenceFrame.J2000
   )

   # Lunar position
   moon_pos = system.get_body_position("MOON", et)  # km

   # Solar state (position+velocity)
   sun_state = system.get_body_state("SUN", et)  # [km, km/s]

   # Gravitational parameter
   gm_moon = system.get_gm("MOON")

Ephemeris dynamics
~~~~~~~~~~~~~~~~~~

:class:`~e2m2e.algorithm.dynamics.ephemeris_dynamics.EphemerisDynamics`
implements the restricted N-body problem, computing multi-body gravitational
accelerations from SPICE data.

.. note::

   ``EphemerisDynamics`` is demoted to an internal implementation
   (`e2m2e.algorithm.dynamics.ephemeris_dynamics` remains importable). New code
   should prefer ForceModel's force-decomposition path (:doc:`forces`).

**Physical model:** with ``system.origin`` as coordinate origin (usually Earth),
the origin body exerts central gravity while remaining bodies exert third-body
perturbations:

.. math::

   \mathbf{a} = -\frac{\mu_0 \mathbf{r}}{|\mathbf{r}|^3}
   - \sum_i \mu_i \left[
     \frac{\mathbf{r} - \mathbf{r}_i}{|\mathbf{r} - \mathbf{r}_i|^3}
     + \frac{\mathbf{r}_i}{|\mathbf{r}_i|^3}
   \right]

where :math:`\mathbf{r}_i` is the i-th body's position and :math:`\mu_i` its GM.

.. code-block:: python

   from e2m2e.algorithm.dynamics.ephemeris_dynamics import EphemerisDynamics

   dynamics = EphemerisDynamics(system)

   # Propagate
   import numpy as np
   state0 = np.array([r1, r2, r3, v1, v2, v3])  # km, km/s
   t_span = (0, 86400)  # seconds
   result = dynamics.propagate(state0, t_span, with_stm=True)

   # Results
   print(result["states"].shape)  # (n_points, 6)
   print(result["stm"].shape)     # (n_points, 6, 6)

With ``with_stm=True``, propagation always takes the
``propagate_with_stm_py`` Rust fast path: missing Rust extension raises outright
(no Python fallback). Behavioral contract: missing kernels or truncated
trajectories (early Rust-side exits) raise ``RuntimeError`` — never silent
truncated returns.

**Integrator config:**

- Default integrator: ``DOP853`` (8th-order Runge-Kutta)
- Default max step: 60 s
- Adaptive steps tuned automatically per propagation duration

vs CR3BP dynamics
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Property
     - CR3BP_Dynamics
     - EphemerisDynamics
   * - Frame
     - Rotating (nondimensional)
     - Inertial (J2000, dimensional)
   * - Units
     - Nondimensional (DU, TU)
     - Physical (km, s)
   * - Bodies
     - 2 primaries
     - N bodies (configurable)
   * - Accuracy
     - Idealized circular orbits
     - High-fidelity ephemeris data
   * - Use case
     - Concept design, periodic orbits
     - High-fidelity mission design
