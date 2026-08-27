Systems
=======

The system layer defines celestial geometry, gravity, and kinematics models —
the context for all subsequent computation.

The System base class
~~~~~~~~~~~~~~~~~~~~~

:class:`~e2m2e.algorithm.dynamics.system.System` is the abstract base defining
three questions a system must answer:

1. Which coordinate frame? Identified by ``frame``.
2. Which units? Decided by ``unit_system``.
3. Gravitational parameters of bodies? Queried via
   ``gravitational_parameter(body)``.

``coordinate_system`` is not in the base interface; only ``EphemerisSystem``
optionally holds one (for frame conversion and ForceModel propagation).

CR3BP system
~~~~~~~~~~~~

:class:`~e2m2e.algorithm.dynamics.CR3BP_System` describes the Circular
Restricted Three-Body Problem: two primaries revolve around their common
barycenter in circles; the third body's mass is negligible.

**Create a system:**

.. code-block:: python

   from e2m2e.algorithm.dynamics import CR3BP_System
   from e2m2e.data.constants import Datum

   system = CR3BP_System(
       mu=Datum.DE421.mu,
       primary="Earth",
       secondary="Moon",
   )._with_default_scales()

For the Earth-Moon branch ``_with_default_scales()`` applies the DE421 self-
consistent datum: distance = ``Datum.DE421.char_length_km`` (384400 km), period
= ``2π × Datum.DE421.char_time_s`` (TU ≈ 375190 s ≈ 27.28 days). Built-in
defaults cover Earth-Moon, Sun-Earth, Sun-Jupiter; other combos raise
``ValueError`` and need explicit ``set_characteristic_scales()``.

**Mass parameter μ:**

.. math::

   \mu = \frac{m_2}{m_1 + m_2}

- Earth-Moon: μ ≈ 0.01215
- Sun-Earth: μ ≈ 3.0039×10⁻⁶
- Sun-Jupiter: μ ≈ 9.535×10⁻⁴

**Libration points:**

.. code-block:: python

   system.compute_libration_points()

   print(system.L1)  # L1 coordinates [x, 0, 0]
   print(system.L2)
   print(system.L3)
   print(system.L4)  # triangular points
   print(system.L5)

Of the five libration points, L1/L2/L3 are collinear (on the x axis); L4/L5 are
triangular.

**Jacobi constant:**

.. math::

   C_J = 2\Omega - v^2

where Ω is the pseudo-potential and speed = ‖v‖. The Jacobi constant is CR3BP's
only integral of motion.

.. code-block:: python

   state = [0.8, 0, 0, 0, 0.6, 0]
   CJ = system.get_jacobi_constant(state)
   print(f"C_J = {CJ}")

**Unit conversion:**

.. code-block:: python

   # Nondimensional → physical
   phys = system.dimensionless_to_physical(dimensionless_state)

   # Physical → nondimensional
   dim = system.physical_to_dimensionless(physical_state)

BCR4BP system
~~~~~~~~~~~~~

:class:`~e2m2e.algorithm.dynamics.BCR4BPSystem` describes the Bicircular
Restricted Four-Body Problem: solar point-mass perturbation added atop the
Earth-Moon synodic rotating frame of CR3BP. Under the bicircular approximation,
Earth-Moon revolves circularly about its barycenter (the CR3BP assumption) while
the Sun also moves on a coplanar circle about the barycenter in the synodic
frame — an analytic function of time requiring no ephemeris:

.. math::

   \mathbf{r}_s(t) = a_s (\cos\theta,\ \sin\theta,\ 0),\quad
   \theta = \theta_0 + \omega_s t

ω_s = n_s − 1 < 0: n_s is the Sun's nondimensional inertial revolution rate;
subtracting the synodic frame's own rate 1 yields the Sun's retrograde rate in
the synodic frame. The system is time-periodic with period ``T = 2π/|ω_s|``,
about one synodic month.

**Create:**

.. code-block:: python

   from e2m2e.algorithm.dynamics import BCR4BPSystem

   system = BCR4BPSystem.earth_moon(sun_phase0=0.0)

   print(system.sun_position(0.0))        # Sun position at t=0 (nondimensional)
   print(system.gravitational_parameter("sun"))   # Solar nondimensional mass m_s

The two systems' Earth-Moon scale datums differ: ``earth_moon()`` uses constant
``EARTH_MOON_DISTANCE_KM`` (DU = 384405 km) with period 27.32 days;
``CR3BP_System._with_default_scales``'s Earth-Moon branch uses DE421 self-
consistent values (DU = 384400 km, period ≈ 27.28 days). ``earth_moon()`` takes
standard sun parameters:

- m_s = GM_sun / GM_EMB ≈ 328900.56 (both GMs from DE440)
- a_s = mean Earth-Sun distance / Earth-Moon distance ≈ 389.17 (distance per GMAT nominalSun)
- ω_s = 27.32/365.25 − 1 ≈ −0.9252 (Julian-year derivation; negative = retrograde)

**Differences from CR3BP:** BCR4BP has no Jacobi integral (explicitly time-
dependent solar terms); ``compute_libration_points`` returns the corresponding
CR3BP's points as reference positions only. Companion dynamics:
:class:`~e2m2e.algorithm.dynamics.BCR4BP_Dynamics`, see :doc:`dynamics`. On
accuracy, the bicircular approximation vs ephemeris (Earth+Moon+Sun point
masses) diverges ~1e3 km over one day of extrapolation, dominated by the Moon's
circular-orbit approximation.

Ephemeris system
~~~~~~~~~~~~~~~~

:class:`~e2m2e.algorithm.dynamics.ephemeris_system.EphemerisSystem` queries body
ephemerides via SPICE kernels, using J2000 inertial frames and physical units
(km, s, km/s). See :doc:`ephemeris`.

.. code-block:: python

   from e2m2e.algorithm.dynamics import EphemerisSystem
   from e2m2e.data.templates.enums import ReferenceFrame
   from e2m2e.data.kernels.manager import SPICEManager

   spice = SPICEManager()
   spice.load_kernel("kernels/de440s.bsp")

   system = EphemerisSystem(
       bodies=["EARTH", "MOON", "SUN"],
       spice=spice,
       origin="EARTH",
       frame=ReferenceFrame.J2000,
   )

CR3BP vs Ephemeris systems
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Property
     - CR3BP_System
     - EphemerisSystem
   * - Frame
     - Rotating (nondimensional)
     - Inertial (J2000, dimensional)
   * - Units
     - Nondimensional (DU, TU)
     - Physical (km, s)
   * - Bodies
     - 2 primaries
     - N bodies (configurable)
   * - Libration points
     - Yes (L1–L5)
     - No
   * - Autonomy
     - Autonomous (time-free)
     - Non-autonomous (epoch-dependent)
   * - Use case
     - Concept design, periodic families
     - High-fidelity mission design
