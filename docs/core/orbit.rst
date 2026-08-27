Orbits
======

The orbit data structures of e2m2e.

The Orbit class
~~~~~~~~~~~~~~~

:class:`~e2m2e.data.types.orbit.Orbit` is the orbit data container holding
state and time series.

**Core attributes:**

- ``states``: state series ``[x, y, z, vx, vy, vz]``, shape ``(n_points, 6)``
- ``times``: time series, shape ``(n_points,)``
- ``system``: the associated system object (``CR3BP_System`` or
  ``EphemerisSystem``)
- ``family_type`` / ``parameters``: family type & continuation parameter
  (filled by external algorithms such as continuation)
- ``metadata``: metadata dict (creation time, origin, description, tags)

.. code-block:: python

   from e2m2e.data.types.orbit import Orbit
   import numpy as np

   # Create from a state series
   orbit = Orbit(
       states=np.array([[0.8, 0, 0, 0, 0.6, 0]]),
       times=np.array([0.0]),
       system=system,
   )

**Basic properties:**

``Orbit.__init__`` ends with ``compute_basic_properties()``, automatically
estimating ``period`` at construction (x-axis zero-crossing detection) and
computing ``amplitudes``, ``extrema``, ``mean_state``, ``center``,
``is_periodic``, ``periodicity_error`` — fields declared explicitly in
``__init__``, exposed via property proxies. Differential-correction results go
into pre-declared ``correction_*`` fields (default ``None``). Jacobi constants
and stability remain on-demand computations by external algorithms.

**Serialization:**

.. code-block:: python

   # Save to JSON
   orbit.save_to_file("my_orbit.json")

   # Load from JSON
   orbit2 = Orbit.load_from_file("my_orbit.json", system=system)

Orbit families (OrbitFamily)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A set of same-type ``Orbit``\ s indexed by continuous parameters (Jacobi
constant, amplitude…). A family is the *result* of continuation, not the method
generating it.

.. code-block:: python

   # Continuation returns a family
   from e2m2e.algorithm.solver import Continuation

   continuation = Continuation(corrector=corrector)
   result = continuation.natural_continuation(
       seed_orbit=seed_dro,
       param_range=(0.14, 0.9),
       step_size=0.005,
   )

   # Iterate orbits in the family (the family lives at result.family; OrbitFamily is iterable)
   for orbit in result.family:
       print(f"Period: {orbit.period:.6f}")

CR3BP periodic family types
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Family
     - Related libration point
     - Physical character
   * - Lyapunov
     - L1, L2, L3
     - Planar periodic orbits
   * - Halo
     - L1, L2
     - Three-dimensional periodic orbits
   * - Vertical
     - L1–L5
     - Out-of-plane oscillations
   * - Butterfly
     - L1, L2
     - Symmetric orbits joining two collinear points
   * - Dragonfly
     - L1, L2
     - Asymmetric orbits joining two collinear points
   * - DRO
     - secondary
     - Distant retrograde orbits
   * - RO
     - whole system
     - Periodic orbits satisfying m:n resonance ratios
