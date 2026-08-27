Halo Orbits
===========

Halos are three-dimensional periodic orbits around collinear points (L1/L2),
appearing halo-shaped when viewed from Earth.

Initial guess
~~~~~~~~~~~~~

Richardson's third-order approximation supplies the initial guess:

.. code-block:: python

   from e2m2e.algorithm.family.halo_initial_guess import compute_halo_initial_guess

   # L1 Halo, z amplitude 0.001 (small seeds suit Richardson)
   guess = compute_halo_initial_guess(mu=system.mu, z_amplitude=0.001, L=1, halo_class=0)

   print(f"x0 = {guess['x0']}")
   print(f"vy0 = {guess['vy0']}")
   print(f"Half period = {guess['T_half']}")

``halo_class`` picks the branch:

- ``0``: northern (z0 > 0)
- ``1``: southern (z0 < 0)

``L`` selects the point: ``1`` for L1, ``2`` for L2.

Differential correction
~~~~~~~~~~~~~~~~~~~~~~~

The Halo-specific strategy refines the guess into an exact periodic orbit:

.. code-block:: python

   from e2m2e.algorithm.solver import DifferentialCorrection
   from e2m2e.data.types.orbit import Orbit
   import numpy as np

   # 1. Assemble initial state
   initial_state = np.array([
       guess["x0"], 0.0, 0.001,
       0.0, guess["vy0"], 0.0,
   ])

   # 2. Halo strategy
   corrector = DifferentialCorrection(dynamics)
   corrector.setup_halo_orbit_fixed_z0(z0=0.001, libration_point=1)

   initial_guess = Orbit(
       states=initial_state.reshape(1, -1),
       times=np.array([0.0]),
       system=system,
   )
   initial_guess.period = guess["T_half"] * 2

   # 3. Correct
   halo_result = corrector.iterate_correction(initial_guess=initial_guess)
   halo = halo_result.orbit  # corrected orbit (None = failure)
   if halo is not None:
       print(f"Halo period: {halo.period:.6f}")

Family generation
~~~~~~~~~~~~~~~~~

From the corrected seed, continue into a family:

.. code-block:: python

   # 1. Seed first (small-amplitude; continuation amplifies)
   seed_orbit = continuation.generate_halo_seed_orbit(
       libration_point=1,
       amplitude_z=0.001,
       halo_class=0,        # 0=northern, 1=southern
   )

   # 2. Natural-parameter continuation from the seed
   family = continuation.generate_halo_family(
       seed_orbit,
       n_orbits=50,
       z_range=(0.001, 0.15),   # z-amplitude range
   )

   print(f"Generated {len(family)} Halo orbits")

NRHO (near-rectilinear halo orbits)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

High-amplitude Halos approach rectilinear shapes — the Gateway station's
configuration. Increase z amplitude (``amplitude_z > 0.1``) to generate them.

References
~~~~~~~~~~

- Richardson D L. Analytical construction of a periodic solution about the collinear points[J]. *Celestial Mechanics*, 1980, 22(3): 303-320.
- Howell K C. Three-dimensional, periodic, 'halo' orbits in the restricted three-body problem[D]. Stanford University, 1983.
