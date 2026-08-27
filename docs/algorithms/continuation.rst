Continuation
============

Continuation starts from one known orbit and steps along a family parameter to
generate neighbors.

Basic principle
~~~~~~~~~~~~~~~

Given a converged periodic orbit (the seed), continuation nudges initial
conditions stepwise along a parameter (Jacobi constant, amplitude…),
re-converging each step via differential correction, producing a family.

Natural continuation
~~~~~~~~~~~~~~~~~~~~

Natural continuation advances along a single parameter, seeding each step with
the current solution:

.. code-block:: python

   from e2m2e.algorithm.solver import DifferentialCorrection, Continuation

   corrector = DifferentialCorrection(dynamics)
   corrector.setup_2D_symmetric_x_fixed_x0(x0=initial_state[0])

   continuation = Continuation(corrector=corrector)
   result = continuation.natural_continuation(
       seed_orbit=seed_dro,       # seed orbit
       param_range=(0.14, 0.9),   # parameter sweep range
       step_size=0.005,           # step size
   )

   print(f"Generated {len(result.family.orbits)} orbits")

Parameters:

- ``seed_orbit``: a converged periodic orbit
- ``param_range``: sweep range (e.g., x0 range)
- ``step_size``: per-step increment

Pseudo-arclength continuation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adds an arclength constraint atop natural continuation, handling turning
points — suited to families with non-monotonic amplitude–parameter relations
(e.g., Halo's amplitude-period curve).

.. code-block:: python

   family = continuation.pseudo_arclength_continuation(
       seed_orbit=seed_orbit,
       n_orbits=50,              # new members on this branch
       step_size=0.005,          # pseudo-arclength step ΔS
       direction="positive",     # direction; call twice for both sides
   )

Halo continuation
~~~~~~~~~~~~~~~~~

Halos have dedicated orchestration combining Richardson's analytic approximation
with continuation:

.. code-block:: python

   # 1. Seed (small amplitude suits Richardson; continuation amplifies)
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

Failure handling
~~~~~~~~~~~~~~~~

When a step's correction fails to converge, the continuer shrinks the step and
retries; once the floor (``min_step_size``) is hit without convergence, that
direction terminates. Only converged orbits join the family.

After continuation, diagnose via the ``Continuation`` instance's
``continuation_stats`` attribute: dict with ``total_steps``,
``successful_steps``, ``failed_steps``.
