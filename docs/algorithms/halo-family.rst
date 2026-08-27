Halo Family Orchestration
=========================

The Halo family orchestration module (``e2m2e.algorithm.family.halo_family``)
split Halo-specific logic out of ``continuation.py``: seed generation, natural-
parameter family continuation, pseudo-arclength continuation.

Seed generation
~~~~~~~~~~~~~~~

``generate_halo_seed_orbit()`` grows one Halo seed orbit from a continuation
instance:

.. code-block:: python

   from e2m2e.algorithm.family.halo_family import generate_halo_seed_orbit

   seed = generate_halo_seed_orbit(
       continuation=continuation,
       libration_point=1,
       amplitude_z=0.001,   # seeds should be small-amplitude (Richardson's accuracy is best there); continuation amplifies
       halo_class=0,
   )

Parameters:

- ``continuation``: a ``Continuation`` instance
- ``libration_point``: point number (1 or 2)
- ``amplitude_z``: z-direction amplitude
- ``halo_class``: branch (0=northern, 1=southern)

Family generation
~~~~~~~~~~~~~~~~~

``generate_halo_family()`` continues naturally along z-amplitude from the seed:

.. code-block:: python

   family = continuation.generate_halo_family(
       seed_orbit=seed,
       n_orbits=50,
       z_range=(0.001, 0.15),   # z-amplitude range
   )

   print(f"Generated {len(family)} Halo orbits")

Internals: each converged orbit seeds the next, fixing target z0 for pointwise
correction while stepping toward the ``z_range`` boundary; failed corrections
shrink the step and retry, terminating when the floor is hit.

Pseudo-arclength continuation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For non-monotonic parameter–amplitude Halo branches (amplitude rising then
falling), use pseudo-arclength:

.. code-block:: python

   from e2m2e.algorithm.family.halo_family import halo_pseudo_arclength_continuation

   family = halo_pseudo_arclength_continuation(
       continuation=continuation,
       seed_orbit=seed,
       n_orbits=50,               # new members per branch
       direction="both",          # both sides
       step_size=0.0045,          # pseudo-arclength step ΔS
   )

Returns an ``OrbitFamily`` containing the seed plus each branch's new members.
