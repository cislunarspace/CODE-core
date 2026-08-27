DPO Family
==========

The DPO (Distant Prograde Orbit) family consists of prograde distant orbits near
the Moon's L2. Opposite to retrograde DROs, DPOs move along the Moon's
revolution direction in the rotating frame.

Design method
~~~~~~~~~~~~~

Designed within the CR3BP framework:

1. **Seed generation**: small-amplitude initial guesses near L2
2. **Differential correction**: symmetric strategies refine the periodic orbit
3. **Natural continuation**: grow the family along amplitude

Usage
~~~~~

Via Facade tier-1:

.. code-block:: python

   from e2m2e.api import Facade

   facade = Facade()
   result = facade.design_orbit(
       orbit_type="DPO",
       collinear_point=2,
       amplitude=15000.0,  # km
       epoch=[2024, 1, 1, 0, 0, 0.0],
       duration=365.25 * 86400.0,  # one-year arc (seconds)
   )

Or lower-level:

.. code-block:: python

   from e2m2e.api.models import DesignOrbitRequest
   from e2m2e.algorithm.design import design_orbit

   request = DesignOrbitRequest(
       orbit_type="DPO",
       collinear_point=2,
       amplitude=15000.0,
       epoch=[2024, 1, 1, 0, 0, 0.0],
   )
   result = design_orbit(request)

Ephemeris correction
~~~~~~~~~~~~~~~~~~~~

DPOs are unstable-family: no DRO-style correct-one-rev-and-drift path exists.
``DesignOrbitRequest`` validation dispatches ``correction_method`` per family:
DPO defaults to ``segmented`` (full-arc segmented shooting) when unspecified;
explicit conflicting values like ``two_level``/``standard`` warn and rewrite to
``segmented``, with actual method recorded in result/response
``correction_method`` fields. Inter-rev quasi-periodic drift remains
``station_keeping``'s job.

A default-20000-km DPO has period ≈ 23 days. Keeping unstable-direction error
within per-segment shooting convergence bounds, production sampling uses 64
uniform-time patch points per rev, at most two revs merged into one segment,
nodes pinned in time. This covers GUI-default magnitudes (~30 days, 1-hour
output steps); longer arcs' maintainability isn't promised by this design stage.

Properties
~~~~~~~~~~

- Prograde, aligned with the Moon's revolution direction
- Near libration point L2
- Suitable for lunar-farside communication relay design
