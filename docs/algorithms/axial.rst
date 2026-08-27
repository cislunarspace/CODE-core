Axial Orbit Family
==================

The Axial family is another class of periodic orbits near CR3BP collinear
libration points, built on Gómez's Type B bifurcation (Gómez et al., 2001).

Unlike Halo orbits (bifurcating from the eigenvalues of L1/L2), Axial orbits
branch through Type B instability of near-point dynamics.

Theory
~~~~~~

Gómez et al. (2001) proved several bifurcation families of periodic orbits near
collinear points:

- **Type A**: standard Halo bifurcation (out-of-plane amplitude growing from zero)
- **Type B**: Axial bifurcation — different shapes, moving mainly along the axis

Within certain parameter ranges Axials offer coverage Halos don't, fitting
specific mission needs.

Usage
~~~~~

Via the Facade tier-1 interface:

.. code-block:: python

   from e2m2e.api import Facade

   facade = Facade()
   result = facade.design_orbit(
       orbit_type="Axial",
       collinear_point=2,
       amplitude=20000.0,  # km
       epoch=[2024, 1, 1, 0, 0, 0.0],
       duration=365.25 * 86400.0,  # one-year arc (seconds)
   )

Or via lower-level APIs:

.. code-block:: python

   from e2m2e.api.models import DesignOrbitRequest
   from e2m2e.algorithm.design import design_orbit

   request = DesignOrbitRequest(
       orbit_type="Axial",
       collinear_point=2,
       amplitude=20000.0,
       epoch=[2024, 1, 1, 0, 0, 0.0],
   )
   result = design_orbit(request)

References
~~~~~~~~~~

- Gómez, G., Llibre, J., Martínez, R., & Simó, C. (2001). *Dynamics and Mission Design Near Libration Points*. World Scientific.
