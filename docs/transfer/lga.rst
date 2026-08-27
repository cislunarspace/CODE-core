LGA Lunar Gravity Assist
========================

The LGA module designs lunar-gravity-assist indirect transfers via conic
patching: lunar flybys rotate the spacecraft's velocity vector in magnitude and
direction, buying orbital changes cheaply.

Design method
~~~~~~~~~~~~~

Three legs:

1. **Departure**: hyperbolic escape from the departure orbit
2. **Flyby**: velocity-vector change inside the lunar sphere of influence
3. **Arrival**: hyperbolic insertion onto the target orbit

Usage
~~~~~

.. code-block:: python

   from e2m2e.algorithm.transfer.lga import search_lga_trajectories, LgaSearchParams
   from e2m2e.data.templates import ConvergenceState

   # departure_state / target_state: CR3BP nondimensional states; system/dynamics: CR3BP system
   candidates = search_lga_trajectories(
       departure_state=departure_state,
       target_state=target_state,
       system=system,
       dynamics=dynamics,
       params=LgaSearchParams(
           tof_range=(15.0, 45.0),   # flight-time range (days)
           max_total_dv=4.0,         # total Δv cap (km/s)
       ),
   )

   if candidates.status is not ConvergenceState.CONVERGED:
       print(f"Search incomplete: {candidates.status.value}, {candidates.cause.value}: {candidates.message}")

   for c in candidates:
       dv_km_s = c.total_dv * system.characteristic_velocity  # nondimensional → km/s
       print(f"Total Δv: {dv_km_s:.4f} km/s, TOF: {c.tof_sec / 86400:.2f} d, "
             f"perilune altitude: {c.perilune_alt_km:.1f} km")

References
~~~~~~~~~~

- Cui, H. et al. (2025). Transfer orbit design for cislunar space missions.
