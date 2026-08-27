WSB Low-Energy Solar-Assisted Transfer
======================================

The WSB (Weak Stability Boundary) module implements low-energy indirect lunar
transfers under solar gravity assist, dispatched as ``"WSB"`` by the
:func:`~e2m2e.algorithm.transfer.transfer_orbit` orchestrator with ballistic grid
search + ThreeBodyLambert arrival refinement closing the loop.

Principle
~~~~~~~~~

WSB transfers exploit the Sun-Earth weak-stability-boundary region: leaving the
lunar SOI into a distant solar-perturbed heliocentric arc, solar gravity near
apogee drops Moon-relative speed below capture threshold — ballistic capture at
near-zero Δv (Belbruno & Miller 1993).

Capture uses Belbruno's two-body Kepler energy :math:`H_2`:

.. math::

   H_2 = \frac{|\mathbf{v}|^2}{2} - \frac{\mu_{\text{moon}}}{r}

:math:`H_2 < 0` at perilune means bound to the Moon (ballistic capture).

Two-step solve:

1. **Ballistic grid search**: BCR4BP forward propagation over a solar-phase ×
   departure-phase × TOF grid, screening candidates on perilune altitude and
   :math:`H_2`. Default ``backend="rust"`` — propagation, section detection,
   screening parallelized Rust-side with Rayon; Python implementation runs only
   under explicit ``backend="python"`` (equivalence comparison,
   ``ProcessPoolExecutor``), never auto-fallback.
2. **Arrival refinement**: ThreeBodyLambert shooting refines the best candidate's
   Moon-centered arrival leg.

Usage
~~~~~

.. code-block:: python

   from e2m2e.algorithm.transfer import transfer_orbit
   from e2m2e.algorithm.transfer.wsb import WsbSearchParams

   result = transfer_orbit(
       "WSB",
       wsb_search_params=WsbSearchParams(
           tof_range=(90.0, 150.0),   # TOF range (days)
           max_total_dv=4.0,          # total Δv cap (km/s)
       ),
       departure_state=departure_state,
       target_state=target_state,
   )

   print(f"Status: {result.status.name}")
   print(f"Total Δv: {result.delta_v:.4f} km/s")
   print(f"TOF: {result.details.tof_sec / 86400.0:.2f} days")

Use cases
~~~~~~~~~

- Low-energy Earth-Moon transfers: lunar capture at minimal Δv for fuel-limited missions
- Economical options for landers & orbital inserters
- Low-fuel backup channel for emergency orbit recovery
