Station Keeping
===============

.. sectionauthor:: e2m2e

Station keeping: orbit design delivers a nominal orbit; in flight, navigation
errors, thrust execution errors, and SRP parameter errors drift the real
trajectory away from it, and station-keeping control laws periodically compute
corrective maneuvers to pull the orbit back. ``e2m2e/algorithm/station_keeping/``
provides target-point strict/loose control and special-point control laws,
with error models, momentum management, and three-orbit Monte Carlo evaluation,
orchestrated by ``controller`` and reached through the Facade ``control_orbit``
entry point.

Calibration Basis of Parameters
-------------------------------

The two empirical conclusions below originally lived in code comments and were
moved here after comment-level cleanup removed the historical war-story
narratives (calibration measurements: `#541
<https://github.com/cislunarspace/e2m2e/issues/541>`__); the values stated are
the current code defaults.

Monte Carlo evaluation uses screening-level tolerance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Monte Carlo evaluation targets statistical features — mean/variance-level
judgments at ±30% magnitude — not the exact position of one trajectory.
Screening-level propagation tolerance therefore suffices: research-grade
tolerance (1e-12 full chain) does not improve the conclusions, it only costs
time. Measured calibration:

- ``tolerance=1e-12``: 4 controls × 2 samples, about 156 s;
- loosened to ``1e-10``: about 3× speedup, statistical conclusions unchanged.

The default ``rtol``/``atol`` of ``PropagatorFactory`` (both 1e-10)
take the loose side; tighten separately only for a future
single-sample precise-trajectory re-verification.

Strict target-point control needs max_iter of at least 6
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nonlinear effects are strong for strict target-point control on NRHOs: the
differential-correction iteration cap ``max_iter`` of
:class:`~e2m2e.algorithm.station_keeping.target_point.StrictTargetPointLaw`
defaults to 6:

- a cap of 2 fails to converge;
- 6 is the measured working floor;
- larger values only cost time.

The ``tight_max_iter`` default used when Monte Carlo evaluation builds the
strict law (``control_mode=2``) is likewise 6.
