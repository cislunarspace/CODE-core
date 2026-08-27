Multiple Shooting
=================

Multiple shooting splits long trajectories into segments with continuity
constraints at nodes, reducing integration sensitivity and improving
convergence.

Basic principle
~~~~~~~~~~~~~~~

Split the orbit into N segments (patch points), integrate each independently,
impose position continuity at joints:

.. math::

   \mathbf{x}_i(t_{i+1}) = \mathbf{x}_{i+1}(t_{i+1})

Adjusting per-segment initial states and (optionally) time nodes satisfies all
constraints simultaneously.

Standard multiple shooting
~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~e2m2e.algorithm.solver.multiple_shooting.MultipleShooting` is the
standard implementation.

.. code-block:: python

   from e2m2e.algorithm.solver import MultipleShooting, sample_patch_points
   from e2m2e.data.templates import ConvergenceState

   ms = MultipleShooting(dynamics=dynamics)

   # Sample patch points from the seed orbit
   t_patch, state_patch = sample_patch_points(seed_orbit, n_points=5)

   result = ms.correct(
       t_patch=t_patch,
       state_patch=state_patch,
       max_iter=50,
       tolerance=1e-10,
       var_time=True,    # allow adjusting time nodes
   )

   if result.status == ConvergenceState.CONVERGED:
       print(f"Converged, max residual {result.max_residual:.2e}")
       print(f"Corrected patch points: {result.state_patch}")

Parameters:

- ``t_patch``: time-node array, shape ``(N,)``
- ``state_patch``: state array, shape ``(N, 6)``
- ``max_iter``: iteration ceiling
- ``tolerance``: position-residual convergence tolerance
- ``var_time``: allow time-node adjustment (more freedom when True)

Perilune-aware sampling
~~~~~~~~~~~~~~~~~~~~~~~

High perilune speeds / ill-conditioned STMs can defeat uniform-time sampling.
``design_orbit`` picks strategies per family (not exposed on request models):

- **Halo**: perilune clustering
  (:func:`~e2m2e.algorithm.solver.multiple_shooting.sample_patch_points_perilune_clustered`)
- **NRHO**: uniform time (#473; step 1 ``revs_per_group=1``). Dropping
  near-perilune nodes
  (:func:`~e2m2e.algorithm.solver.multiple_shooting.sample_patch_points_drop_near_perilune`)
  stays as a utility forcing inclusion of epoch ``t=0``; no longer production default

Perilune clustering: integrate one rev to locate perilune, densify nodes within
windows flanking it:

.. code-block:: python

   from e2m2e.algorithm.solver import sample_patch_points_perilune_clustered

   t_patch, state_patch = sample_patch_points_perilune_clustered(
       orbit,
       dynamics,
       n_base=8,              # uniform-time nodes outside the window
       n_perilune=5,          # clustered nodes inside the window (incl. perilune itself)
       perilune_window=0.15,  # window half-width as fraction of period
   )

Dropping near-perilune nodes: non-epoch nodes fall on complementary arcs beyond
the exclusion zone; epoch ``t=0`` forced in (avoiding holes in segmented
ephemeris grid prefixes):

.. code-block:: python

   from e2m2e.algorithm.solver import sample_patch_points_drop_near_perilune

   t_patch, state_patch = sample_patch_points_drop_near_perilune(
       orbit,
       dynamics,
       n_points=8,
       drop_window=0.12,  # exclusion half-width as fraction of period
   )

Both return time-ascending ``(t_patch, states)``. Non-CR3BP dynamics (no
``mu``) degrade to uniform-time sampling; ``drop_near`` also falls back when
deduplication leaves too few points.

Ephemeris correction
~~~~~~~~~~~~~~~~~~~~

Ephemeris patch-point correction no longer goes through Python
``MultipleShooting``: the design chain uniformly uses Rust multiple shooting
``e2m2e.integrators.multiple_shooting_correct_py`` (default for segmented &
stable orbits), velocity residuals weighted by ``vel_weight`` onto the same
scale as position. The old Python dispatcher (``ephemeris_correction`` package)
and ``TwoLevelMultipleShooting`` are deleted.

References
~~~~~~~~~~

- Montenbruck O, Gill E. *Satellite Orbits*, Chapter 7.
- Multiple shooting in astrodynamics: decompose long arcs into short-segment sequences, lowering sensitivity to initial guesses.
