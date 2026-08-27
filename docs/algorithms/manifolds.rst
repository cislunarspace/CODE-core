Invariant Manifolds & Poincaré Sections
=======================================

Invariant manifolds (stable/unstable) are asymptotic orbit families around
periodic orbits — the fundamental building blocks of low-energy transfers. This
page covers manifold computation, Poincaré-section tooling, and the
manifold-stitching low-energy transfer pipeline.

Manifold computation
~~~~~~~~~~~~~~~~~~~~

Seed-generation principle: real eigenvalues of a periodic orbit's monodromy
matrix M (one-period STM) give manifold directions — stable takes eigenvectors
with abs(λ) < 1, unstable abs(λ) > 1; unit-circle eigenvalues (including the
periodic direction λ=1) aren't hyperbolic and are dropped. Sample n_points
phases uniformly along the orbit, ferry eigenvectors there via STMs from first
point to each phase, normalize position parts, apply ±ε nondimensional offsets:
those are seeds; integrate stable backwards, unstable forwards to grow tubes.

Usage of :class:`~e2m2e.algorithm.manifold.manifolds.InvariantManifold`:

.. code-block:: python

   from e2m2e.algorithm.manifold import InvariantManifold, ManifoldKind

   # orbit is periodic: bound to system with known period (first point suffices)
   epsilon = 50.0 / 384405.0   # nondimensional offset, typically 50 km / DU
   manifold = InvariantManifold(orbit, ManifoldKind.STABLE, "-", epsilon)

   # Phase-swept seeds, shape (n_points, 6)
   seeds = manifold.seeds(12)

   # Batch-propagate arcs (stable integrates backwards; t_span in absolute value)
   tube = manifold.propagate(4.0)
   print(f"Arcs: {len(tube.trajectories)}")

``branch`` ∈ ``"+"``/``"-"`` — the two perturbation directions toward either side
of the orbit. The returned
:class:`~e2m2e.algorithm.manifold.manifolds.ManifoldTube` carries the orbit
reference, manifold kind, branch, ε; ``trajectories`` lists arcs (nondimensional
CR3BP states).

Passing ``section`` to ``propagate`` truncates each arc at its first section
crossing, appending the refined crossing state as arc endpoint:

.. code-block:: python

   from e2m2e.algorithm.manifold import PoincareSection

   section = PoincareSection.periapsis("earth", orbit.system)
   tube = manifold.propagate(4.0, section=section)

Poincaré sections
~~~~~~~~~~~~~~~~~

:class:`~e2m2e.algorithm.manifold.sections.PoincareSection` is defined by the
zero level-set of s(state); two constructors:

- ``PoincareSection.plane(axis, value)``: planar section s = state[axis] − value;
  ``axis`` indexes state components (0=x…5=vz)
- ``PoincareSection.periapsis(center, system)``: perilune/periapsis section
  s = r·v with r relative to center body (primary/secondary names,
  case-insensitive)

Crossing detection is post-hoc: dense sampling during propagation (manifold
default step 0.005 nondimensional time), evaluating the section per sample,
Brent-refining within sign-change intervals over linearly-interpolated states.
Planar residuals reach below 1e-10.

``crossings()`` detects crossings across all tube arcs, returning
:class:`~e2m2e.algorithm.manifold.sections.SectionCrossings`:

.. code-block:: python

   crossings = section.crossings(tube)
   print(crossings.states.shape)          # (k, 6), refined crossing states
   print(crossings.times.shape)           # (k,), crossing times
   print(crossings.trajectory_index)      # (k,), owning-arc index per crossing

``crossings()`` is post-hoc (propagate first, then find crossings on samples).
For in-integration detection (e.g., stop at first section arrival), generate
scipy-semantic events via
:meth:`~e2m2e.algorithm.manifold.sections.PoincareSection.event` for
``Dynamics.propagate(events=...)``:

.. code-block:: python

   event = section.event(direction=-1, terminal=True)
   result = dynamics.propagate(y0, (0.0, 10.0), events=[event])

See :doc:`../core/dynamics`, event detection.

Stitching & low-energy transfer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pairing crossing points of two tubes on one section yields transfer guesses.
:func:`~e2m2e.algorithm.transfer.low_energy.patch_manifolds` outputs candidates
ascending by weighted cost ``w_r·|Δr| + w_v·|Δv|``
(:class:`~e2m2e.algorithm.transfer.low_energy.PatchCandidate`):

.. code-block:: python

   from e2m2e.algorithm.transfer import patch_manifolds

   # Departure unstable + arrival stable manifolds, same section
   candidates = patch_manifolds(tube_a, tube_b, section, weights=(1.0, 1.0))
   best = candidates[0]
   print(f"|Δr|={best.delta_r:.4e}, |Δv|={best.delta_v:.4e}")

:func:`~e2m2e.algorithm.transfer.low_energy.design_low_energy_transfer` chains the
pipeline: departure-unstable × arrival-stable (globally best over ± branch
combinations) propagate to the secondary's periapsis section; pick the best
candidate; departure leg uses the manifold arc directly; beyond the stitch point
:class:`~e2m2e.algorithm.transfer.three_body_lambert.ThreeBodyLambert` shoots closure
onto the target. Impulses: departure (onto departing manifold), stitching (at
section), arrival (onto target manifold).

.. code-block:: python

   from e2m2e.algorithm.transfer import OrbitTerminal, design_low_energy_transfer
   from e2m2e.data.templates import ConvergenceState

   sol = design_low_energy_transfer(OrbitTerminal(departure_orbit), target_orbit)

   if sol.status == ConvergenceState.CONVERGED:
       print(f"Arcs: {len(sol.arcs)}")           # 2
       print(f"Departure: {sol.arcs[0].delta_v:.6f} km/s")
       print(f"Stitching: {sol.arcs[1].delta_v:.6f} km/s")
       print(f"Arrival: {sol.arrival_delta_v:.6f} km/s")
       print(f"Total: {sol.total_delta_v:.6f} km/s")
       print(f"Transfer time: {sol.transfer_time:.1f} s")

Returns a two-arc :class:`~e2m2e.algorithm.transfer.config.TransferSolution` in
physical units. CR3BP-only today; ephemeris conversion (CR3BP closed solution →
ephemeris model) isn't wired yet — ``epoch`` is reserved for it. End-to-end
benchmark: ``tests/algorithm/transfer/test_low_energy.py``, an intra-family L1
Lyapunov mid-to-large-amplitude case with tens-of-m/s stitches.

.. automodule:: e2m2e.algorithm.manifold.manifolds
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: e2m2e.algorithm.manifold.sections
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: e2m2e.algorithm.transfer.low_energy
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
