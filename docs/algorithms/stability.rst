Stability Analysis
==================

Stability analysis computes the eigenvalue spectrum of periodic orbits'
monodromy matrices to classify their stability.

Floquet theory
~~~~~~~~~~~~~~

A periodic orbit's stability follows from its monodromy matrix's eigenvalues —
the STM over one full period:

.. math::

   \mathbf{M} = \boldsymbol{\Phi}(T, 0)

with T the orbit period. Eigenvalues λ satisfy:

- abs(λ) < 1: stable direction
- abs(λ) > 1: unstable direction
- abs(λ) = 1: center direction (neutrally stable)

Hamiltonian symplecticity pairs eigenvalues as (λ, 1/λ).

Usage
~~~~~

:class:`~e2m2e.algorithm.stability.StabilityAnalysis` analyzes one periodic
orbit:

.. code-block:: python

   from e2m2e.algorithm.stability import StabilityAnalysis

   # Pass orbit at construction; dynamics optional when orbit carries a system
   analyzer = StabilityAnalysis(orbit, dynamics)

   # analyze() takes no arguments; returns an immutable OrbitStability result
   result = analyzer.analyze()

   print(f"Eigenvalues: {result.eigenvalues}")
   print(f"Stability indices: {result.stability_indices}")
   print(f"Classification: {result.classification['stability_type']}")

``analyze()`` returns frozen dataclass ``OrbitStability`` with fields:

- ``monodromy_matrix``: monodromy matrix, shape (6, 6)
- ``eigenvalues``: Floquet multipliers
- ``stability_indices``: dict keyed ``nu1``/``nu2``/``nu3``/``broucke``
- ``classification``: classification result (incl. ``stability_type``, ``is_stable`` …)
- ``bifurcation``: bifurcation analysis results
- ``numerical_errors``: numerical error estimates

Stability indices
~~~~~~~~~~~~~~~~~

Broucke's definition: for each reciprocal pair (λ, 1/λ), sum and take the real part,

.. math::

   \nu = \lambda + \frac{1}{\lambda}

- \|ν\| < 2: mode is stable (eigenvalues on unit circle)
- \|ν\| > 2: mode unstable (larger = less stable)

``stability_indices`` provides per-mode ``nu1``/``nu2``/``nu3`` plus
``broucke`` = \|ν1\| + \|ν2\|. Largest eigenvalue magnitude:
``classification["max_eigenvalue_magnitude"]``.

Batch analysis
~~~~~~~~~~~~~~

Analyze family members one by one:

.. code-block:: python

   from e2m2e.algorithm.stability import StabilityAnalysis

   results = []
   for orbit in family:
       result = StabilityAnalysis(orbit, dynamics).analyze()
       results.append(result)
       nu_max = max(v for v in result.stability_indices.values() if v is not None)
       print(f"Period={orbit.period:.4f}, stability index={nu_max:.6f}")

Bifurcation detection
~~~~~~~~~~~~~~~~~~~~~

The ``bifurcation`` field comes from ``analyze_bifurcation()``, classifying by
eigenvalue proximity to +1/-1/unit circle (``BifurcationType`` enum):

- Near +1 → saddle-node (``SADDLE_NODE``)
- Near -1 → period-doubling (``PERIOD_DOUBLING``)
- Complex pair near circle → torus (``TORUS``)

Family-wide bifurcation detection via static methods:

.. code-block:: python

   # Returns bifurcations near +1 across the family
   bifurcations = StabilityAnalysis.detect_bifurcation_in_family(
       orbits=family,
       dynamics=dynamics,
       tolerance=1e-8,
   )

   # Or locate nearest to a target x0; None when absent
   bp = StabilityAnalysis.find_nearest_bifurcation(
       orbits=family,
       dynamics=dynamics,
       target_x0=0.85,
   )

Numerical error checks
~~~~~~~~~~~~~~~~~~~~~~

Monodromy matrices come from numerical integration; ``numerical_errors``
reports two residuals verifying integration adequacy:

- ``determinant_error``: \|det(M) − 1\| — symplectic determinants stay 1
- ``symplectic_error``: norm of ‖MᵀJM − J‖ — symplecticity residual

Markedly large residuals ⇒ tighten tolerances and re-analyze.

References
~~~~~~~~~~

- Hairer E, Nørsett S P, Wanner G. *Solving Ordinary Differential Equations I*, Chapter IV.8.
- Howell K C. *Three-dimensional, periodic, 'halo' orbits*, 1983.
