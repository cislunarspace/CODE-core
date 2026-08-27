Normal Form Reduction Pipeline
==============================

The normal form reduces the complex dynamics near CR3BP libration points into
action-angle characterizing parameters, letting designers describe an orbit that
would otherwise need six time-evolving state variables with a few constants.

``NormalFormPipeline`` chains the reduction into one call:

    ephemeris orbit initial values → dynamical substitute orbit → quasi-Floquet
    transform → center-manifold reduction → characterizing parameters

Features
~~~~~~~~

- Ephemeris-orbit → characterizing parameters in one line
- The returned ``NormalFormResult`` exposes both reduction diagnostics and
  coordinate-transform entries
- Libration point (L1–L5), epoch, expansion order configured via ``NormalFormContext``

Usage
~~~~~

.. code-block:: python

   from e2m2e.algorithm.normal_form import NormalFormContext, NormalFormPipeline
   from e2m2e.algorithm.dynamics import CR3BP_System, LibrationPoint

   system = CR3BP_System(mu=1.215058560962404e-2, primary="Earth", secondary="Moon")
   context = NormalFormContext(
       system=system,
       libration_point=LibrationPoint.L1,
       epoch=2451545.0,
       order=4,
   )

   # rho-frame initial values [ρ, ρ̇] (nondimensional)
   x0 = [1e-3, -1e-3, 0.0, 0.0, 1e-4, -1e-4]

   result = NormalFormPipeline(context).reduce(x0)

   # rho coords → characterizing parameters [q1, p1, I2, θ2, I3, θ3]
   param = result.catalog_transformer.rho_to_param(x0, t=0.0)

``result`` is an immutable container providing whole-pipeline convergence
diagnostics (``success``, ``substitute_residual``, ``message``, ``metadata``)
and exposing the four sub-results as first-class fields for consumers digging
into one tier. ``residual`` is a backward-compat alias of ``substitute_residual``
— prefer the latter in new code.

Persistence
~~~~~~~~~~~

``NormalFormResult`` serializes to disk for archiving and reuse:

.. code-block:: python

   from e2m2e.algorithm.normal_form import NormalFormResult

   # Save as .npz (all sub-results + context params serialized)
   result.save("result.npz")

   # Rebuild; catalog_transformer reconstructed from three sub-results automatically
   result = NormalFormResult.load("result.npz")

``catalog_transformer`` isn't stored separately — rebuilt on load from the
dynamical-substitute / quasi-Floquet / center-manifold sub-results, numerically
equivalent on ``rho_to_param`` / ``param_to_rho`` to the original.

Tunable knobs
~~~~~~~~~~~~~

Constructing ``NormalFormPipeline`` can override defaults:

- ``quasi_floquet_method``: matrix (default; 36-dim direct integration) or
  lie_algebra (21-dim sp(6) parameterization, symplectic by construction)
- ``center_max_order``: center-manifold Lie-transform truncation order, default 10
- ``center_steps``: reduction-step tuple, default ("invariant", "center")
- ``dynamical_kwargs``: overrides passed to the substitute corrector (e.g., {"t_total": 8.0})

Submodules
~~~~~~~~~~

The pipeline calls these reducers in turn; reach one tier's results via
``result.ds_result`` / ``result.qf_result`` / ``result.cm_result``:

- Dynamical substitute: ``DynamicalSubstituteCorrector``
- Quasi-Floquet transform: ``QuasiFloquetReducer``
- Center manifold: ``CenterManifoldReducer``
- Characterizing-parameter catalog: ``LibrationCatalogTransformer``
