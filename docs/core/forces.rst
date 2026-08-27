Force Models
============

The forces subpackage provides configurable, composable spacecraft
perturbation-force models: buildable from config dicts, serializable to JSON,
integrated with ephemeris dynamics for propagation.

Core concepts
~~~~~~~~~~~~~

- **PhysicalModel**: abstract base of all force models defining
  ``compute_acceleration(t, state, system)``. Two optional hooks:
  ``compute_jacobian(t, state, system)`` returning analytic ∂a/∂r (default
  ``None``; ``ForceModel`` falls back to finite differences — Python fallback
  paths were removed with issue #378, production runs require Rust); and
  ``to_rust_spec(system)`` serializing the force into a tuple accepted by the
  Rust compiled path (default ``None`` = not compilable).
- **ForceModel**: the composition container assembling multiple
  ``PhysicalModel``\ s into one propagation's equations of motion; registers,
  enables/disables by name; propagates via Rust integrators.
- **ForceEntry**: per-force registry record inside a container holding ``name``,
  ``force``, ``enabled``.

Supported force types
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Type
     - Description
     - Config type name
   * - PointMassGravity
     - Central body point-mass gravity
     - ``PointMassGravity``
   * - ThirdBodyGravity
     - Third-body perturbation (incl. indirect term)
     - ``ThirdBodyGravity``
   * - IndirectTerm
     - Standalone indirect-term correction
     - ``IndirectTerm``
   * - GravityField
     - Spherical-harmonics gravity field (body-agnostic; EGM96/GRGM900C & custom .gfc/.cof)
     - ``GravityField``
   * - DragModel
     - Atmospheric drag (injected density model)
     - ``DragModel``
   * - SolarRadiationPressure
     - SRP (cannonball + optional shadow model)
     - ``SolarRadiationPressure``
   * - EcomSolarRadiationPressure
     - ECOM empirical SRP (9-coefficient DYB; DFH-compatible)
     - ``EcomSolarRadiationPressure``
   * - FiniteBurn
     - Continuous thrust (closed DSL: constant/pulse profile + fixed direction)
     - ``FiniteBurn``
   * - VariableMassFiniteBurn
     - Variable-mass continuous thrust (mass as 7th state)
     - none (constructed directly, not via config registry)
   * - RelativisticCorrection
     - Post-Newtonian corrections (Schwarzschild / Lense-Thirring / de Sitter)
     - ``RelativisticCorrection``

Also ``ImpulsiveBurn`` (instant Δv at an epoch) — not a ``PhysicalModel``, no
acceleration accumulation, not config-registerable; applied via
``propagate_maneuvers`` (below).

.. warning::

   When using ``GravityField`` for the Moon (including degree=0 central term),
   you must separately add lunar indirect term ``IndirectTerm("MOON")`` and must
   NOT also add ``ThirdBodyGravity("MOON")``: the latter would double-count the
   Moon's point mass with GravityField's central term.

Built-in formulas (summary)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**PointMassGravity**: :math:`\mathbf{a} = -\mu\,\mathbf{r}/|\mathbf{r}|^3` — origin-body two-body gravity only.
**ThirdBodyGravity**: direct + indirect terms referencing SPICE positions;
indirect subtraction keeps the frame origin fixed.
**GravityField**: fully-normalized harmonics via Pines recursion; Earth solid
tides (Step1+Step2+pole+permanent), Moon k₂ = 0.024116 tide.
**DragModel**: computed in ITRF;
:math:`\mathbf{a} = -\tfrac{1}{2}\rho\,(C_d A/m)\,|\mathbf{v}_{rel}|\,\mathbf{v}_{rel}`
with exponential atmosphere (US76 segments), Cd default 2.2.
**FiniteBurn**: :math:`T(t)/m \cdot \hat{\mathbf{d}}`; direction frames inertial /
VNB / LVLH; round-trip configs only through the closed DSL.
**VariableMassFiniteBurn**: same but mass = state[6], burned at
:math:`\dot m = -T/(I_{sp} g_0)`; propagation auto-switches to 7-D augmented
Rust path.
**RelativisticCorrection**: Schwarzschild + Lense-Thirring + de Sitter, GMAT-
aligned.

Registry mechanism
~~~~~~~~~~~~~~~~~~

Auto-naming by class name with disambiguation (``GravityField_2`` …);
explicit names required unique; enable/disable/get/remove/list by name:

.. code-block:: python

   fm.add_force(GravityField("EARTH", degree=2, order=0))          # auto name
   fm.add_force(DragModel(atmosphere=ExponentialAtmosphere(), area=10.0, mass=1000.0), name="drag")
   fm.disable("drag")
   fm.get_force("drag")
   fm.remove_force("drag")

Propagation interface
~~~~~~~~~~~~~~~~~~~~~

``ForceModel.propagate`` drives adaptive stepping on Rust ``rk_step``: options
``t_eval``, ``with_stm`` (42-dim augmented integration — STM components excluded
from step-error control, matching GMAT), ``initial_step``, ``max_steps``,
``method=RkMethod.PD45`` default.

Events unsupported (raises ``NotImplementedError`` on non-None events): use
``Dynamics.propagate`` or post-hoc detection instead. For impulsive maneuvers:
``propagate_maneuvers(initial_state, t_span, burns)`` sorts ``ImpulsiveBurn``\ s by
epoch, coasts between, applies ``state[3:6] += delta_v``, returns extra ``burns``
key.

Compiled Rust fast path: with spice enabled, propagation enters
``propagate_compiled`` — one serialization then full loop in Rust. No silent
fallbacks: missing extension raises ``RustExtensionUnavailableError``; any
enabled force whose ``to_rust_spec()`` is None raises ``NotImplementedError``
(ADR 0020 decision 4). The STM fast path additionally excludes
``RelativisticCorrection`` / ``VariableMassFiniteBurn`` for now (explicit error).

Rust ephemeris pre-sampling cache: ``enable_ephem_cache(targets, frame_pairs,
et_start, et_end, dt, sxform_pairs)`` pre-samples body states + frame matrices
into in-memory cubic splines so inner-loop lookups skip FFI; C² continuity avoids
step-size collapse at grid nodes (~1e-3 km terminal accuracy at 600 s pxform
grids). Wrap propagation in try/finally with ``disable_ephem_cache()``.

Config-driven construction
~~~~~~~~~~~~~~~~~~~~~~~~~~

Build from a dict: ``{"version": 1, "forces": [{"name", "type", "enabled",
"params"}...]}``, nested models (atmosphere/shadow) as recursive ``{type,
params}``. Entry points: ``ForceModel.from_config(config, system)``,
``fm.to_config()`` (normalized values; round-trip contract =
``to_config(from_config(c)) == c`` after re-serialization),
``dump_force_config(fm, path)`` / ``load_force_config(path, system)``. Full LEO
walkthrough in the Chinese section below (identical code).

Common errors
~~~~~~~~~~~~~

- Unknown name → ``KeyError`` from get/disable/remove.
- Duplicate explicit name → ``ValueError: force name '...' already exists``.
- Version mismatch → ``unsupported config version 2; expected 1``.
- Unknown type → ValueError listing known types.
- User-written callables in ``FiniteBurn`` propagate fine but raise
  ``NotSerializableError`` on ``to_config()`` — use the DSL kinds
  (constant/pulse).

Solar radiation pressure & shadow model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**SRP** (cannonball, Montenbruck & Gill):
:math:`\mathbf{a} = \text{flux} \cdot P_{1AU}(1\text{AU}/r)^2 \frac{C_R A}{m} \hat{\mathbf{u}}`
with :math:`P_{1AU} = 4.56\times10^{-6}` N/m²; params: area & mass (required),
cr (default 1.5), shadow (default None = full illumination).
**ConicalShadowModel** implements GMAT ShadowState conical algorithm
(M&G §3.4.2): four branches — full sun / umbra (flux=0) / penumbra (exact disc-
overlap area, eq. 3.92–3.94) / annular. Multi-occulter composition per GMT-6543:
any umbra → 0; disjoint partials → inclusive-exclusive; overlapping → min.
Default occluder radii: Earth 6378.1363, Moon 1737.4, Sun 695700 km. Both
models serialize through ``force_config`` and offer pure-function test paths
without SPICE.
