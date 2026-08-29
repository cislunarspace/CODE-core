Integrator Families & Configuration
===================================

e2m2e's integrators come in four families — adaptive single-step Runge-Kutta,
adaptive high-order Gauss-Radau (IAS15), fixed-step multistep Adams,
fixed-step second-order Cowell — for first-order
:math:`\dot{y}=f(t,y)` systems or direct second-order :math:`\ddot{x}=a(t,x)`.
The Rust side is a workspace of six crates: ``e2m2e-propagation`` (pure-math
integrators: Butcher tables, RK/IAS15/ABM/Cowell, solve_ivp), ``e2m2e-forces``
(N-body & STM variational equations, compiled force models), ``e2m2e-spice``
(CSPICE FFI; embeds CSPICE when spice feature is on), ``e2m2e-integrators``
(PyO3 bindings; maturin's sole packaging target producing the
``e2m2e._integrators`` extension), plus the level-set reachability crate
``e2m2e-levelset`` (ToolboxLS port) and HJB dynamics crate
``e2m2e-hjb-dynamics``. Python does type conversion and init helpers only.

Overview
~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Family
     - Methods
     - Order
     - Steps
     - Equations
     - Typical use
   * - RK
     - PD45 / PD78 / RK89
     - 5 / 8 / 9
     - adaptive
     - first-order :math:`\dot{y}=f(t,y)`
     - general propagation; accuracy baselines
   * - Adams (multistep)
     - ABM
     - 4
     - fixed
     - first-order :math:`\dot{y}=f(t,y)`
     - smooth RHS, long arcs, known steps
   * - Cowell (double integration)
     - Störmer-Cowell
     - 8
     - fixed
     - second-order :math:`\ddot{x}=a(t,x)`
     - gravity-only orbits; avoids state doubling
   * - IAS15 (Gauss-Radau)
     - IAS15
     - 15
     - adaptive
     - first-order :math:`\dot{y}=f(t,y)`
     - high-precision long arcs; close encounters; STM + parameter sensitivity

Runge-Kutta family
~~~~~~~~~~~~~~~~~~

Three embedded explicit RK methods share one ``rk_step`` entry, differing only in
Butcher tables; each step computes high- and low-order solutions whose difference
estimates local error for adaptive stepping.

.. list-table::
   :header-rows: 1

   * - Enum
     - Full name
     - Main/embedded order
     - Stages
     - Source
   * - ``RkMethod.PD45``
     - Prince-Dormand 5(4)
     - 5 / 4
     - 7
     - Dormand & Prince 1980
   * - ``RkMethod.PD78``
     - Prince-Dormand 8(7)13M
     - 8 / 7
     - 13
     - Hairer-Wanner / GMAT
   * - ``RkMethod.RK89``
     - Verner 9(8)
     - 9 / 8
     - 16
     - GMAT R2026a

PD45: low-order, low-overhead default. PD78/RK89 take fewer steps at equal
tolerance (>5× step savings typical) — suited to long arcs / cross-checks.

Key parameters:

- ``tol``: relative tolerance; acceptance threshold scales as ``tol * max(1, ||y||)``
  for consistent behavior across nondimensional and physical units.
- ``h0``: initial trial step; controller converges within a few steps.
- ``result.error``: L2 difference of high vs embedded solutions (local error estimate).
- ``result.h_next``: suggested next step ``h * clamp(0.9·(tol/error)^(1/(p+1)), 0.1, 5)``.
- ``state_error_dim``: restrict step-error accounting to first N dims
  (e.g., 6 during STM-augmented propagation so 36 STM entries don't dominate).

IAS15 (adaptive Gauss-Radau)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

15th-order implicit Gauss-Radau predictor-corrector with compensated
summation, implemented from the published algorithms (Rein & Spiegel 2015;
Everhart 1985 — REBOUND/ASSIST are GPL, no code referenced). Built for
high-precision long-arc extrapolation: round-off accumulates per Brouwer's law
(:math:`\sqrt{n}`) instead of linearly, and the step shrinks automatically near
close encounters. Selected via ``ForceModel.propagate(integrator="ias15")``;
STM propagation and force-model parameter sensitivity are supported.

Key parameters (differ from the RK family):

- ``tol``: single relative tolerance with IAS15 paper semantics (sampled
  relative-acceleration error magnitude), not the RK rtol/atol pair;
  ``initial_step`` is ignored (built-in start heuristic); ``method=`` is
  RK-only.
- ``with_stm``: state-transition-matrix augmentation along the trajectory.
- ``sens_params``: first-order variational columns for force-model parameters
  (``"srp_cr"``, ``"drag_cd"``); requires ``with_stm=True`` and each label must
  uniquely match one enabled force; results appear as ``sensitivity`` with
  shape ``(n_points, 6, n_params)``.

Ephemeris noise floor: under ephemeris force models (SPICE sampling) the
acceleration's effective smoothness is ~1e-11 relative, so the 7th-order
divided-difference error estimate stops shrinking with step size. The engine
detects this stagnation and raises the effective tolerance instead of
rejecting steps indefinitely. Purely analytic force models follow ``tol``
strictly.

ABM (fixed-step multistep)
~~~~~~~~~~~~~~~~~~~~~~~~~~

ABM = 4-step 4th-order PECE predictor-corrector; two RHS evaluations per step but
needs four derivative samples of history. Fixed-step (history tied to h — restart
via ``initialize_abm_history`` on change); startup by RK89;
low per-step function-evaluation cost on long arcs.

Cowell (Störmer-Cowell)
~~~~~~~~~~~~~~~~~~~~~~~

Directly integrates :math:`\ddot{x} = a(t, x)`: half the state dimensions, 8th
order acting directly on position. Position-only output (``cowell_step`` returns
``x_new``); acceleration-only RHS (no velocity-dependent forces); needs 8
acceleration + 2 position history samples; startup default ``n_startup=7`` ≥ 7.

Decision tree
~~~~~~~~~~~~~

.. code-block:: text

   RHS contains velocity terms (drag, thrust direction)?
   ├── yes → first-order methods only (RK, ABM, or IAS15)
   │        max long-arc precision / close encounters / parameter sensitivity?
   │        ├── yes → IAS15 (15th-order Gauss-Radau)
   │        ├── adaptive/unknown steps → RK (PD45/PD78/RK89)
   │        └── fixed known steps → ABM (low call cost)
   └── no → gravity-only
            fewest states / highest position accuracy?
            ├── yes → Cowell (8th-order double integration)
            └── no → RK or ABM

Accuracy & efficiency
~~~~~~~~~~~~~~~~~~~~~

Normalized LEO+J2 benchmark (1-day arc, tol 1e-13): PD45 ≈ 5000 steps × 7 evals;
PD78 ≈ 800 × 13; RK89 ≈ 800 × 16; ABM(h=0.002) ≈ 500 steps at 2 evals. Higher
orders cut total evaluations ~3×. IAS15 (tol 1e-13, analytic force model)
closes a 10-rev circular orbit to <1e-7 km and holds 100-rev relative energy
drift <1e-9 (Brouwer-law round-off).
