Integrator Families & Configuration
===================================

e2m2e's integrators come in three families — adaptive single-step Runge-Kutta,
fixed-step multistep Adams, fixed-step second-order Cowell — for first-order
:math:`\dot{y}=f(t,y)` systems or direct second-order :math:`\ddot{x}=a(t,x)`.
The Rust side is a workspace of six crates: ``e2m2e-propagation`` (pure-math
integrators: Butcher tables, RK/ABM/Cowell, solve_ivp), ``e2m2e-forces``
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
   ├── yes → first-order methods only (RK or ABM)
   │        adaptive/unknown steps?
   │        ├── yes → RK (PD45/PD78/RK89)
   │        └── no → ABM (fixed step, low call cost)
   └── no → gravity-only
            fewest states / highest position accuracy?
            ├── yes → Cowell (8th-order double integration)
            └── no → RK or ABM

Accuracy & efficiency
~~~~~~~~~~~~~~~~~~~~~

Normalized LEO+J2 benchmark (1-day arc, tol 1e-13): PD45 ≈ 5000 steps × 7 evals;
PD78 ≈ 800 × 13; RK89 ≈ 800 × 16; ABM(h=0.002) ≈ 500 steps at 2 evals. Higher
orders cut total evaluations ~3×.
