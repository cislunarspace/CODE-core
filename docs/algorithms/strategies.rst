Correction Strategies
=====================

Strategy functions separate configuration logic from the iterative solver. Each
strategy returns an immutable ``CorrectionConfig`` that
``DifferentialCorrection`` consumes to run Newton iterations.

Strategy functions
~~~~~~~~~~~~~~~~~~

Available strategies:

**Planar (2D symmetric):**

- ``symmetric_2d_fixed_x0``: fix x0, adjust y_dot0 & half-period. For Lyapunov, DRO.
- ``symmetric_2d_fixed_t``: fix half-period, adjust x0 & y_dot0. For fixed periods.
- ``symmetric_2d_fixed_y0``: fix y0, adjust x_dot0 & half-period. For resonant orbits (RO).

**Spatial (3D symmetric):**

- ``symmetric_3d_fixed_x0``: fix x0, adjust z0, y_dot0 & half-period.
- ``symmetric_xz_fixed_x0``: XZ-symmetric, x0 fixed.
- ``symmetric_xz_fixed_z0``: XZ-symmetric, z0 fixed.
- ``axial_fixed_vz0``: Axial orbits (Gómez Type B); x-symmetric, vz0 fixed,
  adjusting x0, y_dot0 & half-period.

**Halo-specific:**

- ``halo_fixed_z0``: fix z amplitude & libration point; adjust x0, y_dot0 &
  half-period.
- ``halo_fixed_x0``: fix x coordinate & libration point.

**L4/L5 triangular strategies (planar, no symmetry, full-period closure):**

- ``spo_fixed_x0``: L4/L5 short-period (SPO), x0 fixed; adjust y0, vx0, vy0 &
  full period.
- ``lpo_fixed_x0``: L4/L5 long-period (LPO), same framework as SPO
  (large members are horseshoe-shaped).

CorrectionConfig
~~~~~~~~~~~~~~~~

``CorrectionConfig`` is a frozen dataclass:

.. code-block:: python

   from e2m2e.algorithm.family.strategies import halo_fixed_z0

   config = halo_fixed_z0(z0=0.01, libration_point=1)
   print(config.setup_type)          # "halo_orbit_fixed_z0"
   print(config.free_variables)      # ["x0", "y_dot0", "T_half"]
   print(config.target_conditions)   # {"y": 0.0, "x_dot": 0.0, "z_dot": 0.0}

How strategies meet correctors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   ┌─────────────────┐     CorrectionConfig      ┌─────────────────────┐
   │  Strategy Func  │ ──────────────────────────> │ DifferentialCorrection│
   │ (e.g. halo_*)   │    (immutable config)       │   ._apply_config()    │
   └─────────────────┘                             └─────────────────────┘
                                                          │
                                                          │ iterate_correction()
                                                          ▼
                                                   ┌──────────────┐
                                                   │   Orbit      │
                                                   │ (periodic)   │
                                                   └──────────────┘

New strategies require no solver changes; configs serialize, compare, test.
