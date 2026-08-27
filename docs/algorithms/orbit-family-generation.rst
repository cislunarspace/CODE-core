Unified Orbit Family Generation
===============================

``Facade.orbit_family_generation()`` generates eight family types — Halo, NRHO,
Axial, Lissajous, SPO, LPO, Horseshoe, DRO — through one interface. The Facade
returns its dedicated Pydantic response ``FamilyGenerationResponse``; it
inherits :class:`e2m2e.data.types.orbit.OrbitFamily` to keep existing reading
interfaces while carrying the ``status/cause/message`` triple directly.
``n_orbits`` caps member count without guaranteeing fullness; on numeric
soft failures the same response retains already-converged members. The
algorithm-layer entry keeps using
:class:`e2m2e.algorithm.results.FamilyGenerationResult` for soft failure.

Parameter contract
~~~~~~~~~~~~~~~~~~

Common parameters: ``orbit_type``, ``libration_point``, ``n_orbits``; DRO is
Moon-centered — requests must not carry ``libration_point`` (explicit presence
rejects, consistent with other families rejecting cross-family fields). All
other fields are per-family:

.. list-table::
   :header-rows: 1

   * - Family
     - Libration point
     - Family params
     - Method
   * - Halo
     - L1/L2
     - ``max_amplitude_km``, ``sampling_mode=natural-z0`` (north +, south −)
     - Natural-parameter continuation at fixed z0
   * - NRHO
     - L1/L2
     - ``north_south``, perilune height, ``continuation_direction=toward-moon``
     - L1: single Rust PAL; L2: fold then fixed-x0 continuation
   * - Axial
     - L1/L2
     - ``max_amplitude_km``, ``continuation_direction=increase-amplitude``
     - Walking with vz0 as Type-B family parameter
   * - Lissajous
     - L1/L2/L3
     - In/out-of-plane amplitudes, two phases, ``sampling_mode=linear-amplitudes``
     - Amplitude sampling (not continuation)
   * - SPO
     - L4/L5
     - amplitude range, ``continuation_direction`` (increase/decrease-x0), match tol.
     - Planar full-period PAL
   * - LPO
     - L4/L5
     - amplitude range, ``continuation_direction`` (increase/decrease-x0), match tol.
     - Planar full-period PAL
   * - Horseshoe
     - L4/L5
     - amplitude range, ``continuation_direction`` (increase/decrease-x0), match tol.
     - Large-amplitude classification of the LPO chain
   * - DRO
     - none (Moon-centered)
     - amplitude range, ``sampling_mode=natural-x0``
     - Single x0 natural continuation from the standard seed (bidirectional across-seed walking)

Request models fill defaults and validate conditional ranges per family.
Ranges are queryable before constructing requests:

.. code-block:: python

   from e2m2e.api.models import FamilyGenerationRequest

   ranges = FamilyGenerationRequest.valid_ranges("HORSESHOE", libration_point=4)
   print(ranges["min_amplitude_km"].format_interval())  # [50000.0, 110000.0]

   options = FamilyGenerationRequest.valid_options("LPO")
   print(options["continuation_direction"])
   # ("decrease-x0", "increase-x0")

DRO amplitude reuses single-orbit ``design_dro``'s definition (mean of min/max
lunar distance over one period, km) with matching request envelope
(1737–110000 km). Members come from one x0 natural continuation: below the
standard-seed amplitude (~90786 km) walk moonward, above walk earthward,
straddling seeds walk both ways; members return ascending by amplitude;
member parameters carry geometry only (``amplitude_km`` etc.), no libration point.

Periodic vs quasi-periodic semantics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Halo/NRHO/Axial/SPO/LPO/Horseshoe/DRO members are strictly periodic
(``family.periodicity == "periodic"``). Lissajous's in/out-plane frequencies are
irrational-ratio: members are quasi-periodic bounded multi-point trajectories on
Rust's nonlinear center-reduced flow without periodic closure; the unified
container distinguishes explicitly:

.. code-block:: python

   family = facade.orbit_family_generation(
       orbit_type="LISSAJOUS",
       libration_point=2,
       amplitude_in_km=2400.0,
       amplitude_out_km=7200.0,
       phase_in=0.01,
       phase_out=0.55,
       n_orbits=3,
   )

   assert family.is_quasi_periodic
   assert family.metadata["periodicity"] == "quasi-periodic"

A Lissajous member's ``period`` is the nominal in-plane one, not full-trajectory
closure. Downstream must neither re-propagate ``states[0]`` as a strict periodic
initial state nor apply periodic-closure assertions.

Numeric boundary
~~~~~~~~~~~~~~~~

Python only validates requests, dispatches families, rewraps domain results.
Rust's single family-generation entry hides seed construction, CR3BP propagation
+ STM, differential correction, PAL Newton, step control, member filtering,
collinear-point center modes, Lissajous trajectory sampling, and perilune-distance
/ L4-L5 radial / out-of-plane amplitude measurement — no Python numeric fallback.
Lissajous advances with the full CR3BP nonlinear potential gradient inside the
4-D center subspace, freezing hyperbolic directions; equivalence with the single-
orbit entry's high-order normal form is not claimed.

Example
~~~~~~~

This request generates up to five L4 SPO members between 5,000–20,000 km:

.. code-block:: python

   from e2m2e.api.facade import Facade

   family = Facade().orbit_family_generation(
       orbit_type="SPO",
       libration_point=4,
       min_amplitude_km=5000.0,
       max_amplitude_km=20000.0,
       n_orbits=5,
   )

   for orbit in family:
       print(orbit.parameters["amplitude_km"], orbit.period)

Runnable version: ``examples/main_family_generation.py``.
