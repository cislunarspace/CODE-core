Glossary
========

CR3BP terms
~~~~~~~~~~~~~~

.. glossary::

   CR3BP (Circular Restricted Three-Body Problem)
      Circular Restricted Three-Body Problem.

   Mass parameter
      μ = m₂/(m₁+m₂): the smaller body's mass over total mass.

   Libration points
      Five equilibrium points (L1–L5) of the rotating frame, specific to the
      CR3BP model. L1–L3 are collinear; L4/L5 triangular. e2m2e says "libration
      point", not "Lagrange point".

   Jacobi constant
      The only integral of motion in CR3BP.

   Halo orbit
      Periodic orbit around a libration point.

   DPO (Distant Prograde Orbit)
      Distant Prograde Orbit — prograde distant family near lunar L2.

   Axial orbit
      Collinear-point periodic family based on Gómez's Type B bifurcation,
      distinct from Halo's Type A mechanism.

   State transition matrix
      Matrix propagating orbital perturbations (STM).

   Differential correction
      Iteratively correcting initial conditions to solve for a periodic orbit.

   Continuation
      Tracing an orbit family from a known solution.

Frames & models
~~~~~~~~~~~~~~~

.. glossary::

   Coordinate system
      Mathematical reference frame assembled from axes + origin
      (``CoordinateSystem``), describing vectors such as position/velocity.
      When chosen as the motion baseline it is the reference frame;
      when used for integration, the integration frame.

   Reference frame
      Coordinate system chosen as the baseline for describing motion. In e2m2e:
      where states live and propagate (``system.coordinate_system``).

   Integration frame
      The frame in which integration executes — coincident with the reference
      frame in e2m2e.

   Frame names
      ``ReferenceFrame`` enum (J2000, INERTIAL, ROTATING, SYNODIC…): names a
      frame; identifies only, performs no conversion.

   Intermediate model
      Four fidelity tiers: two-body < CR3BP < intermediate < high-fidelity
      ephemeris. Intermediate = CR3BP + partial perturbations, below ephemeris.

   ECOM (Empirical CODE Orbit Model)
      Empirical CODE Orbit Model: 9-coefficient empirical SRP model, more
      accurate than a simple cannonball model.

Transfer design
~~~~~~~~~~~~~~~

.. glossary::

   LGA (Lunar Gravity Assist)
      Lunar Gravity Assist — using lunar gravity to rotate the velocity vector,
      saving fuel.

   HMN (Hohmann Transfer)
      Hohmann transfer — minimum-energy two-impulse transfer between coplanar
      circular orbits.

   Lambert problem
      Given two positions and flight time, solve the Keplerian arc joining them.

Propagation & integration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. glossary::

   Propagation
      The whole process of integrating equations of motion to a state history;
      ``Dynamics.propagate``'s duty (step control, result extraction).
      Propagation calls integration.

   Integration
      One low-level numerical step by an ``integrator``. Integration is
      single-step; propagation spans everything.

Equations of motion / dynamics equations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. glossary::

   Kinematic equations
      Equations relating kinematic variables (position, velocity) to time
      (dr/dt = v); about time, not force.

   Dynamic equations
      Equations relating kinematics to force, typically dv/dt = F/m. Full set =
      kinematic + dynamic; ``Dynamics`` integrates their combined state-derivative ODE.

Normal form
~~~~~~~~~~~

.. glossary::

   Normal form
      Layered coordinate transformations eliminating nonlinear couplings near
      CR3BP libration points, reducing dynamics to a few nearly-invariant
      parameters. One-line entry via ``NormalFormPipeline``; context in
      ``NormalFormContext``, results in ``NormalFormResult``.

   rho frame
      6-dim nondimensional relative frame centered on a libration point, state
      ``[ρ, ρ̇]``; all reduction steps happen here.

   Dynamical substitute orbit
      Nearest-to-periodic trajectory in the perturbed (ephemeris) system,
      corrected to closure within a time window by multiple shooting. Produced
      by ``DynamicalSubstituteCorrector``; see
      ``DynamicalSubstituteResult.substitute_orbit``.

   Generating function W
      Generating function of the near-identity map between rho and quasi-Floquet
      coordinates; see ``DynamicalSubstituteResult.W_poly``.

   Quasi-Floquet transform matrix
      Time-varying symplectic matrix ``B(t)`` turning time-varying linearization
      around the substitute into constant-coefficient real normal form (one
      hyperbolic + two center directions), satisfying ``BᵀJB = J``. Solved by
      ``QuasiFloquetReducer``; see ``QuasiFloquetResult``.

   Center manifold
      Invariant manifold of pure center motion after killing hyperbolic-center
      coupling; orbits on it don't escape along hyperbolic directions.
      ``CenterManifoldReducer`` decouples via high-order Lie transforms;
      generating functions at ``CenterManifoldResult.W_series``.

   Characterizing parameters
      Endpoint products of reduction — action-angle coordinates
      ``(q1, p1, I2, θ2, I3, θ3)``, ideally integrals of motion. Invertible chain
      with rho coordinates (``rho ↔ EM ↔ DS ↔ QF ↔ CM ↔ param``), provided by
      ``LibrationCatalogTransformer``.

MBSE
~~~~

.. glossary::

   MBSE (Model-Based Systems Engineering)
      Model-Based Systems Engineering: formal models spanning
      requirements/design/analysis/verification/validation, relations explicitly
      traceable. e2m2e borrows the mindset for component registration,
      requirement traceability, data models, and diagram generation — see
      ``docs/reference/mbse/``.

Spatiography terms
~~~~~~~~~~~~~~~~~~

.. glossary::

   Spatiography
      A "geography of space" (Strughold 1958): dynamical partition of the
      Earth–Moon environment into provinces. Implemented in
      ``e2m2e/algorithm/spatiography/`` per Rosengren et al. 2026 (the
      "Primer"), ADR 0041.

   Five provinces
      terrestrial → cislunar (inner secular / outer resonant) → circumlunar
      → translunar → heliocentric. ``cislunar`` is a band-level name only —
      never an umbrella for the whole Earth–Moon domain; the system-level
      umbrella is "geolunar space" / Earth–Moon system space (Primer §2.6).

   Laplace radius
      Distance where lunisolar secular torques equal Earth's oblateness
      precession — the terrestrial/cislunar onset (Primer Eq. 98,
      r_L ≈ 48812 km). The selenocentric analogue ρ_L ≈ 3846 km bounds the
      lunar-figure-dominated zone (Eq. 124).

   Spheres of influence
      Mutually non-equivalent radial proxies around a secondary (Primer
      §5.4.2): Hill (gateway stability), Laplace–Tisserand (patched-conic
      switching), Chebotarev (direct-force parity), Battin (first-order
      asymmetric). e2m2e implements all four plus the angle-dependent
      activity surface.

   Tidal parity
      Lunisolar tidal parity a_TP ≈ 1.17 a☾ (Primer Eq. 127): orbit-averaged
      quadrupole crossover from lunar-internal to solar-external secular
      dominance. A secular marker, not a gateway or a switching surface.

   Resonance ladder
      Nominal centers of low-order (|k−k_b| ≤ 4) mean-motion commensurabilities:
      interior/exterior lunar and solar sets in geocentric (a, e) space
      (Primer Table 1), exterior-terrestrial set in selenocentric distance
      (Primer Table 2).

   Deliberate overlap
      Adjacent partition zones intentionally overlap near their edges
      (Primer Table 4 note); classifiers therefore return ordered
      multi-labels instead of one exclusive zone.

Orbit taxonomy terms
~~~~~~~~~~~~~~~~~~~~

.. glossary::

   Orbit taxonomy
      The 42-label classification of CR3BP periodic orbits (ADR 0042):
      27 libration-point + 4 moon-centered + 11 resonant labels, adopted
      verbatim from the unreleased STK CODE (Cislunar Orbit Designer)
      component. Criteria are e2m2e's own analytic definitions.

   Taxonomy label
      Structured label (category / family / libration point / hemisphere /
      resonance) with a snake_case canonical string such as
      ``halo_l2_northern`` as the serialization key. Classification is
      measured from the trajectory — never copied from design-side family
      names, which remain as provenance.

   Unclassified
      A legal converged classification outcome (empty label list + reason:
      non-periodic, quasi-periodic, or no matching label) — not an error.

   Resonance ratio p:q
      p satellite revolutions per q lunar revolutions, T/T☾ = q/p (2:1 is
      interior) — the same orientation as the spatiography resonance
      ladders' k:k_b.

   Measured stamping
      Catalog ingest runs the classifier on member trajectories and stores
      ``taxonomy_labels`` alongside the design-side family label; a mismatch
      logs a warning and never fails (ADR 0042 decision 5).
