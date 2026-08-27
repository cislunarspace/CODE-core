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
