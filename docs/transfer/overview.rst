Transfer Design Overview
========================

e2m2e provides multiple transfer-design methods, from simple coplanar transfers
(Hohmann) to complex gravity-assist transfers (LGA).

Transfer modules
~~~~~~~~~~~~~~~~

- :doc:`lambert`: two-body Lambert solver with short/long-way and multi-rev solutions
- :doc:`hmn`: direct Hohmann transfer (HMN) — minimum energy between coplanar circular orbits
- :doc:`lga`: lunar gravity assist (LGA), conic patching
- :doc:`wsb`: solar-gravity-assisted low-energy transfer (WSB), H₂<0 ballistic capture
- :doc:`low_thrust`: low thrust — Q-law guess + shooting/collocation
- :doc:`search`: grid search scanning feasible windows in parameter space
- :doc:`optimization`: NLP optimization refining transfers from search results
- :doc:`terminal`: terminal-condition abstractions
- :doc:`propulsion`: propulsion modeling

Search–optimize two-step method (DRO→RO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on Cui et al. (2025)'s search-plus-optimize scheme for DRO→RO transfer
design.

Pipeline:

1. **Search**: scan parameter space for feasible windows
2. **Optimize**: NLP refinement on the best candidates

**Transfer types:**

- ``DIRECT``: direct transfer
- ``LGA``: lunar gravity assist
- ``EXTERNAL``: external transfer

Design variables
~~~~~~~~~~~~~~~~

- **α (alpha)**: tangential velocity ratio, steering departure direction
- **T**: transfer time
- **t_ins**: apogee-to-insertion time

Objective
~~~~~~~~~

Minimize total impulse:

.. math::

   J = \Delta v_1 + \Delta v_2

with :math:`\Delta v_1` departure and :math:`\Delta v_2` insertion impulses.

Constraints
~~~~~~~~~~~

- Position continuity at insertion
- Velocity parallelism with target velocity direction
- Collision avoidance against Earth & Moon

End-to-end example
~~~~~~~~~~~~~~~~~~

The full flow — search, optimize, cost analysis:

.. code-block:: python

   from e2m2e.algorithm.dynamics.system import CR3BP_System
   from e2m2e.algorithm.dynamics.dynamics import CR3BP_Dynamics
   from e2m2e.data.constants import Datum
   from e2m2e.algorithm.transfer import (
       Transfer,
       TransferSearch,
       TransferConfig,
       DROTRONLPOptimizer,
       NLPOptimizationVariables,
       load_orbit_from_json,
   )

   # 1. Dynamics (μ from DE421, ADR 0022)
   system = CR3BP_System(
       mu=Datum.DE421.mu, primary="Earth", secondary="Moon"
   )._with_default_scales()
   system.compute_libration_points()
   dynamics = CR3BP_Dynamics(system)

   # 2. Load departure (DRO) & target (RO)
   dro_orbit = load_orbit_from_json("data/dro_orbit.json")
   ro_orbit = load_orbit_from_json("data/ro_orbit.json")

   # ========== Path A: high-level (search + optimize in one call) ==========
   transfer = Transfer(dynamics)
   transfer.set_orbit(start=dro_orbit, end=ro_orbit)

   result = transfer.optimize(
       initial_guess={"alpha": 1.0, "transfer_time": 15.0, "t_ins": 5.0},
       alpha_range=(0.5, 2.5),
       use_relaxed_velocity=True,
       velocity_angle_tol=0.05,
   )

   if result.success:
       print(f"Total Δv: {result.total_delta_v:.6f} DU")
       print(f"Transfer time: {result.transfer_time:.2f} TU")

   # ========== Path B: lower-level (search & optimize separately) ==========
   searcher = TransferSearch(dynamics)

   results = searcher.search(
       alpha_min=0.5, alpha_max=2.5, n_alpha=101, n_departure=200,
       max_transfer_time=30.0, intersection_threshold=1e-3,
       min_distance_threshold=1e-3,
       collision_earth_radius=200.0 / 384405.0,
       collision_moon_radius=100.0 / 384405.0,
       integration_dt=0.01,
       departure_orbit=dro_orbit, arrival_orbit=ro_orbit,
       verbose=True,
   )

   feasible = searcher.get_feasible_results()
   print(f"Feasible candidates: {len(feasible)}")

   best = min(feasible, key=lambda r: r["dv_departure"] + r["dv_insertion"])

   optimizer = DROTRONLPOptimizer(
       system=system, dynamics=dynamics,
       departure_orbit=dro_orbit, arrival_orbit=ro_orbit,
       departure_state=best["departure_state"],
   )

   nlp_result = optimizer.optimize(
       initial_guess=NLPOptimizationVariables(
           alpha=best["alpha"], transfer_time=best["transfer_time"],
           t_ins=best.get("t_ins", 5.0),
       ),
       alpha_range=(0.5, 2.5),
       transfer_time_range=(1.0, 30.0),
       t_ins_range=(0.0, 10.0),
       use_relaxed_velocity_constraint=True,
       velocity_angle_constraint=0.05,
   )

   if nlp_result.success:
       print(f"Optimized total Δv: {nlp_result.total_delta_v:.6f} DU")
       print(f"Optimized time: {nlp_result.transfer_time:.2f} TU")

   from e2m2e.algorithm.transfer.cost import compute_transfer_cost

   cost = compute_transfer_cost(
       departure_state=nlp_result.departure_state,
       initial_velocity=nlp_result.departure_state[3:] * best["alpha"],
       final_velocity=nlp_result.final_state[3:],
       insertion_velocity=nlp_result.insertion_state[3:],
   )
   print(f"Cost split: Δv1={cost.dv1:.6f}, Δv2={cost.dv2:.6f}, total={cost.total:.6f}")

   import matplotlib.pyplot as plt

   traj = nlp_result.transfer_trajectory
   fig, ax = plt.subplots(figsize=(8, 6))
   ax.plot(traj[:, 0], traj[:, 1], label="Transfer", color="blue")
   ax.scatter(dro_orbit.states[:, 0], dro_orbit.states[:, 1],
              s=1, label="DRO", color="green", alpha=0.5)
   ax.scatter(ro_orbit.states[:, 0], ro_orbit.states[:, 1],
              s=1, label="RO", color="red", alpha=0.5)
   ax.set_xlabel("x (DU)")
   ax.set_ylabel("y (DU)")
   ax.legend()
   ax.set_title("DRO-RO Transfer Trajectory")
   plt.tight_layout()
   plt.show()
