Transfer Window Search
======================

:class:`~e2m2e.algorithm.transfer.transfer_search.TransferSearch` scans transfer
parameter space — step one of the search-optimize two-step method.

Algorithm
~~~~~~~~~

Grid search: uniform sampling over parameter ranges:

1. Set departure & arrival orbits
2. Sample departure points uniformly in time along the departure orbit
3. For each departure point, sample α uniformly
4. For each α, compute injection velocity and forward-propagate the transfer arc
5. Screen collision/intersection/distance constraints, recording feasible solutions

Search variables
~~~~~~~~~~~~~~~~

- **α (alpha)**: tangential velocity ratio in ``[alpha_min, alpha_max]``
- **Departure point**: position on the departure orbit; count set by ``n_departure``

Configuration
~~~~~~~~~~~~~

Parameters live on :class:`~e2m2e.algorithm.transfer.config.TransferConfig`'s
``search_*`` fields, with backward-compatible property proxies on
``TransferSearch``:

.. code-block:: python

   from e2m2e.algorithm.transfer import TransferSearch, TransferConfig
   from e2m2e.algorithm.dynamics.dynamics import CR3BP_Dynamics

   # Option 1: via TransferConfig's search_* fields
   cfg = TransferConfig(
       search_alpha_min=0.5,
       search_alpha_max=2.5,
       search_n_alpha=101,
       search_n_departure=200,
       search_max_transfer_time=30.0,
       search_intersection_threshold=1e-3,
       search_min_distance_threshold=1e-3,
       search_collision_earth_radius=200.0 / 384405.0,
       search_collision_moon_radius=100.0 / 384405.0,
       search_integration_dt=0.01,
   )
   searcher = TransferSearch(dynamics, config=cfg)

   # Option 2: attribute assignment (backward compatible)
   searcher = TransferSearch(dynamics)
   searcher.alpha_min = 0.5
   searcher.alpha_max = 2.5
   searcher.n_alpha = 101

Running a search
~~~~~~~~~~~~~~~~

``search()`` takes all parameters at once:

.. code-block:: python

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

   print(f"Total candidates: {len(results)}")
