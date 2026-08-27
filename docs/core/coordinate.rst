Coordinate Frames
=================

The coordinate layer converts vectors (position, velocity, …) between reference
frames.

Core concepts
~~~~~~~~~~~~~

- **Axes**: the orientation part of a frame. Given epoch ``et``, returns rotation
  matrix ``R`` such that ``r_icrf = R @ r_axes``.
- **Origin**: the position part. ``state(et)`` returns the origin's absolute
  state in ICRF/J2000.
- **CoordinateSystem**: a complete mathematical reference frame assembled from an
  Axes plus an Origin.

.. code-block:: python

   from e2m2e.algorithm.coordinate import CoordinateSystem, ICRSAxes, ITRFSpiceAxes, CelestialBodyOrigin

   # ICRF inertial frame, Earth-centered
   icrf = CoordinateSystem(
       axes=ICRSAxes(),
       origin=CelestialBodyOrigin(body="EARTH", spice=spice),
   )

Frame families live in the ``e2m2e.algorithm.coordinate`` subpackage — import
from there.

Axes types
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Type
     - Description
     - Use
   * - ``ICRSAxes``
     - Inertial ICRF frame (identity)
     - Ephemeris propagation default
   * - ``ITRFSpiceAxes``
     - SPICE-backed high-precision ITRF93 (needs LSK/PCK kernels)
     - Spherical-harmonics gravity expansion
   * - ``ITRFApproxAxes``
     - Low-precision approximate ITRF
     - Atmospheric drag
   * - ``VNBAxes``
     - Dynamic axes (Velocity-Normal-Binormal)
     - Thrust direction
   * - ``LVLHAxes``
     - Dynamic axes (Local Vertical-Local Horizontal)
     - Station keeping

Dynamic axes
~~~~~~~~~~~~

VNBAxes and LVLHAxes are dynamic: rotation matrices depend on epoch *and* the
spacecraft's instantaneous state. Call ``update(et, state)`` to refresh the
internal direction cache before use.

State transformation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Transform states between frames
   state_itrf = icrf.transform_state(
       state_j2000, from_cs=j2000_cs, to_cs=itrf_cs, et=et
   )

   # Transform vectors (no origin translation)
   accel_j2000 = icrf.transform_vector(
       accel_itrf, from_cs=itrf_cs, to_cs=j2000_cs, et=et
   )

Frames and units
~~~~~~~~~~~~~~~~

States in one frame may be expressed in different unit systems. The
``UnitSystem`` enum tags dimensionality:

- ``DIMENSIONLESS``: nondimensional units (DU, TU, VU)
- ``SI``: SI (km, s, km/s)

``Orbit`` states are interpreted by the bound ``System``: ``frame`` and
``unit_system`` come from the ``System`` base, tagging reference frame and
dimensions; conversions run through ``CoordinateSystem`` objects (an ephemeris
system may hold a default ``coordinate_system``).
