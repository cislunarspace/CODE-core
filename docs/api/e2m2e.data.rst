e2m2e.data package
==================

Data layer: physical constants, coordinate frames and ephemeris kernels, DFH
templates and orbit types.

.. automodule:: e2m2e.data
   :no-index:

e2m2e.data.catalog package
--------------------------

Orbit catalog (ADR 0031): record files are the source of truth; the SQLite
index is a derived artifact that can be rebuilt wholesale. Submodule split:
record defines the record format and segment-array key conventions, store
provides the ``CatalogStore`` engine, index maintains the derived index, and
baseline handles first-use import of the bundled baseline dataset (ADR 0036).
The export surface is re-exported uniformly by the package ``__init__``.


e2m2e.data.catalog.record module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.data.catalog.record
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.data.catalog.store module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.data.catalog.store
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: get


e2m2e.data.catalog.index module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SQLite-based derived index: stores only filter dimensions and file pointers;
record files remain the single source of truth, and the index can be rebuilt
wholesale from the store after deletion. The table schema is an implementation
detail, not an external contract; queries go through ``CatalogStore``, not
this module directly.


e2m2e.data.catalog.baseline module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.data.catalog.baseline
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.data.catalog_baseline package
-----------------------------------

Precomputed baseline dataset bundled with the package (JSON metadata + npz
array segments), covering default samples for every family and libration
point; imported into the user catalog on first use by
``e2m2e.data.catalog.baseline`` (ADR 0036). This package itself exposes no
Python interface.

.. automodule:: e2m2e.data.catalog_baseline
   :no-index:


e2m2e.data.constants package
----------------------------

.. automodule:: e2m2e.data.constants
   :no-index:


e2m2e.data.constants.universal module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.data.constants.universal
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.data.constants.bodies module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.data.constants.bodies
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.data.constants.datums module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.data.constants.datums
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.data.constants.sources module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.data.constants.sources
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.data.frames package
-------------------------

.. automodule:: e2m2e.data.frames
   :no-index:


e2m2e.data.frames.r2s2 module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.data.frames.r2s2
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.data.frames.spice_frames module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.data.frames.spice_frames
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.data.frames.eop module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.data.frames.eop
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.data.frames.leap_seconds module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.data.frames.leap_seconds
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.data.frames.gmat_fixture module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.data.frames.gmat_fixture
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.data.kernels package
--------------------------


e2m2e.data.kernels.manager module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.data.kernels.manager
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.data.kernels.provider module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.data.kernels.provider
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.data.kernels.ephem_cache module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.data.kernels.ephem_cache
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.data.templates package
----------------------------

.. automodule:: e2m2e.data.templates
   :no-index:


e2m2e.data.templates.systems module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.data.templates.systems
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.data.templates.force_config module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.data.templates.force_config
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.data.templates.perturbations module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.data.templates.perturbations
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.data.templates.seed module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.data.templates.seed
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.data.templates.enums module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.data.templates.enums
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.data.types package
------------------------

.. automodule:: e2m2e.data.types
   :no-index:


e2m2e.data.types.epoch module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.data.types.epoch
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.data.types.state module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.data.types.state
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.data.types.orbit module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.data.types.orbit
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.data.types.trajectory module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.data.types.trajectory
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.data.types.maneuver module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.data.types.maneuver
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.data.types.sk_statistic module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.data.types.sk_statistic
   :members:
   :undoc-members:
   :show-inheritance:

