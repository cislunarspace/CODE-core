e2m2e.api package
=================

Task-level tier-1 interface (ADR 0014): CR3BP initial guesses, ephemeris
correction, long-arc propagation, transfer design and spacetime transforms
are exposed as Facade methods for upper-level planning systems (CLI / MCP tools).

.. automodule:: e2m2e.api
   :no-index:

e2m2e.api.catalog_ingest module
-------------------------------

Pure builders from artifact-bearing Facade results to catalog records
(ADR 0031). With no artifact, no record is created and ``None`` is returned.

.. automodule:: e2m2e.api.catalog_ingest
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: build_design_record, build_family_record


e2m2e.api.config module
-----------------------

.. automodule:: e2m2e.api.config
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.api.facade module
-----------------------

.. automodule:: e2m2e.api.facade
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.api.models module
-----------------------

.. automodule:: e2m2e.api.models
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.api.cli package
---------------------

.. automodule:: e2m2e.api.cli
   :no-index:


e2m2e.api.cli.main module
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.api.cli.main
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.api.mcp package
---------------------

.. automodule:: e2m2e.api.mcp
   :no-index:


e2m2e.api.mcp.server module
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.api.mcp.server
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.api.mcp.tools module
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.api.mcp.tools
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.api.sidecar package
-------------------------

GUI sidecar stdio protocol (ADR 0035): request/response/progress are JSON
text lines reusing the MCP unified envelope, with large arrays attached as
binary frames. The tool surface is exactly the ``mcp_exposed`` methods on
the Facade.

.. automodule:: e2m2e.api.sidecar
   :no-index:


e2m2e.api.sidecar.frames module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.api.sidecar.frames
   :members:
   :undoc-members:
   :show-inheritance:

