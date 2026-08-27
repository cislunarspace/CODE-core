e2m2e.api package
=================

任务级一档接口（ADR 0014）：把 CR3BP 初猜、星历修正、长期预报、转移设计、
时空转换等能力封装为 Facade 方法，供上层规划系统（CLI / MCP 工具）调用。

.. automodule:: e2m2e.api
   :no-index:


e2m2e.api.catalog_ingest module
-------------------------------

产物型 Facade 结果到 catalog 记录的构建纯函数（ADR 0031）。无产物时
不建记录，返回 ``None`` 。

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

GUI sidecar stdio 协议（ADR 0035）：请求/响应/进度为 JSON 文本行，复用
MCP 统一信封，大数组经二进制帧附加传输。工具面即 Facade 上 mcp_exposed
的方法。

.. automodule:: e2m2e.api.sidecar
   :no-index:


e2m2e.api.sidecar.frames module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.api.sidecar.frames
   :members:
   :undoc-members:
   :show-inheritance:

