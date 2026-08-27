e2m2e.data package
==================

数据层：物理常数、坐标系与星历内核、DFH 模板与轨道类型。

.. automodule:: e2m2e.data
   :no-index:


e2m2e.data.catalog package
--------------------------

轨道库 catalog（ADR 0031）：记录文件是事实来源，SQLite 索引是可全量
重建的派生物。子模块分工：record 定义记录格式与段数组键约定，store
提供存储引擎 ``CatalogStore`` ，index 维护派生索引，baseline 负责随包
基线数据集的首用导入（ADR 0036）。导出面统一由包 __init__ 再输出。


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

基于 SQLite 的派生索引：只存过滤维度与文件指针，记录文件仍是唯一事实
来源，删除后可由 store 全量重建。表结构为实现细节，不作为外部契约；
查询入口走 ``CatalogStore`` ，不直接使用本模块。


e2m2e.data.catalog.baseline module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: e2m2e.data.catalog.baseline
   :members:
   :undoc-members:
   :show-inheritance:


e2m2e.data.catalog_baseline package
-----------------------------------

随包分发的预生成基线数据集目录（JSON 元数据 + npz 数组段），覆盖各族、
各平动点的默认样本；首用由 ``e2m2e.data.catalog.baseline`` 导入到用户库
（ADR 0036），本包自身不含 Python 接口。

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

