轨道库与族生成
==============

目标
----

用 ``Catalog`` 生成一条轨道族并把成员入库，再查询取回。轨道库解决“设计产物
存哪、怎么找”：每条记录一条完整轨迹（元数据 + 数组段），``catalog.db`` 只是
派生索引，``records/*.json + *.npz`` 是事实来源。

前置
----

- 装好 e2m2e；
- 轨道库 **默认关闭** ：必须显式给 ``Config`` 配 ``catalog_dir``，库操作才可用
  （否则抛 ``OrbitError``，错误码 ``CATALOG_NOT_CONFIGURED``）；
  ``catalog_enabled=True`` 才会把任务产物自动入库；
- 族生成是纯 CR3BP 计算，不需要 SPICE 内核。

代码
----

.. code-block:: python

   import tempfile
   from pathlib import Path

   from e2m2e.api import Facade
   from e2m2e.api.config import Config

   catalog_dir = Path(tempfile.mkdtemp(prefix="e2m2e-tutorial-"))
   facade = Facade(Config(catalog_dir=str(catalog_dir), catalog_enabled=True))

   family = facade.catalog.orbit_family_generation(
       orbit_type="DRO",
       min_amplitude_km=2000.0,
       max_amplitude_km=8000.0,
       n_orbits=3,
   )

   print(family.status, family.cause, family.message)
   print("生成成员：", family.generated_members, "/", family.requested_members)
   print("family_id：", family.family_id)

   summary = facade.catalog.catalog_query(orbit_family="dro")
   print("库内命中：", len(summary.records))
   record = facade.catalog.catalog_get(record_id=summary.records[0].record_id)
   print("记录轨迹数组键：", sorted(k for k in record.arrays if k.startswith("cr3bp/")))

输出解读
--------

- ``orbit_family_generation`` 支持
  ``HALO/NRHO/AXIAL/LISSAJOUS/SPO/LPO/HORSESHOE/DRO`` 八族；字段按族适用
  （如 NRHO 用 ``north_south``／``perilune_height_max_km``，LISSAJOUS 用
  ``amplitude_in_km``／``amplitude_out_km``）。缺省值由 model_validator 按
  族填充，DRO 不得携带 ``libration_point``（月心族不绑定平动点）。
- 库开时族产物 **逐成员** 入库（一轨一记录），返回批次标识 ``family_id``；
  经 ``catalog_query(family_id=…)`` 可整族取回。
- ``catalog_query`` 全字段可选、逻辑与；``catalog_get`` 取完整记录（含
  ``request`` 请求快照、``arrays`` 数组段，``cr3bp/`` 前缀是 CR3BP 参考
  轨道），并有 ``to_ephemeris_table()``／``to_orbit()`` 便捷方法。
- 其余工具：``catalog_tag``（写教学标签，整体替换）、``catalog_export``
  （查询子集打包导出）、``catalog_terminology``（分类学标签图例与闭值集）、
  ``catalog_delete``（不可撤销）、``catalog_sweep``（参数空间批量扫描，
  网格维度三选一）。
- 术语（``orbit_family`` 取值、``transfer_type`` 闭值集）以
  ``catalog_terminology`` 与 :doc:`/glossary` 为准。

延伸
----

- CLI：``e2m2e orbit-family-generation --help`` 等；MCP：``orbit_family_generation``
  与 7 个 ``catalog_*`` 工具；族生成同为长任务（worker 子进程执行，见
  :doc:`transfer-design`）。
