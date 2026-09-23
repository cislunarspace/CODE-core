时空坐标转换
============

目标
----

用 ``Facade.spacetime_transform`` 在四种变换之间转换状态：
``j2000_to_synodic``／``synodic_to_j2000``（惯性 ↔ 地月会合旋转系）与
``gcrs_to_ebcrs``／``ebcrs_to_gcrs``（GCRS ↔ 相对论性地心系，r2s2 后端）。
顺带用 ``Facade.valid_ranges`` 查询各任务的合法参数区间。

前置
----

- 装好 e2m2e；会合系变换按参考历元读星历（SPICE 内核可用）；
  GCRS↔EBCRS 需 ``ephemeris_path`` 指向历表。

代码
----

.. code-block:: python

   from e2m2e.api import Facade

   facade = Facade()

   result = facade.spacetime_transform(
       states=[[384400.0, 0.0, 0.0, 0.0, 1.02, 0.0]],
       times=[0.0],
       transform_type="j2000_to_synodic",
       et0_jd=2461620.5,
   )

   print(result.status, result.cause, result.transform_type)
   print(result.states)

   ranges = facade.valid_ranges()
   print(sorted(ranges.design_orbit)[:5])

输出解读
--------

- ``states`` 是与输入逐行对应的转换后状态；会合系输出为质心原点、物理单位
  km／km·s⁻¹。
- ``times`` 的语义按 ``transform_type`` 区分：GCRS↔EBCRS 用 JD_TDB 绝对时刻；
  会合系转换用无量纲会合时间 ``t_syn``（``0`` 即参考历元 ``et0_jd``）。
- ``details`` 回显变换的辅助量（如参考历元下的会合角速度）。
- ``valid_ranges`` 无参数，返回 ``design_orbit``（逐 ``orbit_type`` 的字段
  区间）、``family_generation_ranges`` 与 ``family_generation_options``
  （族生成的离散选项），是构造请求前查约束的权威入口。

典型用途：``design_orbit``／``orbit_propagation`` 产出的惯性星历先经
``j2000_to_synodic`` 转成会合系，再喂给 ``LGA``／``WSB`` 转移搜索或
``spatiography_classify``。

延伸
----

- CLI：``e2m2e spacetime-transform --help``、``e2m2e valid-ranges --help``；
- MCP：工具名 ``spacetime_transform``、``valid_ranges``。
