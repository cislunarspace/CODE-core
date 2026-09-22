分区分析（spatiography）
========================

目标
----

用 ``Spatiography`` 的五个工具做地月空间分区分析：解析尺度、区域分类、边界
几何、共振图集、天图。全部为无状态计算，输入状态、输出几何与场数据。

前置
----

- 装好 e2m2e。分区分析基于解析公式与点质量传播，不需要 SPICE 内核
  （``spatiography_dynamical_map`` 的传播在内部独立完成）。

代码
----

.. code-block:: python

   from e2m2e.api import Facade

   facade = Facade()
   sp = facade.spatiography

   scales = sp.spatiography_scales(elements=["hill_radius_moon_km", "soi_laplace_moon_km"])
   print(scales.status, scales.scales)

   zones = sp.spatiography_classify(
       states=[[80000.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
       frame="synodic_barycentric_km",
   )
   print(zones.zone_ids, [zones.legend[str(z)] for z in zones.zone_ids[0]])

   boundaries = sp.spatiography_boundaries(kind="synodic_planar", resolution=360)
   print(boundaries.state_frame, len(boundaries.elements), "个边界元素")

   atlas = sp.spatiography_resonance_atlas(products=["vzlk_portrait"])
   print(atlas.status, sorted(atlas.vzlk))

   chart = sp.spatiography_dynamical_map(zone="SC", n_a=2, n_e=2, span_years=1.0)
   print(chart.status, chart.span_years, "年", len(chart.fate_legend), "类命运")

输出解读
--------

- ``spatiography_scales``：Hill 半径、Laplace 声速面、Battin 边界等解析尺度
  （物理单位 km），同时给出 L1–L5 精确位置、临界 Jacobi 值 C1–C5 与共振
  名义中心阶梯（Table 1/2 全表）。``elements`` 空表 = 全部。
- ``spatiography_classify``：逐状态给区域 id（重叠带多值、升序），
  ``legend`` 是 id → 名称（terrestrial / cislunar_inner_secular /
  cislunar_outer_resonant / circumlunar / translunar / heliocentric）；
  ``diagnostics`` 逐状态给地心距、月心距、Jacobi 常数等诊断量。
  ``frame`` 必须声明（物理 km 或无量纲）。
- ``spatiography_boundaries``：输出可绘制的边界元素（圆、折线、点、根数
  平面曲线），``state_frame`` 标注数据系——前端只做归一与绘制。
- ``spatiography_resonance_atlas``：三种产品——Gallardo 共振半宽包络
  ``gallardo_widths``、拱线驻定 loci ``secular_loci``、vZLK 相图
  ``vzlk_portrait``；``vzlk`` 携带临界倾角与 vZLK 时间尺度。
- ``spatiography_dynamical_map``：按六域（SC/CR/CG/IT/OT/TF）在 (a/a☾, e)
  网格上传播 EM／EMS 点质量模型，输出 MEGNO Ȳ 场与八类命运场。分辨率与
  积分窗按需收缩——本例取最小网格演示；全量制图走 ``scripts/``。

延伸
----

- CLI：``e2m2e spatiography-classify --help`` 等（下划线转连字符）；
- MCP：五个 ``spatiography_*`` 工具。
