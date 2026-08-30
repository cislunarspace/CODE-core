Spatiography (cislunar partition)
=================================

.. sectionauthor:: e2m2e

地月空间分区（spatiography）：把地球-月球环境按动力学边界划分为五省
terrestrial → cislunar（内带长期主导 / 外带共振结构化）→ circumlunar →
translunar → heliocentric，边界量全部闭式解析。理论依据为 Rosengren et al.
2026《The Astrodynamics Primer on Cislunar and Translunar Space》§5，
工程裁决与复现陷阱清单见 ADR 0041
（``docs/adr/0041-spatiography-partition.md``）。

模块布局（``e2m2e/algorithm/spatiography/``）
---------------------------------------------

- ``constants``：:class:`~e2m2e.algorithm.spatiography.PrimerConstants`——
  论文自洽常数集（SPICE GM + Simon 1994 月根数 + GGM02 J2），由
  ``constants.toml`` 的 ``[primer]`` 节装配。**不得与 DE421 GM 或
  384405 km 地月距离混搭**。
- ``scales``：闭式尺度纯函数——地心/月心 Laplace 半径、Hill 球、四种
  影响球（Laplace–Tisserand / Chebotarev / Battin 非对称 / 角依赖活动面）、
  日月潮汐平权 a_TP、Tisserand 参数、GEO 参考半径。
- ``resonances``：Table 1/2 共振梯（内月 9 + 外月 9 + 日 5+1 + 月心外地球
  16 条名义中心与开普勒周期）；Gallardo 半解析共振半宽（式 100–104）——
  共面切片、2ρ_H 近遇截断、Simon 1994 月根数，计算设置对齐论文 Fig. 8，
  输出 (a, e) 平面半宽包络。
- ``secular``：长期共振 loci 与 vZLK 骨架（式 64–78）——外支/月内支
  四极长期前置频率（式 47/56）、cislunar 拱线驻定水平线（式 76）、
  translunar 混合驻定倾角曲线（式 78，两极限 1/5 与 (1−e²)/5）、
  vZLK 相图（式 65–68 的 c2 等值线）、特征频率与时间尺度（式 69–71）、
  有效性筛选（J2 在 r_L 内抑制、α = a/a☾ → 1 破坏双平均）。
- ``regions``：五省分类器（Table 1 / Table 4 双口径，重叠带多标签）、
  Jacobi 五拓扑 Case I–V 与临界值（平动点精确求根）。
- ``megno``：MEGNO 混沌指标（式 142）——Rust 内核（14 维增广：状态 +
  切变分 + 两累加器）的调度入口与 scipy 参照实现（仅等价性对照）。
- ``cartography``：六域两层天图管线（§7.3 / Table 4）——命名场景
 （2027-08-02 日全食统一初值切片）、Table 4 六域网格驱动、EM/EMS
  双模型（Rust 地心点质量内核：固定开普勒椭圆天体 + 局部切变分 +
  命运事件在环检测）与逐区对照结论（骨架存续/排干/重组）。
- ``fate``：命运诊断量与八类分类器（§7.2）——事件面复用
  ``Dynamics.propagate`` 的 scipy 语义；MEGNO 阈值带/撞月判据/优先序
  为登记自由参数（ADR 0041 Phase 3b）。
- ``boundaries``：可视化几何——会合系（质心原点，z=0）圆族 + Battin
  闭合曲线 + L1–L5；(a, e) 根数平面走廊曲线族（掠地线、Hill 远点线、
  月 Hill 相遇走廊、GEO 穿越线、共振竖线、Tisserand 等值线；元素空间
  crossing diagnostics 而非物理面）。

MCP 工具（经 Facade 派生）
--------------------------

- ``spatiography_scales``：全部解析尺度 + 平动点精确解 + Jacobi 临界值 +
  共振梯 + 常数溯源。
- ``spatiography_classify``：逐状态五省分类与诊断（osculating a、Jacobi、
  Case、开颈列表）；``frame`` 请求字段携带 ADR 0040 state_frame 词汇
  （``synodic_barycentric_km`` / ``synodic_barycentric_nd``）。
- ``spatiography_boundaries``：边界几何数据（``state_frame`` 为
  ``synodic_barycentric_km`` 或 ``element_space_ae``）；前端只做单位归一
  与绘制。
- ``spatiography_resonance_atlas``：共振图集（Primer §4.2–§4.4 / §5.3）。
- ``spatiography_dynamical_map``：六域两层天图——网格 (a/a☾, e) ×
  EM/EMS 模型 → Ȳ 场 + 八类命运场 + 逃逸时刻/最小距离诊断场。
  大数组走 sidecar 二进制帧（五帧：Ȳ/命运 id/逃逸时刻/最小月心距/
  最小地心距，E2M2 帧，ADR 0035 需求驱动扩展点）。全量制图走
  ``scripts/spatiography_map_production.py``（手动/发版前）。
  Gallardo 半宽包络（envelope_ae/vertical_ae，``element_space_ae``）、
  拱线驻定 loci（locus_ai，``element_space_ai``）、vZLK 相图
  （portrait_curve，``vzlk_phase_plane`）与 vZLK 标量（临界倾角、
  ν_vZLK、t_vZLK）；后三个标签为 ADR 0041 Phase 3 登记的新词汇。
  1:1 共振带宽系统性高估（论文 §5.3 声明），不得当 gateway 边界用。

黄金值速查（tests/algorithm/spatiography 锚定）
-----------------------------------------------

=====================================  ==============  =================
量                                     值              出处
=====================================  ==============  =================
r_L（地心 Laplace 半径）                48812.40 km     Primer Eq. 98
ρ_L（月心 Laplace 半径）                3846 km         Primer Eq. 124
ρ_H（月 Hill 球，近似式）               61364 km        Primer Eq. 110
(r_SOI)^☾（月 Laplace–Tisserand）       66010 km        Primer Eq. 116
(ρ_B)^☾ 朝地 / 背地（Battin）           52009 / 64201   Primer Eq. 118
a_TP（日月潮汐平权）                    447948 km       Primer Eq. 127
T☾（派生，禁硬编码 27.346）             27.34460 d      Table 1 caption
Δa(2:1☾, e=0.5)（Gallardo 半宽）        ≈ 24030 km      Primer Eq. 100–104
I_c（vZLK 临界倾角）                     39.2315°        Primer Eq. 64
translunar 驻定倾角极限（a→a☾）          63.4°/116.6°    Primer Eq. 79/80
=====================================  ==============  =================

Phase 3（ADR 0041 增补）三批已全部落地：3a 解析骨架（#578）、
3b MEGNO + fate（#579）、3c 六域制图管线（#580）。
