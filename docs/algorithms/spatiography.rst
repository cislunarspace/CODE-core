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
  16 条名义中心与开普勒周期）。
- ``regions``：五省分类器（Table 1 / Table 4 双口径，重叠带多标签）、
  Jacobi 五拓扑 Case I–V 与临界值（平动点精确求根）。
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
=====================================  ==============  =================

后续批次（见 ADR 0041 第 6 节）：Gallardo 共振半宽、长期共振 loci、vZLK
相图、MEGNO + fate 数值制图。
