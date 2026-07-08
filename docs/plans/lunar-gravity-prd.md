# PRD：月球非球形引力——天体无关的 GravityField 与潮汐重构

## 问题陈述

e2m2e 与 DFH 满配对比（L1 Halo，0.5 年）显示：七体点质量（地月日 + 大行星）外推 7 天末态位置差达 44.7 km，而加入大行星第三体后差异几乎不变——证明大行星对 cislunar 可忽略。真正的差异来自 e2m2e 缺少的摄动，其中**月球非球形引力是主导项**：月球是 cislunar 轨道的近旁天体，其非球形引力（GRGM900C 到 360×360）直接改变轨道动力学。

当前 `GravityField` 地球专用：`_get_input_coordinate_system` 硬编码 `ITRFApproxAxes()`（`gravity_field.py:437`），`body` 参数仅用作原点天体名，不驱动 body-fixed 轴切换；`gravity_file.py` 只解析 ICGEM `.gfc` 格式，不认 GMAT 的 `.cof` 格式；潮汐代码（`earth_tide.py`）写死 SUN/MOON/EARTH。GMAT 的 `GravityField` 是天体无关的，月球和地球用同一套 Pines 球谐递推，差别只在系数文件、GM、参考半径和 body-fixed 轴。

## 方案

把 `GravityField` 改造为天体无关，支持任意天体（地球、月球、后续火星等）的球谐引力场；潮汐模块重构为天体无关的 Step1 通用版 + 地球专用项保留；扩展 COF 格式解析器以直接使用 GMAT 的月球引力场数据。

### 改动清单

1. **`GravityField` 天体无关化**：
   - `_get_input_coordinate_system` 按 `body` 切换 body-fixed 轴：地球用 SPICE `ITRF93`（`ITRFSpiceAxes(frame="ITRF93")`），月球用 SPICE `MOON_PA`（`ITRFSpiceAxes(frame="MOON_PA")`）。
   - 地球从 `ITRFApproxAxes`（低精度教学）升级到 SPICE `ITRF93`（高精度，需 `earth_latest_high_prec.bpc`）。
   - 默认文件路径按 `body` 判断：地球默认 `egm96_to10.gfc`，月球要求显式提供 `gravity_file`（或默认 `grgm900c.cof`）。
   - `input_frame` 参数语义改为 SPICE frame 名，默认值按 `body` 推导。

2. **COF 格式解析器**：`gravity_file.py` 新增 `load_cof_file`，移植 GMAT `LM_LoadCof`（`HarmonicGravity.cpp:755-799`）的解析逻辑。COF 格式：`POTFIELD<NNN><MMM>` 头 + `RECOEF` 系数行；注意单位（GM 为 m³/s²×1e9，半径为 m×1e3）。返回与 `load_gfc_file` 相同的 `GravityFileData` 结构。

3. **潮汐重构为天体无关**：
   - `solid_tide_step1` 重构为通用版：接收扰动体列表（位置 + GM）+ Love number 表 + 中心天体 GM/参考半径，对每个扰动体累加 ΔC/ΔS。公式（IERS TN32 eqn.1）本身天体无关。
   - 地球专用项（Step2 频率相关、极潮、永久潮汐）保留在 `earth_tide.py`，仅 `body=EARTH` 且 `tide_mode` 非 none 时触发。
   - 月球固体潮：Love number k₂=0.024116（来自 `grgm900c.tide`），扰动体=地球，走天体无关的 Step1。
   - `GravityField._effective_coefficients` 按 `body` 分流潮汐逻辑。

4. **内核文件**：拷贝三个内核到 `kernels/`：`earth_latest_high_prec.bpc`（地球 ITRF93）、`SPICELunaCurrentKernel.bpc` + `SPICELunaFrameKernel.tf`（月球 MOON_PA，基于 DE421）。`SPICEManager` 加载逻辑需识别月球 FK + BPC。

5. **月球引力场数据**：从 GMAT 拷贝 `grgm900c.cof`（360×360）和 `grgm900c.tide` 到 `e2m2e/core/forces/data/`。

## 用户故事

1. 作为**轨道设计师**，我想要在 `ForceModel` 里加 `GravityField("MOON", degree=50, gravity_file="grgm900c.cof")`，以便外推月心轨道或 cislunar 轨道时考虑月球非球形引力。
2. 作为**DFH 对标用户**，我想要 e2m2e 开启月球非球形引力后，和 DFH 满配（含月球非球形）的差异大幅缩小，以便验证力模型对齐。
3. 作为**库维护者**，我想要 `GravityField` 天体无关，以便后续加火星、金星等天体时不用新建子类。
4. 作为**精度敏感用户**，我想要地球也升级到 SPICE ITRF93（高精度地固系），以便地球非球形引力的 body-fixed 变换不再依赖低精度近似。

## 实现决策

- **body-fixed 轴**：地球 `ITRF93`（需 `earth_latest_high_prec.bpc`），月球 `MOON_PA`（需 `SPICELunaCurrentKernel.bpc` + `SPICELunaFrameKernel.tf`，基于 DE421 PA 系）。`ITRFSpiceAxes(frame=...)` 现成可用，不新建子类。
- **COF 解析**：移植 GMAT `LM_LoadCof`，返回 `GravityFileData`，与 GFC 解析器输出统一。GMAT COF 的 GM 单位 m³/s²×1e9、半径单位 m×1e3，解析时换算到 km³/s²、km。
- **潮汐分层**：`solid_tide_step1` 抽通用版（扰动体列表 + Love number），地球 Step2/极潮/永久潮保留地球专用。月球潮汐走通用 Step1 + 月球 Love number（k₂=0.024116），扰动体=地球。
- **球谐递推**：不改。现有 Pines 风格递推（`gravity_field.py` L318-356）已天体无关，只依赖 degree/order/系数/参考半径/GM。
- **`require_inertial_frame`**：不涉及。`GravityField` 不用它（自行做坐标变换）。
- **默认文件**：地球默认 `egm96_to10.gfc`，月球默认 `grgm900c.cof`（拷到 `forces/data/`）。`body="MOON"` 不给文件不再静默加载地球 EGM96。

## 测试决策

- **单元测试**：COF 解析器读 `grgm900c.cof`，验证 GM（≈4902.8 km³/s²）、参考半径（1738.0 km）、阶次（360×360）、若干 Cnm/Snm 值。
- **月球引力单点测试**：在月心轨道某点，`GravityField("MOON", degree=10)` 的加速度与 GMAT `Harmonic::CalculateField`（同一系数、同一点）逐字一致。
- **body 切换测试**：`GravityField("EARTH")` 用 ITRF93 轴，`GravityField("MOON")` 用 MOON_PA 轴，确认 `_get_input_coordinate_system` 按 body 返回不同轴。
- **天体无关潮汐测试**：地球潮汐（Step1+Step2+极潮）行为不变（回归测试）；月球潮汐（Step1 + 月球 Love number）单点 ΔC/ΔS 与手算一致。
- **集成验收**：e2m2e `ForceModel`（地球引力场 + 月球引力场 10×10 + 日月行星第三体 + SRP）vs DFH 满配（`EPHEMERIDES_DAC.TXT`），7 天末态位置差应从当前 ~44 km 大幅缩小（目标 < 5 km）。
- **先例**：`tests/core/forces/test_gravity_field.py`、`tests/core/forces/test_third_body_gravity.py`（自洽性测试范式）、`scripts/compare_with_dfh.py`（DFH 对比脚本）。

## 不在范围内

- 月球 ME 系（Mean Earth/Polar Axis）——只做 PA 系（MOON_PA），ME 是 PA 的常量旋转，后续按需。
- 火星/金星等其他天体的引力场——架构支持，但不引入数据文件和测试。
- 地球引力场文件升级（EGM96 n≤2 → 完整 10×10）——独立后续工作。
- 地球引力场从 EGM96 升级到 EGM2008——独立后续。
- 海洋潮汐（地球的 ocean tide）——只做固体潮。

## 关键决策溯源

| 决策 | 选择 | 否决方案及理由 |
|------|------|------|
| 月球固连系 | MOON_PA（DE421 PA 系，高精度） | 否决 IAU_MOON（pck00010.tpc 三角多项式近似，精度低）；否决"两条都支持"（工作量翻倍） |
| GravityField 改造 | 改天体无关，按 body 切轴 | 否决新建 MoonGravityField 子类（单一实现的子类有成本）；否决参数注入（把"月球该用什么轴"推给用户） |
| 数据格式 | 扩展 COF 解析器 | 否决下载 GFC（引入数据差异变量）；否决 COF→GFC 转换（多一步且要处理单位） |
| 潮汐范围 | 重构为天体无关（Step1 通用版 + 地球专用项保留） | 用户明确选择 C（全量重构），实现上用 A（分层）落地——Step1 通用化，Step2/极潮地球专用 |
| 地球 body-fixed | 升级到 SPICE ITRF93 | 用户明确选择 B（地球也升级），需 earth_latest_high_prec.bpc |
| 交付范围 | 一次性全部 | 否决分片——潮汐重构无法在没有月球引力场时独立验证 |

## 补充说明

- 本轮是"把 GMAT 优点纳入 e2m2e"系列的第二步（第一步是 ThirdBodyGravity，#181/#182/#183）。DFH 满配对比（`scripts/compare_with_dfh.py` + DFH 满配输出）作为验收基准。
- GMAT 参照实现：`GravityField`（天体无关）、`Harmonic::CalculateField`（Pines 递推）、`LM_LoadCof`（COF 解析）、`IncrementSolidTide`（天体无关固体潮）、`BodyFixedAxes`（月球 PA 系从 DE 文件 libration）。
- 内核兼容性：MOON_PA 基于 DE421，e2m2e 用 DE430 行星历表。两者月球轨道差异 ~0.5m，对 cislunar 外推可忽略。
