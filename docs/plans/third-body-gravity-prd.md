# PRD：为力分解路径补 ThirdBodyGravity

## 问题陈述

e2m2e 有两条并行的传播路径（CONTEXT.md "动力学"节）：

1. **解析 N 体路径**（`EphemerisDynamics`）：地月日三体，闭式公式 `a = -μ₀r/r³ - Σμᵢ[(r-rᵢ)/|r-rᵢ|³ + rᵢ/|rᵢ|³]`。是 CR3BP→星历转换的载体。
2. **力分解路径**（`ForceModel`）：多个 `PhysicalModel` 可组合、可配置、可序列化。架构上已搭好 SRP/阻力/相对论/潮汐/推力，是 GMAT 对标路径。

力分解路径缺一个关键力——**第三体点质量引力**。它的现有力清单里，`PointMassGravity` 只算中心引力 `-μr/r³`（`point_mass_gravity.py:56-60`），不查第三体位置、不带间接项；没有独立的第三体力类。

后果：力分解路径算不了 cislunar 轨道。cislunar 的主导摄动是月球（和太阳）的第三体引力，这个力现在进不来。`compare_with_gmat.py` 的 LEO 对标场景里也只有 `GravityField + Drag + SRP`，不涉及第三体。

要让力分解路径成为 cislunar 高保真外推的统一载体，必须先补上第三体力。

## 方案

新增一个力模型 `ThirdBodyGravity`，落在 `e2m2e/core/forces/third_body_gravity.py`，与现有 `PointMassGravity`、`GravityField`、`SolarRadiationPressure` 等并列。

每个第三体天体一个独立实例（GMAT `PointMassForce` 式）：

```python
fm = ForceModel(system)
fm.add_force(PointMassGravity("EARTH"), name="earth_central")
fm.add_force(ThirdBodyGravity("MOON"), name="moon")
fm.add_force(ThirdBodyGravity("SUN"), name="sun")
```

这组力在物理上与 `EphemerisDynamics` 的闭式 N 体公式等价——中心引力 + 第三体直接项 + 第三体间接项，应给出同一条轨迹。

## 用户故事

1. 作为**轨道设计师**，我想要用力分解路径外推一条 NRHO，以便在统一的力模型框架下叠加 SRP、相对论等摄动。
2. 作为**迁移用户**，我想要 `ThirdBodyGravity` 的接口与 GMAT `PointMassForce` 一一对应（每天体一个实例），以便对照配置。
3. 作为**库维护者**，我想要一个自洽性验收测试证明力分解路径与 `EphemerisDynamics` 给出一致结果，以便把第三体力作为后续 cislunar 力（月球非球形、N-plate SRP）的前置依赖。

## 实现决策

- **新类**：`ThirdBodyGravity(body: str)`，继承 `PhysicalModel`。
- **不暴露 origin 参数**：第三体力的间接项只在"力原点 = system 原点"时物理正确。`compute_acceleration` 内部读 `system.origin`（`EphemerisSystem.origin` 已暴露，`ephemeris_system.py:50`），调 `system.get_body_position(self._body, t)`（已自动用 origin 作 observer，`ephemeris_system.py:84-96`）取 `r_ob`。暴露 origin 参数等于给用户一个"算错"的旋钮。
- **不约束天体清单**：不校验 `body in system.bodies`。第三体力查位置走 SPICE 单天体查询，与 `EphemerisSystem.bodies` 无关（后者只约束 `get_gm_values` 等批量查询）。内核有该天体星历即算，没有抛 SPICE 错。
- **GM 来源**：`system.gravitational_parameter(self._body)`，与 `PointMassGravity` 一致（`point_mass_gravity.py:48-54`）。
- **公式**（逐字对齐 `EphemerisDynamics` 第三体分支，`ephemeris_dynamics.py:117-138`）：

  ```
  r_ob  = system.get_body_position(body, t)        # 第三体相对原点
  r_bsc = state[:3] - r_ob                         # 第三体→航天器（e2m2e 约定）
  acc   = -μ * (r_bsc/|r_bsc|³ + r_ob/|r_ob|³)     # 直接项 + 间接项
  ```

  符号约定与 `EphemerisDynamics` 完全一致；物理上等价于 GMAT `PointMassForce`（直接项 `-μ·relPos/|relPos|³` + 间接项 `μ/r³·rv`）。
- **坐标变换**：无。第三体位置查询和加速度合成都在参考系（`system.coordinate_system`，惯性系）下完成，与 `PointMassGravity` 一致，不需切到 body-fixed 系。
- **序列化**：接入 `force_config.py` 的 builder 注册（参考 `PointMassGravity` 的注册范式），配置可往返。
- **不引入雅可比**：`ForceModel` 当前不支持 STM（`force_model.py:494-500`），`PhysicalModel` 抽象接口只有 `compute_acceleration`（`physical_model.py:23-40`）。雅可比是后续 STM 支持的独立工作，不在本范围。

## 测试决策

- **主验收测试（自洽性）**：`tests/core/forces/test_third_body_gravity.py`。同一 cislunar 初值（复用 `test_ephemeris_dynamics.py` 的 `dro_state`，月球距离附近），两条路径各跑一段：
  - 路径 1：`EphemerisDynamics(system)`，闭式 N 体。
  - 路径 2：`ForceModel(system)` + `PointMassGravity("EARTH")` + `ThirdBodyGravity("MOON")` + `ThirdBodyGravity("SUN")`。
  
  判据：末状态位置差 < 1 km（1e-12 积分容差、~9 天弧段下的合理累积误差阈值；实现后跑出实际差异再敲定具体数值）。
  
  这个测试直接证明"第三体力实现正确"——两条路径对同一物理量应有相同结果，差异只来自积分器（scipy DOP853 vs Rust PD45）和数值精度。
- **单元测试**：`ThirdBodyGravity` 单点的加速度应与 `EphemerisDynamics._compute_acc_and_jacobian` 中对应第三体分支的增量一致（取月球距离处的点，单独算 MOON 的加速度，应等于 `EphemerisDynamics` 在该点对 MOON 的贡献）。
- **不在范围**：外部 GMAT 对标（搭脚本、保证两边配置一致的工作量大、易因配置差异而非算法错误失败）；守恒量测试（"守恒量漂移小"≠"和真值一致"，证明力弱）。

## 不在范围内

- 月球非球形引力（要改 `GravityField` 架构支持任意 body 切换 + 月球固连系 + 月球重力场文件，独立大改动）。
- N-plate / box-wing SRP（独立升级）。
- 积分器一致性（修正"自称 RK89 实跑 PD45"、开放 PD78/RK89 选项，独立工作）。
- 相对论 / 潮汐默认启用（架构已就位，启用是另一件事）。
- 改造解析 N 体路径 `EphemerisDynamics`（维持原职）。

## 关键决策溯源

这些决策在 grill-with-docs 会话中逐项与用户确认，记录如下，便于未来读者理解"为什么这么选"：

| 决策 | 选择 | 否决的方案及理由 |
|------|------|------|
| 场景 | cislunar 高保真外推 | 否决 LEO/轨道确定/通用补齐——cislunar 是 e2m2e 的定位 |
| 传播路径 | 力分解路径（ForceModel） | 否决落在 `EphemerisDynamics`——破坏其"自洽快速 N 体"定位，可组合性差 |
| 范围 | 只补第三体力类 | 否决连带月球非球形/N-plate/积分器——违背"精准改动"，各自独立推进 |
| 接口形态 | 每天体独立实例 | 否决容器式 `ThirdBodyForce`——违反"一个力一个对象"范式，序列化复杂 |
| origin | 从 system 查不暴露 | 否决显式 origin 参数——给用户"算错"的旋钮 |
| 天体约束 | 不约束信任 SPICE | 否决要求声明在 `bodies`——`bodies` 不是白名单，硬塞耦合 |
| 验证 | 自洽性 vs `EphemerisDynamics` | 否决外部 GMAT 对标——配置不一致易导致非算法失败 |
| 命名 | `ThirdBodyGravity` | 否决 `PointMassForce`（与现有 `PointMassGravity` 名近职异，易混）、`ThirdBodyForce`（风格不统一） |
