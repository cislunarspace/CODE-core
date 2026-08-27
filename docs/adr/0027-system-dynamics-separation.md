# ADR 0027: System/Dynamics separation retained — dynamics directory unsplit, two classes unmerged / System/Dynamics 分离保留：dynamics 目录不拆分、两类不合并

[English](#adr-0027-systemdynamics-separation-retained--dynamics-directory-unsplit-two-classes-unmerged) | [简体中文](#中文)

## English

**Status**: Adopted
**Date**: 2026-08-16
**Related Issues**: #430 (dynamics split evaluation), #438 (LibrationPoint
layering)
**Related**: ADR 0011 (five-layer architecture), ADR 0012 (dependency
direction), ADR 0026 (decision 1 & its follow-ups — this entry grew from that
audit)

### Context

While auditing the test suite, ADR 0026 raised a question: two kinds of things
mix inside `e2m2e/algorithm/dynamics/` — the System side
(`CR3BP_System`/`EphemerisSystem`, physical system definitions: μ, bodies,
characteristic scales) looks close to the data layer, while the Dynamics side
(`CR3BP_Dynamics`/`EphemerisDynamics`, constructing and integrating equations
of motion) is quintessential algorithm layer. #430 filed it: should System move
to data?

During triage, maintainers further questioned separation itself: System and
Dynamics were manually separated; if they're always constructed together, is
separation superfluous? Four candidate paths emerged: System to data, merge the
two, move only the `LibrationPoint` enum, or status quo.

Before adjudicating, a per-module verified data-flow investigation produced
`docs/architecture/system-dynamics-dataflow.md` (the "dataflow doc"),
verifying both class hierarchies' construction, held state, consumers, and one
propagation's data path module by module. This entry records the ruling;
structural-fact details defer to the dataflow doc.

### Decision

1. **System and Dynamics stay separated, both under
   `e2m2e/algorithm/dynamics/`.** System doesn't move to data; the classes
   don't merge.
2. **The dataflow doc enters the repo as this decision's structural
   explanation** (`docs/architecture/system-dynamics-dataflow.md`; registered
   in Sphinx toctree).
3. `LibrationPoint` enum's layer ownership is an independent pure-data-symbol
   question, deferred to #438 — not adjudicated here.

### Rationale

#### Domain layer: the model ladder

Earth-Moon orbit design's basic working path is a dynamics-model ladder:
design periodic orbits in CR3BP (idealized) for initial guesses, then BCR4BP
(solar perturbation added), finally ephemeris N-body (real force environment)
where low-accuracy results seed multiple-shooting corrections yielding
high-accuracy quasi-periodic orbits. The `design_orbit` docstring's three-stage
main chain (CR3BP design → ephemeris multiple shooting → nominal ephemeris) is
this path in code form.

Each ladder rung is a System+Dynamics pair: System is the model's context
(CR3BP's μ/characteristic scales/libration points, BCR4BP's solar parameters,
ephemeris's body list/SPICE/frames); Dynamics is equations and integration
under that model. The System/Dynamics split isn't directory tidiness — it
mirrors domain structure in code.

The ladder is unfinished: Hill three-body, elliptical restricted three-body,
quasi-bicircular (QBCP) await implementation. The seam formed by the two class
hierarchies plus the `System` base is their future extension slot: each new rung
arrives as a new pair, while frame conversion, shooting, family generation and
other consumers keep their connections unchanged.

#### Structural layer: three empirical facts

Verification process and sources for all below are in the dataflow doc.

1. **System has a dozen-plus independent consumers constructing no Dynamics**:
   force models (`ForceModel` holds system and mandates `coordinate_system`),
   the low-thrust trio, coordinate conversion (`SynodicJ2000System`,
   `rho_bridge`), station keeping & prediction, normal_form, and even the data
   layer's `Orbit` (duck-typed system reference for unit conversion).
   Separation lets context-only code avoid depending on the propagation
   machinery.
2. **One System instance serves multiple Dynamics and consumer paths.** In
   `design_orbit`, a single CR3BP instance simultaneously feeds dynamics
   construction, Jacobi computation, time conversion, and frame conversion —
   four paths; stability analysis and invariant manifolds rebuild their own
   Dynamics from `orbit.system` on demand.
3. **Lifecycles differ by an order of magnitude.** Systems live with the data
   (`Orbit` holds references surviving serialization round-trips); Dynamics
   live with the task: constructed, integrator-config overridden per mission,
   propagated, cache read, discarded. Merging would let consumers sharing a
   system stomp each other's integrator configs.

#### Clarification: nominal polymorphism is currently thin

The `System` base promises three members (`frame`/`unit_system`/
`gravitational_parameter`) but only `gravitational_parameter` is genuinely
polymorphic across both CR3BP and ephemeris implementations; other accesses
are structural duck-typing onto implementation members
(`origin`/`coordinate_system`/`spice`/`mu` via `getattr`/`hasattr`; low-thrust
tests pass plain `SimpleNamespace`). This entry keeps that seam as-is without
exaggeration: separation's load-bearing reasons are the three empirical facts,
not the nominal abstraction. If duck-typing proves insufficient when new models
arrive, a new ADR widens the contract then.

#### Why alternatives were rejected

**System to data layer**: System is a computational object, not data: libration
points solved via `fsolve` on nonlinear equations; stability analysis does
eigendecomposition; plus unit conversions and info printing. No data-layer
subdirectory (constants/frames/kernels/templates/types/catalog…) has precedent
for hosting computational objects. Also `compute_stability_index` shares
`pseudo_potential_hessian` with Dynamics' `compute_jacobian_A` — moving System
drags potential along, creating new ownership questions. Same root as ADR 0026
decision 1's coordinate ruling: "system definitions feel like data" is
functional-class intuition, and functional class ≠ code layer.

**Merging the two**: each structural fact inverts into cost: context-only
consumers forced onto propagation machinery; multi-consumer shared objects
stomping configs; data-living objects saddled with task-living caches.
Inheritance also degrades: `BCR4BPSystem` extends `CR3BP_System`, while
`BCR4BP_Dynamics` extends `Dynamics` directly (time-dependent Jacobian, no
Jacobi integral, four extra sun parameters at the Rust entry). Two non-mirrored
inheritance trees welded into one: BCR4BP would inherit CR3BP's system data
while swapping out nearly all dynamical behavior.

### Consequences

#### Added

- This ADR.
- `docs/architecture/system-dynamics-dataflow.md`: structural explanation of
  System/Dynamics flows; source material for this decision's structural
  rationale.

#### Unchanged

- `e2m2e/algorithm/dynamics/` layout and every line of code; System/Dynamics
  interfaces, implementations, tests stay as-is.

#### Costs

- The thin nominal contract of the `System` base gets documented-and-retained:
  functions taking abstract type signatures still probe via
  `getattr`/`hasattr`. Deliberate trade-off: widening's benefit doesn't repay
  present risk; evaluated together when new models arrive.

## 中文

**状态**：已采纳
**日期**：2026-08-16
**关联 Issue**：#430（dynamics 拆分评估）、#438（LibrationPoint 层级评估）
**关联**：ADR 0011（五层架构）、ADR 0012（依赖方向）、ADR 0026（决策 1 及其后续工作，本条由其审计而来）

### 背景

ADR 0026 审计测试套件时提出疑问：`e2m2e/algorithm/dynamics/` 里两类东西混放：System 一侧（`CR3BP_System`/`EphemerisSystem`，物理系统定义：μ、天体、特征尺度）看着接近 data 层，Dynamics 一侧（`CR3BP_Dynamics`/`EphemerisDynamics`，构造并积分运动方程）是地道的 algorithm 层。#430 把这个问题立案：System 是否应迁 data 层？

分诊中维护者进一步质疑分离本身：System 与 Dynamics 当初是手动分开的，若两者总是一起构造，分开是否多余？候选路径因此有四条：System 迁 data、两类合并、只挪 `LibrationPoint` 枚举、维持现状。

裁决前做了逐模块核实的数据流调研，产出 `docs/architecture/system-dynamics-dataflow.md`（下称数据流文档），逐模块核实了两棵类层次的构造、持有状态、消费面与一次传播的数据路径。本篇记录裁决，结构层事实的细节以数据流文档为准。

### 决策

1. **System 与 Dynamics 保持分离，同留 `e2m2e/algorithm/dynamics/`。** System 不迁 data 层，两类不合并。
2. **数据流文档作为本决策的结构说明入库**（`docs/architecture/system-dynamics-dataflow.md`，已注册进 Sphinx toctree）。
3. `LibrationPoint` 枚举的层级归属是独立的纯数据符号问题，转 #438 单独评估，不在本篇裁决。

### 理由

### 领域层：模型阶梯

地月轨道设计的基本工作路径是动力学模型阶梯：先在 CR3BP（理想化模型）里设计周期轨道出初猜，再到 BCR4BP（叠加太阳摄动），最后在星历 N 体（真实力学环境）里以低精度结果为迭代初值做多重打靶修正，得高精度拟周期轨道。`design_orbit` 模块 docstring 的三段主链（CR3BP 设计、星历多重打靶修正、标称星历输出）就是这条路径的代码形态。

每一级模型都是一对 System + Dynamics：System 是该模型的上下文（CR3BP 的 μ/特征尺度/平动点，BCR4BP 的太阳参数，星历的天体列表/SPICE/坐标系），Dynamics 是该模型下的方程与积分。System/Dynamics 之分不是目录洁癖，是这个领域结构在代码里的镜像。

阶梯尚未走完：Hill 三体、椭圆限制性三体、拟双圆（QBCP）等模型尚未实现。两棵类层次与 `System` 基类构成的接缝，就是它们将来进库的扩展槽：每新一级模型是一对新的 System+Dynamics，坐标转换、打靶、轨道族等消费面的接法不变。

### 结构层：三条实证

以下结论的核实过程与出处见数据流文档。

1. **System 有十余个不构造 Dynamics 的独立消费者**：力模型（ForceModel 持有 system、强制要求 `coordinate_system`）、低推力三件套、坐标转换（`SynodicJ2000System`、`rho_bridge`）、轨道保持与预报、normal_form、数据层的 `Orbit`（鸭子类型持有 system 引用做单位换算）。分离使这些只要上下文的代码不必依赖传播机器。
2. **一个 System 实例服务多个 Dynamics 与多路消费者。** `design_orbit` 里同一个 CR3BP 实例同时喂动力学构造、Jacobi 计算、时间换算、坐标转换四路；稳定性分析与不变流形从 `orbit.system` 按需重建各自的 Dynamics。
3. **生命周期差一个量级。** System 随数据长存（`Orbit` 持有引用，序列化再加载后仍在）；Dynamics 随任务生灭：构造、按任务覆写积分器配置、传播、读缓存、丢弃。合并会让共享 system 的消费者互踩积分器配置。

### 澄清：名义多态目前很薄

`System` 抽象基类承诺三个成员（`frame`/`unit_system`/`gravitational_parameter`），但真正跨 CR3BP 与星历两个实现兑现的多态只有 `gravitational_parameter` 一项；其余访问是对实现侧成员（`origin`/`coordinate_system`/`spice`/`mu`）的结构式鸭子类型（`getattr`/`hasattr` 兜底，低推力测试直接传 `SimpleNamespace`）。本篇按现状保留这条缝，不夸大它：分离的承重理由是上面三条实证，不是名义抽象。将来新模型进库时若鸭子类型不够用，再立新 ADR 加宽契约。

### 反方案为何被排除

**System 迁 data 层**：System 不是数据对象而是计算对象：平动点靠 `fsolve` 解非线性方程，稳定性分析做特征值分解，还有单位换算与信息打印。data 层各子目录（constants/frames/kernels/templates/types/catalog 等）没有承载计算对象的先例。且 `compute_stability_index` 与 Dynamics 的 `compute_jacobian_A` 共用 `pseudo_potential_hessian`，迁 System 须连带 potential，制造新的归属问题。这与 ADR 0026 决策 1 对 coordinate 的裁决同源：系统定义像 data 是功能类直觉，功能类与代码层级是两个独立的轴。

**两类合并**：结构层三条实证各自反转成代价：上下文消费者被迫依赖传播机器、多消费者共享同一对象时互踩配置、随数据长存的对象背上随任务生灭的缓存。继承关系也会变坏：`BCR4BPSystem` 继承 `CR3BP_System`，而 `BCR4BP_Dynamics` 直接继承 `Dynamics`（雅可比含时、无 Jacobi 积分、Rust 入口多四个太阳参数），两棵本不互为镜像的继承树焊成一棵，BCR4BP 要继承 CR3BP 的系统数据却换掉几乎全部动力学行为。

### 结果

### 新增

- 本篇 ADR。
- `docs/architecture/system-dynamics-dataflow.md`：System/Dynamics 数据流的结构说明，本决策结构层理由的材料源。

### 不变

- `e2m2e/algorithm/dynamics/` 目录结构与全部代码一行未动；System/Dynamics 的接口、实现、测试维持现状。

### 代价

- `System` 基类名义契约薄的现状被明文化保留，按抽象类型签名的函数仍靠 `getattr`/`hasattr` 探测实现侧成员。这是有意的取舍：加宽契约的收益不足以抵偿现在动它的风险，留给新模型进库时一并评估。
