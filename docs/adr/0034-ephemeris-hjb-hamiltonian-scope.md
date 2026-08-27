# ADR 0034: Ephemeris force-model Hamiltonian scope — planar full-ephemeris, non-autonomous, harmonics/SRP deferred / 星历力模型 Hamiltonian 的范围：平面全星历、非自治、球谐与光压缓议

[English](#adr-0034-ephemeris-force-model-hamiltonian-scope--planar-full-ephemeris-non-autonomous-harmonicssrp-deferred) | [简体中文](#中文)

## English

**Status**: Adopted
**Date**: 2026-08-21
**Numbering note**: numbers 0032/0033 were taken by ADRs merged with #497/#499;
this entry's actual decision date is 2026-08-21, numbered in sequence.
**Related Issue**: #498 (ephemeris force-model Hamiltonian)
**Related**: ADR 0013 (definition-level verification), ADR 0016 (ephemeris
cache), ADR 0020 (failure policy), ADR 0027 (model ladder), ADR 0032 (HJB
dynamics crate & binding entry), ADR 0033 (value-function product contract &
query interface); subsystem architecture at
`docs/architecture/hjb-subsystem.md`, solve-chain dataflow at
`docs/architecture/hjb-hamiltonian-dataflow.md`

### Context

Issue #498's body demands plugging the ephemeris N-body full-fidelity force
model (Earth-Moon 10×10 harmonics, solar third body, variable-mass SRP) into
the levelset grid solver to solve HJB directly. `hjb-subsystem.md` §3, merged
the same day (effective with PR #500), reached the opposite conclusion:
high-order gravity and SRP are intrinsically three-dimensional forces; the grid
layer's (dimension cap 5, no z-axis) fidelity ceiling is forces planarly
expressible in the synodic frame, and 3D high fidelity belongs to tier-two
neural networks. Two documents hours apart with opposite conclusions — an
unconverged decision left by parallel sessions.

After triage verified the physics, the conflict's full picture:

- All three mission segments (GEO, Halo, NRHO) travel far off-plane in the
  Earth-Moon synodic frame: GEO sits in the equatorial plane inclined 18°–29°
  to lunar orbit plane, out-of-plane amplitudes above ten thousand km; Halo/NRHO
  out-of-plane motion is definitional. No near-planar slice approximates the
  whole mission.
- Section projection (taking in-plane components of 3D forces at z=0) builds
  tables fine, but product control laws' domain won't cover real trajectories:
  value tables have no z axis — off-plane states can't be queried; closed-loop
  replay (verification tier 4) then can't run at all, projected errors barely
  even measurable.
- The grid layer's job in two-tier architecture is baseline + training data. As
  a baseline, model fidelity contributes nothing (cross-checks test solvers not
  models); as training data, CR3BP lacks **time dependence** (stationary model),
  not spatial fidelity — real ephemeris dynamics is time-varying, and tier-two
  networks must learn to handle that.

The maintainer ruled accordingly: #498 takes the planar full-ephemeris route
(decision 1 below), neither implementing the issue body's full-fidelity section
projection nor closing.

### Decision

1. **#498 scope: planar full-ephemeris Hamiltonian.** Real lunar ephemeris
   (position & velocity via `EphemCache` tables) defines a time-varying pulsating
   synodic frame; force models are two-primary point masses + solar third body
   (BCR4BP convention, Sun's in-plane projection of ephemeris position);
   non-autonomous with time explicit in dynamics. State is 5-dim
   `(x, y, vx, vy, m)` nondimensional synodic coordinates (4-dim dropping mass =
   constant-thrust-acceleration variant, distinguished by parameters). Fully
   compatible with `hjb-subsystem.md` §3 and its family-table row "ephemeris
   force model (4 or 5 dims, non-autonomous)" — **no revision of master's
   architecture docs**.
2. **Harmonics & SRP demoted to deferred experiments, out of #498 acceptance.**
   Adding section-projected forces onto decision 1's skeleton is small work,
   reserved as follow-up experiments: decide retention using tier-3 verification
   data (value function magnitude/iso-surface structure vs CR3BP solutions).
   Experiments get their own issue — not elaborated here.
3. **Time-varying synodic frame construction**: the frame derives from lunar
   instantaneous ephemeris (rotation + pulsation); `ω(t)`, `ω̇(t)`, pulsation all
   derive from cached lunar position/velocity — no second ephemeris query path.
   Solver t ↔ SPICE et epoch mapping, cache coverage of the whole solve window
   (out-of-range hard failure per ADR 0020 semantics), backward-in-time reversal
   land per `hjb-hamiltonian-dataflow.md`'s time-semantics section; value-function
   products' `times` semantics, state order, nondimensionalization enter metadata
   per ADR 0033 decision 3.
4. **Force models & cache path**: the ephemeris Hamiltonian lives in
   `e2m2e-hjb-dynamics` (ADR 0032 decision 1), receiving a `CompiledForce` list +
   cache time range at construction — ADR 0033's construction-time injection
   realized thus: injection sits at the **configuration** level (force list, time
   range, epoch mapping as constructor params), while query paths inside
   `CompiledForce` still go through `EphemCache`'s process singleton (ADR 0016),
   constructors responsible for enabling cache coverage of the solve window
   beforehand. No injection-style refactor changes.
5. **Variable-mass SRP contract (`SRPVariableMass`/
   `acceleration_with_mass`) lands independently of #498.** It exists only in an
   uncommitted local workspace while geo-nrho's `lowthrust_rs` already calls it —
   a load-bearing dependency of existing solvers requiring prompt independent PR.
   Under decision 2 it stays off #498's critical path.
6. **#498 repo-internal acceptance in three tiers**:
   (a) Degradation cross-check: replacing ephemeris inputs with circularized
       stationary values must degrade dynamics term-by-term to `Cr3bpSynodic`
       (#497 implementation);
   (b) Force consistency: identical (t, state) — Hamiltonian-internal post-frame-
       transform forces match direct `compute_total_acceleration` calls pointwise;
   (c) Coarse-fine regression (verification tier 3): small grids solve both
       ephemeris & CR3BP value functions; magnitude & iso-surface structure agree;
       exceeding tolerance = implementation error.
   Closed-loop replay (tier 4) runs manually on geo-nrho side, never entering
   e2m2e pytest (acceptance independent of external repo code).
7. **Crate ownership, binding entry, product contract, time interpolation**:
   from ADR 0032 (decisions 1, 3) and ADR 0033 (decisions 2, 3, 4); no re-decision
   here.

### Rationale

1. **Ruling (c) over (a)**: section projection's control-law domain can't cover
   real trajectories (no z axis to query); verification tier 4 becomes
   unexecutable; products resist even error measurement; missions are off-plane
   throughout, so projection isn't locally damaging only. (a)'s output — exact
   solution of an ad-hoc projected model — sits awkwardly research-wise and would
   overturn that day's merged §3.
2. **Ruling (c) over (b)**: closing #498 freezes the grid layer on stationary
   models forever. Tier-two networks must handle time-varying dynamics; if tier
   one provides no non-autonomous baseline, time-varying signals get neither
   training nor spot-checks. Planar full-ephemeris precisely patches this gap
   within the dimension ceiling: **upgrading the grid layer along the temporal-
   fidelity axis rather than crashing into the spatial-fidelity wall**.
3. **Decision 3's pulsating frame**: distance pulsation makes primaries drift in
   fixed-scale frames, destabilizing grid axes' physical meaning; pulsating
   coordinates pin primaries at fixed nondimensional positions — standard for
   elliptical restricted three-body and full-ephemeris models, costing explicit
   time-varying terms ω̇/pulsation corrections which is exactly this issue's wanted
   non-autonomy, not extra burden.
4. **Decision 4's config injection**: `CompiledForce::acceleration` signatures carry
   no cache parameter; query-path injection means touching whole call chains +
   Python bindings for a benefit (multiple caches per process) nobody needs today;
   constructor params satisfy ADR 0033's pure-table-reads intent.
5. **Decision 6(a)'s degradation check** is the strongest assertion: the ephemeris
   version in circularized-stationary limit must reproduce term-by-term the already-
   cross-checked #497 CR3BP dynamics, anchoring new implementation correctness onto
   verified code — matching ADR 0013's verification-by-definition strategy.

### Consequences

- This entry precedes #498's specification; issue #498 carries triage record +
  agent brief, labeled `enhancement` + `ready-for-agent`.
- The family-table's ephemeris-force-model row lands concretely via this entry;
  the doc itself un-revised.
- Once the variable-mass SRP contract PR (decision 5) lands, geo-nrho's
  `lowthrust_rs` sheds its uncommitted-workspace dependency. (2026-08-23 addendum:
  PR landed; `SRPVariableMass` + `acceleration_with_mass` now in
  `crates/e2m2e-forces`.)
- Harmonics/SRP section-projection experiments (decision 2), if started, track via
  new issue; their conclusions may revise decision 2.

## 中文

**状态**：已采纳
**日期**：2026-08-21
**编号说明**：0032、0033 已由随 #497、#499 合并的两篇 ADR 占用；本篇实际决策时间 2026-08-21，编号顺延。
**关联 Issue**：#498（星历力模型 Hamiltonian）
**关联**：ADR 0013（定义级验证）、ADR 0016（星历缓存）、ADR 0020（失败处理）、ADR 0027（模型阶梯）、ADR 0032（HJB 动力学 crate 与绑定入口）、ADR 0033（值函数产品契约与查询接口）；子系统架构见 `docs/architecture/hjb-subsystem.md`，求解链数据流见 `docs/architecture/hjb-hamiltonian-dataflow.md`

### 背景

#498 正文要求把星历 N 体全保真力模型（地月 10×10 球谐、太阳第三体、
变质量光压）接入 levelset 网格求解器直接求解 HJB。同日合并的
`hjb-subsystem.md` §3（随 PR #500 生效）给出了相反结论：高阶引力与
光压本质是三维力，网格层（维度上限 5，无 z 轴）的保真度天花板是
会合系内能平面表达的力，三维高保真由第二层神经网络承接。两份文档
相隔数小时、结论相反，是并行会话留下的未收敛决策。

分诊期间把物理事实核实清楚后，冲突的全貌是：

- 任务三段（GEO、Halo、NRHO）在地月会合系里全都大幅离面：GEO 在
  赤道面，与月球轨道面倾角 18°~29°，离面振幅上万公里；Halo 与
  NRHO 的离面是其定义性特征。不存在对任务全程近似成立的平面切片。
- 截面投影（z=0 处取三维力的面内分量）造表无碍，但产物的控制律
  定义域盖不住真实轨迹：值函数表没有 z 轴，离面状态查不了；
  闭环回放（验证阶梯第 4 级）因此跑不起来，投影误差连实测都难。
- 网格层在两级架构里的职务是基准与训练数据。作基准，模型保真度
  无贡献（对拍的是解法不是模型）；作训练数据，CR3BP 缺的是
  **时间依赖性**（定常模型）而非空间保真度，真实星历动力学是
  时变的，第二层神经网络必须学会处理这一点。

维护者据此裁决：#498 走平面全星历路线（下述决策 1），不按 issue
正文上全保真截面投影，也不关闭。

### 决策

1. **#498 范围：平面全星历 Hamiltonian。** 真实月球星历（位置与
   速度经 `EphemCache` 查表）定义时变脉动会合系；力模型为两主星
   点质量 + 太阳第三体（BCR4BP 口径，太阳取星历位置的面内投影）；
   非自治，时间显式进入动力学。状态为 5 维 `(x, y, vx, vy, m)`
   无量纲会合系坐标（4 维去质量退化为恒推力加速度版本，由参数
   区分）。与 `hjb-subsystem.md` §3 及族谱表中的星历力模型
   （4 或 5 维，非自治）一行完全相容，**不修订 master 现有架构文档**。
2. **球谐与光压降级为缓议实验项，不进 #498 验收。** 在决策 1 的
   骨架上加截面投影力是小改动，留作后续实验：做完以第 3 级验证
   （与 CR3BP 解对拍值函数量级与等值面结构）的数据决定去留。
   实验项另开 issue，不在本篇展开。
3. **时变会合系的构造口径**：会合系由月球瞬时星历定义（旋转 +
   脉动），`ω(t)`、`ω̇(t)`、脉动项全部从缓存的月球位置/速度导出，
   不出现第二次星历查询路径。求解器 t ↔ SPICE et 的历元映射、
   缓存区间覆盖整个求解窗（越界硬失败，ADR 0020 语义）、倒向
   求解的时间取反，均按 `hjb-hamiltonian-dataflow.md` 的时间语义
   节落实；值函数产品的 `times` 语义、状态维顺序、无量纲化口径
   按 ADR 0033 决策 3 进元数据。
4. **力模型与缓存路径**：星历版 Hamiltonian 住
   `e2m2e-hjb-dynamics`（ADR 0032 决策 1），构造时接收
   `CompiledForce` 列表与星历缓存时间范围，这是 ADR 0033 构造时
   注入的落实口径：注入落在**配置**层面（力模型列表、时间范围、
   历元映射为构造参数），查询路径上 `CompiledForce` 内部仍经
   `EphemCache` 进程级单例（ADR 0016），构造方负责在求解前
   `enable` 覆盖求解窗的缓存。不改注入式重构。
5. **变质量 SRP 契约（`SRPVariableMass`/`acceleration_with_mass`）
   独立于 #498 落地。** 它目前只存在于未提交本地工作区，而
   geo-nrho 的 `lowthrust_rs` 已在调用它，是现有求解器的承重
   依赖，须尽快提交为独立 PR。决策 2 之下它不在 #498 关键路径上。
6. **#498 仓库内验收三级**：
   (a) 退化对拍：把星历输入替换为圆化定常值时，动力学逐项退化为
       `Cr3bpSynodic`（#497 实现）；
   (b) 力一致性：同一 (t, state) 下，Hamiltonian 内部经系变换后
       的力与 `compute_total_acceleration` 直接调用的结果逐点一致；
   (c) 粗细模型回归（验证阶梯第 3 级）：小网格分别求星历版与
       CR3BP 版值函数，量级与等值面结构一致，超出容差即实现错误。
   闭环回放（第 4 级）在 geo-nrho 侧手动执行，不进 e2m2e pytest
   （验收不依赖外部仓库代码）。
7. **crate 归属、绑定入口、产品契约、时间插值**：从 ADR 0032
   （决策 1、3）与 ADR 0033（决策 2、3、4），本篇不重复决策。

### 理由

1. **裁决 (c) 而非 (a)**：截面投影的控制律定义域盖不住真实轨迹
   （无 z 轴可查），验证阶梯第 4 级无法执行，产物连误差都测不了；
   且任务三段全程离面，投影并非只伤局部。(a) 的产出是一个
   即席投影模型的精确解，研究定位尴尬，还要推翻当天合并的 §3。
2. **裁决 (c) 而非 (b)**：关闭 #498 会让网格层永远停在定常模型。
   第二层神经网络必须处理时变动力学，第一层若不提供非自治基准，
   时变信号的训练与点检全部缺席。平面全星历恰好在维度天花板之内
   补上这块短板：**在时间保真度轴上升级网格层，而非撞上空间
   保真度轴的墙**。
3. **决策 3 的脉动会合系**：月距脉动使固定尺度会合系里两主星位置
   随时间漂移，网格轴的物理含义随之不稳；脉动坐标把两主星钉在
   固定无量纲位置，是椭圆限制性三体与 full-ephemeris 模型的标准
   做法，代价是动力学多出 ω̇、脉动修正等显式时变项，这正是本
   issue 要的非自治性，不是额外负担。
4. **决策 4 的配置注入口径**：`CompiledForce::acceleration` 签名
   无缓存参数，改成查询路径注入要动整条调用链与 Python 绑定，
   收益（同进程多套缓存并存）目前无真实需求；构造参数注入已满足
   ADR 0033 关于求解阶段纯查表的意图。
5. **决策 6(a) 的退化对拍**是最强断言：星历版在圆化定常极限下
   必须逐项还原 #497 已对拍过的 CR3BP 动力学，把新实现的正确性
   锚定在已验证代码上，符合 ADR 0013 按定义验证的策略。

### 结果

- 本篇为 #498 的规格前置；issue #498 附分诊记录与 agent brief，
  标 `enhancement` + `ready-for-agent`。
- `hjb-subsystem.md` 族谱表中的星历力模型一行由本篇落实为具体
  口径；该文档不修订。
- 变质量 SRP 契约的独立 PR（决策 5）落地后，geo-nrho
  `lowthrust_rs` 摆脱对未提交工作区的依赖。（2026-08-23 补记：
  该 PR 已落地，`SRPVariableMass` 与 `acceleration_with_mass`
  现已在 `crates/e2m2e-forces` 中。）
- 球谐/光压截面投影实验项（决策 2）如启动，以新 issue 跟踪，
  其结论可能修订本篇决策 2。
