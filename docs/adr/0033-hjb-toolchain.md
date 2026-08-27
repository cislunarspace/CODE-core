# ADR 0033: HJB low-thrust toolchain — value-function product contract and online query interface / HJB 小推力工具链：值函数产品契约与在线查询接口

[English](#adr-0033-hjb-low-thrust-toolchain--value-function-product-contract-and-online-query-interface) | [简体中文](#中文)

## English

**Status**: Adopted
**Date**: 2026-08-21
**Related Issues**: #497, #498, #499, #501
**Related**: ADR 0011 (five-layer architecture), ADR 0012 (dependency
direction), ADR 0013 (verification by definition), ADR 0031 (catalog record
format), ADR 0032 (HJB dynamics crate & generic binding entry); geo-nrho
`docs/hjb-dependency-architecture.md` (downstream dependency architecture)

### Context

geo-nrho follows Bellman optimality via a two-level HJB route: offline,
e2m2e-levelset solves the HJ equation on structured grids for a value function;
online, closed-loop control derives from its gradient. The downstream dependency
architecture doc (geo-nrho `docs/hjb-dependency-architecture.md` §2.3) listed four
capabilities proposed for upstreaming: #497 (CR3BP synodic Hamiltonian), #498
(ephemeris force-model Hamiltonian), #499 (gradient interface + discrete operating-
point mapping).

Three seams needed fixing before work or later deepening reworks:

1. **Crate ownership** (fixed meanwhile by ADR 0032 during this entry's drafting):
   original HJB dynamics live in new crate e2m2e-hjb-dynamics — levelset wholesale
   inherits ToolboxLS's ACM non-commercial license; merging originals would sweep
   them under it. Decision 1 aligns without re-deciding.
2. **Value-function product contract.** Putting gradient queries at Python means
   interface↔solver decoupling — contracts shift onto data formats: double-
   integrator products use nondimensional time, synodic coordinates; ephemeris
   products will be ET seconds, possibly different frames, possibly mass-carrying.
   Without fixed format semantics, every Hamiltonian sprouts its own reader. geo-nrho
   has a `ProductMeta` prototype (`produced_by`/`frame`/`units`/`maturity`) but its key
   naming diverges from ADR 0031.
3. **Status of the time dimension.** The double integrator is autonomous — geo-nrho's
   nearest-time-snapshot sufficed by luck; ephemeris models are non-stationary, where
   no time interpolation is simply wrong. Interface design must pin this early.

### Decision

#### 1. levelset stays a pure-math leaf; dynamics ownership defers to ADR 0032

e2m2e-levelset keeps only ToolboxLS counterparts and math examples (`Advection`
etc.); dynamics inject via the `Hamiltonian` trait. Original dynamics (#497's CR3BP
synodic Hamiltonian, #498's ephemeris force-model Hamiltonian) live in
**e2m2e-hjb-dynamics** (Apache-2.0); rationale in ADR 0032 (license boundary).
#498's ephemeris adapter depends inside that crate on e2m2e-forces' `CompiledForce`
and e2m2e-spice's ephemeris cache, injected at construction — pure table reads
during solving, zero CSPICE contact, never into Python callbacks. No new
`e2m2e-forces → e2m2e-levelset` dependency edge.

#### 2. Python exposure solely through e2m2e-integrators, via ADR 0032's generic entry

Levelset solving capability exposes uniformly through e2m2e-integrators under the
ABI-stamp process: one generic entry (dynamics identifier + parameter table + grid
definition + terminal conditions — ADR 0032 decision 3's `solve_hjb_py`), no
per-dynamics dedicated signatures. Bindings do array shuffling, parameter
validation, error translation only; Hamiltonian assembly and control minimization
live entirely inside Rust crates, testable Rust-side.

#### 3. Value-function product contract aligned to ADR 0031

Value-function products = JSON metadata + NPZ arrays, reusing catalog record
machinery rather than inventing formats:

- `schema_version` starts at 1 with no cross-version compatible reads; required keys
  follow ADR 0031's `_META_REQUIRED_KEYS` system (`source_tool`, status triple,
  `request` snapshot, `source_record_id`, etc.).
- Numeric conventions go explicitly into metadata: state-dim order, per-dim physical
  meaning, nondimensionalization (characteristic length/time/mass or none),
  `times` semantics (ET seconds vs synodic nondimensional). Products of differing
  conventions are distinguished by metadata fields, not reader guesswork — same
  spirit as ADR 0031 distinguishing dynamical models by segment presence.
- Value-function products enter catalog as a new record type, `source_record_id`
  pointing at the target-orbit record serving as terminal constraint. Ingestion is
  solver-side (#497/#498); consumers (#499's gradient interface) need only npz
  reading — no catalog dependency.
- geo-nrho's `ProductMeta` is the downstream prototype whose `frame`/`units`/
  `force_model` semantics this contract absorbs; geo-nrho migrates keys toward ADR
  0031 naming (`produced_by` → `source_tool` etc.) on its side.

#### 4. Gradient query interface at the Python algorithm layer; time interpolation mandatory

Value functions are numpy arrays in consumers' hands already; online query rates
are low (control-period scale currently) — no Rust justification:

- Location: Python algorithm layer (`e2m2e/algorithm/`), pure numpy/SciPy per
  ADR 0012 dependency direction.
- Dimension-agnostic: interface takes only `axes`/`values`/`times` + query points;
  assumes neither state dimensionality nor physical meaning.
- Spatial interpolation uses tensor-product splines (e.g., cubic
  `RegularGridInterpolator`); gradients are **analytic derivatives of the
  interpolant**; center-difference-then-interpolate on grids (geo-nrho's current
  `_grid_gradient`) is forbidden — that is precisely this issue's targeted error
  source.
- Time interpolation mandatory, at least linear; autonomous systems are merely the
  degenerate-but-still-correct special case.
- Performance isn't this interface's goal. If future closed-loop sims make queries a
  bottleneck, moving to Rust is an independent decision not touching this contract.

#### 5. Discrete operating-point mapping: data models migrate, constants parameterized, algorithm carries minimum-arc constraint

- `ThrustLevel` (0/60/100%), `ThrustArc`, `ThrustArcSequence` migrate verbatim from
  geo-nrho's `thrust_arcs.py` into e2m2e's Python low-thrust layer, sharing
  operating-point definitions with `LowThrustCollocation`; geo-nrho deletes its local
  copy and imports instead.
- Mission constants become constructor parameters: `MIN_ARC_DURATION_S` (geo-nrho
  currently 3600 s), `MAX_THRUST_N`, `ISP_S` — no module-level constants remain.
- Mapping must handle minimum-arc constraints (merging/splitting), not geo-nrho's
  current per-segment nearest-level approach which hard-errors when collocation
  segments run denser than minimum arcs — unusable.
- Acceptance self-contained within e2m2e: CR3BP continuous-throttle solution
  (generated by `LowThrustShooting`/`LowThrustCollocation`) → mapping → re-propagation
  → terminal residuals under L1 thresholds (384 km / 1 m/s order). No geo-nrho case
  dependencies, honoring ADR 0013 and `.out-of-scope/`'s verification-independence
  principle.

### Rationale

1. **Crate ownership defers to ADR 0032**: levelset's ACM license would cover
   merged originals — harder constraint than saving one scaffold. Centralizing
   dynamics in e2m2e-hjb-dynamics also preserves levelset's faithful-porting
   stance. Mixing domain logic into bindings (integrators) breaks their thin-FFI
   positioning — excluded.
2. **Contract aligned to 0031 rather than a new format**: catalog already solved the
   identical problem (multi-model artifacts, convention ambiguity, lineage) and gifts
   lineage mechanics: `source_record_id` pointing at terminal-constraint orbits
   directly mitigates immutable-boundary-condition pain — changing targets equals
   changing lineage pointers + re-solving; product relationships stay traceable.
3. **Gradient interface at Python**: consumers, data shape, query frequency all live
   Python-side; Rustification's benefit fails. Dimension-agnostic + mandatory-time
   design avoids rework upon ephemerization.
4. **Data models migrate verbatim**: geo-nrho already validated the contract;
   redesigning manufactures two equivalent but different concepts.

### Consequences

**Added**: catalog value-function record type (#498); Python gradient query
interface (#499) + discrete operating-point mapping module (#501).

**Changed**: none among prior decisions. #497 landed per ADR 0032
(e2m2e-hjb-dynamics + generic binding entry); decisions 1–2 align. geo-nrho
deletes `thrust_arcs.py` after #501 lands; `ProductMeta` keys align to ADR 0031
later (both geo-nrho-side actions).

**Unchanged**: e2m2e-levelset kernel & pure-math position; e2m2e-integrators thin
binding position; existing catalog record format (value-function records add a type,
not modify existing schema).

**Costs**: catalog needs segment conventions + validation for value-function
records.

**Out of scope**: three research-grade difficulties (adaptive grids, quantitative
error evolution, re-solve strategies under mutable boundary conditions) aren't solved
here; this entry only guarantees engineering architecture doesn't block them.

### Revision (2026-08-21, implementation feedback)

1. **Renumbered to 0033; decisions 1–2 aligned to landed ADR 0032**: drafted as
   0032, colliding with the ADR 0032 merged via #497 (HJB dynamics crate + binding
   generic entry) and disagreeing on decision 1; renumbered at merge, decisions
   1–2 realigned (dynamics in e2m2e-hjb-dynamics; bindings via generic
   `solve_hjb_py`).
2. **#501 added to related issues**: triage split #499 in two — #499 keeps only the
   gradient interface (decision 4); discrete mapping became #501 (decision 5).
   Decision 3's ingestion ownership unchanged.
3. **Decision 4 implementation form**: tensor-product splines assembled via
   per-axis not-a-knot cubic solves into `NdBSpline`; gradients via analytic `nu`
   derivatives, C² continuous (excluding local sliding stencils: their gradients jump
   at cell boundaries while closed-loop control consumes gradients directly for
   direction & switching functions). Cost: stateless function contract rebuilds
   splines per touched time snapshot per call (~0.5 s/snapshot at 41⁴-scale grids);
   acceptable at control-period query rates. If dense closed-loop sims bottleneck,
   introducing a coefficient-cached interpolator object is independent — contract
   unchanged.
4. **Decision 5's end-to-end case fixed as Earth-ephemeris two-body LEO two revs**
   (`GravityField` degree-0, matching existing low-thrust test fixtures), not CR3BP:
   self-contained, fast, geo-nrho-independent intent unchanged. 384 km / 1 m/s was an
   L1 mission-scale gauge — too loose for LEO; tests tightened ~10× against measured
   residuals (~0.35 km / 0.0004 m/s) as regression assertions. CR3BP end-to-end
   strengthened after #497.
5. **Decision 5 gains three interface refinements**: `validate` adds optional
   `levels` checking level legality (brief acceptance); `sequence_from_controls`
   input is segment boundary times `(N+1,)`, uniform and non-uniform both accepted;
   `ThrustLevel` enum didn't migrate — level sets parameterize as a `levels` tuple
   per issue text or user-defined conventions (default 0/60/100%).

## 中文

**状态**：已采纳
**日期**：2026-08-21
**关联 Issue**：#497、#498、#499、#501
**关联**：ADR 0011（五层架构）、ADR 0012（依赖方向）、ADR 0013（按定义验证）、ADR 0031（catalog 记录格式）、ADR 0032（HJB 动力学 crate 与绑定通用入口）；geo-nrho `docs/hjb-dependency-architecture.md`（下游依赖架构）

### 背景

geo-nrho 项目按 Bellman 最优性原理走两级 HJB 路线：离线用 e2m2e-levelset 在结构网格上解 HJ 方程得值函数，在线由值函数梯度生成闭环控制。下游依赖架构文档（geo-nrho `docs/hjb-dependency-architecture.md` §2.3）已列出四项建议上游化的能力，对应 issue #497（CR3BP 会合系 Hamiltonian）、#498（星历力模型 Hamiltonian）、#499（值函数梯度接口与离散工况映射）。

开工前有三条缝不定，后续深化会返工：

1. **crate 归属**（本篇起草期间由 ADR 0032 先行定死）：原创 HJB 动力学住新 crate e2m2e-hjb-dynamics：levelset 整体继承 ToolboxLS 的 ACM 非商业许可，原创代码并入会被罩进同一条款。本篇决策 1 与之对齐，不重复决策。
2. **值函数产品契约**。梯度查询接口放在 Python 层意味着接口与求解器解耦，契约随之转移到数据格式上：双积分器产物是无量纲时间、会合系坐标，星历产物将是 ET 秒、可能换参考系、可能加质量维。格式语义不定，每种 Hamiltonian 就会长出一个读取特例。geo-nrho 已有 `ProductMeta` 雏形（`produced_by`/`frame`/`units`/`maturity`），但与 ADR 0031 的键名体系不一致。
3. **时间维度地位**。双积分器是自治系统，geo-nrho 现状取最近时间快照尚能凑合；星历模型非定常，时间不插值就是错的。接口设计须把这一点提前固化。

### 决策

### 1. levelset 保持纯数学叶子；动力学归属从 ADR 0032

e2m2e-levelset 只保留 ToolboxLS 对应物与数学示例（`Advection` 等），动力学经 `Hamiltonian` trait 注入。原创动力学（#497 的 CR3BP 会合系 Hamiltonian、#498 的星历力模型 Hamiltonian）住 **e2m2e-hjb-dynamics**（Apache-2.0），理由见 ADR 0032（许可边界）。#498 的星历适配器在该 crate 内依赖 e2m2e-forces 的 `CompiledForce` 与 e2m2e-spice 的星历缓存，构造时注入，求解阶段纯查表、不碰 CSPICE、不进 Python 回调。不新增 `e2m2e-forces → e2m2e-levelset` 依赖边。

### 2. Python 暴露只经 e2m2e-integrators，走 ADR 0032 的通用入口

levelset 求解能力经 e2m2e-integrators 按 ABI 戳流程统一暴露：单一通用入口（动力学标识 + 参数表 + 网格定义 + 终端条件，ADR 0032 决策 3 的 `solve_hjb_py`），不为每种动力学各写专用签名。绑定层只做数组搬运、参数校验与错误转换，Hamiltonian 组装、控制极小化等领域逻辑一律在 Rust crate 内完成并可在 Rust 侧测试。

### 3. 值函数产品契约对齐 ADR 0031

值函数产品 = JSON 元数据 + NPZ 数组段，复用 catalog 记录体系而非另起格式：

- `schema_version` 自 1 起，不兼容跨版本读取；必备键沿用 ADR 0031 的 `_META_REQUIRED_KEYS` 体系（`source_tool`、状态三元组、`request` 快照、`source_record_id` 等）。
- 数值口径显式进元数据：状态维顺序、各维物理含义、无量纲化口径（特征长度/时间/质量或无）、`times` 语义（ET 秒或会合系无量纲时间）。口径不同的产品靠元数据字段区分，不靠读取方猜测，与 ADR 0031 用段存在性区分动力学模型同一精神。
- 值函数产品作为 catalog 新记录类型入库，`source_record_id` 指向作为终端约束的目标轨道记录。入库动作属求解端（#497/#498）；消费端（#499 的梯度接口）只要求能读该格式的 npz，不依赖 catalog 存在。
- geo-nrho `ProductMeta` 是下游原型，其 `frame`/`units`/`force_model` 字段语义被本契约吸收；geo-nrho 迁移时键名向 ADR 0031 对齐（`produced_by` → `source_tool` 等），属 geo-nrho 侧工作。

### 4. 梯度查询接口放 Python 算法层，时间插值必选

值函数在消费者手中本就是 numpy 数组，在线查询频率低（当前为控制周期级），无 Rust 化必要：

- 落点：Python 算法层（`e2m2e/algorithm/`），纯 numpy/SciPy 实现，遵循 ADR 0012 依赖方向。
- 维度无关：接口只吃 `axes`/`values`/`times` 与查询点，不假设状态维数或物理含义。
- 空间插值用张量积样条（如 `RegularGridInterpolator` 三次），梯度为**插值函数的解析导数**；禁止网格上中心差分再插值的路线（geo-nrho `_grid_gradient` 现状），后者正是本 issue 要消除的误差来源。
- 时间插值必选，至少线性；自治系统只是它退化仍正确的特例。
- 性能不是本接口的目标。若未来闭环仿真把查询打成瓶颈，平移 Rust 是独立决策，不影响本契约。

### 5. 离散工况映射：数据模型搬迁，常数参数化，算法含最短弧约束

- `ThrustLevel`（0/60/100%）、`ThrustArc`、`ThrustArcSequence` 从 geo-nrho `thrust_arcs.py` 原样迁入 e2m2e Python 低推力层，与 `LowThrustCollocation` 共享工况定义；geo-nrho 侧删除本地副本改为导入。
- 任务级常数参数化：`MIN_ARC_DURATION_S`（geo-nrho 现值 3600s）、`MAX_THRUST_N`、`ISP_S` 改为构造参数，不留模块级常量。
- 映射算法必须处理最短弧约束（合并/切分），不是 geo-nrho 现有的逐段最近档位，后者在配点段密于最短弧时直接报错，不可用。
- 验收在 e2m2e 内自包含：CR3BP 连续油门解（`LowThrustShooting`/`LowThrustCollocation` 生成）→ 映射 → 重传播 → 终端残差满足 L1 门槛（384 km / 1 m/s 量级）。不依赖 geo-nrho 算例，遵循 ADR 0013 与 `.out-of-scope/` 确立的验证不依赖外部研究代码原则。

### 理由

1. **crate 归属从 ADR 0032**：levelset 的 ACM 非商业许可会罩住并入的原创代码，这是比少一份 crate 脚手架更硬的约束；动力学集中住 e2m2e-hjb-dynamics 也让 levelset 保持忠实移植定位。绑定层（integrators）混领域逻辑则破坏其薄 FFI 定位，排除。
2. **契约对齐 0031 而非另起格式**：catalog 已解决同一个问题（多模型产物、口径歧义、谱系），且白得谱系机制：`source_record_id` 指向终端约束轨道，正是边界条件不可变难题在记录层的缓解：换目标轨道等于换谱系指针重解，产品间关系可追溯。
3. **梯度接口放 Python**：消费者、数据形态、查询频率三者都在 Python 侧，Rust 化的收益不成立；把接口做维度无关、时间必选，星历化时无需返工。
4. **数据模型原样搬迁**：契约已被 geo-nrho 验证过，重设计一套只会制造两个等价但不同的概念。

### 结果

**新增**：catalog 值函数记录类型（#498）；Python 梯度查询接口（#499）与离散工况映射模块（#501）。

**变更**：无对既有决策的变更。#497 已按 ADR 0032 落地（e2m2e-hjb-dynamics + 通用绑定入口），本篇决策 1、2 与之对齐；geo-nrho 的 `thrust_arcs.py` 在 #501 落地后删除本地副本，`ProductMeta` 键名后续向 ADR 0031 对齐（均为 geo-nrho 侧动作）。

**不变**：e2m2e-levelset 内核与其纯数学定位；e2m2e-integrators 薄绑定定位；现有 catalog 记录格式（值函数记录是新增类型，不改既有 schema）。

**代价**：catalog 需为值函数记录增加段约定与校验。

**范围外**：研究层面的三个难题（自适应网格、误差定量演化、边界条件可变的重解策略）不由本篇解决；本篇只保证工程架构不挡它们的路。

### 修订（2026-08-21，实施反馈）

1. **编号改 0033，决策 1、2 对齐已合入的 ADR 0032**：本篇起草时编号 0032，与随 #497 合入 master 的 ADR 0032（HJB 动力学归属新 crate 与绑定层通用入口）撞号且决策 1 相左；合并时改号 0033，决策 1、2 按 ADR 0032 对齐（动力学住 e2m2e-hjb-dynamics、绑定走通用入口 `solve_hjb_py`）。
2. **关联 Issue 补 #501**：分诊后 #499 拆分为两个 issue：#499 只含值函数梯度接口（决策 4），离散工况映射为 #501（决策 5）。决策 3 的入库归属不变。
3. **决策 4 的实现形态**：张量积样条取逐轴 not-a-knot 三次求解组装 `NdBSpline`，梯度取 `nu` 解析导数，C² 连续（排除了局部滑动模板路线：其在网格单元边界梯度跳变，而闭环控制的方向与开关函数直接消费梯度）。实现代价：无状态函数契约下每次调用为触及的每个时间快照重建样条（41⁴ 量级网格约 0.5 s/快照）；当前控制周期级查询可接受。若密集闭环仿真把查询打成瓶颈，引入带系数缓存的插值器对象是独立决策，不改函数契约。
4. **决策 5 的端到端算例定为地球星历系二体 LEO 两圈**（GravityField 零阶，与低推力测试既有 fixture 一致），非 CR3BP：自包含、快、不依赖 geo-nrho 的意图不变。384 km / 1 m/s 是 L1 任务级口径，对 LEO 算例过松，测试按实测残差（约 0.35 km / 0.0004 m/s）收紧十余倍作回归断言。CR3BP 端到端验证随 #497 落地后补强。
5. **决策 5 补充三条接口细化**：`validate` 增加可选 `levels` 参数校验档位合法性（brief 验收口径）；`sequence_from_controls` 的输入为段边界时刻 `(N+1,)`，均匀与非均匀时间节点同样接受；`ThrustLevel` 枚举未随迁，档位集合按 issue 原文或用户自定义档位口径参数化为 `levels` 元组（默认 0/60/100%）。
