# ADR 0032: HJB dynamics in a new crate and binding-layer generic entry / HJB 动力学归属新 crate 与绑定层通用入口

[English](#adr-0032-hjb-dynamics-in-a-new-crate-and-binding-layer-generic-entry) | [简体中文](#中文)

## English

**Status**: Adopted
**Date**: 2026-08-21
**Related Issue**: #497
**Related**: ADR 0011 (five-layer architecture), ADR 0012 (dependency
direction), ADR 0013 (verification by definition), ADR 0016 (EphemCache)

### Context

After completing the ToolboxLS port, e2m2e-levelset carried only example
Hamiltonians (constant advection, Burgers, double integrator). geo-nrho's two-
level dynamic programming route requires plugging Earth-Moon synodic CR3BP
dynamics into the HJB solver (#497), with an explicit later deepening sequence:
5D mass-carrying, solar third body, ephemeris force models. This is the first
original dynamics entering that subsystem — code ownership, license boundary,
binding shape, and dimension ceiling must be fixed before work starts, or every
deepening reworks. The full system shape & division live in
docs/architecture/hjb-subsystem.md; this entry only fixes decisions.

### Decision

1. Original HJB dynamics live in the new crate **e2m2e-hjb-dynamics**, licensed
   under workspace Apache-2.0. e2m2e-levelset keeps only ToolboxLS
   counterparts.
2. Dynamics enter the solver via the `Hamiltonian` trait; parameters are fields
   of implementing structs — no callbacks, no Python inside Rust hot loops.
3. Python bindings use a single generic entry: dynamics identifier + parameter
   table + grid definition + terminal conditions. No per-dynamics binding
   functions. The binding layer validates key existence and values; invalid
   input raises explicit errors.
4. Grid-layer state dimensions cap at five. Three-dimensional high-fidelity
   problems (six dims + time) belong to downstream two-level neural-network
   tiers, never solved on structured grids.
5. State semantics documented: standard nondimensional synodic frame; angular
   speed identically 1, not a parameter; μ fixed at construction; state order
   (x, y, vx, vy); four-dimensional states lift to three-dimensional as z = vz = 0
   sections.

### Rationale

1. e2m2e-levelset wholesale inherits ToolboxLS's ACM non-commercial license;
   merging original dynamics would sweep them under it, damaging availability as
   general infrastructure. An independent crate also preserves levelset's
   faithful-porting positioning — each module maintainable against MATLAB
   originals.
2. The Hamiltonian trait is ToolboxLS's function-handle protocol naturalized
   (see crate README's protocol mapping table), proven sufficient through four
   verification phases. Dimension-agnostic with explicit time parameter:
   mass-carrying and non-autonomous deepening don't touch the seam.
3. The counterexample is solve_planar_lowthrust_hjb_py: signature pinned to a
   double integrator. Following that pattern, every new family member bumps the
   ABI stamp again and maintains another dedicated signature. The generic entry
   pins ABI change to the single addition of the entry itself.
4. At 40 nodes/dim, six-dimensional grids need ~33 GB per array — the dimension
   cap is arithmetic fact. Two-tier division is the demand side's settled plan
   (geo-nrho architecture doc §1.2/3.1); this entry promotes it from downstream
   convention to e2m2e-side architectural constraint.
5. Level-two networks train on level-one solutions; both sides' interpretations
   of state order, nondimensionalization, and frames must match verbatim.
   Conventions living only in code will misalign when mission trajectory data
   arrives.

### Consequences

- Added: crate e2m2e-hjb-dynamics (landed with #497; first member four-
  dimensional planar CR3BP); docs/architecture/hjb-subsystem.md.
- Changed: e2m2e-integrators gains generic HJB solving bindings; ABI stamp
  increments.
- Unchanged: e2m2e-levelset module structure & licensing; ADR 0012/0016 rules.
- Costs: generic bindings' parameter tables are weakly typed at FFI boundaries;
  misspelled keys surface only at runtime, mitigated by binding-layer validation
  + explicit errors; one more crate scaffold to maintain.

### Revision (2026-08-21, #497 implementation review)

Consequences add distribution implications: from #497,
e2m2e-integrators depends on e2m2e-levelset, so released wheels contain ACM
non-commercial licensed code. Per e2m2e-levelset README's license section,
future release notes must state this difference; commercial use requires
contacting ToolboxLS's original author.

## 中文

**状态**：已采纳
**日期**：2026-08-21
**关联 Issue**：#497
**关联**：ADR 0011（五层架构）、ADR 0012（依赖方向）、ADR 0013（按定义验证）、ADR 0016（EphemCache）

### 背景

e2m2e-levelset 完成 ToolboxLS 移植后只带示例 Hamiltonian（常速平流、Burgers、双积分器）。geo-nrho 的两级动态规划路线要求把地月会合系 CR3BP 动力学接入 HJB 求解器（#497），且后续有明确的深化序列：五维含质量、太阳第三体、星历力模型。这是第一次往该子系统加入原创动力学，代码归属、许可边界、绑定层形状与维度边界需要在动工前定死，否则每次深化都要返工。系统形态与分工的完整描述见 docs/architecture/hjb-subsystem.md，本篇只固化决策。

### 决策

1. 原创 HJB 动力学放新 crate e2m2e-hjb-dynamics，许可随 workspace 的 Apache-2.0。e2m2e-levelset 只保留 ToolboxLS 对应物。
2. 动力学经 Hamiltonian trait 进入求解器，参数走实现结构体的字段，不用回调，Python 不进入 Rust 热循环。
3. Python 绑定为单一通用入口：动力学标识加参数表加网格定义加终端条件。不为每种动力学各写一个绑定函数。参数表在绑定层做存在性与取值校验，非法输入报明确错误。
4. 网格层状态维度上限为五。三维高保真问题（六维状态加时间）由下游两级方案的神经网络层承接，不在结构网格层求解。
5. 状态语义文档化：标准无量纲会合系，角速度恒为 1 不作参数，μ 构造时固定；状态顺序 (x, y, vx, vy)；四维状态按 z = vz = 0 截面提升为三维状态。

### 理由

1. e2m2e-levelset 整体继承 ToolboxLS 的 ACM 非商业许可，原创动力学代码并入会被罩进同一条款，损害其作为通用基础设施的可用性。独立 crate 同时保持 levelset 忠实移植的定位，每个模块可对照 MATLAB 原版维护。
2. Hamiltonian trait 是 ToolboxLS 函数句柄协议的自然物化（见 crate README 的协议映射表），已在四个验证阶段证明够用。trait 维度无关且显式带时间参数，含质量与非自治深化无需改接缝。
3. 现状反例是 solve_planar_lowthrust_hjb_py：签名按双积分器写死。照此模式，动力学家族每添一个成员就要动一次 ABI 戳并维护一份专用签名。通用入口把 ABI 变更固定在入口新增的一次。
4. 每维 40 节点时六维网格单数组约 33 GB，维度上限是算术事实。两级分工是需求侧既定方案（geo-nrho 架构文档 1.2、3.1 节），本篇把它从下游约定升格为 e2m2e 侧的架构约束。
5. 第二级神经网络以第一级低维解为训练数据，两边对状态顺序、无量纲化、帧的解释必须逐字一致。约定只活在代码里，接任务轨迹数据时会对不上。

### 结果

- 新增：crate e2m2e-hjb-dynamics（随 #497 落地，首个成员为四维平面 CR3BP）；docs/architecture/hjb-subsystem.md。
- 变更：e2m2e-integrators 增加通用 HJB 求解绑定，ABI 戳递增。
- 不变：e2m2e-levelset 的模块结构与许可条款；ADR 0012、0016 的规则。
- 代价：通用绑定的参数表在 FFI 边界弱类型，键名拼错只能运行期发现，靠绑定层校验与明确报错缓解；新 crate 增加一份脚手架维护成本。

### 修订（2026-08-21，#497 实现评审）

结果节补充分发后果：e2m2e-integrators 自 #497 起依赖 e2m2e-levelset，发布的 wheel 自此包含 ACM 非商业许可代码。按 e2m2e-levelset README 许可节的要求，后续发布说明须注明此许可差异，商业使用需联系 ToolboxLS 原作者。
