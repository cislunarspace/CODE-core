# ADR 0030: algorithm/forces stays at the algorithm layer — Python config/orchestration surface, numerics in crates / algorithm/forces 留在 algorithm 层：Python 配置/编排面，数值在 crates

[English](#adr-0030-algorithmforces-stays-at-the-algorithm-layer--python-configorchestration-surface-numerics-in-crates) | [简体中文](#中文)

## English

**Status**: Adopted
**Date**: 2026-08-17
**Related Issue**: #429 (forces source-layer ownership)
**Related**: ADR 0011 (five-layer architecture), ADR 0012 (dependency
direction), ADR 0004 (ForceModel config-driven), ADR 0021 (test functional
categories), ADR 0026 (follow-ups — forces ownership; this entry grew from that
audit), ADR 0027 (same-type precedent: evaluated and retained + ADR)

### Context

ADR 0026 consolidated forces tests into `tests/numerical/forces` but left
source-layer ownership to #429: the `e2m2e/algorithm/forces` Python package
self-describes as parameter validation + to_rust_spec serialization,
corresponding to Rust `e2m2e-forces`'s `CompiledForce`; the migration ledger
already records forces numerics as sunk. An audit could still read tests at
numerical, sources at algorithm, and self-description resembling integrators as
misplaced — then ask: should it move out of algorithm? If so, should numerical-
layer Python take the shape of a new `e2m2e/numerical/`, or sit beside
`integrators.py`?

#429's constraints state: the five-layer architecture's numerical layer is
`crates/`; Python currently has only thin binding wrappers, no numerics
directory. If status quo holds, explain why and close. This entry records the
ruling.

### Decision

1. **`e2m2e/algorithm/forces` stays put, in the algorithm layer.** The Python
   side is force-model configuration & orchestration surface, not a numeric
   core.
2. **No new `e2m2e/numerical/` (or any Python numerical-layer directory).**
   The five-layer architecture's numerical layer remains `crates/`; no sixth
   layer gets invented for this.
3. **Forces doesn't move beside `integrators.py`, nor get flattened into a
   top-level single file/thin module.** forces is a config-driven domain
   package (ADR 0004), different in kind from binding re-exports.

### Rationale

#### Responsibility: computation vs orchestration already separated

Force-model numerics (spherical harmonics, tides, SRP, third-body, drag, STM…)
live in the `e2m2e-forces` crate. Python's `PhysicalModel` / `ForceModel` do only
parameter validation, `to_rust_spec` serialization, config round-trips, and post-
construction calls into `integrators`' compiled propagation entries; accelerations
and Jacobians keep no Python reference implementations.

This matches architecture docs and README's division: crates compute; Python
constructs problems, calls Rust, interprets results.
`numerics-migration-status` already marks `algorithm/forces` (numerics) as sunk.
This decision doesn't change responsibility division — only **directory
ownership**: numerics already sunk and where the Python config surface lives are
two separate matters.

#### Structure: why not out of algorithm

1. **Consumers are in algorithm.** design, station_keeping, transfer,
   propagation all construct force models inside algorithm chains and propagate.
   Config surface and domain orchestration share a layer, avoiding cross-layer
   moves done purely for directory symmetry.
2. **Real runtime dependencies on algorithm exist.** The config surface depends
   on System context and coordinate (frames, axes, origins); containers call
   integrators. Moving wholesale out of algorithm would either create
   numerical-side → algorithm reverse dependencies (violating ADR 0012) or
   force System/coordinate extraction too — a #430-scale migration, with
   #430/ADR 0027 having already ruled System stays in algorithm.
3. **Five layers have no Python-numerics directory slot.** ADR 0011's numerical
   layer is `crates/`; Python's numerics entry is only the parallel thin binding
   wrapper (`integrators.py` → `_integrators`); there is no `e2m2e/numerical/`.
   forces is a multi-type domain package carrying config schema and containers —
   not a single-file re-export; flattening beside `integrators.py` would crush
   the package structure. A sixth layer changes looks only, zero behavior gain.
4. **Test directories can't back-infer source layers.** ADR 0021/0026:
   functional-class markers and code layering are two axes.
   `tests/numerical/forces` verifies Rust force-model numeric contracts — not a
   demand that Python sources enter numerical. Same root as coordinate tests
   marked `data` while source stays algorithm.

#### Why alternatives were rejected

**Moving out of algorithm (beside integrators, or new numerical/).** Would first
require defining the numerical-layer Python shape that isn't written into the
five layers; untangling System/coordinate dependencies or accepting violating
ones; touching many algorithm consumers and many test imports. Behavior
unchanged; the sole benefit is directory symmetry.

**Flattening to top-level single file/thin module because it resembles
integrators.** forces is a config-driven domain package (ADR 0004), not a
binding-layer re-export; similar duty ≠ comparable form.

### Consequences

#### Added

- This ADR.

#### Changed

- forces entry in `docs/architecture/numerics-migration-status.md`: from "to be
  independently assessed by #429" to citing this ADR's adjudicated statement.

#### Unchanged

- `e2m2e/algorithm/forces` directory, interfaces, implementation, test paths —
  untouched line by line.
- Numerics in `e2m2e-forces`, Python config/orchestration only — unchanged.
- ADR 0011 five-layer architecture & ADR 0012 dependency-direction texts.

#### Costs

- Test tree stays `tests/numerical/forces`; source tree stays
  `e2m2e/algorithm/forces`. The two axes (functional class vs layer) coexist
  and invite misreading; this ADR shares explanatory duty with ADR 0026 to
  prevent repeat misjudgments.

## 中文

**状态**：已采纳
**日期**：2026-08-17
**关联 Issue**：#429（forces 源码层级归属）
**关联**：ADR 0011（五层架构）、ADR 0012（依赖方向）、ADR 0004（ForceModel 配置驱动）、ADR 0021（测试功能类目）、ADR 0026（后续工作中 forces 归属，本条由其审计而来）、ADR 0027（同型先例：评估后维持 + ADR）

### 背景

ADR 0026 把 forces 测试收拢到 `tests/numerical/forces`，但把源码层级归属留给 #429：`e2m2e/algorithm/forces` 的 Python 包自述为参数验证 + to_rust_spec 序列化，对应 Rust `e2m2e-forces` 的 `CompiledForce`；数值清单已把 forces 数值登记为已下沉。审计仍可能把测试在 numerical、源码在 algorithm、自述像 integrators 读成放错层，并追问：是否应迁出 algorithm？若迁，数值层 Python 的形态是新建 `e2m2e/numerical/`，还是与 `integrators.py` 并列？

#429 的约束写明：五层架构的数值层是 `crates/`，Python 侧目前只有绑定薄封装、没有数值目录；若维持现状，说明理由并关闭即可。本篇记录裁决。

### 决策

1. **`e2m2e/algorithm/forces` 维持现状，留在 algorithm 层。** Python 侧是力模型配置与编排面，不是数值核。
2. **不新建 `e2m2e/numerical/`（或任何 Python 数值层目录）。** 五层架构的数值层仍是 `crates/`；不为此发明第六层。
3. **不把 forces 迁到与 `integrators.py` 并列，也不压成顶层单文件/薄模块。** forces 是配置驱动的领域包（ADR 0004），与绑定 re-export 形态不同。

### 理由

### 职责层：计算与编排已分离

力模型数值（球谐、潮汐、SRP、三体、大气、STM 等）在 `e2m2e-forces` crate。Python 的 `PhysicalModel` / `ForceModel` 只做参数验证、`to_rust_spec` 序列化、配置往返，以及构造问题后调用 `integrators` 的编译传播入口；加速度与雅可比不保留 Python 参考实现。

这与架构文档和 README 的分工一致：crates 算，Python 构造问题、调 Rust、解释结果。`numerics-migration-status` 已把 `algorithm/forces`（数值）标为已下沉。本决策不改职责划分，只定**目录归属**：数值已下沉与 Python 配置面放哪是两件事。

### 结构层：为何不迁出 algorithm

1. **消费面在 algorithm。** design、station_keeping、transfer、propagation 等在算法链里构造力模型并传播。配置面与领域编排同层，避免仅为目录对称做跨层搬家。
2. **对 algorithm 有真实运行时依赖。** 配置面依赖 System 上下文与 coordinate（坐标系、轴、原点），容器再调 integrators。整体迁出 algorithm，要么制造数值侧 → algorithm 反向依赖（违反 ADR 0012），要么迫使 System/coordinate 一并外提，后者是 #430 级大迁移，且 #430 / ADR 0027 已裁决 System 留在 algorithm。
3. **五层无 Python 数值目录槽。** ADR 0011 的数值层是 `crates/`；Python 数值入口只有并列的绑定薄封装（`integrators.py` → `_integrators`），没有 `e2m2e/numerical/`。forces 是多类型、带配置 schema 与容器的领域包，不是单文件 re-export；硬与 `integrators.py` 并列会压扁包结构。新建第六层只换目录观感，行为零收益。
4. **测试目录不能反推源码层。** ADR 0021、0026：功能类标记与代码层级是两个轴。`tests/numerical/forces` 验证的是 Rust 力模型数值契约，不是要求 Python 源码进 numerical。与 coordinate 测试标 `data`、源码却留 algorithm 同源。

### 反方案为何被排除

**迁出 algorithm（旁靠 integrators 或新建 numerical/）。** 须先定未写入五层的数值层 Python 形态；还要解开 System/coordinate 依赖或接受违规依赖；牵动多个 algorithm 消费者与大量测试 import。行为不变，收益仅为目录对称。

**因像 integrators 而压成顶层单文件/薄模块。** forces 是配置驱动领域包（ADR 0004），不是绑定层 re-export；职责相近不等于形态可类比。

### 结果

### 新增

- 本篇 ADR。

### 变更

- `docs/architecture/numerics-migration-status.md` 中 forces 条目：由 #429 独立评估改为引用本 ADR 的已裁决表述。

### 不变

- `e2m2e/algorithm/forces` 目录、接口、实现与测试路径一行未动。
- 数值在 `e2m2e-forces`、Python 只做配置/编排的职责划分不变。
- ADR 0011 五层架构与 ADR 0012 依赖方向规则本文不变。

### 代价

- 测试树仍是 `tests/numerical/forces`，源码树仍是 `e2m2e/algorithm/forces`。两轴（功能类 vs 层级）并存，容易被误读；本 ADR 与 ADR 0026 共同承担这份说明义务，避免再次误判。
