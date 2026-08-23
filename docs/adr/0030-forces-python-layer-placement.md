# ADR 0030：algorithm/forces 留在 algorithm 层：Python 配置/编排面，数值在 crates

**状态**：已采纳
**日期**：2026-08-17
**关联 Issue**：#429（forces 源码层级归属）
**关联**：ADR 0011（五层架构）、ADR 0012（依赖方向）、ADR 0004（ForceModel 配置驱动）、ADR 0021（测试功能类目）、ADR 0026（后续工作中 forces 归属，本条由其审计而来）、ADR 0027（同型先例：评估后维持 + ADR）

## 背景

ADR 0026 把 forces 测试收拢到 `tests/numerical/forces`，但把源码层级归属留给 #429：`e2m2e/algorithm/forces` 的 Python 包自述为参数验证 + to_rust_spec 序列化，对应 Rust `e2m2e-forces` 的 `CompiledForce`；数值清单已把 forces 数值登记为已下沉。审计仍可能把测试在 numerical、源码在 algorithm、自述像 integrators 读成放错层，并追问：是否应迁出 algorithm？若迁，数值层 Python 的形态是新建 `e2m2e/numerical/`，还是与 `integrators.py` 并列？

#429 的约束写明：五层架构的数值层是 `crates/`，Python 侧目前只有绑定薄封装、没有数值目录；若维持现状，说明理由并关闭即可。本篇记录裁决。

## 决策

1. **`e2m2e/algorithm/forces` 维持现状，留在 algorithm 层。** Python 侧是力模型配置与编排面，不是数值核。
2. **不新建 `e2m2e/numerical/`（或任何 Python 数值层目录）。** 五层架构的数值层仍是 `crates/`；不为此发明第六层。
3. **不把 forces 迁到与 `integrators.py` 并列，也不压成顶层单文件/薄模块。** forces 是配置驱动的领域包（ADR 0004），与绑定 re-export 形态不同。

## 理由

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

## 结果

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
