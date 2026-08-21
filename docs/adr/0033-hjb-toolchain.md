# ADR 0033：HJB 小推力工具链——值函数产品契约与在线查询接口

**状态**：已采纳
**日期**：2026-08-21
**关联 Issue**：#497、#498、#499、#501
**关联**：ADR 0011（五层架构）、ADR 0012（依赖方向）、ADR 0013（按定义验证）、ADR 0031（catalog 记录格式）、ADR 0032（HJB 动力学 crate 与绑定通用入口）；geo-nrho `docs/hjb-dependency-architecture.md`（下游依赖架构）

## 背景

geo-nrho 项目按 Bellman 最优性原理走两级 HJB 路线：离线用 e2m2e-levelset 在结构网格上解 HJ 方程得值函数，在线由值函数梯度生成闭环控制。下游依赖架构文档（geo-nrho `docs/hjb-dependency-architecture.md` §2.3）已列出四项建议上游化的能力，对应 issue #497（CR3BP 会合系 Hamiltonian）、#498（星历力模型 Hamiltonian）、#499（值函数梯度接口与离散工况映射）。

开工前有三条缝不定，后续深化会返工：

1. **crate 归属**（本篇起草期间由 ADR 0032 先行定死）：原创 HJB 动力学住新 crate e2m2e-hjb-dynamics——levelset 整体继承 ToolboxLS 的 ACM 非商业许可，原创代码并入会被罩进同一条款。本篇决策 1 与之对齐，不重复决策。
2. **值函数产品契约**。梯度查询接口放在 Python 层意味着接口与求解器解耦，契约随之转移到数据格式上：双积分器产物是无量纲时间、会合系坐标，星历产物将是 ET 秒、可能换参考系、可能加质量维。格式语义不定，每种 Hamiltonian 就会长出一个读取特例。geo-nrho 已有 `ProductMeta` 雏形（`produced_by`/`frame`/`units`/`maturity`），但与 ADR 0031 的键名体系不一致。
3. **时间维度地位**。双积分器是自治系统，geo-nrho 现状取最近时间快照尚能凑合；星历模型非定常，时间不插值就是错的。接口设计须把这一点提前固化。

## 决策

### 1. levelset 保持纯数学叶子；动力学归属从 ADR 0032

e2m2e-levelset 只保留 ToolboxLS 对应物与数学示例（`Advection` 等），动力学经 `Hamiltonian` trait 注入。原创动力学——#497 的 CR3BP 会合系 Hamiltonian、#498 的星历力模型 Hamiltonian——住 **e2m2e-hjb-dynamics**（Apache-2.0），理由见 ADR 0032（许可边界）。#498 的星历适配器在该 crate 内依赖 e2m2e-forces 的 `CompiledForce` 与 e2m2e-spice 的星历缓存，构造时注入，求解阶段纯查表、不碰 CSPICE、不进 Python 回调。不新增 `e2m2e-forces → e2m2e-levelset` 依赖边。

### 2. Python 暴露只经 e2m2e-integrators，走 ADR 0032 的通用入口

levelset 求解能力经 e2m2e-integrators 按 ABI 戳流程统一暴露：单一通用入口（动力学标识 + 参数表 + 网格定义 + 终端条件，ADR 0032 决策 3 的 `solve_hjb_py`），不为每种动力学各写专用签名。绑定层只做数组搬运、参数校验与错误转换，Hamiltonian 组装、控制极小化等领域逻辑一律在 Rust crate 内完成并可在 Rust 侧测试。

### 3. 值函数产品契约对齐 ADR 0031

值函数产品 = JSON 元数据 + NPZ 数组段，复用 catalog 记录体系而非另起格式：

- `schema_version` 自 1 起，不兼容跨版本读取；必备键沿用 ADR 0031 的 `_META_REQUIRED_KEYS` 体系（`source_tool`、状态三元组、`request` 快照、`source_record_id` 等）。
- 数值口径显式进元数据：状态维顺序、各维物理含义、无量纲化口径（特征长度/时间/质量或"无"）、`times` 语义（ET 秒或会合系无量纲时间）。口径不同的产品靠元数据字段区分，不靠读取方猜测——与 ADR 0031 用段存在性区分动力学模型同一精神。
- 值函数产品作为 catalog 新记录类型入库，`source_record_id` 指向作为终端约束的目标轨道记录。入库动作属求解端（#497/#498）；消费端（#499 的梯度接口）只要求能读该格式的 npz，不依赖 catalog 存在。
- geo-nrho `ProductMeta` 是下游原型，其 `frame`/`units`/`force_model` 字段语义被本契约吸收；geo-nrho 迁移时键名向 ADR 0031 对齐（`produced_by` → `source_tool` 等），属 geo-nrho 侧工作。

### 4. 梯度查询接口放 Python 算法层，时间插值必选

值函数在消费者手中本就是 numpy 数组，在线查询频率低（当前为控制周期级），无 Rust 化必要：

- 落点：Python 算法层（`e2m2e/algorithm/`），纯 numpy/SciPy 实现，遵循 ADR 0012 依赖方向。
- 维度无关：接口只吃 `axes`/`values`/`times` 与查询点，不假设状态维数或物理含义。
- 空间插值用张量积样条（如 `RegularGridInterpolator` 三次），梯度为**插值函数的解析导数**；禁止"网格上中心差分再插值"的路线（geo-nrho `_grid_gradient` 现状），后者正是本 issue 要消除的误差来源。
- 时间插值必选，至少线性；自治系统只是它退化仍正确的特例。
- 性能不是本接口的目标。若未来闭环仿真把查询打成瓶颈，平移 Rust 是独立决策，不影响本契约。

### 5. 离散工况映射：数据模型搬迁，常数参数化，算法含最短弧约束

- `ThrustLevel`（0/60/100%）、`ThrustArc`、`ThrustArcSequence` 从 geo-nrho `thrust_arcs.py` 原样迁入 e2m2e Python 低推力层，与 `LowThrustCollocation` 共享工况定义；geo-nrho 侧删除本地副本改为导入。
- 任务级常数参数化：`MIN_ARC_DURATION_S`（geo-nrho 现值 3600s）、`MAX_THRUST_N`、`ISP_S` 改为构造参数，不留模块级常量。
- 映射算法必须处理最短弧约束（合并/切分），不是 geo-nrho 现有的逐段最近档位——后者在配点段密于最短弧时直接报错，不可用。
- 验收在 e2m2e 内自包含：CR3BP 连续油门解（`LowThrustShooting`/`LowThrustCollocation` 生成）→ 映射 → 重传播 → 终端残差满足 L1 门槛（384 km / 1 m/s 量级）。不依赖 geo-nrho 算例，遵循 ADR 0013 与 `.out-of-scope/` 确立的"验证不依赖外部研究代码"原则。

## 理由

1. **crate 归属从 ADR 0032**：levelset 的 ACM 非商业许可会罩住并入的原创代码，这是比"少一份 crate 脚手架"更硬的约束；动力学集中住 e2m2e-hjb-dynamics 也让 levelset 保持忠实移植定位。绑定层（integrators）混领域逻辑则破坏其薄 FFI 定位，排除。
2. **契约对齐 0031 而非另起格式**：catalog 已解决同一个问题（多模型产物、口径歧义、谱系），且白得谱系机制——`source_record_id` 指向终端约束轨道，正是"边界条件不可变"难题在记录层的缓解：换目标轨道等于换谱系指针重解，产品间关系可追溯。
3. **梯度接口放 Python**：消费者、数据形态、查询频率三者都在 Python 侧，Rust 化的收益不成立；把接口做维度无关、时间必选，星历化时无需返工。
4. **数据模型原样搬迁**：契约已被 geo-nrho 验证过，重设计一套只会制造两个等价但不同的概念。

## 结果

**新增**：catalog 值函数记录类型（#498）；Python 梯度查询接口（#499）与离散工况映射模块（#501）。

**变更**：无对既有决策的变更。#497 已按 ADR 0032 落地（e2m2e-hjb-dynamics + 通用绑定入口），本篇决策 1、2 与之对齐；geo-nrho 的 `thrust_arcs.py` 在 #501 落地后删除本地副本，`ProductMeta` 键名后续向 ADR 0031 对齐（均为 geo-nrho 侧动作）。

**不变**：e2m2e-levelset 内核与其纯数学定位；e2m2e-integrators 薄绑定定位；现有 catalog 记录格式（值函数记录是新增类型，不改既有 schema）。

**代价**：catalog 需为值函数记录增加段约定与校验。

**范围外**：研究层面的三个难题（自适应网格、误差定量演化、边界条件可变的重解策略）不由本篇解决；本篇只保证工程架构不挡它们的路。

## 修订（2026-08-21，实施反馈）

1. **编号改 0033，决策 1、2 对齐已合入的 ADR 0032**：本篇起草时编号 0032，与随 #497 合入 master 的 ADR 0032（HJB 动力学归属新 crate 与绑定层通用入口）撞号且决策 1 相左；合并时改号 0033，决策 1、2 按 ADR 0032 对齐（动力学住 e2m2e-hjb-dynamics、绑定走通用入口 `solve_hjb_py`）。
2. **关联 Issue 补 #501**：分诊后 #499 拆分为两个 issue——#499 只含值函数梯度接口（决策 4），离散工况映射为 #501（决策 5）。决策 3 的入库归属不变。
3. **决策 4 的实现形态**：张量积样条取逐轴 not-a-knot 三次求解组装 `NdBSpline`，梯度取 `nu` 解析导数，C² 连续（排除了局部滑动模板路线——其在网格单元边界梯度跳变，而闭环控制的方向与开关函数直接消费梯度）。实现代价：无状态函数契约下每次调用为触及的每个时间快照重建样条（41⁴ 量级网格约 0.5 s/快照）；当前控制周期级查询可接受。若密集闭环仿真把查询打成瓶颈，引入带系数缓存的插值器对象是独立决策，不改函数契约。
4. **决策 5 的端到端算例定为地球星历系二体 LEO 两圈**（GravityField 零阶，与低推力测试既有 fixture 一致），非 CR3BP：自包含、快、不依赖 geo-nrho 的意图不变。384 km / 1 m/s 是 L1 任务级口径，对 LEO 算例过松，测试按实测残差（约 0.35 km / 0.0004 m/s）收紧十余倍作回归断言。CR3BP 端到端验证随 #497 落地后补强。
5. **决策 5 补充三条接口细化**：`validate` 增加可选 `levels` 参数校验档位合法性（brief 验收口径）；`sequence_from_controls` 的输入为段边界时刻 `(N+1,)`，均匀与非均匀时间节点同样接受；`ThrustLevel` 枚举未随迁，档位集合按 issue 原文"或用户自定义档位"口径参数化为 `levels` 元组（默认 0/60/100%）。
