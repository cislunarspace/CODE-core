# HJB Subsystem Architecture / HJB 子系统架构

[English](#english) | [简体中文](#简体中文)

## English

Start from a concrete scenario. A spacecraft performs a low-thrust transfer in
cislunar space; flight control must answer at every step: in the current
state, which direction to thrust and how hard. Direct and indirect methods
re-solve an optimal control problem at every query — slow and unstable. The
HJB route does it differently: offline, solve the Hamilton-Jacobi equation
once on a state grid to obtain a numerical table of the value function V(x,
t); online, look up the value-function gradient and compute the control
directly from optimality conditions. One solve, fast queries. That is what
this subsystem does.

This page describes the target shape of e2m2e's HJB subsystem: who provides
what, where the seams are, and how far deepening goes. The demand-side
two-level dynamic programming scheme lives in the geo-nrho project's
docs/hjb-dependency-architecture.md; this is the corresponding supply-side
architecture. Decision snapshots: ADR 0032, 0033, 0034.

### 1. Subsystem positioning and two-level division of labor

Low-thrust trajectory rapid planning has two levels. Level one solves the HJB
equation backwards on structured grids, producing a low-dimensional value
function table. Level two approximates the value function with a neural
network over the full seven-dimensional state (six state dims plus time) for
online control. Level two exists because level one cannot escape the curse of
dimensionality: structured-grid storage grows exponentially with dimension;
seven dims on a grid is infeasible.

e2m2e delivers level one plus verification tooling, concretely four things:

- Grid solver: e2m2e-levelset, the Rust port of ToolboxLS, providing upwind
  schemes, Lax-Friedrichs dissipation, TVD Runge-Kutta time integration.
- Dynamics: a set of structs implementing the Hamiltonian trait that plug a
  concrete problem's vector field into the solver.
- Python bindings: solve entry points exposed via e2m2e-integrators so
  downstream orchestration can run offline solves.
- Verification tools: propagators and compiled force models for closed-loop
  replay and coarse-vs-fine model comparison.

e2m2e does not deliver level two. Neural network training, online policy, and
mission-layer terminal constraints belong to downstream projects (currently
geo-nrho). This boundary deserves emphasis: level one's output is a value
function grid file; how level two consumes it is governed by e2m2e only as
file format and state semantics — training and inference are out of scope.

### 2. The Hamiltonian seam and the dynamics family

The seam between solver and dynamics is e2m2e-levelset's Hamiltonian trait,
corresponding to ToolboxLS's `hamFunc` and `partialFunc` callbacks, with two
methods:

- `hamiltonian(t, grid, phi, p)`: computes H(x, t, p) over the whole grid,
  with the control already analytically eliminated by optimality conditions.
- `partial_bound(t, grid, phi, p_min, p_max, dim)`: per-dimension dissipation
  coefficient envelope max|∂H/∂p_dim|.

Dynamics parameters become fields of the implementing struct, fixed at
construction — no callbacks, no Python inside Rust hot loops. The trait is
dimension-agnostic and carries an explicit time parameter; these two
properties are structural preconditions for later deepening.

Dynamics is not one implementation but a family, a spectrum ordered by
fidelity:

| Impl | State dims | Autonomy | Force model | Landing |
|---|---|---|---|---|
| Cr3bpSynodic | 4: x, y, vx, vy | autonomous | two primaries point-mass, centrifugal, Coriolis | #497 |
| Mass extension | 5: add m | autonomous | same, control becomes thrust, acceleration decays with mass | mass axis landed with #498 (ephemeris convention); CR3BP convention follow-up issue |
| Bcr4bp | 4 or 5 | non-autonomous, explicit t | adds solar gravity | follow-up issue |
| Ephemeris force model | 4 or 5 | non-autonomous | e2m2e-forces compiled force models, ephemeris via EphemCache lookup (ADR 0016) | #498 (ADR 0034) |

New family members do not replace old ones. Coarse models hold two permanent
values: computation is cheap, suited to tuning grid and scheme parameters; and
they serve as regression baselines for fine models — an ephemeris force-model
solution should first match the CR3BP solution in magnitude and structure;
mismatch means implementation error, caught before comparing against mission
data.

Code ownership: family members live in the new crate e2m2e-hjb-dynamics
(Apache-2.0), not in e2m2e-levelset. Two reasons. Licensing: levelset inherits
ToolboxLS's ACM non-commercial license wholesale, so original code placed
there would fall under the same terms. Positioning: levelset keeps the purity
of faithful porting, every module maintainable against its MATLAB original;
dynamics is e2m2e's original layer — different concerns. Dependency direction
is one-way: hjb-dynamics depends on levelset's traits; under the `ephemeris`
feature it additionally depends on e2m2e-forces / e2m2e-spice ephemeris cache
types; reverse dependency forbidden, consistent with ADR 0012's dependency
direction spirit.

### 3. Dimension ceiling

Structured-grid storage = node count × bytes per node. With 40 nodes per
dim, double precision, single array:

| State dims | Nodes | Single array memory |
|---|---|---|
| 4 | 2.6×10⁶ | ≈ 0.02 GB |
| 5 | 1.0×10⁸ | ≈ 0.8 GB |
| 6 | 4.1×10⁹ | ≈ 33 GB |

The solver holds several same-shaped arrays simultaneously (φ, per-dim
gradients, dissipation coefficients), so real usage is several times one
array. The conclusion is hard: **grid-layer state dimensions cap at five**.
Six-dimensional infeasibility is arithmetic fact, not something engineering
optimization can route around.

This ceiling decides where high-fidelity forces go. High-order gravity of
Earth and Moon (e.g. 10×10) and SRP are intrinsically three-dimensional
forces: zonal terms cannot be faithfully projected into planar synodic frames,
and SRP direction depends on the Sun's three-dimensional position.
Three-dimensional forces mean six-dimensional states, beyond the ceiling.
Hence three-dimensional high-fidelity problems are not solved at the grid
layer; they are carried by level-two neural networks whose training data comes
from level-one low-dimensional solutions and mission trajectories. The
grid-layer force-model fidelity ceiling is forces expressible planarly in the
synodic frame: two-primary point masses and solar third-body gravity. Level
one's solutions thereby mean optimal value functions under approximate models
— priors and training signals for level two, not final products.

### 4. Python binding layer

Bindings are exposed through e2m2e-integrators' pyo3 cdylib; large arrays
cross as flat Vec<f64> plus shape; ABI stamps increment per abi-version.txt —
existing processes, unchanged.

Per ADR 0032 decision 3, the entry shape was changed to the generic entry
`solve_hjb_py`. Parameters come in four groups: terminal conditions and
integration controls (terminal cost, time interval, CFL number, step ceiling),
grid definition (per-dim bounds and node counts), dynamics identifier (string:
planar_double_integrator, cr3bp_synodic, ephemeris_planar), and dynamics
parameter table (`HashMap<String, f64>`, values keyed). Rust constructs the
corresponding Hamiltonian impl from the identifier; expected dimension follows
from it; missing keys or invalid values raise explicit errors at the binding
layer.

`solve_planar_lowthrust_hjb_py`, previously used by geo-nrho, remains as a
compat wrapper with signatures pinned to the double integrator (drift_accel,
max_accel, fuel_weight listed individually), forwarding internally to
solve_hjb_py. The generic entry pins the ABI change to that one addition;
afterwards new dynamics only touch Rust-side construction branches. The cost:
the parameter table is weakly typed across the FFI boundary; misspelled keys
surface only at runtime, so the binding layer must validate existence and
values rather than silently ignoring them.

### 5. State semantics and coordinate conventions

Levels one and two hand off through value function grid files; both sides'
interpretation of state must agree verbatim. Conventions below are documented
rather than living only in code:

- Nondimensionalization: characteristic length is the primary-to-secondary
  distance, characteristic time the inverse of synodic angular velocity, so
  angular speed is identically 1 and is not a dynamics parameter. The mass
  ratio μ is the sole nondimensional dynamics parameter, fixed when
  constructing the Hamiltonian, matching `CR3BP_System` in
  e2m2e/algorithm/dynamics.
- State order: (x, y, vx, vy). Synodic frame origin at the barycenter, x-axis
  from primary toward secondary, rotating with the system. Matches STATE_ORDER
  in geo-nrho's algorithm/dp.py.
- Time semantics: in autonomous systems time is only integration direction and
  the value function may take a time-independent form (e.g. TTR); in
  non-autonomous systems time is genuine dependence of the value function. The
  binding entry expresses both cases with the same time-interval parameter.
- Lifting convention: the four-dimensional planar state is the z = vz = 0
  section of three-dimensional space states. Lifting to mission states goes
  through frame rotation (EphemCache's frame matrices, ADR 0016) from the
  synodic frame to the mission frame. Level two depends on this convention
  when using level-one solutions as training data.

### 6. Verification tiering

Verification has four tiers, inside-out:

1. Solver self-contained gates (existing): upwind scheme convergence orders,
   Burgers equation against Hopf-Lax exact solution, dual-integrator reachable
   set against analytic solutions — 22 cases total, see the e2m2e-levelset
   README. This tier depends on no external dynamics.
2. Dynamics correctness: a new Hamiltonian's zero-control vector field matches
   propagate_cr3bp_py's CR3BP dynamics pointwise on sampled states; closed-loop
   behavior checked against Lagrange point stability and orbit periods. #497's
   verification section is this tier.
3. Coarse-fine regression: once the ephemeris force-model Hamiltonian
   (EphemerisPlanar, #498) lands, CR3BP solutions serve as regression baseline,
   checking consistency of value function magnitudes and iso-surface structure.
   Coarse models are fine models' first assertion.
4. Closed-loop replay: value-function gradients generate control laws fed into
   e2m2e-propagation's compiled force models; verify the closed-loop trajectory
   reaches the terminal set. This tier corresponds to the verification
   constraints of geo-nrho's architecture doc, binding the HJB solution to an
   independent integrator so scheme-dissipation drift cannot silently
   accumulate.

None of the four tiers rely on MATLAB reference data, consistent with
e2m2e-levelset's self-contained verification principle during porting and with
ADR 0013's verification-by-definition strategy.

## 简体中文

先从一个具体场景说起。航天器在地月空间做小推力转移，飞控每一步都要回答：当前状态下往哪个方向推、推多大。直接法和间接法每问一次就解一遍最优控制问题，慢且不稳定。HJB 路线换个做法：离线把 Hamilton-Jacobi 方程在状态网格上解一次，得到值函数 V(x, t) 的数值表；在线查值函数梯度，按最优性条件直接算出控制。单次求解，快速调用。这个子系统就是干这件事的。

本文描述 e2m2e 侧 HJB 子系统的目标形态：谁提供什么，接缝在哪，深化到哪一步为止。需求侧的两级动态规划方案见 geo-nrho 项目的 docs/hjb-dependency-architecture.md，本文是与之对应的供给侧架构。决策快照见 ADR 0032、0033、0034。

### 1. 子系统定位与两级分工

小推力轨迹快速规划分两级。第一级在结构网格上反向求解 HJB 方程，产出低维值函数表。第二级用神经网络逼近值函数，处理完整七维状态（六维状态加时间），输出在线控制。第二级存在的理由是第一级绕不开维数灾难：结构网格的存储随维度指数增长，七维在网格上不可行。

e2m2e 承担第一级和验证工具，具体四样：

- 网格求解器：e2m2e-levelset，ToolboxLS 的 Rust 移植，提供迎风格式、Lax-Friedrichs 耗散、TVD Runge-Kutta 时间积分。
- 动力学：一组实现 Hamiltonian trait 的结构体，把具体问题的向量场接进求解器。
- Python 绑定：经 e2m2e-integrators 暴露求解入口，供下游编排离线求解。
- 验证工具：传播器与编译力模型，用于闭环回放与粗细模型对照。

e2m2e 不承担第二级。神经网络训练、在线策略、任务层终端约束属于下游项目（当前是 geo-nrho）。这条边界值得强调：第一级的输出是值函数网格文件，第二级怎么用它，e2m2e 只管文件格式与状态语义，不管训练与推理。

### 2. Hamiltonian 接缝与动力学家族

求解器与动力学之间的接缝是 e2m2e-levelset 的 Hamiltonian trait，对应 ToolboxLS 的 hamFunc 与 partialFunc 两个回调，含两个方法：

- `hamiltonian(t, grid, phi, p)`：在全网格上计算 H(x, t, p)，其中控制已按最优性条件解析消去。
- `partial_bound(t, grid, phi, p_min, p_max, dim)`：逐维给出耗散系数包络 max|∂H/∂p_dim|。

动力学参数作为实现结构体的字段在构造时固定，不用回调，不让 Python 进入 Rust 热循环。trait 维度无关且显式带时间参数，这两条是后续深化的结构性前提。

动力学不是一个实现，而是一个家族，按保真度排成谱系：

| 实现 | 状态维数 | 自治性 | 力模型 | 落地 |
|---|---|---|---|---|
| Cr3bpSynodic | 4：x, y, vx, vy | 自治 | 两主星点质量、离心、科氏 | #497 |
| 含质量扩展 | 5：加 m | 自治 | 同上，控制改为推力，加速度随质量衰减 | 含质量轴随 #498 落地（星历口径），CR3BP 口径后续 issue |
| Bcr4bp | 4 或 5 | 非自治，显含 t | 加太阳引力 | 后续 issue |
| 星历力模型 | 4 或 5 | 非自治 | e2m2e-forces 编译力模型，星历经 EphemCache 查表（ADR 0016） | #498（ADR 0034） |

谱系里新实现不替换旧实现。粗模型有两个长期价值：一是计算便宜，适合调网格与格式参数；二是作细模型的回归基准，星历力模型解出来先和 CR3BP 解对量级与结构，对不上就是实现错了，不必等到和任务数据对比才发现。

代码归属：家族成员放在新 crate e2m2e-hjb-dynamics（Apache-2.0），不放进 e2m2e-levelset。理由两条。一是许可：levelset 整体继承 ToolboxLS 的 ACM 非商业许可，原创代码放进去会被罩进同一条款。二是定位：levelset 保持忠实移植的纯粹性，每个模块都能对照 MATLAB 原版维护；动力学是 e2m2e 的原创层，两回事。依赖方向单向：hjb-dynamics 依赖 levelset 的 trait，ephemeris feature 下再依赖 e2m2e-forces 与 e2m2e-spice 的星历缓存类型，反向依赖不允许，与 ADR 0012 的依赖方向精神一致。

### 3. 维度上限

结构网格的存储是节点数乘以每节点字节数。按每维 40 节点、双精度、单数组估算：

| 状态维数 | 节点数 | 单数组内存 |
|---|---|---|
| 4 | 2.6×10⁶ | 约 0.02 GB |
| 5 | 1.0×10⁸ | 约 0.8 GB |
| 6 | 4.1×10⁹ | 约 33 GB |

求解器同时持有 φ、各维梯度、耗散系数等多个同形数组，实际占用是单数组的数倍。结论是硬性的：**网格层状态维度上限为五**，六维不可行是算术事实，不是工程优化能绕开的。

这条上限决定高保真力的去向。地球与月球的高阶引力（如 10×10 阶次）和光压本质上是三维力：带谐项在平面会合系里无法忠实投影，光压方向依赖太阳的三维位置。三维意味着六维状态，超出上限。因此三维高保真问题不在网格层求解，由第二级神经网络承接，其训练数据来自第一级低维解与任务轨迹。网格层力模型的保真度天花板，是会合系内能平面表达的力：两主星点质量、太阳第三体引力。第一级解的物理意义由此界定为近似模型下的最优值函数，它是第二级的先验与训练信号，不是最终产品。

### 4. Python 绑定层

绑定经 e2m2e-integrators 的 pyo3 cdylib 暴露，大数组以扁平 Vec<f64> 加形状跨界，ABI 戳随 abi-version.txt 递增，这些是既有流程，不变。

入口形状已按 ADR 0032 决策 3 改为通用入口 `solve_hjb_py`。参数分四组：终端条件与积分控制（终端代价、时间区间、CFL 系数、步长上限）、网格定义（各维上下界与节点数）、动力学标识（字符串：planar_double_integrator、cr3bp_synodic、ephemeris_planar）、动力学参数表（`HashMap<String, f64>` 按键取值）。Rust 侧按标识构造对应的 Hamiltonian 实现，期望维数由标识决定，参数表缺键或取值非法时在绑定层报出明确错误。

此前 geo-nrho 在用的 solve_planar_lowthrust_hjb_py 保留为兼容包装，签名按双积分器写死（drift_accel、max_accel、fuel_weight 逐个列出），内部转发到 solve_hjb_py。通用入口把 ABI 变更固定在入口新增的那一次，此后新增动力学只动 Rust 侧的构造分支。代价是参数表在 FFI 边界上是弱类型的，拼错的键名要到运行期才暴露，绑定层必须做存在性与取值校验，不能静默忽略。

### 5. 状态语义与坐标约定

第一级与第二级靠值函数网格文件交接，两边对状态的解释必须逐字一致。约定如下，落成文档而不是只活在代码里：

- 无量纲化：特征长度取两主星间距，特征时间取会合系角速度的倒数，因此角速度恒为 1，不作为动力学参数。质量比 μ 是唯一的无量纲动力学参数，构造 Hamiltonian 时固定，取值与 e2m2e/algorithm/dynamics 的 CR3BP_System 保持一致。
- 状态顺序：(x, y, vx, vy)。会合系原点在两主星质心，x 轴由主星指向次主星，随系统旋转。与 geo-nrho algorithm/dp.py 的 STATE_ORDER 一致。
- 时间语义：自治模型中时间只是积分方向，值函数可取时间无关形式（如 TTR）；非自治模型中时间是值函数的真实依赖。绑定入口对两种情形用同一时间区间参数表达。
- 提升约定：四维平面状态是三维空间状态的 z = vz = 0 截面。提升到星历任务状态时，经帧旋转（EphemCache 的帧矩阵，ADR 0016）从会合系变换到任务帧。第二级用第一级解作训练数据时依赖这个约定。

### 6. 验证分层

验证分四层，由内到外：

1. 求解器自足门控（已有）：迎风格式收敛阶、Burgers 方程对 Hopf-Lax 精确解、双积分器可达集对解析解，共 22 项用例，见 e2m2e-levelset README。这层不依赖任何外部动力学。
2. 动力学正确性：新 Hamiltonian 的零控向量场与 propagate_cr3bp_py 的 CR3BP 动力学在采样状态上逐点一致；闭环行为对照 Lagrange 点稳定性与轨道周期。#497 的验证节即此层。
3. 粗细模型回归：星历力模型 Hamiltonian（EphemerisPlanar，#498）就位后以 CR3BP 解为回归基准，检查值函数的量级与等值面结构一致性。粗模型是细模型的第一道断言。
4. 闭环回放：值函数梯度生成控制律，代入 e2m2e-propagation 的编译力模型积分，验证闭环轨迹到达终端集。这层对应 geo-nrho 架构文档的验证约束，把 HJB 解与独立积分器绑在一起，防止格式耗散造成的偏差静默累积。

四层都不依赖 MATLAB 基准数据，与 e2m2e-levelset 移植时的自足验证原则一致，也与 ADR 0013 的按定义验证策略一致。
