# Ephemeris Force-Model Hamiltonian: Solve-Chain Data Flow (#498) / 星历力模型 Hamiltonian：求解链数据流（#498）

[English](#english) | [简体中文](#简体中文)

## English

This page is the companion research document for ADR 0034. For the
subsystem's overall architecture (two-level division of labor, Hamiltonian
family spectrum, dimension ceiling, binding entries, state semantics,
verification tiering), `docs/architecture/hjb-subsystem.md` on master is
authoritative; this page does not repeat it. It covers only two areas the
other does not: internal facts about force models and ephemeris caching, and,
once ADR 0034 decision 1's planar full-ephemeris Hamiltonian lands, which path
the data of one solve flows along. Line numbers refer to master (`c3af80f`).

### Force-model side: CompiledForce status facts

`crates/e2m2e-forces/src/forces/compiled.rs` is the compiled force-model
enum: Python serializes each force into a tuple (`to_rust_spec`), Rust
rebuilds it via `force_from_tuple`, and the integration inner loop never
crosses back to Python. Facts relevant to #498:

- The interface is pointwise `acceleration(et, state6, observer)`
  (compiled.rs:195) plus the summing `compute_total_acceleration`
  (compiled.rs:364). Batching needs no new interface: ephemeris quantities
  depend on t only, not on nodes; look up the cache once per t per RK substep
  and reuse across all grid nodes — exactly the batch semantics #498 wants.
- #498's planar full-ephemeris scope uses only the `PointMass` and
  `ThirdBody` variants. `ThirdBody`'s solar position queries the ephemeris via
  spk_accel; the cache applies in the inner loop (next section).
- **Variable-mass contract**: `SRPVariableMass { area, cr, shadow_bodies }`
  stores no mass; paired `acceleration_with_mass(et, state, mass, observer)`
  reads current mass from the augmented state. Under ADR 0034 decision 2 it
  was out of #498's scope at writing time; it existed only in uncommitted
  workspace then and has since been merged independently per decision 5
  (#507, Python side `VariableMassSolarRadiationPressure`);
  geo-nrho's `lowthrust_rs` calls this variant.

### Ephemeris side: EphemCache status facts

`crates/e2m2e-spice/src/ephem_cache.rs` (ADR 0016): before integration, needed
body states and frame matrices are pre-sampled on a uniform time grid via
cspice and stored in memory as cubic splines; during solving we look up tables
and never touch cspice. Cubic splines were chosen for C² continuity so
adaptive integrators do not shrink step sizes. Key structural facts:

- **Process-level singleton**: `static CACHE: RwLock<Option<EphemCache>>`,
  installed by `enable(cache)`. `RwLock` is deliberate design: parallel
  segments read pure numeric splines concurrently across threads, and read
  locks never block each other (around ephem_cache.rs:468). ADR 0034 decision
  4 keeps this singleton; injection per construction (ADR 0033) happens at the
  configuration level: the force-model list, time range, and epoch mapping are
  constructor parameters of the Hamiltonian.
- **Two-tier miss semantics** (ADR 0020 decision 4): when not enabled, queries
  return `Ok(None)` and fall back to cspice; after enabling, misses (outside
  range / missing target) hard-fail without exception. The cache interval must
  cover the whole HJB solve window.
- **Everything needed for a time-varying synodic frame is in the cache**:
  `lookup_body_position` / `lookup_body_velocity` give Moon-relative-to-Earth
  position and velocity; rotation rate ω(t), ω̇(t), and libration rate derive
  from them — no second ephemeris query path needed.

### Hamiltonian structure for low-thrust min-fuel

Performance index `J = ψ(x(tf)) + ∫ fuel_weight·δ dt`, control sets
`δ ∈ [0,1]`, `û ∈ S²`. Dynamics
`f = [v, a_forces(r,t) + (T·δ/m)·û, -T·δ/(Isp·g₀)]`.
Minimizing over controls yields analytic optimal laws:

- Thrust direction `û* = -p_v/‖p_v‖` (anti-covariant-velocity);
- Bang-bang throttle: switching function
  `S = fuel_weight - (T/m)·‖p_v‖ - p_m·T/(Isp·g₀)` (includes the mass costate;
  mind T/m's m/s²→km/s² unit conversion); `δ* = 1` when `S < 0`, else 0;
- With control eliminated:
  `H* = p_r·v + p_v·a_forces + min(0, S)`
  (`PlanarDoubleIntegrator` is the massless-dimensional planar precedent:
  `control_gain = fuel_weight - max_accel·‖p_v‖`,
  `H = drift + min(control_gain, 0)`, see e2m2e-hjb-dynamics'
  double_integrator.rs).

The `partial_bound` envelope derivations entered the repo with the
implementation (written into implementation doc comments):
`∂H/∂p_r = v`; `∂H/∂p_v = a_forces + thrust terms`, with the mass dimension
taking T/m at the upper end of the mass grid interval; `∂H/∂p_m` is constant
`-T/(Isp·g₀)` when `δ*=1`.

### Target state: data flow of one planar full-ephemeris HJB solve

Once ADR 0034 decision 1's implementation lands, one solve runs in three
segments.

**Preparation (Python + cspice).** The caller provides solve window
`[et0, etf]`, grid definition, terminal cost, and engine parameters.
`EphemCache::build` pre-samples (MOON, EARTH) and (SUN, EARTH) on a uniform
grid covering the window and calls `enable`; the `CompiledForce` list (two
primaries' `PointMass` + solar `ThirdBody`) and epoch mapping (solver t ↔ SPICE
et) go to the Hamiltonian impl as constructor parameters. Terminal cost ψ is
laid onto the grid via the shape module.

**Solving (Rust hot loop, zero cspice).** Each TVD-RK substep: convert t to et
via epoch mapping, query the cache once for lunar position/velocity, derive
ω(t), ω̇(t), libration rate — **one lookup per t, reused across the grid**;
then node by node over the grid: synodic-frame coordinates compute two-primary
point-mass gravity and solar third-body gravity (in-plane components),
superpose frame-transform-induced Coriolis, centrifugal, ω̇ and libration
corrections, minimize over control per the switching function to get H*;
`partial_bound` supplies the dissipation envelope; LF terms compose `dphi_dt`.
The backward solve is implemented as forward evolution under time reversal
(semantic contract: geo-nrho `hjb-dp-route.md` §二).

**Artifacts (Python).** The value-function grid persists under ADR 0033
decision 3's contract: metadata explicitly records state-dim order
`(x, y, vx, vy, m)`, nondimensionalization conventions, `times` semantics (ET
seconds), and epoch mapping parameters; stored as catalog value-function
records; consumers (#499's gradient interface, time interpolation mandatory)
depend only on that format.

### Degradation cross-checks and verification hooks

ADR 0034 decision 6's three-tier acceptance maps onto three ready anchors:

- **(a) Degradation cross-check**: replace the cache with a circularized
  stationary synthetic ephemeris (constant lunar distance, constant ω); the
  ephemeris dynamics must degrade term-by-term to `Cr3bpSynodic`. Anchored on
  the impl already cross-checked against `cr3bp_eom` in #497.
- **(b) Force consistency**: for identical (t, state), forces inside the
  Hamiltonian (post frame transformation) match direct
  `compute_total_acceleration` calls pointwise;
- **(c) Coarse-fine regression**: small grids solve both ephemeris and CR3BP
  value functions; compare magnitude and iso-surface structure (verification
  ladder tier 3).

### Extension slot: attaching spherical-harmonics/SRP experiments

If ADR 0034 decision 2's experimental items start later, the hook sits at the
solve segment's per-node force evaluation: swap `PointMass` for
`GravityField` (10×10, body-fixed frame matrices also via cache); SRP uses
`acceleration_with_mass` reading the mass-axis coordinate. Evaluate on the z=0
section, discard out-of-plane components; no seam changes, no grid changes.
After experimental results are cross-checked against CR3BP/ephemeris
point-mass solutions, a new ADR revises decision 2.

### Status gap list

This section snapshots gaps at writing time (`c3af80f`); all three have since
progressed, annotated item by item.

1. **Ephemeris Hamiltonian**: not implemented at writing time (#498 body);
   landed with #515: `EphemerisPlanar` in e2m2e-hjb-dynamics;
   `solve_hjb_py` registers the `ephemeris_planar` dynamics.
2. **Variable-mass SRP contract**: only in an uncommitted local workspace at
   writing time; independently committed per ADR 0034 decision 5 (#507):
   `SRPVariableMass` + `acceleration_with_mass` + Python class
   `VariableMassSolarRadiationPressure`. Not on #498's critical path, but
   `lowthrust_rs` depends on it.
3. **Verification ladder tiers 3–4**: tier 3 coarse-fine regression established
   with #498 acceptance
   (`tests/numerical/integrators/bindings/test_hjb_solve.py`);
   tier 4 closed-loop replay remains manual on geo-nrho's side.

## 简体中文

本文是 ADR 0034 的配套调研文档。子系统总体架构（两级分工、
Hamiltonian 族谱、维度上限、绑定入口、状态语义、验证分层）以
master 的 `docs/architecture/hjb-subsystem.md` 为准，本文不重复；
这里只讲它没有覆盖的两块：力模型与星历缓存的内部事实，以及
ADR 0034 决策 1 的平面全星历 Hamiltonian 就位后，一次求解的数据
沿哪条路流动。行号以 master（`c3af80f`）为准。

### 力模型侧：CompiledForce 的现状事实

`crates/e2m2e-forces/src/forces/compiled.rs` 是编译型力模型枚举：
Python 侧把每个 force 序列化成元组（`to_rust_spec`），Rust 侧
`force_from_tuple` 重建，积分内循环全程不跨界回 Python。与 #498
相关的事实：

- 接口是逐点的 `acceleration(et, state6, observer)`
  （compiled.rs:195）与求和的 `compute_total_acceleration`
  （compiled.rs:364）。批量不需要新接口：星历量只依赖 t 不依赖
  节点，每个 RK 子步按 t 查一次缓存、全网格节点复用，即为
  #498 要的批量语义。
- #498 的平面全星历口径只用到 `PointMass` 与 `ThirdBody` 两个
  variant。`ThirdBody` 的太阳位置经 spk_accel 查星历，缓存在内层
  生效（见下节）。
- **变质量契约**：`SRPVariableMass { area, cr,
  shadow_bodies }` 不存质量，配 `acceleration_with_mass(et, state,
  mass, observer)` 从增广状态取当前质量。ADR 0034 决策 2 之下它
  不在 #498 范围；成文时仅存于未提交工作区，此后已按决策 5 独立
  入库（#507，Python 侧对应 `VariableMassSolarRadiationPressure`），
  geo-nrho 的 `lowthrust_rs` 调用该 variant。

### 星历侧：EphemCache 的现状事实

`crates/e2m2e-spice/src/ephem_cache.rs`（ADR 0016）：积分前把要用的
天体状态、帧矩阵在均匀时间网格上经 cspice 预采样，建三次样条存
内存；求解阶段查表，不碰 cspice。选三次样条是因为 C² 连续避免
自适应积分器缩步长。
关键结构事实：

- **进程级单例**：`static CACHE: RwLock<Option<EphemCache>>`，
  `enable(cache)` 安装。选 `RwLock` 是明示意图：并行段多线程
  并发读纯数值样条，读锁互不阻塞（ephem_cache.rs:468 附近）。
  ADR 0034 决策 4 沿用此单例，构造时注入（ADR 0033）落在配置
  层面：力模型列表、时间范围、历元映射是 Hamiltonian 的构造参数。
- **miss 语义分两档**（ADR 0020 决策 4）：未 enable 时查询返回
  `Ok(None)` 回退 cspice；enable 后 miss（区间外/缺目标）一律
  硬失败。缓存区间必须覆盖整个 HJB 求解窗。
- **定义时变会合系所需的量都在缓存里**：`lookup_body_position`
  与 `lookup_body_velocity` 给月球相对地球的位置与速度，旋转角
  速度 ω(t)、ω̇(t) 与脉动率由此导出，不需要第二条星历查询路径。

### 小推力 min-fuel 的 Hamiltonian 结构

性能指标 `J = ψ(x(tf)) + ∫ fuel_weight·δ dt`，控制集
`δ ∈ [0,1]`、`û ∈ S²`。动力学
`f = [v, a_forces(r,t) + (T·δ/m)·û, -T·δ/(Isp·g₀)]`。
对控制取 min 得解析最优律：

- 推力方向 `û* = -p_v/‖p_v‖`（协态负方向）；
- 油门 bang-bang：开关函数
  `S = fuel_weight - (T/m)·‖p_v‖ - p_m·T/(Isp·g₀)`（含质量协态项；
  注意 T/m 的 m/s²→km/s² 单位换算），`S < 0` 时 `δ* = 1`，否则 0；
- 消去控制后 `H* = p_r·v + p_v·a_forces + min(0, S)`
  （`PlanarDoubleIntegrator` 是无质量维的平面版先例：
  `control_gain = fuel_weight - max_accel·‖p_v‖`，`H = drift +
  min(control_gain, 0)`，见 e2m2e-hjb-dynamics 的
  double_integrator.rs）。

`partial_bound` 的包络推导随实现入库（写入实现文档注释）：
`∂H/∂p_r = v`；`∂H/∂p_v = a_forces + 推力项`，含质量维时 `T/m`
取质量网格区间上界；`∂H/∂p_m` 在 `δ*=1` 时为常值
`-T/(Isp·g₀)`。

### 目标态：一次平面全星历 HJB 求解的数据流

ADR 0034 决策 1 的实现就位后，一次求解分三段。

**准备段（Python + cspice）。** 调用方给出求解窗 `[et0, etf]`、
网格定义、终端代价与发动机参数。`EphemCache::build` 对
(MOON, EARTH)、(SUN, EARTH) 在覆盖求解窗的均匀时间网格上预采样
并 `enable`；`CompiledForce` 列表（两主星 `PointMass` + 太阳
`ThirdBody`）与历元映射（求解器 t ↔ SPICE et）作为构造参数交给
Hamiltonian 实现。终端代价 ψ 经 shape 模块铺在网格上。

**求解段（Rust 热循环，零 cspice）。** 积分器每个 TVD-RK 子步：
先按历元映射把 t 换成 et，查一次缓存得月球位置/速度，导出
ω(t)、ω̇(t)、脉动率，**每个 t 只查一次，全网格复用**；然后全
网格逐节点：会合系坐标算两主星点质量引力与太阳第三体引力（面内
分量），叠加系变换诱导的科氏、离心、ω̇ 与脉动修正项，按上面的
开关函数对控制取 min 得 H*；`partial_bound` 给耗散包络，LF 项
合成 `dphi_dt`。倒向求解按时间取反后的正向演化实现（语义对接
见 geo-nrho `hjb-dp-route.md` §二）。

**产物段（Python）。** 值函数网格按 ADR 0033 决策 3 的契约落盘：
元数据显式记录状态维顺序 `(x, y, vx, vy, m)`、无量纲化口径、
`times` 语义（ET 秒）、历元映射参数，作为 catalog 值函数记录
入库；消费端（#499 的梯度接口，时间插值必选）只依赖该格式。

### 退化对拍与验证的挂点

ADR 0034 决策 6 的三级验收对应三处现成锚点：

- **(a) 退化对拍**：把缓存替换为圆化定常合成星历（月距、ω 恒
  定），星历版动力学必须逐项退化为 `Cr3bpSynodic`。锚在 #497
  已与 `cr3bp_eom` 对拍过的实现上。
- **(b) 力一致性**：同一 (t, state) 下，Hamiltonian 内部经系变换
  后的合力与 `compute_total_acceleration` 直接调用的结果逐点
  一致；
- **(c) 粗细模型回归**：小网格分别求星历版与 CR3BP 版值函数，
  量级与等值面结构对照（验证阶梯第 3 级）。

### 扩展槽：球谐/光压实验项怎么挂

若后续启动 ADR 0034 决策 2 的实验项，挂点在求解段的逐节点力
求值处：`PointMass` 换成 `GravityField`（10×10，体固系帧矩阵
同样走缓存），SRP 走 `acceleration_with_mass` 取质量轴坐标。
z=0 截面取值、丢弃面内以外分量，不改接缝、不改网格。实验结论
与 CR3BP/星历点质量解对拍后，由新 ADR 修订决策 2。

### 现状缺口清单

本节为成文时（`c3af80f`）的缺口快照，三项此后均有进展，逐条标注。

1. **星历版 Hamiltonian**：成文时未实现（#498 本体），已随 #515
   落地：`e2m2e-hjb-dynamics` 的 `EphemerisPlanar`，`solve_hjb_py`
   注册 `ephemeris_planar` 动力学。
2. **变质量 SRP 契约**：成文时只在未提交本地工作区，已按
   ADR 0034 决策 5 独立提交（#507）：`SRPVariableMass` +
   `acceleration_with_mass` + Python 类
   `VariableMassSolarRadiationPressure`。不在 #498 关键路径，但
   `lowthrust_rs` 依赖它。
3. **验证阶梯第 3、4 级**：第 3 级粗细模型回归已随 #498 验收
   建立（`tests/numerical/integrators/bindings/test_hjb_solve.py`）；
   第 4 级闭环回放仍在 geo-nrho 侧手动执行。
