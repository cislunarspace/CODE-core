# ADR 0002: Rust integrator core with Python-controlled dynamics / Rust 积分器内核，由 Python 控制动力学

[English](#adr-0002-rust-integrator-core-with-python-controlled-dynamics) | [简体中文](#中文)

## English

**Status**: Adopted
**Date**: 2026-06-11
**Related Issue**: #61

### Context

Issue #60 plans to migrate propagation and force-model capabilities from GMAT
to e2m2e, using a Rust integrator core + Python force models + full
coordinate support. Issue #61 is the first vertical slice of that migration.

The existing `Dynamics` class in `e2m2e/algorithm/dynamics/dynamics.py`
already provides a stable template-method API: `propagate()` orchestrates
integration of the whole trajectory while subclasses override
`_get_eom_func()` and `_get_max_step()`. It currently delegates all integration
to `scipy.integrate.solve_ivp`.

The goal of #61 is to introduce a Rust-based single-step Runge-Kutta engine
(starting from Prince-Dormand 5(4), i.e. PD45), expose it to Python, and do so
without breaking the existing `Dynamics` API or forcing an immediate rewrite
of trajectory-level control logic.

### Decision

1. **Introduce a Rust workspace under `crates/` with maturin as the sole
   build backend.**
   - Root `Cargo.toml` defines a workspace.
   - The first crate is `crates/e2m2e-integrators/`, built with PyO3 +
     maturin.
   - `pyproject.toml`'s `build-system` switches from `hatchling` to `maturin`.

2. **The Rust crate does single-step integration only.**
   - Exposes `rk_step(method, t, y, h, tol, f)` where `f` is a Python callback.
   - Returns a `StepResult` containing `y_new`, `error`, `h_next`.
   - Does **not** do event detection, dense output, or full propagation
     control.

3. **Python-side `Dynamics` keeps trajectory-level control.**
   - In this first slice, `Dynamics.propagate()` continues using
     `scipy.integrate.solve_ivp`.
   - Rust's `rk_step` is used only by dedicated tests and serves as a base
     building block for later slices.

4. **Provide a public thin wrapper at `e2m2e.integrators`.**
   - The Rust extension module installs as `e2m2e._integrators`.
   - A thin Python module `e2m2e.integrators` re-exports `rk_step` and
     `RkMethod` so callers never import the underscore-prefixed internal
     module.

### Rationale

1. **Migration risk must be controlled incrementally.** Replacing
   `solve_ivp` inside `Dynamics.propagate()` in one step would require
   reimplementing adaptive step acceptance, dense output, event detection,
   and stiffness handling in Python. Doing these correctly far exceeds one
   slice and would introduce failure modes the current `solve_ivp` path does
   not have.

2. **Validate algorithms before wiring into default paths.** We want the Rust
   PD45 implementation proven correct before it becomes the default path for
   all propagation. Isolating it behind tests allows comparison against
   `scipy` and analytic solutions without touching production callers.

3. **Build-backend switches are hard to reverse.** Moving from `hatchling` to
   `maturin` changes how wheels are produced and how contributors set up the
   project. This is a weighty, surprising trade-off that should be documented,
   not hidden in `pyproject.toml`.

4. **A workspace reserves room for future crates without imposing them now.**
   Issue #60 mentions Rust force models and coordinate transformations. A
   workspace lets future crates live under `crates/` without later repository
   reorganization.

### Consequences

#### Added

- Repository-root `Cargo.toml` defining a Cargo workspace.
- `crates/e2m2e-integrators/` with `Cargo.toml`, PyO3 bindings, PD45
  coefficients, and inline Rust unit tests.
- `[tool.maturin]` configuration in `pyproject.toml`.
- Public thin wrapper `e2m2e/integrators.py` re-exporting `rk_step` and
  `RkMethod`.
- `tests/numerical/integrators/methods/test_rk_step.py`, Python-side
  correctness and consistency tests.

#### Changed

- `pyproject.toml`'s `build-system` from `hatchling` to `maturin`.
- CI workflow installs the Rust toolchain and runs `maturin develop` before
  Python tests.

#### Unchanged

- Implementation and behavior of `Dynamics.propagate()` in this slice.
- Public `Dynamics`, `CR3BP_Dynamics`, `EphemerisDynamics` APIs.

#### Follow-up work

- Later slices may rewrite `Dynamics.propagate()` to orchestrate Rust's
  `rk_step` from Python, adding dense output and event detection as needed.
- More RK methods (e.g. DOP853) can be added to
  `crates/e2m2e-integrators/src/` without changing the build system.

### Revision (2026-06-14, issue #67)

Decision 2 (Rust crate does single-step integration only) described the scope
of the **first slice**, not a permanent constraint. The integrator-family epic
(#67) extends the Rust crate to three method families:

- **Single-step RK** (`rk_step`): `Pd45`, `Pd78`, `Rk89`, as in the original
  slice.
- **Multistep predictor-corrector** (`multistep_step`):
  Adams-Bashforth-Moulton (`Abm`), fixed step, carrying a **history buffer**
  of derivative samples.
- **Second-order double integration** (`cowell_step`): Störmer-Cowell, fixed
  step, integrating `x'' = a(t, x)` directly from a position+acceleration mixed
  history buffer; outputs positions only.

Decision 1 (workspace + maturin) and decision 4 (public `e2m2e.integrators`
thin wrapper) still hold. The new multistep/second-order families respect the
same boundary: advance one step, return error estimate and suggested step; no
event detection, dense output, or full propagation control.

Note on decision 3: `CR3BP_Dynamics` and `EphemerisDynamics` (system classes)
still use `scipy.solve_ivp`; only `ForceModel` (force-decomposition class)
drives Rust steppers from Python. The original follow-up items are now done:
`ForceModel` orchestrates `rk_step` from Python (adaptive steps + simple event
detection), and the crate gained the multistep and second-order families.

### Revision (2026-07: crate split and spice build conventions)

The single crate splits into four: `e2m2e-integrators` (pyo3 bindings and
build entry, the only maturin packaging target), `e2m2e-propagation` (pure-math
integrators), `e2m2e-forces` (N-body STM, gravity fields), `e2m2e-spice`
(CSPICE FFI). Decision 3 partially lapses: propagation has moved into Rust
(`propagate_compiled`, `propagate_with_stm_py`) because the cspice kernel pool
is a process-level singleton — SPICE-related propagation and STM must compile
into the same extension as force models and cannot stay in the Python
orchestration layer.

Spice-feature build conventions: `cspice-sys` downloads CSPICE sources from
NAIF at build time via `downloadcspice`, no manual install needed (or point
`CSPICE_DIR` at a local installation). `maturin develop` defaults to no spice;
`maturin develop --features spice` includes STM propagation, shooting,
third-body and other Rust fast paths; without spice, Python silently degrades
to slow paths and corresponding tests skip via `importorskip`. **Release
wheels ship without spice for now**: including it would embed CSPICE in wheels
and tie builds to NAIF reachability; licensing and release stability need a
separate evaluation first. CI covers spice-gated code compilation via
`cargo clippy --workspace --features spice`.

> Revision note (2026-08, ADR 0020 decision 4): after spice became a default
> feature this section went stale: `maturin develop` defaults to spice (below),
> silent degradation to slow paths without spice became hard errors (issue
> #378), and corresponding tests' `importorskip` semantics were adjusted.

### Revision (2026-08: spice promoted to default feature)

Spice is now a default feature: crates `default = ["spice"]` plus pyproject
`features=["spice"]` as double insurance; `maturin develop` defaults to spice,
producing no no-spice subset; release wheels carry spice (ADR 0009 delivered).

**Dynamics integration core paths unified on Rust.** Integration
(`rk_step`/`multistep_step`/`cowell_step`/`solve_ivp`), propagation
(`propagate_compiled`/`propagate_with_stm_py`), force models
(PointMass/ThirdBody/GravityField/Drag/SRP/Relativistic), multiple shooting,
and transfer grid search have all moved into Rust. With spice enabled by
default, all core computation paths avoid Python scipy in normal operation.

**Scipy paths retained in the following scenarios** (deliberate design
choices, not oversights):

- **Event detection** (`CR3BP_Dynamics._propagate_with_stm(events=...)`): Rust
  `solve_ivp_events_py` exists but its event semantics don't fully align with
  scipy; event paths choose explicitly `backend="scipy"/"rust"` (ADR 0020
  decision 4) — omitting it errors out, `auto` disallowed; `"scipy"` uses
  scipy `solve_ivp`, `"rust"` uses Rust `solve_ivp_events` (accepting semantic
  differences). BCR4BP likewise (#333's NotImplementedError divergence
  resolved).
- **Defensive fallbacks** (`Dynamics` base and `EphemerisDynamics`
  `_propagate_with_stm`/`_propagate_state_only`): when the Rust extension is
  unavailable there is no fallback to scipy anymore;
  `require_rust_extension` raises `RustExtensionUnavailableError` instead
  (issue #378, ADR 0020 decision 4: missing resources raise).
- **NLP optimization** (`transfer/nlp_copt.py`): when COPT is unavailable the
  default raises (`fallback_to_scipy` defaults `False`); passing `True`
  explicitly falls back to SciPy SLSQP. ADR 0017 keeps NLP at the Python layer.
- **Normal form propagation** (`normal_form/multiple_shooting.py`,
  `dynamical_substitution.py`, `propagation.py`, `quasi_floquet.py`): moved to
  Rust `solve_ivp_py` (#336). QF↔CM high-order Lie flows
  (`coord_trans/qf_cm.py`) sunk as `qf_to_cm_py` / `cm_to_qf_py` (#465,
  12-real-dim split-complex integration); `backend="python"` remains for
  explicit comparison only. `scipy.linalg.expm` (matrix exponential) and
  `scipy.optimize.fsolve` keep awaiting Rust replacements.
- **Libration point solving and initial-value generation**
  (`scipy.optimize.fsolve`/`brentq`): for L1/L2 position solving, Halo orbit
  initial guesses etc. Single calls, low migration value.

## 中文

**状态**：已采纳
**日期**：2026-06-11
**关联 Issue**：#61

### 背景

Issue #60 计划把传播与力模型能力从 GMAT 迁到 e2m2e，采用 Rust 积分器内核 + Python 力模型 + 完整坐标支持。Issue #61 是这次迁移的第一个纵向切片。

`e2m2e/algorithm/dynamics/dynamics.py` 中现有的 `Dynamics` 类已经提供稳定的模板方法 API：`propagate()` 编排整条轨迹的积分，子类覆写 `_get_eom_func()` 与 `_get_max_step()`。它目前把全部积分委托给 `scipy.integrate.solve_ivp`。

#61 的目标是引入一个基于 Rust 的单步 Runge-Kutta 引擎（从 Prince-Dormand 5(4)，即 PD45 起步），并暴露给 Python，同时不破坏现有 `Dynamics` API、也不强迫立刻重写轨迹级控制逻辑。

### 决策

1. **在 `crates/` 下引入 Rust 工作空间，构建后端只用 maturin。**
   - 根 `Cargo.toml` 定义一个工作空间。
   - 第一个 crate 是 `crates/e2m2e-integrators/`，用 PyO3 + maturin 构建。
   - `pyproject.toml` 的 `build-system` 从 `hatchling` 切换到 `maturin`。

2. **Rust crate 只负责单步积分。**
   - 暴露 `rk_step(method, t, y, h, tol, f)`，其中 `f` 是 Python 回调。
   - 返回一个 `StepResult`，含 `y_new`、`error`、`h_next`。
   - **不**做事件检测、稠密输出或完整传播控制。

3. **Python 侧的 `Dynamics` 继续负责轨迹级控制。**
   - 在这第一个切片里，`Dynamics.propagate()` 继续用 `scipy.integrate.solve_ivp`。
   - Rust 的 `rk_step` 只被专用测试使用，并作为后续切片的底层构件。

4. **在 `e2m2e.integrators` 提供公开薄封装。**
   - Rust 扩展模块安装为 `e2m2e._integrators`。
   - 一个薄薄的 Python 模块 `e2m2e.integrators` 重新导出 `rk_step` 与 `RkMethod`，调用者不必导入带下划线前缀的内部模块。

### 理由

1. **迁移风险需逐步控制。** 一步到位地在 `Dynamics.propagate()` 里替换 `solve_ivp`，就要用 Python 重新实现自适应步长接受、稠密输出、事件检测与刚性处理。正确做完这些远超一个切片，还会引入当前 `solve_ivp` 路径没有的失败模式。

2. **在接入默认路径之前先验证算法。** 我们希望先确认 Rust PD45 实现正确，再让它成为所有传播的默认路径。把它隔离在测试之后，可以与 `scipy` 及解析解对比，而不影响生产调用方。

3. **构建后端的切换难以逆转。** 从 `hatchling` 换到 `maturin` 改变了 wheel 的产出方式和贡献者搭建项目的方式。这是一个有分量、会让人意外的权衡，应当记在文档里，而不是只藏在 `pyproject.toml` 中。

4. **工作空间为未来的 crate 预留位置，但不现在强加。** Issue #60 提到 Rust 力模型与坐标变换。工作空间让未来的 crate 可以放在 `crates/` 下，不必日后重组仓库。

### 结果

### 新增

- 仓库根的 `Cargo.toml`，定义 Cargo 工作空间。
- `crates/e2m2e-integrators/`，含 `Cargo.toml`、PyO3 绑定、PD45 系数与内联 Rust 单元测试。
- `pyproject.toml` 中的 `[tool.maturin]` 配置。
- `e2m2e/integrators.py` 公开薄封装，重新导出 `rk_step` 与 `RkMethod`。
- `tests/numerical/integrators/methods/test_rk_step.py`，Python 层正确性与一致性测试。

### 变更

- `pyproject.toml` 的 `build-system` 从 `hatchling` 改为 `maturin`。
- CI 工作流安装 Rust 工具链，并在 Python 测试前运行 `maturin develop`。

### 不变

- 本切片中 `Dynamics.propagate()` 的实现与行为。
- 公开的 `Dynamics`、`CR3BP_Dynamics`、`EphemerisDynamics` API。

### 后续工作

- 后续切片可重写 `Dynamics.propagate()`，从 Python 编排 Rust 的 `rk_step` 调用，按需加入稠密输出与事件检测。
- 可向 `crates/e2m2e-integrators/src/` 增加更多 RK 方法（如 DOP853），无需改构建系统。

### 修订（2026-06-14，issue #67）

决策 2（Rust crate 只负责单步积分）描述的是**第一个切片**的范围，不是永久约束。积分器族 epic（#67）把 Rust crate 扩展为三个方法族：

- **单步 RK**（`rk_step`）：`Pd45`、`Pd78`、`Rk89`，与原切片一致。
- **多步预测-校正**（`multistep_step`）：Adams-Bashforth-Moulton（`Abm`），定步长，携带一个导数采样的**历史缓冲**。
- **二阶双积分**（`cowell_step`）：Störmer-Cowell，定步长，直接积 `x'' = a(t, x)`，从一个位置+加速度混合的历史缓冲出发；只输出位置。

决策 1（工作空间 + maturin）、4（公开的 `e2m2e.integrators` 薄封装）仍然成立。新增的多步/二阶族遵守同样的边界：推进一步、返回误差估计与步长建议；不做事件检测、稠密输出或完整传播控制。

关于决策 3 的说明：`CR3BP_Dynamics` 与 `EphemerisDynamics`（系统类）仍用 `scipy.solve_ivp`；只有 `ForceModel`（力分解类）从 Python 驱动 Rust 步进器。原先的后续工作各项现已落实：`ForceModel` 从 Python 编排 `rk_step`（自适应步长 + 简单事件检测），crate 也增加了多步与二阶族。

### 修订（2026-07，crate 拆分与 spice 构建约定）

单 crate 拆为四个：`e2m2e-integrators`（pyo3 绑定与编译入口，maturin 唯一打包目标）、`e2m2e-propagation`（纯数学积分器）、`e2m2e-forces`（N 体 STM、重力场）、`e2m2e-spice`（CSPICE FFI）。决策 3 随之部分失效：传播已进入 Rust（`propagate_compiled`、`propagate_with_stm_py`），原因是 cspice 内核池是进程级单例，SPICE 相关的传播与 STM 必须和力模型编进同一个扩展，无法留在 Python 编排层。

spice feature 的构建约定：`cspice-sys` 经 `downloadcspice` 在构建时从 NAIF 官网下载 CSPICE 源码，无需手工安装（也可用 `CSPICE_DIR` 指向本机安装）。`maturin develop` 默认不带 spice，`maturin develop --features spice` 才包含 STM 传播、打靶、第三体等 Rust 快速路径；无 spice 时 Python 侧全部静默降级到慢路径，对应测试以 `importorskip` 跳过。**release wheel 暂不带 spice**：带上意味着 wheel 内嵌 CSPICE 且构建依赖 NAIF 官网可达性，许可与发布稳定性需单独评估后再开。CI 以 `cargo clippy --workspace --features spice` 兜底 spice-gated 代码的编译。

> 修订（2026-08，ADR 0020 决策 4）：spice 升为默认 feature 后本节过时：`maturin develop` 默认带 spice（见下节），无 spice 时的静默降级到慢路径改为报错（issue #378），对应测试的 `importorskip` 语义同步调整。

### 修订（2026-08，spice 升为默认 feature）

spice 现为默认 feature：crates `default = ["spice"]` + pyproject `features=["spice"]` 双保险，`maturin develop` 默认带 spice，不再产无 spice 子集；release wheel 已带 spice（ADR 0009 落实）。

**动力学积分核心路径统一走 Rust。** 积分（`rk_step`/`multistep_step`/`cowell_step`/`solve_ivp`）、传播（`propagate_compiled`/`propagate_with_stm_py`）、力模型（PointMass/ThirdBody/GravityField/Drag/SRP/Relativistic）、多重打靶、转移网格搜索均已迁入 Rust。spice feature 默认启用后，正常操作中所有核心计算路径不经过 Python scipy。

**以下场景保留 scipy 路径**（有意的设计选择，非临时遗漏）：

- **事件检测**（`CR3BP_Dynamics._propagate_with_stm(events=...)`）：Rust `solve_ivp_events_py` 已实现但事件检测语义与 scipy 不完全对齐，事件路径按显式 `backend="scipy"/"rust"` 二选一（ADR 0020 决策 4），不传报错、不允许 `auto`；`"scipy"` 走 scipy `solve_ivp`，`"rust"` 走 Rust `solve_ivp_events`（接受语义差异）。BCR4BP 同（#333 的 NotImplementedError 分歧已消除）。
- **防御性回退**（`Dynamics` 基类与 `EphemerisDynamics` 的 `_propagate_with_stm`/`_propagate_state_only`）：Rust 扩展不可用时不再回退 scipy，改由 `require_rust_extension` 抛 `RustExtensionUnavailableError`（issue #378，ADR 0020 决策 4：资源缺失即报错）。
- **NLP 优化**（`transfer/nlp_copt.py`）：COPT 不可用时默认报错（`fallback_to_scipy` 默认 `False`），显式传 `True` 才回退 SciPy SLSQP。ADR 0017 明确 NLP 留在 Python 层。
- **Normal form 传播**（`normal_form/multiple_shooting.py`、`dynamical_substitution.py`、`propagation.py`、`quasi_floquet.py`）：已迁至 Rust `solve_ivp_py`（#336）。QF↔CM 高阶 Lie 流（`coord_trans/qf_cm.py`）已下沉 `qf_to_cm_py` / `cm_to_qf_py`（#465，12 实维分裂复积分）；`backend="python"` 仅作显式对照。`scipy.linalg.expm`（矩阵指数）和 `scipy.optimize.fsolve` 暂无 Rust 替代，保留。
- **平动点解算与初值生成**（`scipy.optimize.fsolve`/`brentq`）：用于 L1/L2 位置解算、Halo 轨道初始猜测等。单次调用，迁移收益低。
