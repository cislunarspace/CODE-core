# ADR 0002：Rust 积分器内核，由 Python 控制动力学

**状态**：已采纳
**日期**：2026-06-11
**关联 Issue**：#61

## 背景

Issue #60 计划把传播与力模型能力从 GMAT 迁到 e2m2e，采用 Rust 积分器内核 + Python 力模型 + 完整坐标支持。Issue #61 是这次迁移的第一个纵向切片。

`e2m2e/core/dynamics.py` 中现有的 `Dynamics` 类已经提供稳定的模板方法 API：`propagate()` 编排整条轨迹的积分，子类覆写 `_get_eom_func()` 与 `_get_max_step()`。它目前把全部积分委托给 `scipy.integrate.solve_ivp`。

#61 的目标是引入一个基于 Rust 的单步 Runge-Kutta 引擎（从 Prince-Dormand 5(4)，即 "PD45" 起步），并暴露给 Python，同时不破坏现有 `Dynamics` API、也不强迫立刻重写轨迹级控制逻辑。

## 决策

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

## 理由

1. **迁移风险需逐步控制。** 一步到位地在 `Dynamics.propagate()` 里替换 `solve_ivp`，就要用 Python 重新实现自适应步长接受、稠密输出、事件检测与刚性处理。正确做完这些远超一个切片，还会引入当前 `solve_ivp` 路径没有的失败模式。

2. **在接入默认路径之前先验证算法。** 我们希望先确认 Rust PD45 实现正确，再让它成为所有传播的默认路径。把它隔离在测试之后，可以与 `scipy` 及解析解对比，而不影响生产调用方。

3. **构建后端的切换难以逆转。** 从 `hatchling` 换到 `maturin` 改变了 wheel 的产出方式和贡献者搭建项目的方式。这是一个有分量、会让人意外的权衡，应当记在文档里，而不是只藏在 `pyproject.toml` 中。

4. **工作空间为未来的 crate 预留位置，但不现在强加。** Issue #60 提到 Rust 力模型与坐标变换。工作空间让未来的 crate 可以放在 `crates/` 下，不必日后重组仓库。

## 结果

### 新增

- 仓库根的 `Cargo.toml`，定义 Cargo 工作空间。
- `crates/e2m2e-integrators/`，含 `Cargo.toml`、PyO3 绑定、PD45 系数与内联 Rust 单元测试。
- `pyproject.toml` 中的 `[tool.maturin]` 配置。
- `e2m2e/integrators.py` 公开薄封装，重新导出 `rk_step` 与 `RkMethod`。
- `tests/integrators/test_rk_step.py`，Python 层正确性与一致性测试。

### 变更

- `pyproject.toml` 的 `build-system` 从 `hatchling` 改为 `maturin`。
- CI 工作流安装 Rust 工具链，并在 Python 测试前运行 `maturin develop`。

### 不变

- 本切片中 `Dynamics.propagate()` 的实现与行为。
- 公开的 `Dynamics`、`CR3BP_Dynamics`、`EphemerisDynamics` API。

### 后续工作

- 后续切片可重写 `Dynamics.propagate()`，从 Python 编排 Rust 的 `rk_step` 调用，按需加入稠密输出与事件检测。
- 可向 `crates/e2m2e-integrators/src/` 增加更多 RK 方法（如 DOP853），无需改构建系统。

## 修订（2026-06-14，issue #67）

决策 2（"Rust crate 只负责单步积分"）描述的是**第一个切片**的范围，不是永久约束。积分器族 epic（#67）把 Rust crate 扩展为三个方法族：

- **单步 RK**（`rk_step`）：`Pd45`、`Pd78`、`Rk89`——与原切片一致。
- **多步预测-校正**（`multistep_step`）：Adams-Bashforth-Moulton（`Abm`），定步长，携带一个导数采样的**历史缓冲**。
- **二阶双积分**（`cowell_step`）：Störmer-Cowell，定步长，直接积 `x'' = a(t, x)`，从一个位置+加速度混合的历史缓冲出发；只输出位置。

决策 1（工作空间 + maturin）、4（公开的 `e2m2e.integrators` 薄封装）仍然成立。新增的多步/二阶族遵守同样的边界：推进一步、返回误差估计与步长建议；不做事件检测、稠密输出或完整传播控制。

关于决策 3 的说明：`CR3BP_Dynamics` 与 `EphemerisDynamics`（系统类）仍用 `scipy.solve_ivp`；只有 `ForceModel`（力分解类）从 Python 驱动 Rust 步进器。原先的"后续工作"各项现已落实：`ForceModel` 从 Python 编排 `rk_step`（自适应步长 + 简单事件检测），crate 也增加了多步与二阶族。

## 修订（2026-07，crate 拆分与 spice 构建约定）

单 crate 拆为四个：`e2m2e-integrators`（pyo3 绑定与编译入口，maturin 唯一打包目标）、`e2m2e-propagation`（纯数学积分器）、`e2m2e-forces`（N 体 STM、重力场）、`e2m2e-spice`（CSPICE FFI）。决策 3 随之部分失效：传播已进入 Rust（`propagate_compiled`、`propagate_with_stm_py`），原因是 cspice 内核池是进程级单例，SPICE 相关的传播与 STM 必须和力模型编进同一个扩展，无法留在 Python 编排层。

spice feature 的构建约定：`cspice-sys` 经 `downloadcspice` 在构建时从 NAIF 官网下载 CSPICE 源码，无需手工安装（也可用 `CSPICE_DIR` 指向本机安装）。`maturin develop` 默认不带 spice，`maturin develop --features spice` 才包含 STM 传播、打靶、第三体等 Rust 快速路径；无 spice 时 Python 侧全部静默降级到慢路径，对应测试以 `importorskip` 跳过。**release wheel 暂不带 spice**：带上意味着 wheel 内嵌 CSPICE 且构建依赖 NAIF 官网可达性，许可与发布稳定性需单独评估后再开。CI 以 `cargo clippy --workspace --features spice` 兜底 spice-gated 代码的编译。

## 修订（2026-08，spice 升为默认 feature）

spice 现为默认 feature：crates `default = ["spice"]` + pyproject `features=["spice"]` 双保险，`maturin develop` 默认带 spice，不再产无 spice 子集；release wheel 已带 spice（ADR 0009 落实）。

**动力学积分核心路径统一走 Rust。** 积分（`rk_step`/`multistep_step`/`cowell_step`/`solve_ivp`）、传播（`propagate_compiled`/`propagate_with_stm_py`）、力模型（PointMass/ThirdBody/GravityField/Drag/SRP/Relativistic）、多重打靶、转移网格搜索均已迁入 Rust。spice feature 默认启用后，正常操作中所有核心计算路径不经过 Python scipy。

**以下场景保留 scipy 路径**（有意的设计选择，非临时遗漏）：

- **事件检测**（`CR3BP_Dynamics._propagate_with_stm(events=...)`）：Rust `solve_ivp_events_py` 已实现但事件检测语义与 scipy 不完全对齐，`CR3BP_Dynamics` 在传入 events 时回退 scipy。（`BCR4BP_Dynamics` 传入 events 时抛 `NotImplementedError`——行为不一致，见 #333。）
- **防御性回退**（`Dynamics` 基类 `_propagate_with_stm`/`_propagate_state_only`、`EphemerisDynamics._propagate`）：Rust 扩展不可用时（`_HAS_RUST_* = False`）回退 `scipy.integrate.solve_ivp`。spice feature 默认启用后此路径在正常操作中不可达，保留作为构建失败时的降级路径。
- **NLP 优化**（`transfer/nlp_scipy.py`）：COPT 不可用时回退 SciPy SLSQP。ADR 0017 明确 NLP 留在 Python 层。
- **Normal form 传播**（`normal_form/multiple_shooting.py`、`dynamical_substitution.py`、`propagation.py`、`quasi_floquet.py`）：已迁至 Rust `solve_ivp_py`（#336）。保留 scipy 的仅剩 `coord_trans/qf_cm.py` 的复值 Lie 级数流——Rust `solve_ivp_py` 仅支持实值。`scipy.linalg.expm`（矩阵指数）和 `scipy.optimize.fsolve` 暂无 Rust 替代，保留。
- **平动点解算与初值生成**（`scipy.optimize.fsolve`/`brentq`）：用于 L1/L2 位置解算、Halo 轨道初始猜测等。单次调用，迁移收益低。

以上保留路径的上一次修订中"scipy 回退路径已移除""所有事情统一走 Rust"的措辞过于绝对，特此修正。
