# ADR 0018: Jacobian interface extended with ∂a/∂v; STM covers velocity dependence / Jacobian 接口扩 ∂a/∂v，状态转移矩阵纳入速度依赖

[English](#adr-0018-jacobian-interface-extended-with-adv-stm-covers-velocity-dependence) | [简体中文](#中文)

## English

**Status**: Adopted
**Date**: 2026-08-07
**Related**: ADR 0002 (Rust integrator core), ADR 0003 (ITRF93 defaults),
ADR 0017 (grid search Rayon), issue #317
**Related code**: `crates/e2m2e-forces/src/forces/{compiled,nbody_stm,augmented_state,compiled_stm}.rs`,
`crates/e2m2e-integrators/src/multiple_shooting.rs`

### Context

The state transition matrix (STM) satisfies the variational equation
``Φ̇ = A·Φ`` where

```text
A = | 0₃ₓ₃   I₃ₓ₃ |
    | ∂a/∂r  ∂a/∂v |
```

The lower-left block ``∂a/∂r`` (acceleration vs position) sums per-force
Jacobians; the lower-right block ``∂a/∂v`` (acceleration vs velocity) is zero
for pure N-body gravity: all body gravities depend only on position. This
repo's STM propagation (CR3BP / EphemerisDynamics / N-body STM / compiled STM /
multiple shooting) long rested on that assumption.

Atmospheric drag breaks it: ``a_drag = −½·ρ(|r|)·BC·|v|·v`` depends on both
position (via density ρ) and velocity (relative velocity itself). Once drag
enters STM propagation, ``A``'s ``∂a/∂v`` block is nonzero, and the old
implicit ``∂a/∂v = 0`` assumption makes the STM silently wrong: no error, just
wrong sensitivities.

The drag Rust port (issue #315 series) brings ``DragModel`` into compiled force
models — the first velocity-dependent force on the STM path. This forces the
force-Jacobian interface to extend from returning ``∂a/∂r`` to returning
``(∂a/∂r, ∂a/∂v)``.

### Decision

Extend the Rust force-model Jacobian interface's return from the pair
``(acc, ∂a/∂r)`` to a **triple** ``(acc, ∂a/∂r, ∂a/∂v)``, threading ``∂a/∂v``
through the whole chain into the variational equation.

1. **`acceleration_and_jacobian` returns the triple.**
   - Type alias ``AccelJacobiResult = Result<([f64;3], [[f64;3];3], [[f64;3];3]), String>``
     (compiled.rs).
   - Each `CompiledForce` variant supplies `∂a/∂v` under its own semantics:
     - `PointMass` / `ThirdBody` / `IndirectTerm` / `GravityField` / `SRP`:
       analytic ``[[0.0;3];3]`` (position-type forces, velocity-independent).
     - `Drag`: central-difference FD homologous with its `∂a/∂r`, perturbing
       all 6 J2000 components (12 accel evaluations); `∂a/∂v` is true-valued.
   - `compute_total_acceleration_and_jacobian` accumulates `total_dadv`
     across forces (isomorphic to `total_jac`).

2. **`stm_derivative` accepts `dadv`.**
   - Signature ``stm_derivative(stm, jac_da_dr, dadv) -> [f64;36]``;
     bottom 3 rows of ``A`` use ``∂a/∂r·Φ[:3] + ∂a/∂v·Φ[3:]``.
   - `nbody_stm`, `augmented_state`, `compiled_stm`, and `multiple_shooting`
     all update their call signatures accordingly.

3. **N-body path has zero `dadv`.**
   - `compute_nbody_acceleration_and_jacobian` returns
     ``dadv = [[0.0;3];3]`` (N-body gravity is velocity-independent). The
     path's behavior is unchanged; the triple merely writes out the implicit
     zero.

4. **Python-side `ForceModel` STM fallback synced (issue #317 item 2.1).**
   - `_compute_total_jacobian` returns ``(∂a/∂r, ∂a/∂v)``;
     `_eom_func_with_stm` assembles ``A[3:,3:] = ∂a/∂v``.
   - Analytic-Jacobian forces (`compute_jacobian` not None): `∂a/∂v = 0`;
     forces without analytic Jacobians take finite differences perturbing both
     position and velocity for true values.
   - This eliminates the future hazard of velocity-dependent forces silently
     erring without SPICE. Issue #378 later removed the Python fallback
     wholesale (no more Python RHS fallbacks, ADR 0020); this clause's code was
     removed with it.

### Impact surface

Extending the interface to triples touches all STM-related Rust modules; ABI
unchanged (internal signatures, not pyfunction boundaries):

| Module | Change |
|---|---|
| `compiled.rs` | `AccelJacobiResult` alias; triple returns from `acceleration_and_jacobian` / `compute_total_acceleration_and_jacobian` |
| `nbody_stm.rs` | `compute_nbody_acceleration_and_jacobian` returns constant-zero `dadv`; `stm_derivative` gains `dadv` parameter |
| `augmented_state.rs` | 42-dim augmented RHS forwards `dadv` to `stm_derivative` |
| `compiled_stm.rs` | `augmented_eom` destructures triple, forwards `dadv` |
| `multiple_shooting.rs` | STM shooting segments forward `dadv` |
| `e2m2e/algorithm/forces/force_model.py` | Python fallback sync (item 2.1; removed later by issue #378) |

### Precedents and positioning

- **More fundamental than ADR 0017.** 0017 is a parallel-execution strategy
  for grid search (Rayon), touching one algorithm path; this ADR is the
  force-Jacobian contract underpinning every STM propagation (shooting,
  correction, STM sampling, grid-search integration). 0017's cited
  precedents are themselves consumers of this contract.
- **Relation to ADR 0002.** Its revision 2 established SPICE-related
  propagation compiling into the Rust extension (cspice pool singleton). This
  ADR doesn't move that boundary; it refines the in-Rust data contract from two
  matrices to three — internal to ADR 0002's kernel, no seam crossing.
- **Relation to ADR 0003.** drag's nonzero `∂a/∂v` stems from its physics
  (relative velocity), frame-independent; the coordinate side's ITRF93 choice
  is recorded separately in ADR 0019.

### Why not other shapes

- **Not struct fields.** The ``(acc, ∂a/∂r, ∂a/∂v)`` triple matches GMAT's
  `CompleteDerivativeCalculations` division: forces supply Ã's lower-left
  block only, here refined from ``3×3`` into ``(∂a/∂r, ∂a/∂v)``. A struct
  (``AccelJacobi { acc, dadr, dadv }``) reads slightly better but touches all
  destructure points; the triple aligns with existing `AccelDrag` /
  `AccelJacobiResult` style, minimal churn.
- **Not optional `∂a/∂v`.** An optional ``Option<[[f64;3];3]>`` would let
  velocity-dependent forces omit filling it — recreating exactly the silent-
  error mode this ADR kills. Mandatory triples fail compilation when missing.

### Consequences

#### Added

- Force-Jacobian interface explicitly includes `∂a/∂v`; the STM variational
  equation handles velocity-dependent forces correctly.
- Drag entering compiled-STM paths no longer corrupts sensitivities silently.
- Python `ForceModel` fallback sync (item 2.1), removing a future hazard
  sans-SPICE (path removed by #378).

#### Unchanged

- N-body / CR3BP / EphemerisDynamics STM behavior (`dadv` identically zero —
  equivalent to old implicit zero).
- pyfunction ABI (`propagate_compiled_stm_py` etc.; argument/return shapes
  unchanged).
- `PhysicalModel.compute_jacobian` still returns only `∂a/∂r` (3×3); Python
  analytic-Jacobian contract unchanged; `∂a/∂v` supplied at `ForceModel` level
  as zeros for analytic forces / FD otherwise. (That Python contract was later
  removed by issue #378.)

#### Trade-offs

- **FD evaluation count doubles (drag).** `drag_accel_and_jacobian` grows from
  perturbing 3 position components (6 evals) to all 6 (12 evals). Drag FD was
  already the heaviest single force on the STM path, but STM cost is dominated
  by RK stepping rather than per-call Jacobians; drag magnitudes are small
  (fast altitude decay) — imperceptible in practice.
- **Python FD path cost.** Forces without analytic Jacobians expand Python-FD
  from 6 evals (position) to 12 (position+velocity). That was a degraded path,
  drag unreachable there; sole consumer then was spherical-harmonics
  `GravityField` (position-type; velocity FD gave zeros), so the extra 6 evals
  were acceptable. (Path removed by issue #378.)

## 中文

**状态**：已采纳
**日期**：2026-08-07
**关联**：ADR 0002（Rust 积分器内核）、ADR 0003（ITRF93 默认值）、ADR 0017（网格搜索 Rayon）、issue #317
**关联代码**：`crates/e2m2e-forces/src/forces/{compiled,nbody_stm,augmented_state,compiled_stm}.rs`、`crates/e2m2e-integrators/src/multiple_shooting.rs`

### 背景

状态转移矩阵（STM）满足变分方程 ``Φ̇ = A·Φ``，其中

```text
A = | 0₃ₓ₃   I₃ₓ₃ |
    | ∂a/∂r  ∂a/∂v |
```

左下块 ``∂a/∂r``（加速度对位置）由各力雅可比叠加；右下块 ``∂a/∂v``（加速度对速度）在纯 N 体引力模型里恒为零：所有天体引力只依赖位置。本仓 STM 传播（CR3BP / EphemerisDynamics / N 体 STM / compiled STM / 多重打靶）长期基于这一假设。

大气阻力打破它：``a_drag = −½·ρ(|r|)·BC·|v|·v``，既依赖位置（通过大气密度 ρ）又依赖速度（相对速度本身）。drag 一旦进入 STM 传播，``A`` 的右下块 ``∂a/∂v`` 非零，旧的 ``∂a/∂v = 0`` 隐式假设会让 STM 静默出错：不抛错，只给出错误的灵敏度。

drag Rust 移植（issue #315 系列）把 ``DragModel`` 纳入 compiled 力模型，STM 路径首次出现速度依赖力。这迫使力-雅可比接口从返回 ``∂a/∂r`` 扩为返回 ``(∂a/∂r, ∂a/∂v)``。

### 决策

把 Rust 力模型雅可比接口的返回值从二元组 ``(acc, ∂a/∂r)`` 扩为**三元组** ``(acc, ∂a/∂r, ∂a/∂v)``，全链路透传 ``∂a/∂v`` 到变分方程。

1. **``acceleration_and_jacobian`` 返回三元组。**
   - 类型别名 ``AccelJacobiResult = Result<([f64;3], [[f64;3];3], [[f64;3];3]), String>``（compiled.rs）。
   - 每个 ``CompiledForce`` variant 在自身语义下给出 ``∂a/∂v``：
     - ``PointMass`` / ``ThirdBody`` / ``IndirectTerm`` / ``GravityField`` / ``SRP``：解析 ``[[0.0;3];3]``（位置型力，速度无关）。
     - ``Drag``：与 ``∂a/∂r`` 同源的中心差分 FD，扰动 J2000 全 6 分量（12 次 accel 评估），``∂a/∂v`` 为真值。
   - ``compute_total_acceleration_and_jacobian`` 逐力叠加 ``total_dadv``（与 ``total_jac`` 同构）。

2. **``stm_derivative`` 接收 ``dadv``。**
   - 签名 ``stm_derivative(stm, jac_da_dr, dadv) -> [f64;36]``，后 3 行 ``A`` 用 ``∂a/∂r·Φ[:3] + ∂a/∂v·Φ[3:]``。
   - ``nbody_stm``、``augmented_state``、``compiled_stm``、``multiple_shooting`` 全部跟进调用签名。

3. **N 体路径 ``dadv`` 恒零。**
   - ``compute_nbody_acceleration_and_jacobian`` 返回 ``dadv = [[0.0;3];3]``（N 体引力不依赖速度）。这条路径行为不变，三元组只是显式写出原本的隐式零。

4. **Python 侧 ``ForceModel`` STM 回退路径同步（issue #317 第 2.1 项）。**
   - ``_compute_total_jacobian`` 返回 ``(∂a/∂r, ∂a/∂v)``；``_eom_func_with_stm`` 组装 ``A[3:,3:] = ∂a/∂v``。
   - 解析雅可比力（``compute_jacobian`` 非 ``None``）``∂a/∂v = 0``；无解析雅可比力（``None``）走有限差分，同时对位置与速度扰动，给出真值。
   - 本项消除未来无 SPICE 下的速度依赖力静默出错的隐患。后续 issue #378 把 Python 回退路径整体移除（不再回退 Python RHS，见 ADR 0020），本条同步代码随之删除。

### 影响面

接口扩三元组触及所有 STM 相关 Rust 模块，ABI 不变（内部函数签名变更，非 pyfunction 边界）：

| 模块 | 变更 |
|---|---|
| `compiled.rs` | `AccelJacobiResult` 类型别名；`acceleration_and_jacobian` / `compute_total_acceleration_and_jacobian` 三元组 |
| `nbody_stm.rs` | `compute_nbody_acceleration_and_jacobian` 返回恒零 `dadv`；`stm_derivative` 加 `dadv` 参数 |
| `augmented_state.rs` | 42 维增广右端项透传 `dadv` 给 `stm_derivative` |
| `compiled_stm.rs` | `augmented_eom` 解三元组、透传 `dadv` |
| `multiple_shooting.rs` | STM 打靶段透传 `dadv` |
| `e2m2e/algorithm/forces/force_model.py` | Python 回退路径同步（第 2.1 项；该路径后于 issue #378 移除） |

### 先例与定位

- **比 ADR 0017 更基础。** 0017 是网格搜索的并行执行策略（Rayon），只影响转移搜索一条算法路径；本 ADR 是力-雅可比契约，所有 STM 传播（打靶、修正、STM 采样、网格搜索内的积分）都依赖它。0017 引用的 ``multiple_shooting`` / ``propagate_compiled`` 先例本身就是本契约的消费者。
- **与 ADR 0002 的关系。** ADR 0002 修订 2 已确立 SPICE 相关传播必须编进 Rust 扩展（cspice 内核池是进程级单例）。本 ADR 不改那条边界，只把 Rust 内部的力-雅可比数据结构从二维扩到三维，属于 ADR 0002 Rust 内核内部的数据契约细化，不跨界。
- **与 ADR 0003 的关系。** drag 的 ``∂a/∂v`` 非零源于其物理（相对速度），与坐标系无关；坐标侧的 ``ITRF93`` 选择另由 ADR 0019 记录。

### 为什么不是别的形状

- **不改成结构体字段。** ``(acc, ∂a/∂r, ∂a/∂v)`` 三元组与 GMAT ``CompleteDerivativeCalculations`` 的分工一致：力只供 Ã 的左下块，这里只是把左下块从 ``3×3`` 细化为 ``(∂a/∂r, ∂a/∂v)`` 两块。结构体（如 ``AccelJacobi { acc, dadr, dadv }``）可读性略好，但会动所有解构点；三元组对齐已有 ``AccelDrag`` / ``AccelJacobiResult`` 风格，改动面最小。
- **不把 ``∂a/∂v`` 留作可选。** 可选 ``Option<[[f64;3];3]>`` 会让速度依赖力漏填重新变成静默错误，这正是本 ADR 要消除的失败模式。强制三元组让缺项编译不过。

### 结果

### 新增

- 力-雅可比接口显式包含 ``∂a/∂v``，STM 变分方程对速度依赖力正确。
- drag 进入 compiled STM 路径后灵敏度不再静默出错。
- Python ``ForceModel`` 回退路径同步（第 2.1 项），消除未来无 SPICE 下的隐患（该路径后于 issue #378 移除）。

### 不变

- N 体 / CR3BP / EphemerisDynamics STM 行为（``dadv`` 恒零，等价于旧隐式零）。
- pyfunction ABI（``propagate_compiled_stm_py`` 等入参与返回形状不变）。
- ``PhysicalModel.compute_jacobian`` 仍只返回 ``∂a/∂r``（3×3），Python 解析雅可比契约不变；``∂a/∂v`` 在 ``ForceModel`` 层按解析力给零、无解析力走 FD 推导。（该 Python 契约后于 issue #378 随回退路径一并移除。）

### 取舍

- **FD 评估数翻倍（drag）。** ``drag_accel_and_jacobian`` 从扰动位置 3 分量（6 次 accel）扩到全 6 分量（12 次 accel）。drag 的 FD 本就是 STM 路径里最重的单力，但 STM 传播主要成本在 RK 步进而非单次雅可比，且 drag 量级小（高度衰减快），实测无感。
- **Python FD 路径成本。** 无解析雅可比力在 Python 回退路径下 FD 从 6 次 accel（位置）扩到 12 次（位置+速度）。该路径是降级路径，drag 不可达，当时唯一消费者是球谐 ``GravityField``（位置型，速度 FD 给零），额外 6 次 accel 是球谐评估，可接受。（该路径后于 issue #378 移除。）
