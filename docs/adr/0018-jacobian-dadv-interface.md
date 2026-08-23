# ADR 0018：Jacobian 接口扩 ∂a/∂v，状态转移矩阵纳入速度依赖

**状态**：已采纳
**日期**：2026-08-07
**关联**：ADR 0002（Rust 积分器内核）、ADR 0003（ITRF93 默认值）、ADR 0017（网格搜索 Rayon）、issue #317
**关联代码**：`crates/e2m2e-forces/src/forces/{compiled,nbody_stm,augmented_state,compiled_stm}.rs`、`crates/e2m2e-integrators/src/multiple_shooting.rs`

## 背景

状态转移矩阵（STM）满足变分方程 ``Φ̇ = A·Φ``，其中

```text
A = | 0₃ₓ₃   I₃ₓ₃ |
    | ∂a/∂r  ∂a/∂v |
```

左下块 ``∂a/∂r``（加速度对位置）由各力雅可比叠加；右下块 ``∂a/∂v``（加速度对速度）在纯 N 体引力模型里恒为零：所有天体引力只依赖位置。本仓 STM 传播（CR3BP / EphemerisDynamics / N 体 STM / compiled STM / 多重打靶）长期基于这一假设。

大气阻力打破它：``a_drag = −½·ρ(|r|)·BC·|v|·v``，既依赖位置（通过大气密度 ρ）又依赖速度（相对速度本身）。drag 一旦进入 STM 传播，``A`` 的右下块 ``∂a/∂v`` 非零，旧的 ``∂a/∂v = 0`` 隐式假设会让 STM 静默出错：不抛错，只给出错误的灵敏度。

drag Rust 移植（issue #315 系列）把 ``DragModel`` 纳入 compiled 力模型，STM 路径首次出现速度依赖力。这迫使力-雅可比接口从返回 ``∂a/∂r`` 扩为返回 ``(∂a/∂r, ∂a/∂v)``。

## 决策

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

## 影响面

接口扩三元组触及所有 STM 相关 Rust 模块，ABI 不变（内部函数签名变更，非 pyfunction 边界）：

| 模块 | 变更 |
|---|---|
| `compiled.rs` | `AccelJacobiResult` 类型别名；`acceleration_and_jacobian` / `compute_total_acceleration_and_jacobian` 三元组 |
| `nbody_stm.rs` | `compute_nbody_acceleration_and_jacobian` 返回恒零 `dadv`；`stm_derivative` 加 `dadv` 参数 |
| `augmented_state.rs` | 42 维增广右端项透传 `dadv` 给 `stm_derivative` |
| `compiled_stm.rs` | `augmented_eom` 解三元组、透传 `dadv` |
| `multiple_shooting.rs` | STM 打靶段透传 `dadv` |
| `e2m2e/algorithm/forces/force_model.py` | Python 回退路径同步（第 2.1 项；该路径后于 issue #378 移除） |

## 先例与定位

- **比 ADR 0017 更基础。** 0017 是网格搜索的并行执行策略（Rayon），只影响转移搜索一条算法路径；本 ADR 是力-雅可比契约，所有 STM 传播（打靶、修正、STM 采样、网格搜索内的积分）都依赖它。0017 引用的 ``multiple_shooting`` / ``propagate_compiled`` 先例本身就是本契约的消费者。
- **与 ADR 0002 的关系。** ADR 0002 修订 2 已确立 SPICE 相关传播必须编进 Rust 扩展（cspice 内核池是进程级单例）。本 ADR 不改那条边界，只把 Rust 内部的力-雅可比数据结构从二维扩到三维，属于 ADR 0002 Rust 内核内部的数据契约细化，不跨界。
- **与 ADR 0003 的关系。** drag 的 ``∂a/∂v`` 非零源于其物理（相对速度），与坐标系无关；坐标侧的 ``ITRF93`` 选择另由 ADR 0019 记录。

## 为什么不是别的形状

- **不改成结构体字段。** ``(acc, ∂a/∂r, ∂a/∂v)`` 三元组与 GMAT ``CompleteDerivativeCalculations`` 的分工一致：力只供 Ã 的左下块，这里只是把左下块从 ``3×3`` 细化为 ``(∂a/∂r, ∂a/∂v)`` 两块。结构体（如 ``AccelJacobi { acc, dadr, dadv }``）可读性略好，但会动所有解构点；三元组对齐已有 ``AccelDrag`` / ``AccelJacobiResult`` 风格，改动面最小。
- **不把 ``∂a/∂v`` 留作可选。** 可选 ``Option<[[f64;3];3]>`` 会让速度依赖力漏填重新变成静默错误，这正是本 ADR 要消除的失败模式。强制三元组让缺项编译不过。

## 结果

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
