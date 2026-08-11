# LPO 设计 Rust 化 + Rayon 并行设计（ADR 0022 前置）

**立项**：#366（L4/L5 多重打靶不收敛）、#367（L4 lpo 卡死默认套件）
**关联**：ADR 0002（Rust 积分器内核）、ADR 0017（转移网格搜索 Rust Rayon）、ADR 0013（验证策略）
**日期**：2026-08-10
**状态**：设计

## 一、问题

`test_lpo_family.py::test_cr3bp_orbit_for_l4_lpo_end_to_end` 卡死默认套件（60s 超时被 kill），根因：

1. `design_lpo` 的分层网格搜索（30 粗 + 15 细 + 局部二分）是 **Python 串行** `_grid_search` for 循环，每个候选点调 `_correct_lpo` → Newton 修正器。
2. 单点传播虽已走 Rust（`propagate_cr3bp_stm`，实测 0.27s/次），但**网格候选点数量多 × 每点多轮修正**，串行累计到几十秒级。
3. 部分候选 x₀ 修正不收敛跑满 50 轮（`DEFAULT_MAX_ITERATIONS`），进一步放大。

用户决策：**放弃 Python 串行计算，方案 B——把整个 LPO 设计链路下沉 Rust + Rayon 并行**。

## 二、要下沉的 Python 逻辑（全部在 `design_lpo` 链路）

| 组件 | Python 位置 | Rust 化要点 |
|---|---|---|
| **初猜** `compute_lpo_initial_guess` | `family/lpo_initial_guess.py` → `triangular_initial_guess._triangular_modes`（`np.linalg.eig` 复特征分解） | 6×6 复特征分解（L4/L5 线性化模态）+ 长周期模态叠加 |
| **传播** `dynamics.propagate` | 已走 Rust `propagate_cr3bp_stm` ✓ | **直接复用** `e2m2e_forces::cr3bp::propagate_cr3bp_stm`（不重写）|
| **修正器** `iterate_full_period_correction` | `solver/differential_correction.py`：Newton 循环（全周期 STM 传播 → 残差 → 雅可比从 STM 取列 → solve → 更新） | **核心工作**：按 LPO 语义重写 Newton 循环在 Rust |
| **网格搜索** `_grid_search` | `family/cr3bp_orbits.py`：30 粗 + 15 细 + 二分 | Rayon `par_iter` 并行候选点 + 保序 |
| **振幅测量** `_l45_distance` | 传播一周 + 距离 min/max | Rust 传播 + 距离扫描 |

## 三、修正器语义（`lpo_fixed_x0` 策略）

- **自由变量**：`[y₀, ẋ₀, ẏ₀, T]`（4 个，x₀ 固定，z₀=ż₀=0 平面约束）
- **约束**：`[Δy=0, Δẋ=0, Δẏ=0]`（3 个，全周期闭合）
- **4 自由 vs 3 约束**：欠定系统，最小二乘求最小范数修正
- 收敛容差 `tolerance=1e-12`，停滞阈值 `stagnation_limit`，发散 `divergence_limit`，`max_iterations=50`
- 收敛后校验 `period ∈ [10, 50]`（nd），否则算失败

## 四、并行架构（照 ADR 0017 / multiple_shooting.rs 范式）

```rust
#[pyfunction]
fn design_lpo_py(...) -> ... {
    py.allow_threads(move || {
        (0..n_pts).into_par_iter()            // 并行候选点
            .map(|i| correct_lpo(x0[i], ...)) // 每个候选 = 独立 Newton 修正
            .collect::<Vec<_>>()               // 保序
    })
}
```

- 每个候选的 Newton 修正器**完全独立**（无共享可变 state）→ 天然可并行
- 内部传播直接调纯 Rust `propagate_cr3bp_stm`，不绕道 PyO3（避免 GIL 串行化）
- `E2M2E_LPO_PARALLEL=0` 强制串行（对照位级一致），沿用 `E2M2E_MS_PARALLEL`/`E2M2E_SEARCH_PARALLEL` 命名

## 五、数学基础与求解器决策

### 轨道计算的数学基础

LPO 设计 = CR3BP 运动方程（ODE）+ 数值积分（Rust 已解决）+ 微分修正（解非线性方程组）。

微分修正找周期轨道：求 `(x₀, T)` 使单圈闭合 `F = φ_T(x₀) - x₀ = 0`（约束分量上）。Newton 每步线性化：

```
J · dX = -F
```

雅可比 `J = ∂F/∂(自由变量)` 从 STM 取列：`∂φ_T/∂x₀ = Φ(T)`，`∂φ_T/∂T = ẋ(T)`。LPO 是 4 变量 × 3 约束**欠定**系统，有无穷多解。**数学上自然的选取是极小范数解**：

```
dX = argmin ‖dX‖  s.t.  J·dX = -F
```

几何意义：在满足方程的所有修正里取离初猜最近的——"停留在初猜附近"，即保形（微分修正保持轨道族连续性的来源，与多重打靶阻尼 LM 的"保形"同一目的）。

### 求解器决策（2026-08-10 确立）

**核心原则：实现数学正确的 min-norm 最小二乘，不纠结 numpy 逐位对齐。**

验证结论（`least_squares_solve` λ=0 vs `np.linalg.lstsq`）：
- `least_squares_solve`（正规方程 + 高斯消元，λ=0）对欠定系统给**某个特解**（主元选择决定，LPO 规模下把一列钉 0），**非 min-norm**。
- numpy `lstsq`（SVD 伪逆）给 **min-norm 解**——那是数学上正确的选择。
- λ>0 时 `least_squares_solve` 收敛到 min-norm（λ=1e-8 → 误差 1.7e-7），但那只是近似。

**决策**：Rust 侧实现**数学正确、数值稳定的 min-norm 最小二乘**（3×4 固定小规模，手写 QR/Householder 或 SVD），向数学看齐而非向 numpy 看齐。与 numpy 结果一致到机器精度——那是因为两者实现同一数学，不是为过对照测试。

### 复用边界（`multiple_shooting.rs`）

- **可复用**：`least_squares_solve` 的**模式**（LM 阻尼 `(JᵀJ+λI)dX=Jᵀ(-F)`）——但 LPO 用真 min-norm（QR/SVD），非 λ>0 近似。
- **不可直接复用**：`build_jacobian_*`（段间残差语义）、`build_residual`（段拼接残差）。LPO 按 `iterate_full_period_correction:907-915` 重写雅可比组装（单弧全周期闭合，含 T 时间项 `ẋ(T)`、`var_idx==c_idx` 的 `-1`）。
- **复特征分解**（`_triangular_modes`）：无现成工具，用 Python 预计算 L4/L5 模态常量传参（L4/L5 模态是系统常量，不随候选点变）。



## 六、验证策略（ADR 0013）

- **Rust vs Python 等价性对照**：同参数分别跑 Rust `design_lpo_py` 与 Python `design_lpo`，对照 `orbit.period`、`orbit.states[0]`、振幅、收敛状态
- **位级一致优先**：`propagate_cr3bp_stm` 已是位级一致先例，Newton 迭代应逐位对齐
- **测试分层**：修正器单元（Rust 内）+ 等价性对照（Python）+ 端到端收敛（L4/L5）
- **环境变量开关**：`E2M2E_LPO_PARALLEL=0` 串行对照

## 七、里程碑（每步独立合入 + 等价性验证）

1. **M1**：Rust LPO Newton 修正器（核心骨架，复用 `least_squares_solve`）+ 单测
2. **M2**：Rust 初猜（复用 Python 预计算 L4/L5 模态）+ 振幅测量下沉
3. **M3**：Rust `design_lpo` 网格搜索 + Rayon 并行 + `E2M2E_LPO_PARALLEL`
4. **M4**：Python 接线 + 等价性测试 + 关闭 #366/#367

## 八、架构合规

- **ADR 0011（五层）**：下沉的是数值层职责内的纯数值原子（传播 + 修正 + 距离），不是编排
- **ADR 0012（依赖方向）**：Python 算法层调 Rust 数值层（`crates/`），合法方向
- **ADR 0002（Rust 边界）**：引用 `multiple_shooting` + `propagate_compiled` 先例，Rust 内核边界扩展到"单弧全周期修正 + 网格搜索"
- **ADR 0013（验证）**：Rust 与 Python 等价性对照，不依赖外部软件
