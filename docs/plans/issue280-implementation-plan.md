# Issue #280 实施计划：TIGHT/SPECIAL 模式量级对齐（修订版）

> **修订说明**：原计划（Phase 0–4）在 #257 合并前编写，根因分析基于旧默认值。当前
> 工作树已包含 Phase 1（TIGHT）和 Phase 2（SPECIAL）的核心改动（未提交），本计划
> 以当前代码状态为起点，聚焦剩余验证与收尾工作。

---

## 当前状态

### 已完成的改动（工作树未提交，117 行）

| 文件 | 改动 | 依据 |
|---|---|---|
| `target_point.py` | `tolerance_km` 1.0→0.1，`max_iter` 2→6 | NRHO 非线性效应需更多迭代；0.1 km 容差比测定轨精度（1.5 km 1σ）小一个量级 |
| `special_point.py` | 新增 `damping_factor`（Armijo 回溯）、`v_c`（雅可比无量纲化）、穿越边界 `frac=0.0` 修复 | 无阻尼牛顿在大扰动下振荡；缺 `v_c` 导致步长偏差 ~2%/步 |
| `monte_carlo.py` | `_make_law` / `_build_simulation` / `run_monte_carlo` 透传新参数；SPECIAL 获取 `v_c` | 参数贯通到公共 API |
| `controller.py` | `control_orbit()` 新增 `tight_tolerance_km`、`tight_max_iter`、`special_damping_factor` | 暴露可调参数 |
| `test_control_laws.py` | +3 个测试：默认值、`v_c` 缩放、阻尼收敛 | 覆盖新增功能 |

### 测试状态

- `tests/algorithms/station_keeping/test_control_laws.py`：**11/11 passed**（含 3 个新增）
- E2E 测试（`tests/dfh/`）：**未运行**（需 SPICE 内核 + Rust 后端）

---

## 根因复核

### TIGHT 偏低 3.2×（3.4 vs 11.0 m/s）

**原假设**：`max_iter=2` 不足以收敛。

**代码验证**：原默认 `max_iter=2`、`tolerance_km=1.0`。NRHO 弧段 28 天，线性初值
加 1 次精化后残差可能仍 > 1 km（tolerance=1.0 km 下直接退出），导致控制量偏小。
改为 `max_iter=6`、`tolerance_km=0.1` 后，迭代空间充足，容差比测定轨精度小一个
量级。**假设成立，改动合理。**

### SPECIAL 偏高 16×（~66 vs 4.0 m/s）

**原假设 1**：无阻尼牛顿振荡发散。

**代码验证**：原代码 `v0 = v0 + dv` 无回溯。大扰动（位置 1.5 km 1σ）下约束残差
`‖g‖` 可达 0.01 量级，牛顿步可能过矫正。新增 Armijo 回溯（`damping_factor < 1` 时
`‖g_new‖ > ‖g_old‖` 则步长乘 `damping_factor`）。**假设成立。**

**原假设 2**：雅可比缺 `v_c` 因子。

**代码验证**：原代码 `jac = rows @ stm_vv`，理论应为 `jac = rows @ stm_vv / v_c`
（`v_c = l_c/t_c ≈ 1.023 km/s`）。缺因子使每步 Newton 偏差 ~2%，8 步累积 ~17%，
是 16× 偏高的贡献因子但非主因。已修正。**假设部分成立，是叠加因素。**

**原假设 3**：穿越边界 `frac=0.0` 回退。

**代码验证**：原代码在无符号变化时 `frac=0.0`，`t_star` 精度退化。已改为
`t_star = t_fine[j]`（取最接近零的点）。**假设成立，是边界情形修正。**

---

## 剩余工作

### Phase 0：诊断脚本（确认修复有效，~1h）

**目标**：用同一标称轨道 + 同一入轨扰动，跑 LOOSE/TIGHT/SPECIAL 三种模式（无误差，
`num_monte_carlo=1`），打印中间量确认量级对齐。

**产物**：`scripts/diagnose_tight_special.py`

```
输入：标称轨道星历（FR1 产物）、固定入轨扰动
输出：
  - 每个控制节点的 Δv 量级（三种模式）
  - TIGHT: ‖dr_free‖, ‖dr_after_iter_k‖（收敛路径）
  - SPECIAL: ‖g‖ (约束残差), t* (穿越时刻), 迭代次数
  - 三种模式总 Δv 汇总表
验收：TIGHT/SPECIAL 总 Δv 量级与 LOOSE 同阶（< 3× 差异）
```

**注意**：此脚本为开发期诊断工具（ADR 0013），放 `scripts/`，不进 CI。

### Phase 3：E2E 回归测试（~1-2h）

**目标**：确认修复在真实力模型 + 误差模型下端到端有效。

**改动文件**：`tests/dfh/test_e2e.py`（或扩展现有 E2E 测试文件）

**新增测试**：

1. **TIGHT E2E**
   - `control_mode=2`，`num_controls=4`，`num_monte_carlo=1`（无误差或最小误差）
   - 断言：输出结构正确（SK_STATISTIC/MANEUVERS/受控星历非空）
   - 断言：SK_STATISTIC 总 Δv 量级合理（< 100 m/s，物理合理性检查）
   - 断言：迭代收敛（可通过 mock 或日志确认 `‖dr‖ < tolerance_km`）

2. **SPECIAL E2E**
   - `control_mode=3`，`num_controls=4`，`num_monte_carlo=1`
   - 断言：输出结构正确
   - 断言：控制后穿越处会合系 `ẋ_syn ≈ 0`（物理定义验证，ADR 0013）
   - 断言：总 Δv 量级合理（< 100 m/s）

3. **LOOSE 回归保护**
   - 确认现有 `test_mode1_loose` 仍通过（不拆东墙补西墙）

**验收**：`pytest tests/dfh/` 全绿。

### Phase 4：文档与收尾（~30min）

1. **更新 `docs/plans/dfh-parity-prd.md` FR2**
   - 验收标准打勾：`[x] TIGHT/SPECIAL 量级对齐（#280）`
   - 注明验证方式：物理定义验证（ADR 0013），非黄金样本

2. **代码注释补充**
   - `target_point.py`：注释说明 `max_iter=6`、`tolerance_km=0.1` 的选取理由
   - `special_point.py`：注释说明 `v_c` 因子和阻尼的物理/数值原因

3. **删除旧计划文件**
   - `docs/plans/issue280-implementation-plan.md`（本文件取代）

4. **Issue #280 更新**
   - 更新验收标准："与 DFH 一致（< 50%）" → "控制律正确实现《控制方案.md》公式，
     用物理定义验证（ADR 0013）"
   - 关联 PR

---

## 依赖关系

```
Phase 0（诊断）──→ Phase 3（E2E）──→ Phase 4（文档/收尾）
     ↑
 [已就绪：Phase 1+2 代码改动已完成，测试通过]
```

Phase 0 和 Phase 3 严格串行：先用诊断脚本确认中间量正确，再跑 E2E 验证端到端。

## 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| 修复后 Δv 仍未对齐 | Phase 0 诊断脚本会暴露具体哪个步骤异常 | 优先跑 Phase 0，不要跳到 Phase 3 |
| `special_damping_factor` 默认 1.0（不阻尼）可能仍导致 SPECIAL 发散 | 用户需手动传 `< 1` 的值 | 考虑将默认值改为 0.5（待诊断确认） |
| E2E 测试依赖 SPICE 内核 + Rust 后端 | CI 环境可能缺失 | 确认 `tests/dfh/` 已有 SPICE fixtures |
| 改动暴露更深层 STM 传播 bug | 范围扩大 | 超出本 issue，记录为新 issue |

## 验收标准（ADR 0013 合规）

| 项目 | 验证方式 |
|---|---|
| TIGHT 位置重合 | 迭代后 `‖r_pred - r_target‖ < 0.1 km`（解析验证） |
| SPECIAL 穿越约束 | 控制后 `‖ẋ_syn‖ < 1e-6`（无量纲，≈ 1e-3 m/s） |
| 物理合理性 | SK_STATISTIC 总 Δv：LOOSE ≈ TIGHT ≤ SPECIAL（同一轨道） |
| 不拆东墙 | LOOSE E2E 测试不回归 |
| 无黄金样本 | 不与 DFH 输出做断言对比（DFH 仅作诊断参考） |
