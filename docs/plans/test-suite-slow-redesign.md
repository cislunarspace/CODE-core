# 测试套件 slow 测试重设计 RFC

**立项**：#361
**关联**：ADR 0021（测试套件按功能类目组织）、ADR 0013（验证策略）、#359（标记重组落地）
**日期**：2026-08-09

## 一、问题

#359 落地了 ADR 0021 的「目录迁移 + 标记重组」，但 **slow 测试的内容未动**——仍按已废的「L3 scenario」旧范式写。直接症状：默认套件（`-m "not slow"`）跑 18 分钟仍未完；54 个 slow 测试里大量是「每断言一次 e2e 管线」的债。

逐文件审查后，根因不是「物理验证需要慢」，而是旧范式残留的六类病灶：

1. **每断言一 e2e**：`design/scenarios/` 的 6 个测试各跑一次 18 天 `design_orbit`，其中 4 个断言的只是 `Ephemeris`/`DesignOrbitResponse` 字段形状——本质是数据结构契约，该归 `data`/`interface` 类，不该靠真传播验证。
2. **fixture 不共享**：`pal_stagnation` 3 个方法各跑一次相同的 80 步延拓；`wsb` 2 个方法各搜一次相同的网格。
3. **bug 回归靠 e2e 间接抓**：`segmented` 回归的 `propagate_compiled` 的 `t_eval[0]≠t0` 是积分器 bug，该有最小复现单元，不该靠 30 天 segmented 设计间接抓。
4. **参数化覆盖 = 重复契约**：`initial_guess/test_lissajous` 的 4 组合断言同一组契约（形状/单调/有界），不是 4 个不同目标。
5. **开发中 feat 占位**：`homotopy` 从未跑通，在默认套件里持续 F，是未完成功能的占位测试。
6. **docstring 旧框架措辞**：4 个文件仍写「属 tests/orbit_design 三层分层中的 L3（scenarios，端到端）」，引用已被 ADR 0021 废除的分层。

## 二、设计原则

落实 ADR 0021，五条：

1. **e2e 解散，按断言归类**（ADR 0021 理由 4）。一个 `design_orbit` 调用里，字段形状/类型 → `data`/`interface`；编排链路收敛 → `orchestration`；物理量取值 → `orchestration`/`theory`。三者分属不同类、不同手段，不塞进同一个 slow e2e。
2. **一次管线调用，多断言共享 fixture**。`segmented`/`frozen_orbit` 已做对（module scope）；`pal`/`wsb` 做反了——纯浪费，与分类无关。
3. **bug 回归找最小复现**。integrator bug 做 integrator 单元，不靠 e2e 间接抓（既慢，又会在 bug 以别的方式复现时漏掉）。
4. **开发中 feat 标记跳过**。`homotopy` 未跑通，module-level skip，待功能完成后启用。
5. **C 类保留，但审「最短弧/最少阶/最少点」**。物理结论（Δe、secular 斜率、Jacobi 守恒）需要足够计算量体现，但当前参数往往留了过大余量。

## 三、「测到什么程度」的操作定义

每个测试必须能回答「我证明了什么事实」，事实的性质决定它的类和手段。答不出「证明了什么」的（如「跑通不崩」），消除或跳过。

| 证明的事实 | 功能类 | 手段 |
|---|---|---|
| 字段形状/类型对 | `data`/`interface` | 构造对象断言，零管线 |
| 数值方法对解析解 | `theory` | 短弧 + 解析式 |
| 编排链路不抛 + 收敛 | `orchestration` | 每类型 1 个瘦身真调用，共享 fixture |
| 某物理量在 X 条件取 Y | `orchestration`/`theory` | 体现结论的最短弧 |
| 某 bug 不再复现 | 视 bug 所属层 | 最小复现条件 |

## 四、处置清单（逐文件，含新测试草图）

### A 类：e2e 解散 + 契约下沉（先改，收益最大、风险最小）

#### `design/scenarios/test_lissajous.py` + `test_triangular.py`（合并）

**现状**：两文件逐行重复，各 6 个 slow 测试 = 12 个，每个跑一次 18 天 `design_orbit`。4 个断言字段形状契约（`output_shape`/`epoch_matches_input`/`initial_state_shape`/`cr3bp_orbit_present`），2 个真集成（`end_to_end_converges`/`amplitude_bounded`）。

**新设计**：

- 字段形状契约**删除**，下沉到：
  - `Ephemeris` 契约（`position_km.shape`/`velocity_mps.shape`/`synodic_position.shape`/`year[0]`）→ `tests/data/test_types_trajectory.py`（已存在，补断言）。
  - `DesignOrbitResponse` 契约（`initial_state.shape==(6,)`/`cr3bp_orbit_present`/`cr3bp_jacobi` 类型）→ `tests/api/`（构造 Response 断言字段，零管线）。
- 集成收敛：1 个 `orchestration` 测试，参数化 `orbit_type ∈ {LISSAJOUS-L1, LISSAJOUS-L2, L4, L5}`，**共享 module fixture**（每个类型跑一次），`duration` 从 18 天缩到 3–5 天（够验证 `correction.converged`），断言收敛 + `ephemeris` 非空。
- 有界性：并入收敛测试同一 fixture，断言 Jacobi 守恒漂移（比「位置 < 1e6 km」更严格、更物理）。

**净效果**：12 slow → 4 个 orchestration 真调用（短弧、共享 fixture）+ 契约下沉 data/api。预计默认套件省下 ~12 次 18 天传播。

#### `transfer/test_wsb.py`（2 个 slow）

**现状**：两个 `@slow` 方法各搜一次相同的 WSB 网格（15 格点）。`test_returns_transfer_design_result` 断言返回类型 + `transfer_type`；`test_wsb_details_populated` 断言 `details` 各字段类型（`isinstance float/bool/int`）。

**新设计**：

- `details` 字段类型契约 → `tests/api/` 或 interface，构造 `WsbTransferDetails` 直接断言字段类型，零搜索。
- 真搜索：1 个 `orchestration` 测试，**共享 module fixture**（搜一次），断言 `transfer_type=="WSB"` + `converged` + `dv` 合理量级（物理），同时复用 fixture 检查 details 非空。

**净效果**：2 slow → 1 orchestration + 契约下沉。WSB 单格点 BCR4BP 多圈传播 inherent 慢，搜索次数减半。

### B 类：fixture 共享 + 参数化收敛

#### `correction/test_pal_stagnation.py`（24 个 slow）

**现状**：8 组合（L1/L2 × 北/南 × 正/负）× 3 方法 = 24 个；3 个方法（`reaches_fold`/`no_stagnation_oscillation`/`extends_x_past_fold`）**各跑一次相同的 80 步延拓**，断言同一 family 的不同方面。8 组合里南北、正负物理对称冗余。

**新设计**：

- 3 方法共享 1 个 family fixture（parametrize 在 fixture 上，每组合跑一次 80 步延拓，三方面断言共用）→ 从 24 次延拓降到 8 次。
- 组合 8 → 代表性 2（如 L1 北正、L2 北正）；南北对称、正负方向对称用少量组合覆盖即可。→ 8 次降到 2 次。
- 三方面断言（z_amp 到折叠点 / z 振幅无 2-周期环 / x 穿过折叠点）合并到每组合一个测试的多断言，或保留 3 测试共享同一 fixture。

**净效果**：24 slow → 2 组合 × 1 family = 2 次 80 步延拓，断言三方面。回归价值不减（bug 根因 `pal_plausible` 误判在代表性组合上同样暴露）。

#### `design/initial_guess/test_lissajous.py::TestLissajousBoundedTrajectory`（4 个 slow）

**现状**：4 组合（L ∈ {1,2} × 振幅 ∈ {(500,2000),(2500,7500)}），各走中心流形约化，断言同一组契约（三元组结构/形状/times 单调/period/有界性）。

**新设计**：

- 契约（三元组结构/形状/times 单调/period>0）→ 1 组合的单元测试，直接调 `compute_lissajous_bounded_trajectory` 断言结构。
- 有界性物理（面内偏移 < 3× 振幅）→ 参数化 2 组合（小振幅 + 大振幅）验证有界性随振幅成立。

**净效果**：4 slow → 1 契约单元 + 2 有界性。契约不再重复跑约化。

### bug 下沉

#### `design/scenarios/test_segmented.py`（2 个 slow）

**现状**：fixture 共享做对（module scope，2 测试复用一次 4 分钟 segmented 设计）。但回归的 bug 是 `propagate_compiled` 的 `eval_idx` 初始化假设 `t_eval[0]==t0`——**积分器层 bug**。

**新设计**：

- bug 下沉：`tests/numerical/integrators/` 加单元测试——构造 `t_eval[0] > t0` 场景，直接调积分器，断言首个输出点状态非初值错置（相邻点 J2000 漂移正常）。秒级，直接抓根因。
- segmented 集成：保留 1 个 `orchestration` 测试（共享 fixture，审 30 天 → 短弧是否够验证星历连续），断言星历相邻点无停滞 + patch point 间距量级。

**净效果**：bug 用快单元直接抓（不再靠 4 分钟 e2e 间接）；segmented 集成保留但瘦身。

### 开发中 feat

#### `correction/test_homotopy_ephemeris_integration.py`

**现状**：homotopy correction 是开发中 feat，从未跑通。文件无 skip 门，4 个测试在默认套件里持续 F。

**新设计**：文件顶部加 module-level 跳过：

```python
pytest.skip(
    "homotopy correction 开发中（feat 未完成），待功能跑通后启用",
    allow_module_level=True,
)
```

**净效果**：默认套件不再背 homotopy 的 F；功能完成后移除 skip、重设计测试（真种子验证残差收敛，而非平凡种子 + placeholder 判据）。

### C 类：保留，审弧长/阶数

#### `normal_form/test_lissajous_bounded.py`（2 个 slow）

**保留**。验证中心流形约化的核心正确性（双曲耦合消除 + 6 周期有界），目标本身重，fixture 已缓存复用。order 5 不降（降阶影响约化质量，验证失意义）。

#### `design/scenarios/test_frozen_orbit_e2e.py`（4 个 slow）

**保留**，审弧长。fixture 已共享（module scope）。断言物理结论（i=75° 无严格冻结、Δe≈−0.019、Δrp≈+61 km、漂移方向、|Δrp| 随 a 增大）。60 天 → 审 30 天是否够体现结论方向（Δrp 减半但正负、单调性不变）；`test_drift_rp_increases_with_a` 的 a8000 第二次传播可同步缩短。

#### `numerical/forces/test_low_thrust_propagation.py::test_low_thrust_spiral_orbit_evolution`（1 个 slow）

**保留**，缩弧。30 天 → 7–10 天，够体现「半长轴 secular 斜率 > 0 + 偏心率 < 0.05」的物理结论。文件内已有的短弧解析对照（`a(t)` 解析解 5% 容差，`theory` 类）不动。

## 五、docstring 旧框架措辞清理

随 PR 一并清理 4 处对已废分层的引用：

- `tests/algorithm/design/scenarios/test_lissajous.py:10-11`（「属 tests/orbit_design 三层分层中的 L3（scenarios，端到端）」）
- `tests/algorithm/design/scenarios/test_triangular.py:10-11`（同上）
- `tests/algorithm/design/scenarios/test_segmented.py:10`（同上）
- `tests/algorithm/design/scenarios/test_frozen_orbit_e2e.py:1`（「（L3）」）

替换为 ADR 0021 的功能类描述（`orchestration`：design_orbit 全链路集成），不引用已废分层。

## 六、分批 PR 计划

按「收益最大、风险最小」排序，每批独立合入：

- **PR1（A 类 + docstring 清理）**：合并 lissajous/triangular、瘦身 wsb、契约下沉 data/api、清理 4 处 docstring。收益最大（默认套件省 ~14 次 e2e 传播），风险最小（契约下沉是搬运、不碰算法）。
- **PR2（B 类）**：pal_stagnation fixture 共享 + 组合收敛、initial_guess/lissajous 契约下沉。
- **PR3（bug 下沉 + homotopy skip + C 类微调）**：segmented bug 下沉 integrator 单元、homotopy module-level skip、frozen/low_thrust 弧长微调。

每批合入后跑一次默认套件 + slow 套件，确认 `--durations` 下降、无回归。

## 七、验收

- 默认套件（`-m "not slow"`）wall time 显著下降（目标：去除 homotopy F + A/B 类 e2e 后，回到可日常迭代的量级）。
- slow 套件从 54 个降到 ~10 个真物理测试（C 类 + 保留的瘦身集成）。
- 每个保留下来的测试能一句话回答「证明了什么物理/契约事实」。
- `grep -rn 'L3\|三层分层\|scenarios，端到端' tests/` 无残留。
