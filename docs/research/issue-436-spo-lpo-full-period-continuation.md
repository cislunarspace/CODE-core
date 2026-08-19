# Issue #436 一手资料调研：SPO/LPO 全周期延拓方法

**调研对象**：[cislunarspace/e2m2e#436](https://github.com/cislunarspace/e2m2e/issues/436)
**调研快照**：[`1402591`](https://github.com/cislunarspace/e2m2e/commit/14025914956faa90a1d3e24019db5ca6c33647af)，`issue/436`，2026-08-16
**性质**：实施前调研。尚无数值实验的数据不被写成既定结论；本文件不替代 #436 要求的 ADR。

## 问题

#436 要为 #428 的多轨道族生成决定 SPO/LPO 的正式延拓方法，并研究 LPO 非单调段和大振幅 Horseshoe 的穿越能力。Issue 明确要求 ADR 与可复现实验记录；它目前开放、标为 `ready-for-human`，没有评论，也没有关联 PR 或提交，时间线唯一交叉引用为 #428。[#436](https://github.com/cislunarspace/e2m2e/issues/436)；[#436 时间线](https://github.com/cislunarspace/e2m2e/issues/436/timeline)

本调研只核对 GitHub 一手材料、当前源代码、测试、文档和 Git 历史，提出实验与实施方案；不实现算法或修改 API。#428 也明确把本研究同 Facade 多族分派分开，并将 Lissajous 定义为拟周期参数采样而不是本问题的周期延拓对象。[#428 分诊评论](https://github.com/cislunarspace/e2m2e/issues/428#issuecomment-5305368148)

## 已核实的现状

### API 与算法入口

- `FamilyGenerationRequest` 预留了 Halo、NRHO、Axial、Lissajous、SPO、LPO、Horseshoe，并按共线/三角平动点校验；文档仍写明“第一版仅实现 Halo”。[models.py:661-730](https://github.com/cislunarspace/e2m2e/blob/14025914956faa90a1d3e24019db5ca6c33647af/e2m2e/api/models.py#L661-L730)
- `Facade.orbit_family_generation()` 对非 Halo 返回结构化 `NOT_IMPLEMENTED`，仅分派 `design_halo_family`。[facade.py:498-520](https://github.com/cislunarspace/e2m2e/blob/14025914956faa90a1d3e24019db5ca6c33647af/e2m2e/api/facade.py#L498-L520)
- 注册表是 `design_* -> Orbit` 的单轨分派，`design_halo_family` 才是现有族生成函数。因此现有 `design_spo`、`design_lpo`、`design_horseshoe` 不能证明已有可供 Facade 调用的族生成契约。[family/__init__.py:1-104](https://github.com/cislunarspace/e2m2e/blob/14025914956faa90a1d3e24019db5ca6c33647af/e2m2e/algorithm/family/__init__.py#L1-L104)；[#428 分诊评论](https://github.com/cislunarspace/e2m2e/issues/428#issuecomment-5300033899)

### 修正器与延拓器

- SPO/LPO 都无 x 轴或 XZ 镜面对称，使用全周期闭合。当前配置固定 `x0`，自由变量为 `[y0, vx0, vy0, T_full]`，约束为 `[dy, dvx, dvy]`。[spo.py:18-54](https://github.com/cislunarspace/e2m2e/blob/14025914956faa90a1d3e24019db5ca6c33647af/e2m2e/algorithm/family/strategies/spo.py#L18-L54)；[lpo.py:18-50](https://github.com/cislunarspace/e2m2e/blob/14025914956faa90a1d3e24019db5ca6c33647af/e2m2e/algorithm/family/strategies/lpo.py#L18-L50)
- `iterate_full_period_correction()` 会传播完整周期、由 STM 对 `constraint_indices` 构造闭合雅可比；它不会自动补上未配置的状态残差。[differential_correction.py:819-958](https://github.com/cislunarspace/e2m2e/blob/14025914956faa90a1d3e24019db5ca6c33647af/e2m2e/algorithm/solver/differential_correction.py#L819-L958)
- `Continuation.natural_continuation()` 的 `_sweep()` 硬编码调用半周期 `iterate_correction()`。这就是 SPO/LPO 不能使用该正式入口的直接根因。[continuation.py:193-427](https://github.com/cislunarspace/e2m2e/blob/14025914956faa90a1d3e24019db5ca6c33647af/e2m2e/algorithm/solver/continuation.py#L193-L427)
- 当前 PAL 同样不是通用全周期 PAL：其自由变量固定为 `[rx, rz, vy, T/2]`，终点残差 `[vx, vz, ry]`，雅可比为 `3 x 4`，随后仍选择 Halo 的 XZ 对称修正配置。[continuation.py:26-91](https://github.com/cislunarspace/e2m2e/blob/14025914956faa90a1d3e24019db5ca6c33647af/e2m2e/algorithm/solver/continuation.py#L26-L91)；[continuation.py:429-777](https://github.com/cislunarspace/e2m2e/blob/14025914956faa90a1d3e24019db5ca6c33647af/e2m2e/algorithm/solver/continuation.py#L429-L777)
- 文档只给出了 Halo 背景下的自然延拓/PAL 示例，没有 SPO/LPO 全周期入口。[continuation.rst:39-63](https://github.com/cislunarspace/e2m2e/blob/14025914956faa90a1d3e24019db5ca6c33647af/docs/algorithms/continuation.rst#L39-L63)；[halo-family.rst:48-65](https://github.com/cislunarspace/e2m2e/blob/14025914956faa90a1d3e24019db5ca6c33647af/docs/algorithms/halo-family.rst#L48-L65)

### LPO 与 Horseshoe 的当前能力

- `design_spo()` 在代码中承认小振幅段的振幅-`x0` 映射非单调，采用目标振幅二分加全周期修正；这是单轨反解而不是保存连续成员的延拓。[cr3bp_orbits.py:900-1007](https://github.com/cislunarspace/e2m2e/blob/14025914956faa90a1d3e24019db5ca6c33647af/e2m2e/algorithm/family/cr3bp_orbits.py#L900-L1007)
- `design_lpo()` 明确记录“小振幅椭圆 → 混沌过渡 → 大振幅马蹄”的高度非单调映射，采用 30 点粗网格、15 点局部网格、10 步局部二分，并允许宽容差回退。这既不保证成员连续，也不能穿折或证明分支身份。[cr3bp_orbits.py:1078-1217](https://github.com/cislunarspace/e2m2e/blob/14025914956faa90a1d3e24019db5ca6c33647af/e2m2e/algorithm/family/cr3bp_orbits.py#L1078-L1217)
- Horseshoe 是 `design_lpo()` 的便捷封装。#435 已由 #440 合并：默认值为 100,000 km，LPO/Horseshoe 可声明上限为 110,000 km，并有默认、上限和越界早拒测试。[#435](https://github.com/cislunarspace/e2m2e/issues/435)；[PR #440](https://github.com/cislunarspace/e2m2e/pull/440)；[cr3bp_orbits.py:1219-1257](https://github.com/cislunarspace/e2m2e/blob/14025914956faa90a1d3e24019db5ca6c33647af/e2m2e/algorithm/family/cr3bp_orbits.py#L1219-L1257)
- 110,000 km 是当前网格搜索的保守声明边界，不是物理上限。#435 的材料把旧窗口包络记为约 112,000-120,000 km，并明确没有在该修复中扩大搜索算法；更大范围正是 #436 的待研究事项。[#435](https://github.com/cislunarspace/e2m2e/issues/435)；[#436](https://github.com/cislunarspace/e2m2e/issues/436)

### 测试证据的边界

逐族测试已构造 SPO/LPO 的最小全周期预测-修正循环，各走 5 步 `x0`，检查成员数、参数单调、有限状态和 `closure_error < 1e-3`。测试注释明确：修正器不强制 `dx` 闭合，长弧 LPO 的全 6D 闭合误差地板约 `1e-5`。这证明局部链可走，不能证明全局 PAL 能穿过非单调段。[test_continuation_per_family.py:1-21](https://github.com/cislunarspace/e2m2e/blob/14025914956faa90a1d3e24019db5ca6c33647af/tests/algorithm/design/continuation/test_continuation_per_family.py#L1-L21)；[test_continuation_per_family.py:72-91](https://github.com/cislunarspace/e2m2e/blob/14025914956faa90a1d3e24019db5ca6c33647af/tests/algorithm/design/continuation/test_continuation_per_family.py#L72-L91)；[test_continuation_per_family.py:168-203](https://github.com/cislunarspace/e2m2e/blob/14025914956faa90a1d3e24019db5ca6c33647af/tests/algorithm/design/continuation/test_continuation_per_family.py#L168-L203)

单轨 LPO/Horseshoe 测试则会检查全周期闭合 `<1e-6`、Jacobi 守恒和平面约束；新 PAL 不能把“链未发散”的 `1e-3` 当作其精度契约。[LPO 测试](https://github.com/cislunarspace/e2m2e/blob/14025914956faa90a1d3e24019db5ca6c33647af/tests/algorithm/family/test_lpo_family.py#L126-L172)；[Horseshoe 测试](https://github.com/cislunarspace/e2m2e/blob/14025914956faa90a1d3e24019db5ca6c33647af/tests/algorithm/family/test_horseshoe_family.py#L116-L152)

## 根因与约束

1. **修正和延拓的方程形状不同。** 固定 `x0` 的 4 自由变量/3 残差修正器不是可直接穿折的 PAL 问题。把 `iterate_full_period_correction` 替换进现有 PAL 无法消除 Halo 的坐标、半周期和对称假设。
2. **“全 6D 状态 + 周期”仍缺相位规范。** 自治周期解具有相位退化；必须在原型中明确独立闭合残差、相位条件、族参数和数值秩。否则增广雅可比可能奇异，或只沿同一轨道的相位移动。当前 `3 x 4` XZ 雅可比不能回答这个问题。[continuation.py:26-91](https://github.com/cislunarspace/e2m2e/blob/14025914956faa90a1d3e24019db5ca6c33647af/e2m2e/algorithm/solver/continuation.py#L26-L91)
3. **结果契约不可绕开。** `Continuation` 返回 `ContinuationResult`，族在 `.family`；#434 记录了使用旧 `.orbits` 属性导致 L1 NRHO 的 PAL 编排崩溃。新路径应沿用 `ContinuationResult`、结构化失败和 `OrbitFamily`。[continuation.py:781-801](https://github.com/cislunarspace/e2m2e/blob/14025914956faa90a1d3e24019db5ca6c33647af/e2m2e/algorithm/solver/continuation.py#L781-L801)；[#434](https://github.com/cislunarspace/e2m2e/issues/434)
4. **环境前置条件已满足。** 依 README/Makefile 执行 `make setup && make dev` 后，CSPICE、SPICE 内核和 Rust 扩展均已就绪；`tests/algorithm/design/continuation/test_continuation_per_family.py` 已重跑通过（5 passed）。

## 已决实施方案

实验后采纳的方案见 [ADR 0028](../adr/0028-planar-triangular-full-period-pal.md)：为平面三角平动点族增加专用的全周期 PAL 模块，未知量为 `(x0, y0, vx0, vy0, T)`，以四个平面闭合残差、相位条件和伪弧长条件构成 6 行、5 列的最小二乘 Newton 问题。闭合加相位的数值有效秩应为 4，加入弧长后应为 5；实现不能用机器精度阈值把自治系统的理论零空间误判为满秩。

该模块返回 `ContinuationResult` 与部分 `OrbitFamily`，并把 `SINGULAR_JACOBIAN`、步长耗尽等结局映射为 ADR 0024 的结构化状态。SPO、LPO 与 Horseshoe 共用这一个平面求解器；Horseshoe 仍是 LPO 的成员分类。Halo 保持现有 XZ 对称 PAL 路径。首版不引入任意周期问题适配器，也不把全周期逻辑塞入硬编码半周期的 `Continuation.natural_continuation()`。

#451 已按上述修订落地首个纵向切片：数值迭代位于 Rust 平面全周期 PAL 内核，Python 侧只保留问题构造与结果解释的族生成接缝 `generate_planar_periodic_family`。当前接缝测试覆盖 L4/L5 的 LPO、L4 的 SPO、LPO 周期转向和内核软失败保留部分族；反向初始方向和更长程分支扫描仍待补齐。

网格搜索继续作为单轨按振幅求解的基线，不承担连续族生成语义。实验超过 110,000 km 仅证明当前方法可走到该范围外，公开振幅范围仍保持 #440 的保守声明。

## 实验方案与验收标准

实验脚本须固定提交、地月 CR3BP 常数、积分器后端、容差、种子和步长，输出 CSV/JSON 与每个轨道成员。每条记录至少包括平动点、族、步号、状态、初始状态、周期、`x0`、振幅、Jacobi 常数、全周期闭合残差、PAL 残差、Newton 次数、步长、雅可比条件数及终止原因。

1. **局部基线**：对 L4/L5 的 SPO/LPO，复现现有 5 步全周期链，再以新 PAL 走同区间；比较成员、振幅、周期、闭合残差和分支方向，不能只判断是否返回 `Orbit`。
2. **LPO 穿折**：从小振幅段向大振幅段继续，记录 `x0`、振幅和 Jacobi 常数是否转向、断链或跳支。改变 PAL 步长和方向复跑，以区分真实折叠、初值失败和分支跳转。
3. **Horseshoe 上限**：从 L4/L5 两侧向大振幅端推进，记录最后成功成员的残差和条件数，并与 110,000 km 网格基线比较。报告必须区分网格限制、PAL 奇异、积分/修正发散和新可达成员；不得将最后成功振幅称为物理上限。

本次 #436 已完成 ADR、平面公式、可重放原型和 L4/L5 的初始定量证据。#451 的生产实现与后续验收边界如下：

- [ ] L4/L5 的 SPO、LPO 在两个初始方向均有小规模端到端验证；
- [ ] LPO 周期转向区的长程试验改用不同步长复跑，区分数值步长敏感性与真实分支形状；
- [ ] 新模块测试完整六维闭合、平面约束、Jacobi 漂移、成员连续性和结构化失败；
- [ ] 新路径返回 `ContinuationResult`/`OrbitFamily`，既有 Halo PAL 与 `Continuation` 测试无回归；
- [ ] 只有系统扫描确认后才另行修改 Horseshoe 的公开范围、默认值、API 模型和端到端测试。

## 数值实验结果（2026-08-16）

在通过 `make setup && make dev` 配置 CSPICE、内核和 Rust 扩展后，使用临时脚本 `scripts/research_issue_436_full_period_pal.py`（使命完成后已移除，可在下述固定提交 `14025914956faa90a1d3e24019db5ca6c33647af` 的 git 历史中取得）做了可复现实验。实验固定提交 `14025914956faa90a1d3e24019db5ca6c33647af`、地月 CR3BP 质量比 `0.01215058560962404`、L4、小振幅线性化种子、归一化弧长步长 `0.01`，结果保存为：

- [`issue-436-spo-pal-local-baseline.json`](issue-436-spo-pal-local-baseline.json)：SPO 5 步，最大完整平面闭合误差 `1.58e-11`，最大 Jacobi 漂移 `1.33e-15`。
- [`issue-436-lpo-pal-long.json`](issue-436-lpo-pal-long.json)：L4 LPO 60 步，几何振幅度量从 586 km 增至 238,833 km，最大完整平面闭合误差 `3.30e-10`，最大 Jacobi 漂移 `3.11e-15`。
- [`issue-436-l5-lpo-pal-extended.json`](issue-436-l5-lpo-pal-extended.json)：L5 LPO 20 步，越过 110,000 km 到 138,526 km，最大完整平面闭合误差 `8.17e-10`，最大 Jacobi 漂移 `1.78e-15`。

L4 LPO 在第 9 至 13 步出现周期转向，但 PAL 链没有断裂；L5 的对称分支也越过当前搜索声明范围。有效秩按相对奇异值阈值 `1e-8` 记录为闭合加相位 `4`、增广系统 `5`。这支持“平面全周期 PAL 作为正式方法”的设计，但不构成新的物理振幅上限，也没有覆盖两个初始方向。对应设计决策见 [ADR 0028](../adr/0028-planar-triangular-full-period-pal.md)。

## 开放问题

1. 反向初始切向量及不同弧长步长是否仍沿同一 L4/L5 分支通过周期转向区？当前证据只覆盖 `x0` 初始减小、`ds=0.01`。
2. 当前 SPO/LPO 的三残差固定 `x0` 修正是否保留为单轨设计基线，还是改由新全周期闭合器统一实现？这属于 #428 的实现取舍，不改变 ADR 0028 的族生成方法。
3. 三维非对称周期族的独立闭合约束和相位规范是什么？在出现第二个实际调用方前，不提前泛化平面模块。
4. PAL 若扩大可达范围，何时、以何种可复现实验证据修改 #440 的 110,000 km 声明边界？
5. #428 仍需决定 `OrbitFamily` 统一结果、`n_orbits` 语义与 Lissajous 的结果类型；#436 只提供三角周期族的算法依据。[#428 分诊评论](https://github.com/cislunarspace/e2m2e/issues/428#issuecomment-5305368148)

## Git 历史来源

- [#436：原始任务](https://github.com/cislunarspace/e2m2e/issues/436)
- [#428：多族 Facade 与分诊实验](https://github.com/cislunarspace/e2m2e/issues/428)
- [#435：Horseshoe 范围失配](https://github.com/cislunarspace/e2m2e/issues/435)
- [#440：收紧 LPO/Horseshoe 范围](https://github.com/cislunarspace/e2m2e/pull/440)
- [`5d89693`：LPO/Horseshoe 初始实现](https://github.com/cislunarspace/e2m2e/commit/5d89693caf2fa3f76e29f4ae192d58f375f5f12d)
- [`9dc7860`：SPO 初始实现](https://github.com/cislunarspace/e2m2e/commit/9dc7860964de4bb7fb234db7708d1787eb2db67f)
- [`52479d1`：逐族修正/延拓测试骨架](https://github.com/cislunarspace/e2m2e/commit/52479d14443f7c1fb490f85ed42569775f644be3)
- [`1e6d3e1`：Halo PAL 折叠点回归修复](https://github.com/cislunarspace/e2m2e/commit/1e6d3e1759c9ccf21f0ec7ac3e481bda50cf8509)
