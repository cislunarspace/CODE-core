# 稳定性与分岔分析能力（monodromy / Floquet / 沿族分岔）

e2m2e 不提供轨道稳定性与分岔分析能力。`e2m2e.algorithm.stability`——单轨的 `StabilityAnalysis`（`analyze_bifurcation` / `classify_orbit` / `full_analysis`）、族级的 `detect_bifurcation_in_family`（沿族穿越判据定位与精化，PR #704 交付），连同其闭值集 `StabilityType`／`BifurcationType` 与结果类型 `FamilyBifurcationScan`／`FamilyBifurcationPoint`／`BranchJump`／`MemberAnalysisFailure`——已整体删除：接口层不留、算法层也不留。

## 为什么不在范围内

**能力自落地起零生产消费者。** 删除前实测：`e2m2e/api/` 对 stability 与 bifurcation 零引用，任何调用链（进程内 import、MCP、CLI、stdio sidecar）都取不到它；全仓引用只有三类——测试（`tests/algorithm/stability/`）、一份手工标定脚本、`e2m2e/mbse/` 的追溯元数据。这不是“还没人用”，而是“没有任何路径能到达它”。

**接口层准入的任一款都不成立。** ADR 0043 决策 6 要求“任务级能力，或其内容被既有响应字段引用且没有既有工具能供给”。稳定性/分岔属分析能力（决策 4 原文即如此归类 `orbit_stability`），而 `design_orbit` 路径不产单值矩阵——`compute_state_transition_matrix` 在 `e2m2e/` 内只被本模块与 `family/axial_initial_guess.py` 调用；把它挂进设计响应等于为每次设计多算一次整周期 STM 积分，为无消费者的字段付常态成本。其闭值集一旦跨边界，按 ADR 0044 还要进术语清单并受双向锁定测试长期守护。

**“保留为算法层内部能力”同样是净成本。** 无调用方的模块不被任何真实使用检验，却持续承担维护、评审与文档面；`e2m2e.data.templates` 里那份平行死枚举（`StabilityLabel`／`BifurcationLabel`）已经证明同一概念的第二份定义会漂移。删净比登记“内部能力”更诚实。

**它唯一的实际产出是一次负结论，删除不撤回该结论。** 平面 DRO 族不存在垂直临界点（`.out-of-scope/three-dimensional-dro.md`）由 `scripts/calibrate_dro3d_vertical_critical.py` 标定得出；该脚本保留并改为自含（`vt` 与稳定性指数由同一单值矩阵的分块迹直接给出），证据链仍可复跑。

## 复核路径

本决定由维护者裁定（#687，2026-09-26）。定位变化时按以下路径复议：

- **触发条件**：出现具体消费场景（GUI 需报稳定性标签、站保需按稳定性筛参考轨道、外部调用方要求“这条族在哪个族参数上分岔”），或 `design_orbit` 因其他理由本来就要算单值矩阵（那时边际成本归零）；
- **重新实现，而非恢复历史代码**：先按 ADR 0043 决策 2/3 的分家先例定类归属、按决策 6 定工具面准入；判定反转的标志是“零消费者”变成“有消费者”，不是能力缺口本身；
- **仅存的相关逻辑**：`e2m2e/algorithm/family/axial_initial_guess.py` 的 z-vz 块半迹扫描（Axial 族种子定位，自含、从不属于本能力）；`scripts/calibrate_dro3d_vertical_critical.py`（自含分块迹口径的平面 DRO 垂直临界点标定）；
- **不得以此记录为由**：拒绝站保的拟周期参考轨道语义（`.out-of-scope/invariant-torus-qpit.md`）或 NominalOrbit 契约变更（`.out-of-scope/nominal-orbit-floquet-precompute.md`）——三者边界独立；
- **验证纪律**：ADR 0013 按物理定义验收。单值矩阵的分块迹口径是精确等式（面外对 `ν_out = M[2,2] + M[5,5]`；面内两根满足 `ν² − tr(A_in)·ν + (m_in − 2) = 0`，`A_in` 为 x/y/vx/vy 的 4×4 辛块），复议时不得以特征值排序与“平凡对剔除”这类启发式替代。

## 过往请求

- [#687](https://github.com/cislunarspace/CODE-core/issues/687)：分岔分析的接口层归属与工具面准入（#655 拆分）—— 维护者裁定：能力整体删除，接口层与算法层都不留
- [#688](https://github.com/cislunarspace/CODE-core/issues/688)：族级分岔定位与精化：沿族跟踪 Floquet 乘子、按穿越判据报告 —— 曾以 `Completed` 交付（PR #704），实现随本决定删除
- [#655](https://github.com/cislunarspace/CODE-core/issues/655)：新增三维 DRO 生成、不变环面构造与分岔分支切换（epic 第 3 项后半）
