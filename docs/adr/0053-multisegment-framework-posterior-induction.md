# ADR 0053: 通用多段参数优化框架——后置归纳，三个具体模块交付后一次性抽象回迁

**状态**：已采纳
**日期**：2026-09-27
**相关 issue**：#720（本决策 issue）、#718（迁移总纲地图）、#725（SF conic 档）、#727（SF 星历档）、#723（MGA 连接器）、#726（段框架）
**相关**：ADR 0050（SF 双档保真度，段框架推迟到 #726）、ADR 0005（TwoLevelMultipleShooting 撤销先例）、ADR 0030（数值只在 Rust）、ADR 0024（状态契约）

## 背景

Issue #720 提出通用多段参数优化框架（leg/node 抽象 + 连续性约束组装 +
打靶/minimax/NLP）何时建、以什么角色存在，三个选项：

- **A. 后置归纳**：先交付 SF（conic → 星历）与 MGA 具体模块，共性沉淀后
  一次性抽象成框架并单次回迁。
- **B. 先行奠基**：先建 segment 抽象层，具体模块作为实例化（Copernicus
  原始开发路线）。
- **C. 不建框架**：各模块独立，共享工具函数下沉 solver 层。

代码事实（ADR 0050 代码审计 + 本裁决全量 grep 复核，2026-09-27）：

- `e2m2e/algorithm/` 无任何 Sims-Flanagan / MGA / leg / node / matchpoint
  抽象或实现；SF 与 MGA 代码均未落地，无既有结构可沿用，也无冲突需要
  收敛。
- ADR 0005 的 `TwoLevelMultipleShooting` 是"零实例先行固化接口"的仓库
  先例，2026-08-13 已撤销（实现整体删除，设计链统一到 Rust multiple
  shooting），独立算法安排无消费方存活。
- ADR 0050 已裁决 SF 问题结构（leg / 节点 / 匹配点 / 连续性约束）一次
  建对，并把段框架整体推迟到 #726；逐 leg 混档问题同批移交 #726。

## 决策

1. **选 A：后置归纳。** 先交付 SF（conic → 星历）与 MGA 具体模块，共性
   沉淀后一次性抽象成通用框架并单次回迁；不留两套。
2. **抽象触发点**：#725（SF conic）、#727（SF 星历）、#723（MGA）三个
   issue 全部交付完成后，#726 方可启动。SF 两档本身就是"同一结构换传播
   内核"的第一次归纳，MGA 补充借力约束形态；三者到齐后共性才有足够实例
   照亮。
3. **回迁范围**：仅日心多段新模块（SF 与 MGA）。Earth-Moon CR3BP 低推力
   栈（LowThrustShooting / LowThrustCollocation / qlaw）、
   MultiImpulseTransfer、DRO→RO NLP 保持现状，不纳入。
4. **框架职责：只做问题组装**——leg 切分、节点变量、matchpoint 连续性
   约束、雅可比拼装。NLP 求解复用既有
   `e2m2e/algorithm/transfer/nlp_scipy.py` / `nlp_copt.py`，Rust 只承担
   缺陷与雅可比数值（ADR 0030）；框架不内置求解器分发。ADR 0050 移交
   #726 的逐 leg 混档问题仍由 #726 在此职责边界内处理。
5. **代码归属**：#726 动工时框架放 `e2m2e/algorithm/transfer/` 新子包；
   `e2m2e/algorithm/solver/` 保持纯校正器定位，不接收问题组装代码。
6. **规范术语**：leg（段）、node（控制节点）、matchpoint（匹配点，含其
   连续性约束残差组装），与 ADR 0050 及 Copernicus 文献
   （IAC-06-C1.4.04）一致，三条术语已录入 CONTEXT.md。`segment` 保留给
   既有专用用途（LowThrustSegment、segmented_shooting）。#720/#726 标题
   中的 "segment/node" 用词不改历史 issue，新文档一律用规范术语。
7. **回迁验收判据（grep 可查）**：#726 完成后，SF 与 MGA 模块内不存在
   各自的 leg 切分 / matchpoint 残差组装实现（grep `matchpoint` 及连续
   性约束残差函数名只命中框架子包），两者均实例化框架。

## 理由

1. **A 与现行决策链一致**：ADR 0050 已把段框架推迟到 #726，本决策把
   "推迟"具体化为可执行的触发点与回迁判据，不新增裁决方向。
2. **B 有实际成本的仓库先例**：ADR 0005 在零实例时固化了独立算法接口，
   最终随实现整体删除；三个模块的共性尚未互相照亮时先建抽象，抽象返工面
   与后置归纳的回迁面同量级，却先付了一遍设计成本。
3. **C 等于隐式框架**：不建框架意味着 leg/matchpoint 组装逻辑在 SF 与
   MGA 各写一遍，结构漂移无人守护；决策 4 的组装职责就是 C 试图下沉的
   共享代码，收拢为显式子包比散落的工具函数可审计。
4. **框架不内置求解器分发**：求解入口已经分层存在（nlp_scipy /
   nlp_copt，Rust 数值内核）；组装与求解的接缝是"问题（缺陷/雅可比）"，
   恰好是既有 NLP 层的输入契约，框架重复分发层只会制造第二份求解编排。
5. **`solver/` 保持纯校正器**：问题组装（变量布局、约束拼装）与校正器
   （给定问题迭代收敛）变更节奏不同；混放会让 solver 层继承 transfer 层
   的演化压力。

## 验证

- 触发点由依赖边守护：#726 blocked_by #725、#727、#723（GitHub 原生
  issue 依赖，实时闸门，前置 issue 关闭即自动解除）。
- 判据 7 在 #726 回迁 PR 中机械可查：`git grep matchpoint` 在
  `e2m2e/algorithm/transfer/` 下只命中框架子包与 SF/MGA 的实例化调用。

## 影响

- #726 的启动条件由本决策触发点约束，依赖边随本决策落地补齐。
- CONTEXT.md 新增段（leg）、控制节点（node）、匹配点（matchpoint）三条
  术语。
- 本决策无面向调用方的行为变化，`CHANGELOG.md` 不更新；#726 进代码时
  再记。

## 备选方案

- **B 先行奠基**：Copernicus 原始路线在零实例时固化抽象层，与 ADR 0005
  撤销先例的教训冲突；否决。
- **C 不建框架**：leg/matchpoint 组装在 SF 与 MGA 各写一遍的隐式框架，
  结构漂移无人守护；否决。
- **回迁范围纳入 CR3BP 低推力栈与既有专用路径**：
  LowThrustShooting / LowThrustCollocation / qlaw 与 MultiImpulseTransfer、
  DRO→RO NLP 已有专用结构且行为已验证，强行纳入把单次回迁扩大为全
  transfer 层重构，风险与收益不成比例；否决，保持现状。
