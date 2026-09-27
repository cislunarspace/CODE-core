# Domain Docs

工程技能在探索代码库时如何使用本仓库的领域文档。

## 探索前先读

- 仓库根目录的 **`CONTEXT.md`**：e2m2e 的统一领域术语表，记录规范术语、测试类目、五层架构与语言约定。
- **`docs/adr/`**：读与你即将工作的区域相关的 ADR。ADR 是决策快照，涉及架构分层、数据来源、容差口径、验收口径等硬约束时必读。

本仓库是单一上下文仓库（根目录无 `CONTEXT-MAP.md`，没有按子目录划分的上下文级 `docs/adr/`）。

如果这些文件不存在，**静默继续**。不要标记缺失，不要主动建议创建。当术语或决策真正确立时，`/domain-modeling` 技能（通过 `/grill-with-docs` 或 `/triage` 到达）会按需创建它们。

## 文件结构

```
/
├── CONTEXT.md                        ← 唯一领域术语表
├── docs/adr/                         ← 全仓库 ADR，编号四位递增
│   ├── 0011-five-layer-architecture.md
│   ├── 0021-test-suite-functional-categories.md
│   └── 0055-acceptance-oracle-taxonomy.md
├── e2m2e/                            ← data / algorithm / api / tools / mbse
├── crates/                           ← Rust 数值层
└── tests/                            ← 按分层镜像
```

## 按术语表用词

当你的输出提到领域概念时（issue 标题、重构提案、假设、测试名），按 `CONTEXT.md` 术语表用词。不要漂移到术语表明确避免的同义词。

如果你需要的概念还不在术语表中，这是一个信号：要么你在发明项目不使用的语言（重新考虑），要么存在真实的空白（记录下来留给 `/domain-modeling`）。

## 标注 ADR 冲突

如果你的输出与已有 ADR 矛盾，明确标出，而非默默覆盖：

> _与 ADR 0027（系统动力学分离）矛盾，但值得重新讨论，因为……_

ADR 不会被静默覆盖：决策变化时在原文追加修订小节或新开递增 ADR，并在旧文标注取代关系。
