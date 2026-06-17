# 领域文档

工程类 skill 在探索本仓库时应如何消费领域文档。

## 探索前先读这些

- 仓库根目录的 **`CONTEXT.md`**，或者
- 仓库根目录的 **`CONTEXT-MAP.md`**（如果存在）——它指向每个上下文的 `CONTEXT.md`，只读与当前主题相关的即可。
- **`docs/adr/`** —— 读取与你即将修改的区域相关的 ADR。多上下文仓库中，还要检查 `src/<context>/docs/adr/` 里与上下文相关的决策。

如果上述文件不存在，**静默继续**。不要标记缺失，也不要建议预先创建。生产者 skill（`/grill-with-docs`）会在术语或决策真正被解决时懒创建它们。

## 文件结构

单上下文仓库（最常见）：

```
/
├── CONTEXT.md
├── docs/adr/
│   ├── 0001-event-sourced-orders.md
│   └── 0002-postgres-for-write-model.md
└── src/
```

多上下文仓库（根目录存在 `CONTEXT-MAP.md`）：

```
/
├── CONTEXT-MAP.md
├── docs/adr/                          ← 全系统级决策
└── src/
    ├── ordering/
    │   ├── CONTEXT.md
    │   └── docs/adr/                  ← 与 ordering 上下文相关的决策
    └── billing/
        ├── CONTEXT.md
        └── docs/adr/                  ← 与 billing 上下文相关的决策
```

## 使用术语表的词汇

当你的输出中命名领域概念（issue 标题、重构提案、假设、测试名）时，请使用 `CONTEXT.md` 中定义的术语。不要漂移到期 glossary 明确避免的同义词。

如果你需要的概念尚未出现在 glossary 中，这是一个信号——要么你在发明项目不用的语言（请重新考虑），要么确实存在一个缺口（请记给 `/grill-with-docs`）。

## 标记 ADR 冲突

如果你的输出与现有 ADR 矛盾，请显式指出，而不是默默覆盖：

> _与 ADR-0007（event-sourced orders）矛盾——但值得重新讨论，因为…_
