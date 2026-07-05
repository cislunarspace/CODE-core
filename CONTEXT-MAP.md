# 上下文地图

本仓库是单上下文仓库，所有领域知识放在根目录。

| 上下文 | 位置 | 覆盖 |
|---------|----------|--------|
| `e2m2e` | `./CONTEXT.md` | 整个 e2m2e 库 |

## 架构层次

```text
core/           → 基础层 — CR3BP 系统、轨道数据结构、物理模型
    ↓
algorithms/     → 数值求解 — 微分修正、延拓、稳定性、多重打靶
    ↓
transfer/       → 转移设计 — 网格搜索、NLP 优化
    ↓
visualization/  → 绘图 — 轨道族、转移轨迹

横切：           mbse/ — 由 scripts 触发的文档产物（架构图、需求追溯、数据模型导出），
                            非运行时架构组件（运行期不应被依赖）
```

## 消费领域文档的 skills

下列 skills 在探索前先读 `CONTEXT.md`：

- `/improve-codebase-architecture`
- `/diagnose`
- `/tdd`

下列 skills 读 `docs/adr/` 了解架构决策：

- `/grill-with-docs` — 撰写 ADR
- 上述 skills 在提出变更前都会先查阅现有 ADR
