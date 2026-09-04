# 贡献指南

感谢关注 e2m2e。本仓库接受贡献：报告 Bug、提出功能建议、改进文档与测试、提交代码，都欢迎。

参与前请读完本指南：它说明 Issue 怎么提、PR 怎么交、贡献如何被分类和推进。技术讨论就事论事，对事不对人；提交即表示你同意以 Apache 2.0 许可（与仓库一致）授权你的贡献。

## 提 Issue

提之前先搜索现有 Issue，避免重复。Issue 分五类，各有一套模板，按要提交的内容选择：

| 类型 | 用途 |
|---|---|
| Bug | 记录现有预期行为的失效 |
| Feature | 新增或有意改变可观察行为 |
| Idea | 尚未承诺实施、但具有行动可能的想法 |
| Research | 形成结论、证据或决策 |
| Task | 明确的非 Feature、非 Bug 工作 |

写法上有两条约定：

- **标题写一句中文行动或结果句**，说明要修复或达成什么。不要带类型、优先级等前缀——类型由你选的模板自动打的 `type/*` 标签承载，推进状态由 Project 字段承载。
- **正文一句话说清核心，细节收进模板自带的折叠区**（复现步骤、验收条件等），保持正文一眼可读。

使用问题、想法探讨与一般性讨论走 [Discussions](https://github.com/cislunarspace/e2m2e/discussions)，不占用 Issue。维护者会尽快给 Issue 归型并排入 Project（见下）；需要补充信息时会打上 `needs-info` 标签。

## 提 Pull Request

1. **先开 Issue**：修复与功能类 PR 必须关联一个同仓库 Issue，在描述中写 `Fixes #NN`（合并即自动关闭对应 Issue），仅关联不关闭写 `Related to #NN`；纯文档小修可不挂。
2. **Fork 并建分支**，分支名建议 `fix/<简述>` 或 `feat/<简述>`。
3. **本地验证**：`make test` 与 `make check` 通过；新增或改变行为要有对应测试。
4. **开 PR 指向 `master`**，按模板填写。一个 PR 只做一件事；commit message 用中文 conventional commits（如 `fix(catalog): 修正……`）。
5. **CI 必须绿**：lint 与 test 是必过检查。评审通过后由维护者合并。

## 标签体系

标签回答两个独立的问题：改动是什么意图（`kind/*`），实质影响哪个领域（`area/*`）。打标是维护者的职责，贡献者不必操心。

**`kind/*`——PR 恰好一个**，记录主导意图（顺带的测试或文档不改变主导意图）：

| 标签 | 含义 |
|---|---|
| `kind/feature` | 新增或有意改变行为 |
| `kind/bug-fix` | 修正错误行为 |
| `kind/doc` | 文档为主要意图 |
| `kind/testing` | 只动测试或测试基建，不改产品行为 |
| `kind/cleanup` | 不改行为地维护或简化实现与流程 |
| `kind/dependency` | 更新依赖，无其他主导意图 |

**`area/*`——PR 至少一个**，命名实质影响的持久领域：`area/api`（接口层）、`area/algorithm`（算法编排）、`area/numerical`（Rust 数值层）、`area/catalog`（轨道库）、`area/mcp`、`area/cli`、`area/data`（星历、常数、数据资产）、`area/tools`（日志、可视化辅助）、`area/mbse`、`area/docs`、`area/infra`（构建、CI、发布、脚本）。

领域清单是开放的：现有描述确实盖不住新的持久领域时，维护者新建 `area/<kebab-case>`；不为单个 PR、临时事项或个人建域。

Issue 不用 kind 标签：GitHub 的原生 Issue Type 是组织仓库功能，本仓库（个人账号）没有，分类由五套模板创建时自动打的标签承担——`type/bug`、`type/feature`、`type/idea`、`type/research`、`type/task`；`area/*` 对 Issue 可选。`ready-for-agent`、`ready-for-human`、`needs-triage`、`needs-info` 是分诊标签，记录一项工作交给谁、卡在谁那里，与两轴标签正交。

## Project 流水线

所有 Issue 进入 Project「[e2m2e Issue Management](https://github.com/users/cislunarspace/projects/1)」，按状态推进：

| 状态 | 含义 |
|---|---|
| Inbox | 新到，待分诊 |
| Backlog | 已确认，未排期 |
| Ready | 已排期，可开工 |
| In progress | 实现中 |
| In review | 等评审 |
| Done | 完成（对应 Issue 关闭原因 Completed） |
| No action | 不处理（对应关闭原因 Not planned） |

状态与 Issue 开合自动对应：Done 与 No action 是终态，分别要求 Issue 以 Completed 与 Not planned 关闭；重开的 Issue 回到 Inbox。每条 Issue 还带两个字段：`Priority`（P0–P3，可不设）与 `Start Date`（开工日期，由维护者维护）。
