# Issue tracker: GitHub

本仓库的 issue 和规格存放在 GitHub Issues 中。所有操作使用 `gh` CLI。

## 约定

- **创建 issue**：`gh issue create --title "..." --body "..."`。多行正文用 heredoc。
- **AI 贡献标记**：AI 提交的 issue 与 PR 标题以 `[AI Generated][<类型>]` 开头；类型标签见 `CONTRIBUTING.md`（PR 用 `[FEAT]` / `[FIX]` / `[DOC]` / `[TEST]` / `[CLEANUP]` / `[DEP]`，Issue 用 `[FEAT]` / `[BUG]` / `[IDEA]` / `[RESEARCH]` / `[TASK]`）。AI 写的评论首行用 `> **[AI Generated]** 本评论由 AI 完成。` 或 `> **[AI Assisted]** 本评论由 AI 辅助完成。`
- **读取 issue**：`gh issue view <number> --comments`，用 `jq` 过滤评论，同时获取标签。
- **列出 issue**：`gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'`，按需加 `--label` 和 `--state` 过滤。
- **评论 issue**：`gh issue comment <number> --body "..."`
- **添加 / 移除标签**：`gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **关闭**：`gh issue close <number> --comment "..."`

从 `git remote -v` 推导仓库，`gh` 在 clone 内运行时自动识别。

## Pull request 作为分诊渠道

**PR 作为请求渠道：否。** _（如果本仓库将外部 PR 视为功能请求，改为 `是`；`/triage` 会读取此标记。）_

设为 `是` 时，PR 与 issue 走相同的标签和状态，使用 `gh pr` 等价命令：

- **读取 PR**：`gh pr view <number> --comments`，`gh pr diff <number>` 看 diff。
- **列出待分诊的外部 PR**：`gh pr list --state open --json number,title,body,labels,author,authorAssociation,comments`，只保留 `authorAssociation` 为 `CONTRIBUTOR`、`FIRST_TIME_CONTRIBUTOR` 或 `NONE` 的（去掉 `OWNER`/`MEMBER`/`COLLABORATOR`）。
- **评论 / 打标签 / 关闭**：`gh pr comment`、`gh pr edit --add-label`/`--remove-label`、`gh pr close`。

GitHub 的 issue 和 PR 共享编号空间，所以 `#42` 可能是其中任一，用 `gh pr view 42` 确认，回退到 `gh issue view 42`。

## 当技能说"发布到 issue tracker"时

创建一个 GitHub issue。

## 当技能说"获取相关工单"时

运行 `gh issue view <number> --comments`。

## GitHub Project

- Issue 和 PR 都加入仓库约定的 GitHub Project；Project 是工作状态的来源，label 只表达分类、领域或分诊角色。
- 使用 `gh project item-list <number> --owner <owner> --format json` 查询项目项，使用 `gh project item-edit --id <item-id> --project-id <project-id> --field-id <field-id> --single-select-option-id <option-id>` 更新字段。
- 默认状态流转：`Inbox` → `Backlog` → `Ready` → `In progress` → `In review` → `Done` / `No action`。
- `Done` 对应 Issue 以 `Completed` 关闭；`No action` 对应 Issue 以 `Not planned` 关闭；重开的 Issue 回到 `Inbox`。
- `Priority` 使用 `P0`–`P3`，`Start Date` 由维护者维护。具体 Project、字段 ID 和选项 ID 记录在下节。

### 工作流状态迁移

- 新 Issue：加入 Project，设为 `Inbox`。
- 分诊确认但未排期：`Backlog`；可开始：`Ready`。
- 开始实现：`In progress`；创建 PR：`In review`。
- PR 合并并验证完成：Issue 关闭原因为 `Completed`，Project 设为 `Done`。
- PR 关闭但未合并：不自动关闭 Issue 或设为 `No action`，等待维护者决定。

### Project 配置

ID 于 2026-09-23 经 `gh project field-list 1 --owner ouyangjiahong26 --format json` 实测核对。

- Project：`cislunarspace Issue Management`（https://github.com/users/ouyangjiahong26/projects/1），owner `ouyangjiahong26`、number `1`；与 transfer-orbit-design 共用，Repository 字段区分来源。
- Project ID：`PVT_kwHOCpw4xM4BiZm4`
- Status 字段 ID：`PVTSSF_lAHOCpw4xM4BiZm4zhhRz-U`

  | 选项 | option ID |
  | --- | --- |
  | Inbox | `f75ad846` |
  | Backlog | `c63ae0e5` |
  | Ready | `871e53fc` |
  | In progress | `47fc9ee4` |
  | In review | `7b94b6f5` |
  | Done | `98236657` |
  | No action | `58b611df` |

- Priority 字段 ID：`PVTSSF_lAHOCpw4xM4BiZm4zhhR0V8`

  | 选项 | option ID |
  | --- | --- |
  | P0 | `bff420ad` |
  | P1 | `4679013b` |
  | P2 | `380c4288` |
  | P3 | `826b89d3` |

- Start Date 字段 ID：`PVTF_lAHOCpw4xM4BiZm4zhhR0WA`

## Wayfinding 操作

被 `/wayfinder` 使用。**地图**是一个 issue，其下挂**子** issue 作为工单。

- **地图**：一个带 `wayfinder:map` 标签的 issue，承载 Notes / Decisions-so-far / Fog 正文。`gh issue create --label wayfinder:map`。
- **子工单**：一个 issue，作为 GitHub sub-issue 链接到地图（通过 `gh api` 调 sub-issues 端点）。在未启用 sub-issues 的地方，把子工单加到地图正文的 task list 中，并在子工单正文顶部放 `Part of #<map>`。标签：`wayfinder:<type>`（`research`/`prototype`/`grilling`/`task`）。一旦被认领，工单指派给驱动的 dev。
- **阻塞**：GitHub 的**原生 issue 依赖**：规范的、UI 可见的表示。用 `gh api --method POST repos/<owner>/<repo>/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>` 添加一条边，其中 `<blocker-db-id>` 是阻塞方的数字 **database id**（`gh api repos/<owner>/<repo>/issues/<n> --jq .id`，_不是_ `#number` 或 `node_id`）。GitHub 报告 `issue_dependencies_summary.blocked_by`（仅未关闭的阻塞方，即实时闸门）。在依赖不可用的地方，回退到子工单正文顶部的 `Blocked by: #<n>, #<n>` 行。当所有阻塞方都关闭时，工单解除阻塞。
- **前沿查询**：列出地图下未关闭的子工单（`gh issue list --state open`，范围限定到地图的 sub-issues / task list），丢弃任何有未关闭阻塞方（`issue_dependencies_summary.blocked_by > 0`，或 `Blocked by` 行中的未关闭 issue）或有 assignee 的；按地图中的顺序，第一个胜出。
- **认领**：`gh issue edit <n> --add-assignee @me`：会话的第一次写操作。
- **解决**：`gh issue comment <n> --body "<answer>"`，然后 `gh issue close <n>`，然后在地图的 Decisions-so-far 中追加一条上下文指针（gist + 链接）。
