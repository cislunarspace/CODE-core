# Issue tracker: GitHub

本仓库的 issue 和 PRD 存放在 GitHub Issues 中。所有操作使用 `gh` CLI。

## 约定

- **创建 issue**：`gh issue create --title "..." --body "..."`。多行正文用 heredoc。
- **读取 issue**：`gh issue view <number> --comments`，用 `jq` 过滤评论，同时获取标签。
- **列出 issue**：`gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'`，按需加 `--label` 和 `--state` 过滤。
- **评论 issue**：`gh issue comment <number> --body "..."`
- **添加 / 移除标签**：`gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **关闭**：`gh issue close <number> --comment "..."`

从 `git remote -v` 推导仓库，`gh` 在 clone 内运行时自动识别。

## Pull requests 作为分诊渠道

**PR 作为请求渠道：否。** _（如果本仓库将外部 PR 视为功能请求，改为 `是`；`/triage` 会读取此标记。）_

设为 `是` 时，PR 与 issue 走相同的标签和状态，使用 `gh pr` 等价命令：

- **读取 PR**：`gh pr view <number> --comments`，`gh pr diff <number>` 看 diff。
- **列出待分诊的外部 PR**：`gh pr list --state open --json number,title,body,labels,author,authorAssociation,comments`，只保留 `authorAssociation` 为 `CONTRIBUTOR`、`FIRST_TIME_CONTRIBUTOR` 或 `NONE` 的（去掉 `OWNER`/`MEMBER`/`COLLABORATOR`）。
- **评论 / 打标签 / 关闭**：`gh pr comment`、`gh pr edit --add-label`/`--remove-label`、`gh pr close`。

GitHub 的 issue 和 PR 共享编号空间，所以 `#42` 可能是其中任一，用 `gh pr view 42` 确认，回退到 `gh issue view 42`。

## 当技能说"发布到 issue tracker"时

创建一个 GitHub issue。

## 当技能说"获取相关 ticket"时

运行 `gh issue view <number> --comments`。
