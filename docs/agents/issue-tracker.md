# Issue 跟踪器：GitHub

本仓库的 issue 与 PRD 以 GitHub issue 形式存在。所有操作使用 `gh` CLI。

## 约定

- **创建 issue**：`gh issue create --title "..." --body "..."`。多行正文用 heredoc。
- **读取 issue**：`gh issue view <number> --comments`，可用 `jq` 过滤评论。
- **列出 issue**：`gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'`，可叠加 `--label` 与 `--state` 过滤。
- **评论 issue**：`gh issue comment <number> --body "..."`
- **添加 / 移除标签**：`gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **关闭**：`gh issue close <number> --comment "..."`

`gh` 会自动从 `git remote -v` 推断仓库，因此在 clone 内运行即可。

## 当 skill 说“发布到 issue 跟踪器”

创建一条 GitHub issue。

## 当 skill 说“取回相关工单”

运行 `gh issue view <number> --comments`。
