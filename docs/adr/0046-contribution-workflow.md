# ADR 0046: 贡献流程——模板、type/kind/area 标签与 Project 流水线

**状态**：已采纳（已实施）
**日期**：2026-09-04
**相关**：README.md（贡献节）、CONTRIBUTING.md、`.github/ISSUE_TEMPLATE/`、
`.github/PULL_REQUEST_TEMPLATE.md`、`.github/dependabot.yml`。

## 背景

仓库对外开放贡献（接受外部 PR），但流程缺位：README 贡献节引用的文件
不存在；Issue 与 PR 模板仍是早期英文风格，与 2026-09 的全面中文化相悖；
标签是 GitHub 默认散装集，分不出改动意图与影响领域；Issue 没有承载推进
状态的面板。

另一处参照是 deepseek-harness 的贡献管理体系（Issue 五类模板、
kind/area 两轴标签、七态 Project 流水线、policy 自动化）。结构经过实战
检验，但有两点不适配：该体系「暂不接受外部 PR」的立场与本仓库相反；
其硬校验依赖组织仓库才有的原生 Issue Type，以及 GitHub App 与策略脚本
——本仓库是个人账号仓库，当前体量也撑不起这套维护成本。

## 决策

### 1. 立场：接受外部贡献，Issue 先行

修复与功能类 PR 必须关联同仓库 Issue（`Fixes #NN` 合并即自动关闭）；
使用问题与想法探讨走 Discussions，不占用 Issue。

### 2. Issue 五类，由模板承载

Bug / Feature / Idea / Research / Task 五套中文模板；正文一句话说明
核心，细节收进默认收起的折叠区；标题写一句中文结果句，元信息不进标题。
GitHub 原生 Issue Type 是组织仓库功能，个人仓库没有，故每套模板自动打
`type/*` 标签（type/bug、type/feature、type/idea、type/research、
type/task），使类型在创建之后仍机器可查。

### 3. PR 两轴标签

`kind/*` 恰好一个（闭集：feature、bug-fix、doc、testing、cleanup、
dependency），记录主导意图；`area/*` 至少一个（开放集，命名持久领域），
记录实质影响。两轴独立，不混用；无命名空间的同义标签一律不保留。
打标是维护者职责。

### 4. Project「e2m2e Issue Management」七态流水线

Inbox → Backlog → Ready → In progress → In review → Done / No action。
终态与 Issue 关闭原因一一对应：Done↔Completed，No action↔Not planned；
重开的 Issue 回 Inbox。字段仅 Priority（P0–P3，可不设）与 Start Date。

### 5. 暂不引入 policy 自动化

deepseek-harness 用策略脚本与 workflow 硬校验上述规则，另配专用凭据。
本仓库暂不引入：规则由模板与 CONTRIBUTING 承载，维护者手工执行；PR 量
增大或规则漂移出现时，再评估引入。

## 备选

- **沿用 GitHub 默认标签**（bug、enhancement……）：意图与领域混在一个
  平面上，查询语义不清，且与仓库全面中文化冲突。
- **Issue 也打 kind/\* 标签**：原生 Type（此处为 type/*）已承担分类，
  复制一份产生漂移；deepseek-harness 在有原生 Type 的前提下同样禁止。
- **现在就上 policy 自动化**：硬校验需要专用凭据（PAT 或 App）与脚本
  测试，当前 PR 量支撑不起；规则文档先行的成本更低，自动化随时可补。
- **README 继续承担贡献说明**：README 定位产品入口；流程细节归
  CONTRIBUTING，GitHub 会在新建 Issue/PR 页自动展示该文件。

## 后果

### 新增

- CONTRIBUTING.md；五套中文 Issue 模板与 config.yml（选择器页指向
  Discussions）；中文 PR 模板；type/*、kind/*、area/* 标签体系；
  Project 面板与 Priority、Start Date 两个字段。

### 变更

- 英文 bug_report/feature_request 模板删除。
- dependabot 标签从 `dependencies`+`needs-triage` 改为
  `kind/dependency`+`area/infra`+`needs-triage`。
- 默认标签 bug、enhancement、documentation、wontfix、invalid、question
  删除，语义分别由 type/*、kind/*、Project 终态与 needs-info 承担。

### 不变

- AGENTS.md 的写作与 commit 约定（CONTRIBUTING 引用而不重复）；
  既有分诊标签 ready-for-agent、ready-for-human、needs-triage、
  needs-info；duplicate、good first issue、help wanted 等通用标签。

### 代价

- 规则无强制力：标签与 Project 状态靠维护者手工维护，外部 PR 不合规时
  只能人工指出。这是决策 5 的直接后果，也是重新评估自动化的触发条件。
