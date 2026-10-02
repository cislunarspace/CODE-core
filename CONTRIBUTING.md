# 贡献指南

感谢关注 e2m2e。本仓库接受贡献：报告 Bug、提出功能建议、改进文档与测试、提交代码，都欢迎。

参与前请读完本指南：它说明 Issue 怎么提、PR 怎么交、贡献如何被分类和推进。技术讨论就事论事，对事不对人；提交即表示你同意以 Apache 2.0 许可（与仓库一致）授权你的贡献。AI 生成的 issue 与 PR：标题最前面加 `[AI Generated]` 标记（位于类型标签之前），正文首行注明工具。**未正确标记的，不予受理，不开展进一步工作和实施。**

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

- **标题以类型标签开头，后接一句中文行动或结果句**。Issue 用 `[FEAT]` / `[BUG]` / `[IDEA]` / `[RESEARCH]` / `[TASK]`（与所选模板对应）；PR 用意图标签，取标签体系一节统一表中的标题标签列。改动破坏既有调用方时，在类型标签之后加 `[BREAKING]`。优先级、状态等其余元信息不进标题，由 Project 字段承载。标题同时承担一眼可读的职责：扫读清单的人只看标题就该知道这条工单要做什么。
- **正文完整叙述，写给不了解这条改动来龙去脉的协作者**。要回答哪些问题、怎么行文，见下文[正文写作约定](#正文写作约定)一节；开单模板以提问的形式给出了引导。细节不收进折叠区，人写的文字都放在正文可见区。

使用问题、想法探讨与一般性讨论走 [Discussions](https://github.com/cislunarspace/CODE-core/discussions)，不占用 Issue。维护者会尽快给 Issue 归型并排入 Project（见下）；需要补充信息时会打上 `needs-info` 标签。

## 提 Pull Request

1. **先开 Issue**：修复与功能类 PR 必须关联一个同仓库 Issue，在描述中写 `Fixes #NN`（合并即自动关闭对应 Issue），仅关联不关闭写 `Related to #NN`；纯文档小修可不挂。
2. **Fork 并建分支**，分支名建议 `fix/<简述>` 或 `feat/<简述>`。
3. **本地验证**：`make test` 与 `make check` 通过；新增或改变行为要有对应测试。
4. **开 PR 指向 `master`**，标题带意图标签（见上方标题约定），正文按模板内联的问题集与[正文写作约定](#正文写作约定)完整填写。一个 PR 只做一件事；commit message 用中文 conventional commits（如 `fix(catalog): 修正……`），标题不使用破折号（——）。
5. **CI 必须绿**：lint 与 test 是必过检查。评审通过后由维护者合并。
6. **正文过关再等合并**：评审按写作约定核对正文是否回答完整、行文是否可读，缺项的 PR 由作者在合并前补正。


## 正文写作约定

Issue、PR、评论、commit message 和 CHANGELOG 都是写给协作者读的，不是写给程序或作者自己读的。默认读者了解本仓库所在的领域，但不了解这条改动的来龙去脉：他应该只读正文就能明白发生了什么、为什么、自己接下来能做什么。

正文要回答一组固定的问题，而不是堆砌参数和决策名。PR 正文回答：这个改动解决什么问题，为什么值得做；为什么采用这个方案，否决过哪些替代做法；实际改了什么，按协作者阅读代码的顺序讲清楚来龙去脉；怎么验证的，跑了哪些命令和测试，结果是什么；协作者从哪里继续读，关键文件和后续工作在哪。Issue 正文回答：现状是什么，期望是什么，为什么重要，已经做过哪些排查或有过哪些想法。模板以提问的形式给出了这组问题，逐段回答即可，不必把问题本身抄进正文。

行文用平实的中文陈述句，一段话说完一个完整的意思，段落之间按逻辑承接。能写成连贯段落的就不要拆成小标题加短句；小标题连篇而每段只有一两句话，读者得到的是碎片而不是理解。内容本身就短时直说其事，不要在开头预告要说什么，也不要在结尾总结说过什么。

少用特殊符号。反引号只用于需要逐字精确的东西，比如符号名、命令、文件路径和代码片段；不用加粗或引号做叙述性强调；括号只保留真正插入语的一处，不连续嵌套，需要展开的意思写成从句。术语第一次出现时用一句话解释，不靠引号或加粗标注。

细节不收进折叠区。人写的文字全部放在正文可见区；折叠区只用于原始机器输出，比如长日志、命令输出和大段数据表。

commit message 标题保持中文 conventional commits 的单行格式；标题之外写 body 时，body 必须是完整段落，讲清楚做了什么和为什么，不用条目罗列。改动很小时可以只有标题。

AI 生成的评论遵守同样的行文约束。篇幅按内容需要，短事短说；完整性指的是逻辑完整，不是字数。

CHANGELOG 条目保留现有的条目形状：一条变化一个条目，粗体短标题加冒号，随后用完整的中文句子把变化、动因、边界和用法讲成连贯叙述；参数与符号名按上文的符号边界使用反引号，需要展开的意思用从句而不是嵌套括号；issue 与 ADR 引用保持在段尾。分节取上文标签体系统一表里的 CHANGELOG 分节列，按改动的主导意图归组，破坏性变更单独放进 `BREAKING` 分节、不在其他分节重复。不写版本导语。已发布条目与其分节是不可变历史，保持写成时的口径；`Unreleased` 节里旧风格的存量条目，在最近一次发布前统一按本约定回改。

这些约定由评审把关：评审者按问题集核对 PR 正文是否回答完整，按行文约束提出修改意见。存量的 open PR 与 issue 不因本约定回改，但 open PR 在合并前由作者补正。写作约定没有机器检查；标题标签、关联 Issue 等原有硬性规则仍按原方式执行。

## 标签体系

标签回答两个独立的问题：改动是什么意图（`kind/*`），实质影响哪个领域（`area/*`）。打标是维护者的职责，贡献者不必操心。

**`kind/*`——PR 恰好一个**，记录主导意图（顺带的测试或文档不改变主导意图）。这一套分类贯穿 PR 标签、标题标签、commit 前缀、CHANGELOG 分节与发布说明小标题，各处取用同一张表，不另立词汇：

| kind 标签 | 标题标签 | commit 前缀 | CHANGELOG 分节 | 发布说明小标题 | 含义 |
|---|---|---|---|---|---|
| `kind/feature` | `[FEAT]` | `feat` | `FEAT` | `FEAT` | 新增或有意改变行为 |
| `kind/bug-fix` | `[FIX]` | `fix` | `FIX` | `FIX` | 修正错误行为 |
| `kind/cleanup` | `[CLEANUP]` | `refactor`、`chore` | `CLEANUP` | `CLEANUP` | 不改行为地维护或简化实现与流程 |
| `kind/dependency` | `[DEP]` | `build`、`ci` | `DEP` | `DEP` | 更新依赖，无其他主导意图 |
| `kind/doc` | `[DOC]` | `docs` | `DOC` | `DOC` | 文档为主要意图 |
| `kind/testing` | `[TEST]` | `test` | `TEST` | `TEST` | 只动测试或测试基建，不改产品行为 |

破坏性变更是正交属性，不是第七种 kind：它描述改动对既有调用方的影响，任何 kind 都可能带。标注一律用 `[BREAKING]`，写在 PR 标题的类型标签之后；CHANGELOG 与发布说明里放进 `BREAKING` 分节（发布说明的小标题用大写 `BREAKING`），不在其他分节重复列。

**`area/*`——PR 至少一个**，命名实质影响的持久领域：`area/api`（接口层）、`area/algorithm`（算法编排）、`area/numerical`（Rust 数值层）、`area/catalog`（轨道库）、`area/mcp`、`area/cli`、`area/data`（星历、常数、数据资产）、`area/tools`（日志、可视化辅助）、`area/mbse`、`area/docs`、`area/infra`（构建、CI、发布、脚本）。

领域清单是开放的：现有描述确实盖不住新的持久领域时，维护者新建 `area/<kebab-case>`；不为单个 PR、临时事项或个人建域。

Issue 不用 kind 标签：GitHub 的原生 Issue Type 是组织仓库功能，本仓库（个人账号）没有，分类由五套模板创建时自动打的标签承担——`type/bug`、`type/feature`、`type/idea`、`type/research`、`type/task`；`area/*` 对 Issue 可选。`ready-for-agent`、`ready-for-human`、`needs-triage`、`needs-info` 是分诊标签，记录一项工作交给谁、卡在谁那里，与两轴标签正交。

## Project 流水线

所有 Issue 进入 Project “[cislunarspace Issue Management](https://github.com/orgs/cislunarspace/projects/4)”，按状态推进。该面板与 [transfer-orbit-design](https://github.com/cislunarspace/transfer-orbit-design) 共用，Repository 字段区分来源：

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
