# Triage Labels

技能以五个标准分诊角色来表述。本文件将这些角色映射到本仓库 issue tracker 中实际使用的标签字符串。

| 本技能集内的标签   | 本 tracker 中的标签  | 含义                            |
| ------------------ | -------------------- | ------------------------------- |
| `needs-triage`     | `needs-triage`       | 维护者需要评估此 issue          |
| `needs-info`       | `needs-info`         | 等待报告者补充更多信息          |
| `ready-for-agent`  | `ready-for-agent`    | 规格完整，AFK agent 可直接开始  |
| `ready-for-human`  | `ready-for-human`    | 需要人工实现                    |
| `wontfix`          | `wontfix`            | 不予处理                        |

当技能提到某个角色时（如“打上 AFK 就绪的分诊标签”），使用本表中对应的标签字符串。

## 本仓库现状

- 上表五个标签已在 GitHub 仓库中存在，无需新建；标签颜色、描述由维护者维护，本文件只固定字符串。
- 分诊角色与 `type/*`、`kind/*`、`area/*` 三类标签分工不同：后者表达分类、改动类型与所属层，不是状态角色。每个分诊过的 issue 恰好带一个分类角色和一个状态角色，领域标签按需附加。
- `wontfix` 的拒绝记录写入 `.out-of-scope/` 知识库；已实现而非被拒绝的请求不写入该目录。
