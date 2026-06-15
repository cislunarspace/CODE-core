## Communication language

Use Chinese for all interactions with the user.

## 写作要求

所有面向人读的文本——CONTEXT.md、ADR、issue 评论、PR 描述、agent brief、triage notes、Sphinx 文档——遵守以下原则（原话引用）：

- **善于总结材料**：材料弄全弄准，去粗取精、去伪存真、由此及彼、由表及里，反映事物本质；不堆砌细节、不拼凑清单。
- **不用夸大的修饰词**：不写"权威""强大""完整""单一事实来源"之类的修饰，它们减损力量。
- **注意词语的逻辑界限**：相邻概念要划清（如"配置"与"运行规格"、"力模型"与"力模型聚合"），不混用、不模糊。
- **废话应当尽量除去**。
- **通俗、亲切，由小讲到大，由近讲到远，引人入胜**：先讲读者已知／当前的事物，再推到陌生／抽象的；忌一上来就宏大叙事或先搬死人、外国人。
- **与读者完全平等**：靠分析说服，不要装腔作势来吓人；老老实实办事。

## Agent skills

### Issue tracker

Issues live as GitHub issues. Use the `gh` CLI for all operations. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical triage roles mapped 1:1 to GitHub labels (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repo. One `CONTEXT.md` at the root, ADRs under `docs/adr/`. See `docs/agents/domain.md`.
