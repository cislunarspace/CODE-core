## 交流语言

始终使用中文与用户交流。代码、commit message、PR 描述等技术输出也用中文。

## 写作要求

所有面向人读的文本——注释、CONTEXT.md、ADR、issue 评论、PR 描述、agent brief、triage notes、Sphinx 文档、Agent 回复——遵守以下原则：

- **善于总结材料**：材料弄全弄准，去粗取精、去伪存真、由此及彼、由表及里，反映事物本质；不堆砌细节、不拼凑清单。
- **不用夸大的修饰词**：不写"权威""强大""完整""单一事实来源"之类的修饰，它们减损力量。
- **注意词语的逻辑界限**：相邻概念要划清，不混用、不模糊。
- **废话应当尽量除去**。
- **通俗、亲切，由小讲到大，由近讲到远，引人入胜**：先讲读者已知／当前的事物，再推到陌生／抽象的；忌一上来就宏大叙事或先搬死人、外国人。
- **与读者完全平等**：靠分析说服，不要装腔作势来吓人；老老实实办事。

## 文档语言政策

- 面向人读的文档与注释一律用中文（CONTEXT.md、ADR、issue/PR 正文、README、Sphinx 文档、代码注释与 docstring）。代码标识符与不可避免的外文术语（CR3BP、SPICE、SysML 等）保留原文。
- `CHANGELOG.md`：保留 Keep a Changelog 的英文小节头（Added / Changed / Fixed / Removed），条目内容用中文。
- `CODE_OF_CONDUCT.md`、`SECURITY.md` 等 GitHub 标准模板保留英文，不翻译。

## Agent skills

### Issue tracker

Issues live as GitHub issues. Use the `gh` CLI for all operations. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical triage roles mapped 1:1 to GitHub labels (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repo. One `CONTEXT.md` at the root, ADRs under `docs/adr/`. See `docs/agents/domain.md`.
