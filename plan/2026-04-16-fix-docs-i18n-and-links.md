# 修复文档站中英文切换与链接问题

## 目标
修复 website/ 中所有文档的中英文混乱和链接指向错误，确保中文默认 locale 全为中文、英文 locale 全为英文、链接指向正确。

## 背景
Docusaurus 站点配置 `defaultLocale: 'zh-Hans'`，即 `website/docs/` 下应为中文，`website/i18n/en/` 下应为英文。实际扫描发现多处语言反转、frontmatter 标题语言不一致、过期文件引用等问题。

## 约束与风险
- frontmatter 中的类名（如 `EphemerisSystem`）保留英文作为技术术语，但描述性文字应为对应 locale 语言
- MBSE 模块部分文件为纯 Mermaid 图（无散文内容），frontmatter title 是唯一需要翻译的部分
- `_zh` 后缀的文件不符合 Docusaurus i18n 机制，应合并或移除

## 任务列表

- [x] 1. **修复 ways-of-work 文件的语言反转** `website/docs/ways-of-work/`, `website/i18n/en/.../ways-of-work/`
  - `test-coverage-issues-checklist.md`（默认 locale）当前为英文，需与 `_zh.md` 互换内容
  - `test-coverage-project-plan.md`（默认 locale）当前为英文，需与 `_zh.md` 互换内容
  - 互换后删除 `_zh.md` 文件（默认 locale 已是中文，不需要 `_zh` 后缀）
  - 同步清理 `i18n/en/` 下对应的 `_zh.md` 残留文件
  - 验收：`ways-of-work/*.md` 全为中文，`i18n/en/.../ways-of-work/*.md` 全为英文，无 `_zh` 文件

- [x] 2. **修复中文 docs frontmatter 标题** `website/docs/`
  - `core/coordinate.md`: `"CoordinateTransformation & ReferenceFrame"` → `"坐标变换与参考系"`
  - `core/ephemeris_system.md`: `"EphemerisSystem - 星历系统"` → `"星历系统 EphemerisSystem"`（统一格式）
  - `core/ephemeris_dynamics.md`: `"EphemerisDynamics - 星历动力学"` → `"星历动力学 EphemerisDynamics"`
  - `algorithms/multiple_shooting.md`: `"MultipleShooting - 多重打靶法"` → `"多重打靶法 MultipleShooting"`
  - `reference/api-reference.md`: 章节标题 `"1. Core Module (核心模块)"` 等 → 统一为中文
  - 验收：所有 frontmatter title 和一级标题与中文 locale 一致

- [x] 3. **修复 `reference/algorithms.md` 章节编号错误** `website/docs/reference/algorithms.md`
  - 1.3 排在 1.1 前面，需调整为 1.1 → 1.2 → 1.3 的正确顺序

- [x] 4. **修复英文 locale 中 `reference/algorithms.md` 的过期文件引用** `website/i18n/en/.../reference/algorithms.md`
  - 附录中引用了 `algorithms_en.md`、`continuation_en.md` 等旧路径
  - 更新为实际的 Docusaurus i18n 目录结构

- [x] 5. **修复 MBSE `index.md` 语言问题** `website/mbse/index.md`
  - 默认 locale 的 `index.md` 全文为英文，需翻译为中文

- [x] 6. **修复 MBSE 中文文档 frontmatter 标题** `website/mbse/*.md`
  - 10 个文件的 frontmatter title 为纯英文，需翻译为中文（保留类名/技术术语英文）：
    - `sequence-propagation.md`: → `"传播序列"`
    - `activity-orbit-design.md`: → `"轨道设计活动图"`
    - `requirements.md`: → `"功能需求"`
    - `state-convergence.md`: → `"收敛状态机"`
    - `state-orbit-lifecycle.md`: → `"轨道生命周期状态机"`
    - `bdd-algorithms.md`: → `"算法模块 BDD"`
    - `sequence-correction.md`: → `"修正序列"`
    - `bdd-core.md`: → `"核心模块 BDD"`
    - `activity-differential-correction.md`: → `"微分修正活动图"`
  - `index.md`: 标题已在任务 5 中处理

- [x] 7. **移除英文 locale 中的中文残留文件** `website/i18n/en/.../ways-of-work/test-coverage-project-plan_zh.md`
  - 英文 locale 中存在 `test-coverage-project-plan_zh.md`（纯中文），应删除

- [x] 8. **清理 `halo-roadmap.md` 中的过期引用** `website/docs/ways-of-work/halo-roadmap.md`
  - 引用了不存在的 `halo_en.md`，更新为实际路径描述

- [x] 9. **构建验证** `website/`
  - `npm run build` 成功，零 broken link/anchor
  - 仅剩 vscode-languageserver-types 的第三方 webpack 警告和 KaTeX LaTeX 兼容性提示
  - 验收：通过

## 备注
- `npm run build` 当前可通过（无 broken link），但语言内容有大量不一致
- ways-of-work 文件属于"工作文档"性质，如果不需要出现在 sidebar 中，可考虑移到单独位置或从 sidebars 中排除
