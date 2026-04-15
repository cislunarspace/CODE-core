# 用 Docusaurus 重构文档站并部署到 GitHub Pages

## 目标
用 Docusaurus 替换现有 MkDocs 文档站，迁移所有文档内容（中英双语），配置 GitHub Actions 自动部署到 GitHub Pages。

## 背景
项目已有完整的 MkDocs 文档体系（`docs/` 下约 50 个 markdown 文件，含 Mermaid 图、MathJax 公式），现需用 React 技术栈重构为 Docusaurus 站点。`mkdocs-demo` 分支已有旧的 MkDocs 尝试。

## 约束与风险
- 现有 `docs/` 下的 markdown 内容需保留（作为源内容迁移到 Docusaurus 的 `docs/` 目录）
- Mermaid 图需 Docusaurus 插件支持（`docusaurus-plugin-mdx-mermaid`）
- MathJax/KaTeX 需 `remark-math` + `rehype-katex` 插件
- GitHub Pages 部署需要 `gh-pages` 分支或 GitHub Actions `actions/deploy-pages`
- 仓库为 `cislunarspace/e2m2e`，需注意 `basePath` 配置（如仓库名不为 `.github.io`）

## 任务列表

- [x] 1. **创建新分支** 
  - 从 master 创建 `docs/docusaurus` 分支
  - 验收：分支存在并切换到该分支

- [x] 2. **初始化 Docusaurus 项目** `website/`
  - 在项目根目录下创建 `website/` 目录，用 `npx create-docusaurus@latest` 初始化 classic 模板
  - 配置 `docusaurus.config.js`：站点名、URL、basePath、i18n
  - 验收：`npm run start` 能在本地跑起来

- [x] 3. **迁移文档内容** `website/docs/`, `website/i18n/`
  - 将现有 `docs/` 下的中文文档迁移到 `website/docs/`
  - 将英文文档配置为 i18n 英文 locale
  - 修复 markdown 中的链接引用（MkDocs 相对路径 → Docusaurus 路径）
  - 验收：所有文档页面可正常访问

- [x] 4. **配置插件（Mermaid + KaTeX）** `website/docusaurus.config.js`, `website/package.json`
  - 安装并配置 `@docusaurus/plugin-content-docs`、Mermaid 插件、KaTeX 支持
  - 验收：包含 Mermaid 图和数学公式的页面能正确渲染

- [x] 5. **配置导航栏和侧边栏** `website/docusaurus.config.js`, `website/sidebars.js`
  - 按 mkdocs.yml 的 nav 结构重建导航
  - 配置双语切换
  - 验收：导航结构完整，中英切换正常

- [x] 6. **清理旧 MkDocs 文件**
  - 删除 `mkdocs.yml`、`mkdocs-env/`、`site/`、根目录旧 `docs/`（迁移完成后）
  - 更新 `.gitignore`
  - 验收：无旧文件残留，`.gitignore` 包含 `website/build/` 等

- [x] 7. **配置 GitHub Actions 部署** `.github/workflows/deploy-docs.yml`
  - 创建 deploy-docs workflow，使用 `actions/deploy-pages` 部署到 GitHub Pages
  - 触发条件：push 到 master 分支的 `website/` 目录变更
  - 验收：workflow 文件语法正确

- [x] 8. **本地构建验证** `website/`
  - 运行 `npm run build`，确认无错误
  - 运行 `npm run serve` 预览构建产物
  - 验收：`build/` 目录生成完整静态站点

## 备注
- Docusaurus 的 i18n 使用 JSON 翻译文件 + markdown 文档目录结构
- 英文文档可放在 `website/i18n/en/docusaurus-plugin-content-docs/current/` 下
- GitHub Pages 部署可能需要在仓库 Settings 中启用 Pages 并选择 "GitHub Actions" 作为 source
