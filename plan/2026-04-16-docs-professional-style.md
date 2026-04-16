# 文档站风格重设计：专业技术文档风格

## 目标
将 Docusaurus 文档站从默认 Indigo 模板风格改为 MATLAB/STK 风格的专业技术文档风格（深蓝导航栏 + 白色内容区 + 灰蓝侧边栏 + 克制配色）。

## 任务列表

- [x] 1. **配色与字体** `website/src/css/custom.css`
  - 主色 #0B5FA5，导航栏 #1B2A4A，侧边栏 #F7F9FC
  - 引入 Inter + JetBrains Mono 字体
  - 暗色主题配色 #5BA3D9

- [x] 2. **导航栏与侧边栏样式** `website/src/css/custom.css`
  - 深蓝导航栏、白色文字
  - 灰蓝侧边栏、活跃项左侧蓝色竖条

- [x] 3. **内容区样式** `website/src/css/custom.css`
  - 标题、代码块、表格、KaTeX 公式、admonitions、链接、引用块、TOC、分页

- [x] 4. **页脚样式** `website/src/css/custom.css`
  - 深蓝背景匹配导航栏

- [x] 5. **配置调整** `website/docusaurus.config.js`
  - Prism 暗色主题改为 oneDark，TOC 限制 h2-h3

- [x] 6. **Logo 配色更新** `website/static/img/logo.svg`, `website/static/img/favicon.svg`
  - 颜色从 #4f46e5 更新为 #0B5FA5/#5BA3D9

- [x] 7. **首页重设计** `website/src/pages/index.js`, `website/src/pages/index.module.css`, `website/src/components/HomepageFeatures/`
  - 白色背景 hero + 代码示例，特性卡片带 SVG 图标和边框

- [x] 8. **构建验证** `website/`
  - `npm run build` 通过，零 broken link

## 备注
- 1-4 合并为一次 custom.css 重写（约 300 行 CSS），无需 swizzle 组件
- 首页 JSX 中 code 块引号冲突已修复（用 `{"..."}` 模板字符串）
