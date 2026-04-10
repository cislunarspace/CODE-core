# 修复文档锚点链接问题

## 目标

修复 docs 网页中 90 个锚点链接警告，使所有内部超链接点击后能正确跳转。

## 根因分析

Python-Markdown 默认的 `slugify` 函数会剔除非 ASCII 字符（中文），导致纯中文标题的锚点 ID 被降级为 `_1`, `_2` 等数字编号，与目录中写的 `#快速开始` 等中文锚点不匹配。

此外，`&` 和 `/` 前后的空格被转成了双连字符 `--`，但 slugify 实际生成的是单连字符 `-`。重复标题使用 `_N` 下划线后缀，但目录中写的是 `-N` 连字符后缀。

## 修复方案

1. 在 `mkdocs.yml` 中配置 `slugify_unicode`，保留中文字符
2. 修正 TOC 链接中的双连字符 `--` → `-`
3. 修正重复标题后缀 `-N` → `_N`

## 任务列表

- [x] 1. 修改 `mkdocs.yml` 的 toc 扩展配置，添加 `slugify: !!python/name:markdown.extensions.toc.slugify_unicode`
- [x] 2. 修正 `docs/reference/api-reference.md`：双连字符 3 处 + 重复标题后缀 19 处 + 3.1 标题锚点 1 处
- [x] 3. 修正 `docs/reference/api-reference_en.md`：双连字符 5 处 + 重复标题后缀 19 处 + 3.1 标题锚点 1 处
- [x] 4. 修正 `docs/guides/visualization-guide_en.md`：双连字符 1 处
- [x] 5. 运行 `mkdocs build` 验证所有锚点警告已消除（90 → 0）
- [ ] 6. 提交修复

## 备注

- 修改 4 个文件：mkdocs.yml + 3 个 markdown 文件
- 中文文件的中文锚点链接无需修改，配置 slugify_unicode 后自动生效
- 双连字符涉及的标题：`& ReferenceFrame`, `& BifurcationType`, `& ProjectionPlane`, `/ DROTransferSearch`, `↔ Inertial Transformation`
- 重复标题后缀：`设计原理`, `核心方法`, `使用示例`, `Design Principles`, `Core Methods`, `Usage Example` 等在多个小节中出现
