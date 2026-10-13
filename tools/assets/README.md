# 品牌资产生成工具

本目录由矢量母版生成 README 用的品牌位图，产物可复现：改母版后重跑命令即可，不需要手工修图。

## 目录内容

| 文件 | 用途 |
|---|---|
| `banner.svg` | README 横幅母版（1536×400），右侧三行文字右对齐到 x=1432 |
| `logo.png` | 标志副本，取自本仓库 `docs/_static/logo.png`，作为横幅里的标志来源 |
| `build-assets.mjs` | 由母版光栅化 `assets/banner.png` |
| `package.json` | 光栅化依赖 `@resvg/resvg-js` |

## 常用命令

```bash
npm ci --prefix tools/assets              # 安装生成依赖
npm --prefix tools/assets run banner      # 由母版重生成 assets/banner.png
```

母版里标志写成相对引用 `logo.png` 便于阅读，resvg 不解析相对图片路径，`build-assets.mjs` 在渲染前把该引用替换成 base64 data URI。文字走系统字体，需要本机装有 Noto Sans CJK。

## 改文案或配色

只改 `banner.svg` 里的 `text` 与填充色，然后重跑 `npm --prefix tools/assets run banner`。右侧三行统一用 `text-anchor="end"` 对齐到 x=1432，左侧标志的左边距 104 与之对称，改动其中一侧时保持这个对称关系。
