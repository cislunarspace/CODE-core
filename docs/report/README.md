# e2m2e 技术白皮书 / e2m2e White Paper

本目录是 e2m2e 技术白皮书的 LaTeX 源码，中英双语版本共享同一导言区
（`preamble.tex`）与图表样式。

## 编译 / Build

需要 TeX Live（含 xeCJK/ctex、tcolorbox、tikz 等，完整版即可）。
中文版使用 ctex + Windows 中文字体（`fontset` 自动检测），用 XeLaTeX 编译：

```bash
make          # 同时编译中文版 main-zh.pdf 与英文版 main-en.pdf
make zh       # 仅中文版
make en       # 仅英文版
make clean    # 清理中间文件
```

也可以直接：

```bash
latexmk -xelatex main-zh.tex
latexmk -xelatex main-en.tex
```

## 结构 / Layout

```
docs/report/
├── preamble.tex     # 共享导言区：配色、标题样式、代码框、数学宏
├── main-zh.tex      # 中文版主文件（ctexart）
├── main-en.tex      # 英文版主文件（article + lmodern）
├── zh/              # 中文版各章
├── en/              # 英文版各章
└── Makefile
```

两版章节一一对应：`zh/NN-*.tex` 与 `en/NN-*.tex` 内容相同、语言不同。
改结构时两侧同步修改。
