LATEXMK = latexmk -xelatex -interaction=nonstopmode -halt-on-error

all: zh en

zh: main-zh.pdf

en: main-en.pdf

main-zh.pdf: main-zh.tex preamble.tex zh/*.tex
	$(LATEXMK) main-zh.tex

main-en.pdf: main-en.tex preamble.tex en/*.tex
	$(LATEXMK) main-en.tex

clean:
	latexmk -C main-zh.tex main-en.tex

.PHONY: all zh en clean
