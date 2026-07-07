# Paper

`main.tex` is the write-up of the study. It is self-contained: it uses the
standard `article` class and only common LaTeX packages (`graphicx`,
`booktabs`, `amsmath`, `hyperref`, `natbib`), so it compiles with a stock TeX
installation — no conference style files required.

## Compile

**Overleaf.** Create a new project, upload `main.tex` and the `figures/`
directory (preserving the folder), and compile with pdfLaTeX.

**Local.**

```bash
cd paper
pdflatex main.tex
pdflatex main.tex   # second pass resolves \ref / \cite cross-references
```

## Figures

The figures in `figures/` are snapshots of the plots the paper includes. They
are copies of files in the repository's top-level `results/` directory, which
is the regeneratable source: running the corresponding phase (e.g.
`python run_all.py phase3`) rewrites the PNGs in `results/`. To refresh a
paper figure after re-running a phase, copy it from `results/` into
`figures/`.

## Porting to the ACL / BlackboxNLP template

Swap the `\documentclass` line for the ACL class and move the title block into
its macros; the body, figures, and manual `thebibliography` carry over
unchanged. The bibliography is written inline (`thebibliography`) rather than
via a `.bib` file to keep the source dependency-free; convert to BibTeX if the
venue requires it.
