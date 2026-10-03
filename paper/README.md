# ACRO preprint

LaTeX source for **ACRO: Actor–Critic Rollout Orchestration**.

Build from this directory with [Tectonic](https://tectonic-typesetting.github.io/):

```sh
tectonic root.tex
```

The build produces `root.pdf`. A conventional LaTeX installation can also build the document with `pdflatex`, `bibtex`, and two subsequent `pdflatex` passes.

- `root.tex` is the manuscript entry point.
- `authors.tex` contains the author and affiliation block.
- `sec/` and `tab/` contain manuscript sections and result tables.
- `figures/` and `assets/` contain the graphics used by the document.
- `acro_preprint.sty` preserves the manuscript typography in a public preprint layout.
- `references.bib`, `acro_preprint.bst`, and `root.bbl` provide the bibliography.

See [ASSET_SOURCES.md](ASSET_SOURCES.md) for the environment-image sources.
