# Paper folder — Overleaf-ready, official AAAI 2027 format

Upload the contents of this folder (or a zip of it) to a new Overleaf
project and it compiles as-is with pdfLaTeX:

- `main.tex` — the manuscript, using the official AAAI 2027 author kit
- `aaai2027.sty`, `aaai2027.bst` — the AAAI 2027 style files (unmodified)
- `references.bib` — bibliography (author-year, AAAI format)
- `figures/` — PNG figures copied from `../results/figures/`; re-copy after
  regenerating results (`cp ../results/figures/*.png figures/`)

`main.pdf` is a prebuilt copy for viewing; Overleaf regenerates it.

For the anonymous review upload, change the style line in `main.tex` to
`\usepackage[submission]{aaai2027}` — the option hides the author block
automatically. Section numbering is enabled via `\setcounter{secnumdepth}{2}`
(the kit permits it; the paper cross-references sections).

Local build (BasicTeX at /Library/TeX/texbin; missing packages were
installed user-mode via
`tlmgr --usermode install newtx placeins txfonts courier helvetic times
xstring kastrup fontaxes xkeyval etoolbox tex-gyre`):

    pdflatex main && bibtex main && pdflatex main && pdflatex main
