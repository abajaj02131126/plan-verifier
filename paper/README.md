# Paper folder — Overleaf-ready

Upload the contents of this folder (or a zip of it) to a new Overleaf
project and it compiles as-is with pdfLaTeX:

- `main.tex` — the manuscript (set as the main document; Overleaf picks it
  up automatically)
- `references.bib` — bibliography (plainnat/natbib)
- `figures/` — PNG figures copied from `../results/figures/`; re-copy after
  regenerating results (`cp ../results/figures/*.png figures/`)

`main.pdf` is a prebuilt copy for viewing; Overleaf regenerates it and does
not need it uploaded.

For actual AAAI submission: download the AAAI author kit, replace the
preamble (documentclass through the custom formatting packages) with
`\usepackage{aaai26}` per the kit's template, switch the bibliography style
as the kit requires, and anonymize the author block. The body, tables,
figures, and references port unchanged.

Local build (BasicTeX at /Library/TeX/texbin):

    pdflatex main && bibtex main && pdflatex main && pdflatex main

Note: `main.tex` sets `\ttdefault` to Computer Modern because BasicTeX
lacks Courier metrics; on Overleaf (full TeX Live) you can delete that line
to get standard Times+Courier.