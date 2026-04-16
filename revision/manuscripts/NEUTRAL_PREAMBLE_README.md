# Neutral (venue-independent) LaTeX preamble

**File:** [neutral_preamble.tex](neutral_preamble.tex)

The committed manuscript source under `revision/manuscripts/v2_revision/`
(gitignored) was originally authored against the MDPI `Definitions/mdpi`
document class (target: MDPI Genes). To keep the local PDF build
venue-neutral without rewriting the body, we swap the MDPI preamble
for a standard `\documentclass[11pt,a4paper]{article}` preamble plus
compatibility shims that accept MDPI-specific macros (`\Title`,
`\Author`, `\abstract{}` as a command, `\keyword`, `\authorcontributions`,
`\funding`, `\dataavailability`, `\useofartificialintelligence`,
`\PublishersNote`, etc.) and render them in a neutral style.

## How to apply

The swap is a one-time preamble replacement (run from the repo root):

```bash
cp revision/manuscripts/v2_revision/clop_dit_genes.tex \
   revision/manuscripts/v2_revision/clop_dit_genes.tex.mdpi_backup

python -c "
import re
tex = 'revision/manuscripts/v2_revision/clop_dit_genes.tex'
src = open(tex).read()
pre = open('revision/manuscripts/neutral_preamble.tex').read()
m = re.search(r'^\\\\begin\{document\}', src, re.MULTILINE)
new = pre + '\n' + src[m.start():]
open(tex, 'w').write(new)
"

cd revision/manuscripts/v2_revision && pdflatex -interaction=nonstopmode clop_dit_genes.tex
pdflatex -interaction=nonstopmode clop_dit_genes.tex   # second pass for references
```

## What the neutral preamble does

1. `\documentclass[11pt,a4paper]{article}` — vanilla article class; no
   venue branding, no watermarks, no line numbers.
2. Manually loads packages the MDPI class used to auto-load: inputenc,
   fontenc, amsmath/amssymb/amsthm, graphicx, xcolor, colortbl, array,
   booktabs, multirow, tabularx, longtable, microtype, titlesec,
   etoolbox, caption, placeins, float, lineno, changepage, upgreek,
   setspace, enumitem, natbib (numbers mode to match inline
   `\bibitem`-style bibliography), hyperref, cleveref, url.
3. Defines tabularx column types `C`, `L`, `R` the MDPI class
   provided.
4. Shims MDPI-specific macros to standard equivalents:
   - `\Title{}` / `\Author{}` / `\AuthorNames{}` / `\address{}` /
     `\corres{}` / `\abstract{}` / `\keyword{}` — stored and emitted
     as a plain title block + abstract environment at `\begin{document}`
     via `\AtBeginDocument{\mdpiemitheader}`.
   - `\authorcontributions{}` / `\funding{}` / `\institutionalreview{}` /
     `\informedconsent{}` / `\dataavailability{}` / `\acknowledgments{}` /
     `\conflictsofinterest{}` / `\abbreviations{Title}{body}` /
     `\useofartificialintelligence{}` — rendered as `\section*{...}`
     with the provided content.
   - `\reftitle{}` — rendered as `\section*{}`.
   - `\simplesumm{}` / `\AuthorNames{}` / `\PublishersNote{}` — no-op.
   - `\firstpage` / `\pubvolume` / `\issuenum` / `\articlenumber` /
     `\pubyear` / `\copyrightyear` / `\date{received,revised,accepted,published}` /
     `\doinum` — no-op (venue-specific metadata).
   - `\appendixtitles{}` no-op; `\appendixstart` → `\appendix`;
     `\extralength` declared as a 0pt length.

## Output

Builds a plain 36-page PDF with:
- Standard article title block (title, author, affiliation,
  correspondence, abstract, keywords).
- All 16 original sections, 20+ figures, and all tables render
  unchanged.
- Matter-end sections (Author Contributions, Funding, Data
  Availability, Acknowledgments, Conflicts of Interest,
  Abbreviations) render as `\section*{}` headings.
- References via inline `\bibitem{}` entries with `natbib`
  numeric citations.
