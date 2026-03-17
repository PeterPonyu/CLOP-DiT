#!/usr/bin/env python3
"""Generate an Elsevier-formatted manuscript from the canonical MDPI article.

This keeps the main manuscript body, appendix, figures, and inline bibliography in
sync with `articles/clop_dit_biology.tex`, while replacing the MDPI-specific
front matter with a generic `elsarticle` front matter and a small set of
submission-friendly summary sections.
"""
from __future__ import annotations

from pathlib import Path
import re
import textwrap

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "articles" / "clop_dit_biology.tex"
DST_DIR = ROOT / "articles_elsevier"
DST = DST_DIR / "clop_dit_elsevier.tex"

POLISHED_TITLE = (
    "CLOP-DiT: Structured-metadata-conditioned single-cell latent generation "
    "via contrastive language-omics pretraining and diffusion transformers"
)

POLISHED_PLAIN_LANGUAGE_SUMMARY = (
    "CLOP-DiT links structured biological descriptions—cell type, tissue, "
    "organism, marker genes, and disease context—to synthetic single-cell "
    "states. The pipeline first aligns text descriptions with real cells in a "
    "shared embedding space and then uses a diffusion transformer to generate "
    "new latent states that match the requested description. The generated "
    "cells recover cell identity and marker-gene patterns, but they still "
    "underestimate the full cell-to-cell variability observed in real "
    "single-cell data. We therefore present CLOP-DiT as a proof of concept for "
    "text-guided single-cell generation, with a clear roadmap toward stronger "
    "biological fidelity."
)

POLISHED_ABSTRACT = (
    "We introduce CLOP-DiT, a two-stage framework that combines Contrastive "
    "Language--Omics Pretraining (CLOP) with a conditional Diffusion "
    "Transformer (DiT) trained by flow matching to generate single-cell latent "
    "states from structured biological metadata. CLOP aligns BiomedBERT text "
    "embeddings and scGPT cell embeddings in a shared 512-dimensional space, "
    "and DiT samples latents conditioned on a five-field prompt template "
    "spanning cell type, tissue, organism, marker genes, and disease context. "
    "Across 69 cell types from 80 GEO datasets (220{,}304 cells), CLOP-DiT "
    "reaches 36.9\% KNN accuracy and 81.0\% steering in a high-fidelity "
    "regime, while a high-diversity regime attains a diversity ratio of 0.93 "
    "at 80.7\% steering. A frozen scGPT decoder maps generated latents back to "
    "gene expression with high mean concordance (Pearson $r > 0.999$), but "
    "per-gene variance correlation remains near zero, highlighting limited "
    "preservation of cell-to-cell heterogeneity. Conditioning-field ablation "
    "identifies marker genes as the dominant steering signal, and external "
    "validation against PanglaoDB and CellMarker~2.0 supports biological "
    "plausibility. Generated cells nevertheless remain distinguishable from "
    "real data (discriminator AUC 0.656), a Gaussian baseline outperforms "
    "CLOP-DiT on common metrics, and a pilot rare-cell augmentation study is "
    "negative. CLOP-DiT therefore establishes the feasibility of "
    "structured-metadata-conditioned single-cell latent generation while also "
    "clarifying the biological fidelity gap that future work must close."
)

POLISHED_HIGHLIGHTS = [
    "CLOP-DiT couples contrastive language--omics alignment with diffusion-based latent generation.",
    "The study evaluates 69 deduplicated cell types from 80 GEO datasets containing 220,304 cells.",
    "Structured prompts steer cell identity and marker-gene programs, but variance preservation remains limited.",
    "The results establish feasibility and define a concrete roadmap for biologically faithful single-cell generation.",
]


def extract_macro_args(text: str, macro: str, n_args: int = 1) -> list[str]:
    anchor = text.index("\\" + macro)
    pos = anchor + len(macro) + 1
    args: list[str] = []

    for _ in range(n_args):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text) or text[pos] != "{":
            raise ValueError(f"Expected '{{' after \\{macro}")

        pos += 1
        start = pos
        depth = 1

        while pos < len(text) and depth:
            ch = text[pos]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            pos += 1

        if depth != 0:
            raise ValueError(f"Unbalanced braces while parsing \\{macro}")

        args.append(text[start:pos - 1])

    return args


def extract_between(text: str, start_marker: str, end_marker: str, *, include_end: bool = False) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    if include_end:
        end += len(end_marker)
    return text[start:end]


def normalise_text(fragment: str) -> str:
    fragment = fragment.replace("People\\textquotesingle s", "People's")
    fragment = fragment.replace("\\textquotesingle", "'")
    return fragment.strip()


def clean_fragment(fragment: str) -> str:
    fragment = re.sub(r"\\begin\{linenomath\}\s*", "", fragment)
    fragment = re.sub(r"\\end\{linenomath\}\s*", "", fragment)
    fragment = fragment.replace("\\appendixtitles{yes}", "")
    fragment = fragment.replace("\\appendixstart", "")
    fragment = fragment.replace(
        "\\section[\\appendixname~\\thesection]{Supplementary Material}",
        "\\section{Supplementary Material}",
    )
    fragment = fragment.replace("\\PublishersNote{}", "")
    return normalise_text(fragment)


def make_highlights_block(items: list[str]) -> str:
    lines = ["\\section*{Highlights}", "\\begin{itemize}"]
    lines.extend(f"\\item {item}" for item in items)
    lines.append("\\end{itemize}")
    return "\n".join(lines)


def main() -> None:
    text = SRC.read_text(encoding="utf-8")

    keywords = normalise_text(extract_macro_args(text, "keyword")[0])
    author_contributions = normalise_text(extract_macro_args(text, "authorcontributions")[0])
    funding = normalise_text(extract_macro_args(text, "funding")[0])
    institutional_review = normalise_text(extract_macro_args(text, "institutionalreview")[0])
    informed_consent = normalise_text(extract_macro_args(text, "informedconsent")[0])
    data_availability = normalise_text(extract_macro_args(text, "dataavailability")[0])
    acknowledgments = normalise_text(extract_macro_args(text, "acknowledgments")[0])
    conflict_of_interest = normalise_text(extract_macro_args(text, "conflictsofinterest")[0])
    _, abbreviations_body = extract_macro_args(text, "abbreviations", 2)
    abbreviations_body = normalise_text(abbreviations_body)

    main_body = clean_fragment(
        extract_between(text, "\\section{Introduction}", "\\authorcontributions{")
    )

    appendix = clean_fragment(
        extract_between(
            text,
            "\\appendixtitles{yes}",
            "\\FloatBarrier  % Force all appendix floats to appear before the reference list",
        )
    )

    references = normalise_text(
        extract_between(
            text,
            "\\begin{thebibliography}{999}",
            "\\end{thebibliography}",
            include_end=True,
        )
    )

    keyword_text = re.sub(r"\s*;\s*", r" \\sep ", keywords)
    highlights_block = make_highlights_block(POLISHED_HIGHLIGHTS)

    preamble = textwrap.dedent(
        f"""
        \\documentclass[preprint,12pt]{{elsarticle}}

        \\usepackage[T1]{{fontenc}}
        \\usepackage[utf8]{{inputenc}}
        \\usepackage{{lmodern}}
        \\usepackage{{amsmath,amssymb}}
        \\usepackage{{graphicx}}
        \\usepackage{{booktabs}}
        \\usepackage{{tabularx}}
        \\usepackage{{longtable}}
        \\usepackage{{array}}
        \\usepackage{{placeins}}
        \\usepackage{{microtype}}
        \\usepackage{{textcomp}}
        \\usepackage[colorlinks=true,allcolors=blue]{{hyperref}}

        \\journal{{Elsevier manuscript draft}}

        \\begin{{document}}

        \\begin{{frontmatter}}

        \\title{{{POLISHED_TITLE}}}

        \\author[aff1]{{Zeyu Fu\\fnref{{eq1}}}}
        \\author[aff2]{{Jiawei Fu\\fnref{{eq1}}}}
        \\author[aff3]{{Chunlin Chen\\fnref{{eq1}}}}
        \\author[aff4]{{Keyang Zhang}}
        \\author[aff1]{{Junping Wang}}
        \\author[aff2]{{Tianfei Ran\\corref{{cor1}}}}
        \\ead{{rantianfei@tmmu.edu.cn}}
        \\author[aff1]{{Song Wang\\corref{{cor2}}}}
        \\ead{{swang1981@tmmu.edu.cn}}

        \\fntext[eq1]{{These authors contributed equally to this work.}}
        \\cortext[cor1]{{Corresponding author.}}
        \\cortext[cor2]{{Corresponding author.}}

        \\address[aff1]{{State Key Laboratory of Trauma and Chemical Poisoning, Institute of Combined Injury, Chongqing Engineering Research Center for Nanomedicine, College of Preventive Medicine, Army Medical University, Chongqing 400038, China}}
        \\address[aff2]{{Department of Orthopedics, Xinqiao Hospital, Army Medical University, Chongqing 400037, China}}
        \\address[aff3]{{Department of Rehabilitation Medicine, The First Affiliated Hospital, Sun Yat-sen University, Guangzhou 510080, China}}
        \\address[aff4]{{School of Medicine, Sun Yat-sen University, Shenzhen 518107, China}}

        \\begin{{abstract}}
        {POLISHED_ABSTRACT}
        \\end{{abstract}}

        \\begin{{keyword}}
        {keyword_text}
        \\end{{keyword}}

        \\end{{frontmatter}}

        \\section*{{Plain-language summary}}
        {POLISHED_PLAIN_LANGUAGE_SUMMARY}

        {highlights_block}
        """
    ).strip()

    endmatter = textwrap.dedent(
        f"""
        \\section*{{Author contributions}}
        {author_contributions}

        \\section*{{Funding}}
        {funding}

        \\section*{{Ethics statement}}
        {institutional_review}

        \\section*{{Informed consent}}
        {informed_consent}

        \\section*{{Data availability}}
        {data_availability}

        \\section*{{Acknowledgments}}
        {acknowledgments}

        \\section*{{Conflict of interest}}
        {conflict_of_interest}

        \\section*{{Abbreviations}}
        {abbreviations_body}

        \\FloatBarrier
        """
    ).strip()

    output = (
        preamble
        + "\n\n"
        + main_body
        + "\n\n"
        + endmatter
        + "\n\n"
        + appendix
        + "\n\n"
        + references
        + "\n\n\\end{document}\n"
    )

    DST_DIR.mkdir(parents=True, exist_ok=True)
    (DST_DIR / "figures").mkdir(parents=True, exist_ok=True)
    DST.write_text(output, encoding="utf-8")

    print(f"Wrote {DST}")


if __name__ == "__main__":
    main()
