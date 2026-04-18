#!/usr/bin/env python3
"""Classify rebuttal replies as STRONG or THIN.

STRONG = word count >= 100 AND contains >= 1 numeric anchor.
Numeric anchor = digit(s) followed by a recognised scientific unit token
(e.g. %, pp, pts, ×, -fold, /N, KNN, FD, SWD, Jaccard, coverage, steering).

Usage:
    python scripts/analysis/audit_rebuttal_depth.py <rebuttal_markdown>
    python scripts/analysis/audit_rebuttal_depth.py --validate-regex

Writes a per-reply report to stdout. Designed to gate the Stage 1
rebuttal-deepening pass: every reply must classify as STRONG before
the stage can close.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

WORD_COUNT_THRESHOLD = 100

_LATEX_STRIP_RE = re.compile(
    r"\\(?:text|mathrm|mathit|mathbf|mathsf|msref|operatorname)\{([^}]*)\}"
)
_LATEX_CMD_RE = re.compile(r"\\[,!;:]|\\(?:approx|sim|lesssim|gtrsim|,|!)")
_LATEX_UNIT_MAP = {r"\times": " × ", r"\pm": " pm ", r"\%": " % "}
_LATEX_TOKEN_RE = re.compile(r"\\[a-zA-Z]+")

STATS_KEYWORDS = (
    r"KNN|FD|SWD|Jaccard|coverage|steering|tolerance|Pearson|cosine|"
    r"accuracy|quality|ratio|std-?dev|median|fold|composite"
)

_UNIT_ANCHOR_RE = re.compile(
    r"\d+(?:\.\d+)?"
    r"\s*"
    r"(?:%|pp|pts?|×|x\b|-fold|/\d+|pm)",
    flags=re.IGNORECASE,
)

_PM_PRECEDES_NUM_RE = re.compile(
    r"\bpm\b\s*<?\s*\d+(?:\.\d+)?",
    flags=re.IGNORECASE,
)

_STATS_AFTER_RE = re.compile(
    rf"(?:{STATS_KEYWORDS})[^.\n]{{0,40}}\d+(?:\.\d+)?",
    flags=re.IGNORECASE,
)

_STATS_BEFORE_RE = re.compile(
    rf"\d+(?:\.\d+)?[^.\n]{{0,40}}(?:{STATS_KEYWORDS})",
    flags=re.IGNORECASE,
)


def _normalize(text: str) -> str:
    out = _LATEX_STRIP_RE.sub(r"\1", text)
    for token, replacement in _LATEX_UNIT_MAP.items():
        out = out.replace(token, replacement)
    out = out.replace("$", " ")
    out = _LATEX_CMD_RE.sub(" ", out)
    out = _LATEX_TOKEN_RE.sub(" ", out)
    out = re.sub(r"[{}]", " ", out)
    return re.sub(r"\s+", " ", out)

REPLY_HEADER_RE = re.compile(
    r"^###\s+(?P<id>R\d+\.\d+|Gaussian\s+addendum|[A-Z][A-Za-z0-9 \-]+?)"
    r"(?:\s*\(.*?\))?"
    r"\s*(?:—.*)?$",
    flags=re.MULTILINE,
)


def count_words(text: str) -> int:
    tex_stripped = re.sub(r"\\[a-zA-Z]+(?:\{[^}]*\})?", " ", text)
    tex_stripped = re.sub(r"[${}\\]", " ", tex_stripped)
    return len([w for w in tex_stripped.split() if w])


def find_numeric_anchors(text: str) -> list[str]:
    norm = _normalize(text)
    hits: list[str] = []
    hits.extend(_UNIT_ANCHOR_RE.findall(norm))
    hits.extend(_STATS_AFTER_RE.findall(norm))
    hits.extend(_STATS_BEFORE_RE.findall(norm))
    hits.extend(_PM_PRECEDES_NUM_RE.findall(norm))
    return hits


def classify(reply_text: str) -> tuple[str, int, int]:
    words = count_words(reply_text)
    anchors = len(find_numeric_anchors(reply_text))
    if words >= WORD_COUNT_THRESHOLD and anchors >= 1:
        return "STRONG", words, anchors
    return "THIN", words, anchors


def split_replies(source: str) -> list[tuple[str, str]]:
    headers = list(REPLY_HEADER_RE.finditer(source))
    out: list[tuple[str, str]] = []
    for idx, match in enumerate(headers):
        reply_id = match.group("id").strip()
        start = match.end()
        end = headers[idx + 1].start() if idx + 1 < len(headers) else len(source)
        body = source[start:end].strip()
        if body:
            out.append((reply_id, body))
    return out


def validate_regex_on_paste_ready_samples() -> int:
    """Smoke-test the regex on hand-crafted paste-ready snippets.

    Returns the number of samples that failed to match (0 == pass).
    """
    samples: Iterable[tuple[str, bool]] = (
        ("KNN $\\approx 1.0\\%$, steering $\\approx 47.5\\%$", True),
        ("$36.9\\%$ KNN and $81.0\\%$ steering", True),
        ("$\\sim\\!8.7\\times$ drop", True),
        ("$+4.2$\\,pp positive-pair-cosine gain", True),
        ("Pearson $r_\\text{var} = 0.988$", True),
        ("median std-dev ratio $1.33$", True),
        ("$0.966 \\pm <\\!0.01$", True),
        ("BERT embeddings", False),
        ("CLOP and DiT", False),
        ("the scGPT model", False),
    )
    fails = 0
    for text, should_match in samples:
        hit = bool(find_numeric_anchors(text))
        if hit != should_match:
            fails += 1
            print(f"FP/FN: {text!r} → match={hit}, expected={should_match}")
        else:
            print(f"OK  : {text!r} → match={hit}")
    return fails


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, help="rebuttal markdown")
    parser.add_argument("--validate-regex", action="store_true")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 if any reply is THIN")
    args = parser.parse_args(argv)

    if args.validate_regex:
        fails = validate_regex_on_paste_ready_samples()
        print(f"\n{fails} regex validation failure(s)")
        return 1 if fails else 0

    if not args.path:
        parser.error("path is required unless --validate-regex")

    source = args.path.read_text()
    replies = split_replies(source)
    if not replies:
        print(f"no replies found in {args.path}", file=sys.stderr)
        return 2

    strong = thin = 0
    print(f"reply\tverdict\twords\tanchors")
    for reply_id, body in replies:
        verdict, words, anchors = classify(body)
        if verdict == "STRONG":
            strong += 1
        else:
            thin += 1
        print(f"{reply_id}\t{verdict}\t{words}\t{anchors}")

    print(f"\n{strong} STRONG, {thin} THIN (total {strong + thin})")
    if args.strict and thin:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
