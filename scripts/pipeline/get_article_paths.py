#!/usr/bin/env python3
"""Print ARTICLE_DIR and ARTICLE_TEX for use by build_article.sh.

Does not import src.utils so it can run in environments without torch.
Reads configs/pipeline.yaml and respects CLOPDIT_ARTICLE_DIR / CLOPDIT_ARTICLE_TEX env.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"


def main() -> None:
    article_dir = os.environ.get("CLOPDIT_ARTICLE_DIR")
    article_tex = os.environ.get("CLOPDIT_ARTICLE_TEX")
    if not article_dir or not article_tex:
        if CONFIG_PATH.is_file():
            try:
                import yaml
                with open(CONFIG_PATH) as f:
                    cfg = yaml.safe_load(f) or {}
                article_dir = article_dir or cfg.get("article_dir", "articles")
                article_tex = article_tex or cfg.get("article_tex", "clop_dit_biology.tex")
            except Exception:
                pass
        article_dir = article_dir or "articles"
        article_tex = article_tex or "clop_dit_biology.tex"
    article_dir = Path(article_dir)
    if not article_dir.is_absolute():
        article_dir = PROJECT_ROOT / article_dir
    article_dir = article_dir.resolve()
    print(article_dir)
    print(article_tex)


if __name__ == "__main__":
    main()
