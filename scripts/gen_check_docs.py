#!/usr/bin/env python3
"""Generate docs/scivcd/checks.md from registered scivcd checks.

Usage::

    python scripts/gen_check_docs.py

The script imports ``scivcd.core.registry.iter_checks`` and walks every
registered check to produce a Markdown table.  If ``scivcd`` is not
importable (e.g. the package has not been installed yet) the script exits
gracefully with a warning and leaves the existing checks.md untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo-root detection so the script can be run from any working directory.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_TARGET = REPO_ROOT / "docs" / "scivcd" / "checks.md"

BEGIN_MARKER = "<!-- BEGIN AUTO-GENERATED TABLE -->"
END_MARKER = "<!-- END AUTO-GENERATED TABLE -->"


def _build_table() -> str:
    """Return the Markdown table string for all registered checks."""
    try:
        from scivcd.core.registry import iter_checks  # type: ignore[import]
        import scivcd.checks  # side-effect: registers every detector  # noqa: F401
    except ImportError:
        print(
            "WARNING: scivcd not importable — skipping table generation.\n"
            "Install with: pip install -e scivcd/",
            file=sys.stderr,
        )
        return (
            "| *(scivcd not installed — run `pip install -e scivcd/` and retry)* "
            "| | | | | |"
        )

    header = (
        "| id | severity | category | stage | description | config_keys |\n"
        "|----|----------|----------|-------|-------------|-------------|"
    )

    rows: list[str] = []
    for spec in iter_checks(enabled_only=False):
        config_keys = ", ".join(f"`{k}`" for k in spec.config_keys) or "—"
        rows.append(
            f"| `{spec.id}` "
            f"| {spec.severity.name} "
            f"| {spec.category.name} "
            f"| {spec.stage.name} "
            f"| {spec.description or '—'} "
            f"| {config_keys} |"
        )

    if not rows:
        rows.append(
            "| *(no checks registered — import your check modules first)* "
            "| | | | | |"
        )

    return header + "\n" + "\n".join(rows)


def _splice_table(content: str, table: str) -> str:
    """Replace the auto-generated block inside *content* with *table*."""
    before, _, rest = content.partition(BEGIN_MARKER)
    _, _, after = rest.partition(END_MARKER)
    return f"{before}{BEGIN_MARKER}\n\n{table}\n\n{END_MARKER}{after}"


def main() -> None:
    table = _build_table()

    if not DOCS_TARGET.exists():
        print(f"ERROR: {DOCS_TARGET} not found — run from repo root.", file=sys.stderr)
        sys.exit(1)

    original = DOCS_TARGET.read_text(encoding="utf-8")

    if BEGIN_MARKER not in original or END_MARKER not in original:
        print(
            f"ERROR: marker comments not found in {DOCS_TARGET}.\n"
            "Expected:\n"
            f"  {BEGIN_MARKER}\n"
            f"  {END_MARKER}",
            file=sys.stderr,
        )
        sys.exit(1)

    updated = _splice_table(original, table)

    if updated == original:
        print("checks.md is already up to date.")
        return

    DOCS_TARGET.write_text(updated, encoding="utf-8")
    print(f"Updated {DOCS_TARGET}")


if __name__ == "__main__":
    main()
