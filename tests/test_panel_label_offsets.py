"""Allowlist-scoped panel-label offset enforcement.

Asserts that every ``add_panel_label(`` call in the strict-scan files passes
``x=`` and ``y=`` values that are either:

  1. A subscript of one of the four canonical ``PANEL_OFFSET_*`` constants
     (``PANEL_OFFSET_STD``, ``PANEL_OFFSET_LEFT``, ``PANEL_OFFSET_WIDE``,
     ``PANEL_OFFSET_TIGHT``), or
  2. A bare numeric zero (``0`` / ``0.0`` / ``0.00``) — allowed as a
     judgment-call for column-3 panels (e.g. fig06 G3 ``x=0.00``) where
     subscript syntax would push the label off-canvas.

Allowlist scoped per
``.omc/plans/revision-figure-polish-2026-04-19.md §Allowlist``.
Extending this scan to the remaining 29 visualization files is deferred to a
follow-up A2-full migration PR.

Files excluded from the panel-label scan (in the broader 8-file allowlist but
their ``add_panel_label`` sites were NOT migrated in Phase 3):

  - ``fig11_conditioning.py`` — panel J/K intentionally kept per-file offset
    literals (plan §"Figure 5 J/K").
  - ``fig16_benchmark.py`` — Phase I-a only touched L281/L283/legend; panel
    labels were not in scope.
  - ``fig09_expression_corr.py`` — Phase 4 annotation-loop change only; panel
    labels not migrated.
  - ``fig10_expression_analysis.py`` — same as fig09.
"""

import ast
import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Repository root (two levels up from this file: tests/ -> repo root)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent.resolve()

# ---------------------------------------------------------------------------
# Strict-scan targets: files whose ALL ``add_panel_label`` calls were
# migrated to PANEL_OFFSET_* subscripts in Phase 3.
# ---------------------------------------------------------------------------
_STRICT_SCAN_FILES = [
    _REPO_ROOT / "src/visualization/fig03_training.py",
    _REPO_ROOT / "src/visualization/fig05_metrics.py",
    _REPO_ROOT / "src/visualization/fig06_fidelity.py",
    _REPO_ROOT / "src/visualization/fig07_alignment.py",
]

# ---------------------------------------------------------------------------
# Canonical constant names (the four PANEL_OFFSET_* exported from style.py).
# ---------------------------------------------------------------------------
_CANONICAL_CONSTANTS = {
    "PANEL_OFFSET_STD",
    "PANEL_OFFSET_LEFT",
    "PANEL_OFFSET_FARLEFT",
    "PANEL_OFFSET_WIDE",
    "PANEL_OFFSET_TIGHT",
}


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _is_panel_offset_subscript(node: ast.expr) -> bool:
    """Return True if *node* is ``PANEL_OFFSET_<NAME>[<int>]``."""
    if not isinstance(node, ast.Subscript):
        return False
    # The value being subscripted must be one of the canonical names.
    value = node.value
    if not isinstance(value, ast.Name):
        return False
    if value.id not in _CANONICAL_CONSTANTS:
        return False
    # The slice must be an integer constant (0 or 1).
    slc = node.slice
    if isinstance(slc, ast.Constant) and isinstance(slc.value, int):
        return True
    # Python < 3.9 wraps the slice in an ast.Index node.
    if isinstance(slc, ast.Index):  # type: ignore[attr-defined]
        inner = slc.value  # type: ignore[attr-defined]
        if isinstance(inner, ast.Constant) and isinstance(inner.value, int):
            return True
    return False


def _is_bare_zero(node: ast.expr) -> bool:
    """Return True if *node* is the numeric literal 0 (int or float)."""
    if not isinstance(node, ast.Constant):
        return False
    return node.value == 0 or node.value == 0.0


def _is_allowed_offset_value(node: ast.expr) -> bool:
    """Return True if *node* is an allowed x= / y= value."""
    return _is_panel_offset_subscript(node) or _is_bare_zero(node)


def _collect_violations(source_path: Path) -> list[str]:
    """Parse *source_path* and return a list of human-readable violation
    strings, one per offending keyword argument.

    A violation is any ``x=`` or ``y=`` kwarg in an ``add_panel_label(``
    call whose value is neither a ``PANEL_OFFSET_*[n]`` subscript nor a bare
    zero literal.
    """
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))

    violations: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        # Match ``add_panel_label(...)`` whether called as a bare name or
        # as an attribute (e.g. ``style.add_panel_label(...)``).
        func = node.func
        if isinstance(func, ast.Name):
            func_name = func.id
        elif isinstance(func, ast.Attribute):
            func_name = func.attr
        else:
            continue

        if func_name != "add_panel_label":
            continue

        # Examine x= and y= keyword arguments.
        for kw in node.keywords:
            if kw.arg not in ("x", "y"):
                continue
            if not _is_allowed_offset_value(kw.value):
                lineno = getattr(kw.value, "lineno", getattr(node, "lineno", "?"))
                # Unparse the offending value for the error message.
                try:
                    value_src = ast.unparse(kw.value)
                except AttributeError:
                    # ast.unparse added in Python 3.9; fall back gracefully.
                    value_src = repr(kw.value)
                violations.append(
                    f"  {source_path.name}:{lineno}  {kw.arg}={value_src}"
                )

    return violations


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPanelLabelOffsets:
    """Every add_panel_label call in the strict-scan files must use a
    canonical PANEL_OFFSET_* subscript or a bare zero for x=/y=."""

    @pytest.mark.parametrize("src_file", _STRICT_SCAN_FILES, ids=lambda p: p.name)
    def test_scan_file_exists(self, src_file: Path):
        """Each strict-scan file must exist on disk before we can audit it."""
        assert src_file.exists(), (
            f"Strict-scan file missing: {src_file}\n"
            "If the file was renamed, update _STRICT_SCAN_FILES in this test."
        )

    @pytest.mark.parametrize("src_file", _STRICT_SCAN_FILES, ids=lambda p: p.name)
    def test_all_add_panel_label_calls_use_canonical_offsets(self, src_file: Path):
        """add_panel_label x=/y= args must be PANEL_OFFSET_*[n] or bare 0."""
        if not src_file.exists():
            pytest.skip(f"Source file not found: {src_file}")

        violations = _collect_violations(src_file)

        assert not violations, (
            f"\n{src_file.name} has {len(violations)} add_panel_label call(s) "
            "with non-canonical offset values.\n"
            "Each x= / y= argument must be one of:\n"
            "  PANEL_OFFSET_STD[0/1]     PANEL_OFFSET_LEFT[0/1]\n"
            "  PANEL_OFFSET_FARLEFT[0/1] PANEL_OFFSET_WIDE[0/1]\n"
            "  PANEL_OFFSET_TIGHT[0/1]\n"
            "  or a bare 0 / 0.0 (allowed for column-3 panels per fig06 G3).\n\n"
            "Offending call sites:\n"
            + "\n".join(violations)
        )

    def test_strict_scan_list_covers_expected_four_files(self):
        """Guard against accidental truncation of the strict-scan list."""
        expected_names = {
            "fig03_training.py",
            "fig05_metrics.py",
            "fig06_fidelity.py",
            "fig07_alignment.py",
        }
        actual_names = {p.name for p in _STRICT_SCAN_FILES}
        assert actual_names == expected_names, (
            "The strict-scan list no longer matches the expected 4 files.\n"
            f"Expected: {sorted(expected_names)}\n"
            f"Got:      {sorted(actual_names)}"
        )

    def test_canonical_constant_set_has_five_entries(self):
        """Guard: _CANONICAL_CONSTANTS must name exactly the five style exports."""
        assert len(_CANONICAL_CONSTANTS) == 5
        for name in _CANONICAL_CONSTANTS:
            assert name.startswith("PANEL_OFFSET_"), (
                f"Unexpected entry in _CANONICAL_CONSTANTS: {name!r}"
            )
