"""Visual Conflict Detector — backward-compatible wrapper.

This module re-exports the public API from the modular ``vcd`` package
so that existing ``from visual_conflict_detector import detect_all_conflicts``
statements continue to work without change.

The actual implementation lives in the ``vcd/`` subpackage:

    vcd/
        __init__.py          — orchestrator + public API
        vcd_config.py        — centralised thresholds
        vcd_core.py          — geometry helpers, ArtistInfo, collector
        vcd_checks_text.py   — passes 1, 5, 8, 9
        vcd_checks_artists.py — passes 2-4, 6-7
        vcd_checks_legend.py — passes 10-13, 15, 18
        vcd_checks_colorbar.py — passes 14, 17
        vcd_checks_structure.py — passes 16, 19-21

Usage (unchanged):
    from visual_conflict_detector import detect_all_conflicts
    issues = detect_all_conflicts(fig, label="my_panel", verbose=True)
"""

from vcd import detect_all_conflicts, detect_conflicts_in_file, summarize_issues

__all__ = ["detect_all_conflicts", "detect_conflicts_in_file", "summarize_issues"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Enhanced visual conflict detection audit (21 passes)")
    parser.add_argument("--series", default="dpmm",
                        choices=["dpmm", "topic"])
    args = parser.parse_args()
    print("VCD is now a modular package. Use: python -m vcd --help")
