# Visual Conflict Detection (VCD)

> Subsystem for detecting and resolving visual conflicts in generated figures.

**Location**: `scripts/vcd/`
**Scripts**: 12
**Version**: v6.0+

---

## Overview

The VCD subsystem provides automated visual conflict detection for publication-ready figures. It identifies:
- Overlapping labels
- Truncated content
- Color conflicts
- Font size inconsistencies

---

## Usage

```bash
# Run VCD check on all panels
python scripts/vcd/check_all_panels.py

# Check specific panel
python scripts/vcd/check_panel.py --panel A
```

---

## Integration

VCD is automatically run during:
- `scripts/pipeline/regenerate_report.sh`
- `scripts/pipeline/build_article.sh`

See [Pipeline Index](../pipeline/INDEX.md) for details.

---

## See Also

- [Visualization](../../src/visualization/) - Figure generation modules
- [Article Building](../pipeline/INDEX.md#article-building)
