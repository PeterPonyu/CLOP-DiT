# constants.py — Default prompts, markers, and style for biological validation.

MARKER_GENES = {
    "CD8+ T cells": ["CD8A", "CD8B", "GZMB", "PRF1"],
    "Macrophages": ["CD68", "CD163", "CSF1R", "MSR1"],
    "Epithelial cells": ["EPCAM", "KRT8", "KRT19", "MUC1"],
    "NK cells": ["NKG7", "GNLY", "KLRD1", "FCGR3A"],
}

DEFAULT_TEXT2CELL_PROMPTS = {
    "CD8+ T cells": (
        "CD8+ cytotoxic T lymphocytes from human lung adenocarcinoma tumor "
        "microenvironment. Tumor-infiltrating CD8+ T cells expressing cytotoxic "
        "effector molecules."
    ),
    "Macrophages": (
        "Tumor-associated macrophages from human lung adenocarcinoma. "
        "Myeloid macrophage populations in the cancer microenvironment."
    ),
    "Epithelial cells": (
        "Malignant epithelial cells from human lung adenocarcinoma. "
        "Cancer cells of epithelial origin with tumor-associated expression programs."
    ),
    "NK cells": (
        "Natural killer cells infiltrating human lung adenocarcinoma. "
        "Innate lymphoid NK cells with cytotoxic activity in the tumor microenvironment."
    ),
}

COLORS = {
    "Real": "#2166ac",
    "Generated": "#d6604d",
    "source": "#636363",
    "edited": "#e6550d",
    "target": "#31a354",
}

CELLTYPIST_MODEL = "Human_Lung_Atlas.pkl"
