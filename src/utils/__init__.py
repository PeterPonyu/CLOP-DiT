# utils/__init__.py
try:
    from .helpers import seed_everything, get_device, count_parameters, format_time
except Exception:  # pragma: no cover - allow path/config usage without torch
    seed_everything = None
    get_device = None
    count_parameters = None
    format_time = None
from .logging_config import setup_logging
from .paths import (
    PROJECT_ROOT,
    CACHE_DIR,
    RESULTS_DIR,
    FIG_DIR,
    ARTICLE_FIGURES_DIR,
    CHECKPOINT_DIR,
    CONFIG_DIR,
    LOG_DIR,
    PROCESSED_H5AD_DIR,
    SCGPT_DIR,
    ARTICLE_DIR,
    ARTICLE_TEX,
)
