# logging_config.py — Structured logging setup
"""
Logging configuration for CLOP-DiT.
"""

import logging
import sys
from pathlib import Path


def setup_logging(
    log_file: str = "logs/clopdit.log",
    level: int = logging.INFO,
    console: bool = True,
):
    """Configure logging for the project.

    Parameters
    ----------
    log_file : str
        Path to log file.
    level : int
        Logging level.
    console : bool
        Also log to console.
    """
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    handlers = [logging.FileHandler(log_file, mode="a")]
    if console:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
