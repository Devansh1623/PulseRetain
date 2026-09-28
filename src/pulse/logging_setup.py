"""
pulse/logging_setup.py — Structured logging configuration.

Call `get_logger(__name__)` in every module.
Logs go to both console (rich, coloured) and a rotating file.
No print() calls are used anywhere else in the codebase.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

from pulse.config import load_config


def _ensure_log_dir(log_path: str) -> Path:
    p = Path(load_config()["_project_root"]) / log_path
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def setup_logging(level: Optional[str] = None) -> None:
    """Configure root logger once.  Idempotent — safe to call multiple times."""
    cfg = load_config()
    log_cfg = cfg.get("logging", {})
    effective_level = level or os.getenv("LOG_LEVEL", log_cfg.get("level", "INFO"))
    fmt = log_cfg.get("format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    log_file = _ensure_log_dir(log_cfg.get("file", "logs/pulse.log"))

    root = logging.getLogger()
    if root.handlers:
        return  # already configured

    numeric_level = getattr(logging, effective_level.upper(), logging.INFO)
    root.setLevel(numeric_level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(numeric_level)
    ch.setFormatter(logging.Formatter(fmt))
    root.addHandler(ch)

    # Rotating file handler (10 MB × 5 backups)
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(numeric_level)
    fh.setFormatter(logging.Formatter(fmt))
    root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, setting up logging on first call."""
    setup_logging()
    return logging.getLogger(name)
