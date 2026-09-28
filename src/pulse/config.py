"""
pulse/config.py — Config loader using PyYAML.

Reads config/settings.yaml and returns a dot-accessible dict.
All other modules import `load_config()` rather than hard-coding paths.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


# The project root is three levels up from this file (src/pulse/config.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load settings.yaml once and cache.

    Parameters
    ----------
    path : optional override; useful in tests.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    # Inject project root so all path helpers can resolve relative paths
    cfg["_project_root"] = str(_PROJECT_ROOT)
    return cfg


def resolve_path(key: str, cfg: dict[str, Any] | None = None) -> Path:
    """Return an absolute Path for a key found under cfg['paths']."""
    cfg = cfg or load_config()
    rel = cfg["paths"][key]
    return Path(cfg["_project_root"]) / rel
