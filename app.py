"""
app.py — Streamlit Cloud entrypoint for PulseRetain.

This file exists at the repo root so Streamlit Community Cloud can find it
without any special path configuration. It simply delegates to the real
dashboard module in src/pulse/dashboard.py.

Auto-bootstrap: if the SQLite warehouse is missing (fresh cloud deployment),
it runs the lightweight pipeline steps (ingest → clean → database) to
regenerate it from the synthetic data seed. Model predictions (already
committed as CSV) are loaded directly from disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the Python path
_root = Path(__file__).resolve().parent
_src = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# ── Auto-bootstrap if warehouse DB is missing ──────────────────────────────
from pulse.config import load_config, resolve_path

_cfg = load_config()
_db_path = resolve_path("processed", _cfg) / "pulse_warehouse.db"

if not _db_path.exists():
    import streamlit as st
    with st.spinner("🔄 First-time setup: building data warehouse (takes ~30s)…"):
        try:
            from pulse.pipeline import (
                step_ingest, step_validate, step_clean, step_database
            )
            _raw_df = step_ingest(_cfg)
            step_validate(_raw_df, _cfg)
            _cleaned_df = step_clean(_raw_df, _cfg)
            step_database(_cleaned_df, _cfg)
        except Exception as _e:
            st.error(
                f"Bootstrap failed: {_e}\n\n"
                "Please run `python -m pulse.pipeline --step ingest && "
                "python -m pulse.pipeline --step clean && "
                "python -m pulse.pipeline --step db` locally and commit the outputs."
            )
            st.stop()

# ── Load the real dashboard ────────────────────────────────────────────────
# Import and execute the dashboard module (it runs top-level Streamlit code)
import importlib.util

_dashboard_path = _src / "pulse" / "dashboard.py"
_spec = importlib.util.spec_from_file_location("pulse.dashboard", _dashboard_path)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["pulse.dashboard"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
