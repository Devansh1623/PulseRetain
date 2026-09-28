"""
pulse/pipeline.py — End-to-end pipeline orchestrator.

Runs all steps in order from raw data generation to trained models.
Each step is idempotent: if output files already exist, the step can be
skipped with --skip-if-exists (useful for incremental development).

Usage:
    python -m pulse.pipeline --all
    python -m pulse.pipeline --step ingest
    python -m pulse.pipeline --step clean
    python -m pulse.pipeline --step db
    python -m pulse.pipeline --step features
    python -m pulse.pipeline --step train-sklearn
    python -m pulse.pipeline --step train-tf
    python -m pulse.pipeline --step predict
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from pulse.config import load_config, resolve_path
from pulse.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Individual step runners
# ---------------------------------------------------------------------------

def step_ingest(cfg: dict[str, Any]) -> pd.DataFrame:
    from pulse.ingestion import run_ingestion
    logger.info("═══ STEP: INGEST ═══")
    return run_ingestion(cfg)


def step_validate(raw_df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    from pulse.validation import run_validation
    logger.info("═══ STEP: VALIDATE ═══")
    run_validation(raw_df, cfg)


def step_clean(raw_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    from pulse.cleaning import run_cleaning
    logger.info("═══ STEP: CLEAN ═══")
    return run_cleaning(raw_df, cfg)


def step_database(cleaned_df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    from pulse.database import run_db_pipeline
    logger.info("═══ STEP: DATABASE ═══")
    run_db_pipeline(cleaned_df, cfg)


def step_features(cleaned_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    from pulse.features import run_feature_engineering
    logger.info("═══ STEP: FEATURES ═══")
    return run_feature_engineering(cleaned_df, cfg)


def step_train_sklearn(featured_df: pd.DataFrame, data_hash: str, cfg: dict[str, Any]) -> dict:
    from pulse.modelling_sklearn import run_sklearn_training
    logger.info("═══ STEP: TRAIN SKLEARN ═══")
    return run_sklearn_training(featured_df, data_hash, cfg)


def step_train_tf(featured_df: pd.DataFrame, data_hash: str, cfg: dict[str, Any]) -> dict:
    from pulse.modelling_tf import run_tf_training
    logger.info("═══ STEP: TRAIN TF ═══")
    return run_tf_training(featured_df, data_hash, cfg)


def step_load_predictions(sklearn_results: dict, cfg: dict[str, Any]) -> None:
    """Load best model's predictions into the database for dashboard consumption."""
    from pulse.database import load_predictions, get_connection

    processed_dir = resolve_path("processed", cfg)
    # Use gradient_boost predictions as the primary model displayed in dashboard
    pred_file = processed_dir / "predictions_gradient_boost.csv"
    if not pred_file.exists():
        logger.warning("No gradient_boost predictions found; skipping DB load")
        return

    pred_df = pd.read_csv(pred_file)
    # subscriber_id in predictions is the index from the feature-engineered df
    # Map back to actual subscriber IDs using the cleaned data
    cleaned_file = resolve_path("interim", cfg) / "subscribers_cleaned.csv"
    if cleaned_file.exists():
        cleaned = pd.read_csv(cleaned_file)
        pred_df["subscriber_id"] = cleaned.iloc[pred_df["subscriber_id"].astype(int)]["subscriber_id"].values
    else:
        pred_df["subscriber_id"] = "SUB-" + pred_df["subscriber_id"].astype(str).str.zfill(6)

    load_predictions(pred_df, model_name="gradient_boost", run_id="latest", cfg=cfg)


# ---------------------------------------------------------------------------
# Full pipeline runner
# ---------------------------------------------------------------------------

def run_full_pipeline(cfg: dict[str, Any] | None = None) -> None:
    """Execute all pipeline steps in sequence."""
    cfg = cfg or load_config()

    # 1. Ingest
    raw_df = step_ingest(cfg)

    # 2. Validate
    step_validate(raw_df, cfg)

    # 3. Clean
    cleaned_df = step_clean(raw_df, cfg)

    # 4. Load into database
    step_database(cleaned_df, cfg)

    # 5. Feature engineering
    featured_df = step_features(cleaned_df, cfg)

    # 6. Compute data hash for model provenance
    from pulse.ingestion import data_hash
    dhash = data_hash(cleaned_df)
    logger.info("Data hash: %s", dhash)

    # 7. Train classical models
    sklearn_results = step_train_sklearn(featured_df, dhash, cfg)

    # 8. Train TF model
    step_train_tf(featured_df, dhash, cfg)

    # 9. Load predictions into DB
    step_load_predictions(sklearn_results, cfg)

    logger.info("✓ Full pipeline complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PulseRetain data and ML pipeline"
    )
    parser.add_argument(
        "--step",
        choices=["ingest", "validate", "clean", "db", "features", "train-sklearn", "train-tf", "predict", "all"],
        default="all",
        help="Pipeline step to run (default: all)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config()

    if args.step == "all":
        run_full_pipeline(cfg)
    else:
        # For individual steps, load intermediate outputs where needed
        interim_dir = resolve_path("interim", cfg)
        processed_dir = resolve_path("processed", cfg)

        if args.step == "ingest":
            step_ingest(cfg)

        elif args.step == "validate":
            raw_df = pd.read_csv(resolve_path("raw", cfg) / "subscribers_raw.csv")
            step_validate(raw_df, cfg)

        elif args.step == "clean":
            raw_df = pd.read_csv(resolve_path("raw", cfg) / "subscribers_raw.csv")
            step_clean(raw_df, cfg)

        elif args.step == "db":
            cleaned_df = pd.read_csv(interim_dir / "subscribers_cleaned.csv")
            step_database(cleaned_df, cfg)

        elif args.step == "features":
            cleaned_df = pd.read_csv(interim_dir / "subscribers_cleaned.csv")
            step_features(cleaned_df, cfg)

        elif args.step == "train-sklearn":
            featured_df = pd.read_csv(processed_dir / "subscribers_features.csv")
            from pulse.ingestion import data_hash
            dhash = data_hash(featured_df)
            step_train_sklearn(featured_df, dhash, cfg)

        elif args.step == "train-tf":
            featured_df = pd.read_csv(processed_dir / "subscribers_features.csv")
            from pulse.ingestion import data_hash
            dhash = data_hash(featured_df)
            step_train_tf(featured_df, dhash, cfg)

        elif args.step == "predict":
            step_load_predictions({}, cfg)


if __name__ == "__main__":
    main()
