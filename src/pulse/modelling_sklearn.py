"""
pulse/modelling_sklearn.py — Classical ML pipeline with scikit-learn.

MODELS IMPLEMENTED
──────────────────
1. Naive baseline (stratified majority-class prediction) — sets the floor.
2. Logistic Regression with L2 regularization (linear baseline with calibration).
3. Random Forest — captures non-linearities and feature interactions.
4. Gradient Boosting (HistGradientBoostingClassifier) — handles mixed types
   natively, often the best classical model on tabular data.

All models use an identical ColumnTransformer preprocessing pipeline so that
comparisons are fair. Cross-validation is performed inside the pipeline to
prevent any data from the validation/test set influencing preprocessing.

TARGET: binary classification — churned (0 = retained, 1 = churned).
METRIC: ROC-AUC (primary) + PR-AUC, F1, Precision, Recall at operating threshold.
THRESHOLD: 0.40 (lower than 0.50 default) because the cost of a false negative
           (missing a churning customer) exceeds the cost of a false positive
           (unnecessarily contacting a retained customer).
"""

from __future__ import annotations

import json
import pickle
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pulse.config import load_config, resolve_path
from pulse.features import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from pulse.logging_setup import get_logger

logger = get_logger(__name__)

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Preprocessing transformer
# ---------------------------------------------------------------------------

def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    binary_cols: list[str],
) -> ColumnTransformer:
    """Build a ColumnTransformer that:
    - Standardizes numeric features (zero mean, unit variance)
    - One-hot encodes categorical features (drop='first' to avoid multicollinearity
      in logistic regression; tree models are unaffected)
    - Passes binary flags through unchanged (already 0/1)

    Using a ColumnTransformer inside a Pipeline prevents any leakage between
    the train and test sets — fit() is only called on training data.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_cols),
            ("bin", "passthrough", binary_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified train/val/test split.

    The test set is held out until final evaluation — not used for any
    model selection or hyperparameter tuning decisions.
    """
    cfg = cfg or load_config()
    test_size = cfg["model"]["test_size"]
    val_size = cfg["model"]["val_size"]

    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED
    )

    # Then split train/val from the remaining
    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, stratify=y_temp, random_state=RANDOM_SEED
    )

    logger.info(
        "Split: train=%d, val=%d, test=%d | churn rate: train=%.1f%% val=%.1f%% test=%.1f%%",
        len(X_train), len(X_val), len(X_test),
        y_train.mean() * 100, y_val.mean() * 100, y_test.mean() * 100,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.40) -> dict[str, float]:
    """Compute a comprehensive metric set at the given operating threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Individual model builders
# ---------------------------------------------------------------------------

def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    binary_cols: list[str],
) -> Pipeline:
    """Stratified dummy baseline — predicts the class distribution randomly.
    This sets the floor: any real model must beat this to be worth deploying.
    """
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, binary_cols)
    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", DummyClassifier(strategy="stratified", random_state=RANDOM_SEED)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    binary_cols: list[str],
    cfg: dict[str, Any] | None = None,
) -> Pipeline:
    """Logistic Regression with L2 regularization.

    WHY: interpretable, well-calibrated probabilities, fast to train.
    L2 regularization prevents overfitting on the one-hot encoded categories.
    C=1.0 is the sklearn default and works well for this data size.
    """
    cfg = cfg or load_config()
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, binary_cols)
    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced",
            solver="lbfgs", random_state=RANDOM_SEED,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    binary_cols: list[str],
    cfg: dict[str, Any] | None = None,
) -> Pipeline:
    """Random Forest with balanced class weights.

    WHY: captures non-linear interactions between features (e.g. the
    joint effect of low engagement AND high payment failures) without
    requiring explicit interaction terms. Robust to outliers and scales
    of features (so StandardScaler is less critical for RF, but kept for
    consistency). class_weight='balanced' handles class imbalance.
    """
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, binary_cols)
    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_gradient_boost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    binary_cols: list[str],
    cfg: dict[str, Any] | None = None,
) -> Pipeline:
    """HistGradientBoostingClassifier — modern gradient boosting.

    WHY: faster than XGBoost for this data size, handles missing values
    natively (via surrogate splits), supports class_weight='balanced',
    and often achieves the best performance on tabular data by efficiently
    binning numeric features into histograms before splitting.
    """
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, binary_cols)
    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", HistGradientBoostingClassifier(
            max_iter=400,
            max_depth=6,
            learning_rate=0.05,
            l2_regularization=0.1,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


# ---------------------------------------------------------------------------
# Cross-validation helper
# ---------------------------------------------------------------------------

def cross_validate_model(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
) -> dict[str, float]:
    """Run stratified k-fold CV and return mean ± std for ROC-AUC and PR-AUC."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    roc_scores = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    pr_scores = cross_val_score(pipe, X, y, cv=skf, scoring="average_precision", n_jobs=-1)
    return {
        "cv_roc_auc_mean": round(roc_scores.mean(), 4),
        "cv_roc_auc_std": round(roc_scores.std(), 4),
        "cv_pr_auc_mean": round(pr_scores.mean(), 4),
        "cv_pr_auc_std": round(pr_scores.std(), 4),
    }


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(
    pipe: Pipeline,
    model_name: str,
    metrics: dict[str, Any],
    data_hash: str,
    cfg: dict[str, Any] | None = None,
) -> tuple[str, Path]:
    """Persist model artifact + metadata.  Returns (run_id, artifact_dir)."""
    cfg = cfg or load_config()
    run_id = f"{model_name}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
    artifacts_dir = resolve_path("models", cfg) / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(pipe, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "run_id": run_id,
        "model_name": model_name,
        "trained_at": datetime.utcnow().isoformat(),
        "data_hash": data_hash,
        "metrics": metrics,
        "artifact_path": str(model_path),
    }
    meta_path = artifacts_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    logger.info("Model saved: %s → %s", model_name, artifacts_dir)
    return run_id, artifacts_dir


def load_model(run_id: str, cfg: dict[str, Any] | None = None) -> Pipeline:
    """Load a previously saved model artifact."""
    cfg = cfg or load_config()
    model_path = resolve_path("models", cfg) / run_id / "model.pkl"
    with model_path.open("rb") as f:
        pipe = pickle.load(f)
    return pipe


# ---------------------------------------------------------------------------
# Master training orchestrator
# ---------------------------------------------------------------------------

def run_sklearn_training(
    df: pd.DataFrame,
    data_hash: str,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train all classical models, evaluate, save, and return comparison dict."""
    cfg = cfg or load_config()
    threshold = cfg["model"]["threshold"]
    cv_folds = cfg["model"]["cv_folds"]

    # Determine available feature columns
    available_numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    available_categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    available_binary = [c for c in BINARY_FEATURES if c in df.columns]

    logger.info(
        "Features: %d numeric, %d categorical, %d binary",
        len(available_numeric), len(available_categorical), len(available_binary),
    )

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, cfg)

    # Save splits for TF model and test scripts
    processed_dir = resolve_path("processed", cfg)
    X_test.assign(**{TARGET_COLUMN: y_test}).to_csv(processed_dir / "test_set.csv", index=False)
    X_train.assign(**{TARGET_COLUMN: y_train}).to_csv(processed_dir / "train_set.csv", index=False)

    results = {}

    for model_name, train_fn in [
        ("baseline", train_baseline),
        ("logistic_regression", train_logistic),
        ("random_forest", train_random_forest),
        ("gradient_boost", train_gradient_boost),
    ]:
        logger.info("Training %s ...", model_name)
        try:
            pipe = train_fn(X_train, y_train, available_numeric, available_categorical, available_binary, cfg)
        except TypeError:
            # baseline doesn't accept cfg
            pipe = train_fn(X_train, y_train, available_numeric, available_categorical, available_binary)

        # Validation metrics
        y_prob_val = pipe.predict_proba(X_val)[:, 1]
        val_metrics = compute_metrics(y_val, y_prob_val, threshold)

        # CV (skip for dummy baseline — it's trivially fast but results aren't informative)
        if model_name != "baseline":
            cv_results = cross_validate_model(pipe, X_train, y_train, cv_folds)
            val_metrics.update(cv_results)

        # Test set evaluation (ONLY ONCE — kept separate for final reporting)
        y_prob_test = pipe.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, y_prob_test, threshold)

        # Save model
        run_id, artifact_dir = save_model(pipe, model_name, {"val": val_metrics, "test": test_metrics}, data_hash, cfg)

        # Save predictions for this model
        pred_df = X_test.copy()
        pred_df["subscriber_id"] = X_test.index.astype(str)
        pred_df["churn_probability"] = y_prob_test
        pred_df["churned_actual"] = y_test.values
        pred_df[["subscriber_id", "churn_probability", "churned_actual"]].to_csv(
            processed_dir / f"predictions_{model_name}.csv", index=False
        )

        results[model_name] = {
            "pipe": pipe,
            "run_id": run_id,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "y_prob_test": y_prob_test,
            "y_test": y_test,
        }

        logger.info(
            "%s | val ROC-AUC=%.4f | test ROC-AUC=%.4f",
            model_name, val_metrics["roc_auc"], test_metrics["roc_auc"],
        )

    # Save comparison table
    comparison = []
    for name, res in results.items():
        row = {"model": name}
        row.update(res["test_metrics"])
        comparison.append(row)
    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(processed_dir / "model_comparison.csv", index=False)
    logger.info("Model comparison saved → %s/model_comparison.csv", processed_dir)

    # Also export as Excel for stakeholders
    try:
        comparison_df.to_excel(processed_dir / "model_comparison.xlsx", index=False, sheet_name="Model Comparison")
    except Exception as e:
        logger.warning("Could not save Excel comparison: %s", e)

    return results
