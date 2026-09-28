"""
pulse/validation.py — Schema and data-quality validation.

Every rule is explicit, config-driven, and produces a structured report rather
than silently dropping or modifying data. The cleaning step reads this report
and decides what to do with each issue.

Design choice: validation is a read-only step — it only diagnoses, never fixes.
This separation makes it easy to audit what was wrong with the raw data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pulse.config import load_config, resolve_path
from pulse.logging_setup import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes for the quality report
# ---------------------------------------------------------------------------

@dataclass
class ColumnReport:
    column: str
    dtype: str
    n_total: int
    n_null: int
    pct_null: float
    n_unique: int
    issues: list[str] = field(default_factory=list)


@dataclass
class QualityReport:
    n_rows: int
    n_cols: int
    n_duplicates: int
    pct_duplicates: float
    columns: list[ColumnReport] = field(default_factory=list)
    global_issues: list[str] = field(default_factory=list)

    def summary(self) -> str:
        total_issues = sum(len(c.issues) for c in self.columns) + len(self.global_issues)
        return (
            f"Rows: {self.n_rows} | Duplicates: {self.n_duplicates} ({self.pct_duplicates:.1f}%) "
            f"| Total issues: {total_issues}"
        )


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class DataValidator:
    """Validates a raw subscriber DataFrame against config rules.

    Produces a QualityReport but does not mutate the DataFrame.
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or load_config()
        self.val_cfg = self.cfg["pipeline"]["validation"]

    def validate(self, df: pd.DataFrame) -> QualityReport:
        logger.info("Running data validation on %d rows × %d cols", len(df), len(df.columns))
        report = QualityReport(
            n_rows=len(df),
            n_cols=len(df.columns),
            n_duplicates=int(df.duplicated(subset=["subscriber_id"]).sum()),
            pct_duplicates=round(df.duplicated(subset=["subscriber_id"]).mean() * 100, 2),
        )

        # Global: duplicate rows
        if report.n_duplicates > 0:
            report.global_issues.append(
                f"{report.n_duplicates} duplicate subscriber_id values ({report.pct_duplicates:.1f}%)"
            )

        # Global: impossible date ordering (churn_date < subscription_start_date)
        mask_both = df["churn_date"].notna() & df["subscription_start_date"].notna()
        if mask_both.any():
            sub = df.loc[mask_both].copy()
            sub["_sd"] = pd.to_datetime(sub["subscription_start_date"], errors="coerce")
            sub["_cd"] = pd.to_datetime(sub["churn_date"], errors="coerce")
            bad_dates = (sub["_cd"] < sub["_sd"]).sum()
            if bad_dates > 0:
                report.global_issues.append(
                    f"{bad_dates} rows where churn_date precedes subscription_start_date"
                )

        # Per-column validation
        col_rules = {
            "subscriber_id": {"not_null": True, "unique": True},
            "plan_tier": {"not_null": True, "allowed": self.val_cfg["allowed_plans"]},
            "billing_cycle": {"not_null": True, "allowed": self.val_cfg["allowed_billing_cycles"]},
            "region": {"not_null": True, "allowed": self.val_cfg["allowed_regions"]},
            "acquisition_channel": {"not_null": True, "allowed": self.val_cfg["allowed_acquisition_channels"]},
            "monthly_fee_per_seat": {
                "not_null": True,
                "min": self.val_cfg["monthly_fee_min"],
                "max": self.val_cfg["monthly_fee_max"],
            },
            "seat_count": {
                "not_null": True,
                "min": self.val_cfg["seats_min"],
                "max": self.val_cfg["seats_max"],
            },
            "contract_months": {"not_null": True, "min": 1, "max": 36},
            "tenure_months": {"not_null": True, "min": 0},
            "engagement_score": {"min": 0.0, "max": 100.0},
            "support_tickets_90d": {"not_null": True, "min": 0},
            "payment_failures_12m": {"not_null": True, "min": 0, "max": 12},
            "n_integrations": {"not_null": True, "min": 0},
            "feature_adoption_rate": {"min": 0.0, "max": 1.0},
            "churned": {"not_null": True, "allowed": [0, 1]},
        }

        for col, rules in col_rules.items():
            if col not in df.columns:
                report.global_issues.append(f"Expected column '{col}' is missing")
                continue

            series = df[col]
            n_null = int(series.isna().sum())
            col_report = ColumnReport(
                column=col,
                dtype=str(series.dtype),
                n_total=len(series),
                n_null=n_null,
                pct_null=round(n_null / len(series) * 100, 2),
                n_unique=int(series.nunique(dropna=True)),
            )

            if rules.get("not_null") and n_null > 0:
                col_report.issues.append(f"{n_null} nulls ({col_report.pct_null:.1f}%)")

            if "allowed" in rules:
                non_null = series.dropna()
                bad = ~non_null.astype(str).str.lower().isin(
                    [str(v).lower() for v in rules["allowed"]]
                )
                if bad.sum() > 0:
                    col_report.issues.append(
                        f"{bad.sum()} values not in allowed set: {rules['allowed']}"
                    )

            if "min" in rules:
                numeric = pd.to_numeric(series, errors="coerce")
                below = (numeric < rules["min"]).sum()
                if below > 0:
                    col_report.issues.append(f"{below} values below minimum {rules['min']}")

            if "max" in rules:
                numeric = pd.to_numeric(series, errors="coerce")
                above = (numeric > rules["max"]).sum()
                if above > 0:
                    col_report.issues.append(f"{above} values above maximum {rules['max']}")

            if rules.get("unique") and series.dropna().duplicated().sum() > 0:
                col_report.issues.append(
                    f"{series.dropna().duplicated().sum()} duplicate values (expected unique)"
                )

            report.columns.append(col_report)

        logger.info("Validation complete. %s", report.summary())
        return report


def save_quality_report(report: QualityReport, cfg: dict[str, Any] | None = None) -> Path:
    """Write the quality report to interim/ as JSON and a human-readable TXT."""
    cfg = cfg or load_config()
    interim = resolve_path("interim", cfg)
    interim.mkdir(parents=True, exist_ok=True)

    # JSON version (machine-readable)
    json_path = interim / "data_quality_report.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, default=str)

    # Human-readable summary
    txt_path = interim / "data_quality_report.txt"
    lines = [
        "=== PulseRetain Data Quality Report ===",
        f"Rows: {report.n_rows}  |  Columns: {report.n_cols}",
        f"Duplicate subscriber_ids: {report.n_duplicates} ({report.pct_duplicates:.1f}%)",
        "",
        "--- Global Issues ---",
    ]
    if report.global_issues:
        lines.extend(f"  • {issue}" for issue in report.global_issues)
    else:
        lines.append("  (none)")
    lines += ["", "--- Column Issues ---"]
    for col in report.columns:
        if col.issues:
            lines.append(f"  {col.column} (null: {col.pct_null:.1f}%):")
            lines.extend(f"      - {iss}" for iss in col.issues)
    if not any(c.issues for c in report.columns):
        lines.append("  (none beyond nulls already reported)")

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Quality report saved → %s and %s", json_path, txt_path)
    return json_path


def run_validation(df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> QualityReport:
    """Top-level validation step called by the pipeline orchestrator."""
    cfg = cfg or load_config()
    validator = DataValidator(cfg)
    report = validator.validate(df)
    save_quality_report(report, cfg)
    return report
