"""
pulse/cleaning.py — Data cleaning pipeline.

Each cleaning step is a standalone function with a documented rationale.
The cleaning log records what was changed and why, producing an audit trail.

Design choices:
- Missing engagement_score → imputed with plan-tier median (better than global
  median because engagement varies significantly by plan tier).
- Missing monthly_logins → imputed with global median (no strong group signal).
- Out-of-range monthly_fee → rows flagged and replaced with plan-tier median
  (rare, ~0.5 %; dropping would introduce selection bias).
- Duplicate subscriber_ids → keep the first occurrence (earliest row), as it
  represents the original record; duplicates arise from CRM double-sync.
- Impossible dates (churn before start) → churn_date set to null and churned
  set to 0 (the date is untrustworthy; we cannot infer the true churn date).
- Column types normalized at the end (dates to datetime64, flags to int8).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pulse.config import load_config, resolve_path
from pulse.logging_setup import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Individual cleaning steps
# ---------------------------------------------------------------------------

def _remove_duplicates(df: pd.DataFrame, log: list[str]) -> pd.DataFrame:
    """Keep first occurrence of each subscriber_id.

    Rationale: duplicates originate from CRM double-sync (injected at ingestion).
    The first record is the canonical one; subsequent occurrences are artifacts.
    """
    before = len(df)
    df = df.drop_duplicates(subset=["subscriber_id"], keep="first")
    dropped = before - len(df)
    msg = f"Deduplication: removed {dropped} duplicate rows (kept first occurrence per subscriber_id)"
    log.append(msg)
    logger.debug(msg)
    return df


def _fix_impossible_dates(df: pd.DataFrame, log: list[str]) -> pd.DataFrame:
    """Nullify churn_date and reset churned=0 when churn_date < subscription_start_date.

    Rationale: a churn event cannot precede subscription start. The date is
    unreliable; setting churned=0 is conservative (avoids phantom churn signals).
    """
    df = df.copy()
    df["_sd_tmp"] = pd.to_datetime(df["subscription_start_date"], errors="coerce")
    df["_cd_tmp"] = pd.to_datetime(df["churn_date"], errors="coerce")
    bad = df["_cd_tmp"].notna() & (df["_cd_tmp"] < df["_sd_tmp"])
    n_bad = bad.sum()
    if n_bad > 0:
        df.loc[bad, "churn_date"] = None
        df.loc[bad, "churned"] = 0
        msg = f"Date fix: {n_bad} rows with churn_date < subscription_start_date → churn_date nulled, churned=0"
        log.append(msg)
        logger.debug(msg)
    df = df.drop(columns=["_sd_tmp", "_cd_tmp"])
    return df


def _fix_negative_fees(df: pd.DataFrame, log: list[str], cfg: dict[str, Any]) -> pd.DataFrame:
    """Replace out-of-range monthly_fee_per_seat with plan-tier median.

    Rationale: negative or zero fees are data-entry errors. Plan-tier median
    is a better imputation than global median because pricing is tier-specific.
    Dropping these rows would introduce selection bias (corrupt records are not
    random with respect to plan type in real CRM systems).
    """
    df = df.copy()
    fee_min = cfg["pipeline"]["validation"]["monthly_fee_min"]
    bad_mask = df["monthly_fee_per_seat"] < fee_min
    n_bad = bad_mask.sum()
    if n_bad > 0:
        tier_medians = (
            df.loc[~bad_mask]
            .groupby("plan_tier")["monthly_fee_per_seat"]
            .median()
        )
        df.loc[bad_mask, "monthly_fee_per_seat"] = df.loc[bad_mask, "plan_tier"].map(tier_medians)
        msg = (
            f"Fee fix: {n_bad} out-of-range monthly_fee_per_seat values replaced "
            f"with plan-tier median"
        )
        log.append(msg)
        logger.debug(msg)
    return df


def _impute_engagement(df: pd.DataFrame, log: list[str]) -> pd.DataFrame:
    """Impute missing engagement_score with plan-tier median.

    Rationale: app telemetry is not collected for all plans (Starter may have
    limited tracking). Plan-tier median captures this structural difference.
    ~3 % missing — MCAR-like within each tier, so median imputation is valid.
    """
    df = df.copy()
    null_mask = df["engagement_score"].isna()
    n_null = null_mask.sum()
    if n_null > 0:
        tier_medians = df.groupby("plan_tier")["engagement_score"].median()
        df.loc[null_mask, "engagement_score"] = df.loc[null_mask, "plan_tier"].map(tier_medians)
        msg = f"Imputation: {n_null} null engagement_score → plan-tier median"
        log.append(msg)
        logger.debug(msg)
    return df


def _impute_monthly_logins(df: pd.DataFrame, log: list[str]) -> pd.DataFrame:
    """Impute missing monthly_logins_per_user with global median.

    Rationale: ~1.5 % missing, no strong group structure detected in EDA.
    Median is preferred over mean here because the distribution is right-skewed.
    """
    df = df.copy()
    null_mask = df["monthly_logins_per_user"].isna()
    n_null = null_mask.sum()
    if n_null > 0:
        med = df["monthly_logins_per_user"].median()
        df.loc[null_mask, "monthly_logins_per_user"] = med
        msg = f"Imputation: {n_null} null monthly_logins_per_user → global median ({med:.2f})"
        log.append(msg)
        logger.debug(msg)
    return df


def _cap_outliers(df: pd.DataFrame, log: list[str], cfg: dict[str, Any]) -> pd.DataFrame:
    """Winsorise extreme outliers in numeric columns using IQR fencing.

    Rationale: outliers in support_tickets_90d and seat_count can be real
    (a large enterprise might have 500 seats and 20 tickets) so we cap rather
    than drop. The IQR factor from config (default 3.0) is intentionally wide
    to preserve genuine extreme values while catching data errors.
    """
    factor = cfg["pipeline"]["cleaning"]["cap_outliers_iqr_factor"]
    columns_to_cap = ["support_tickets_90d", "monthly_logins_per_user", "total_contract_value"]
    df = df.copy()
    for col in columns_to_cap:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        n_capped = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_capped > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            msg = f"Outlier cap ({col}): {n_capped} values capped to [{lower:.2f}, {upper:.2f}]"
            log.append(msg)
            logger.debug(msg)
    return df


def _normalize_types(df: pd.DataFrame, log: list[str]) -> pd.DataFrame:
    """Normalize column data types to their canonical forms.

    Converts date strings to datetime64, integer flags to int8, and
    rounds float columns to consistent precision.
    """
    df = df.copy()
    df["subscription_start_date"] = pd.to_datetime(df["subscription_start_date"], errors="coerce")
    df["churn_date"] = pd.to_datetime(df["churn_date"], errors="coerce")
    df["churned"] = df["churned"].astype("int8")
    df["seat_count"] = df["seat_count"].astype("int32")
    df["support_tickets_90d"] = df["support_tickets_90d"].astype("int16")
    df["payment_failures_12m"] = df["payment_failures_12m"].astype("int8")
    df["n_integrations"] = df["n_integrations"].astype("int8")
    df["contract_months"] = df["contract_months"].astype("int8")
    df["tenure_months"] = df["tenure_months"].astype("int16")
    log.append("Type normalization: dates → datetime64, integer flags → int8/16/32")
    return df


# ---------------------------------------------------------------------------
# Main cleaning orchestrator
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Run all cleaning steps in the correct order and return cleaned df + audit log.

    The order matters: remove duplicates first (so group medians are not skewed
    by duplicate records), then fix dates, then impute, then cap outliers.
    """
    cfg = cfg or load_config()
    log: list[str] = []

    logger.info("Starting cleaning pipeline on %d rows", len(df))

    df = _remove_duplicates(df, log)
    df = _fix_impossible_dates(df, log)
    df = _fix_negative_fees(df, log, cfg)
    df = _impute_engagement(df, log)
    df = _impute_monthly_logins(df, log)
    df = _cap_outliers(df, log, cfg)
    df = _normalize_types(df, log)
    df = df.reset_index(drop=True)

    logger.info("Cleaning complete. %d rows remaining. %d steps applied.", len(df), len(log))
    return df, log


def save_cleaned(
    df: pd.DataFrame,
    log: list[str],
    cfg: dict[str, Any] | None = None,
) -> Path:
    """Persist cleaned DataFrame to data/interim/. Also saves audit log."""
    cfg = cfg or load_config()
    interim = resolve_path("interim", cfg)
    interim.mkdir(parents=True, exist_ok=True)

    out = interim / "subscribers_cleaned.csv"
    df.to_csv(out, index=False)
    logger.info("Cleaned data saved → %s", out)

    log_path = interim / "cleaning_audit_log.txt"
    log_path.write_text("\n".join(log), encoding="utf-8")
    logger.info("Cleaning audit log → %s", log_path)
    return out


def run_cleaning(df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Top-level step called by pipeline orchestrator."""
    cfg = cfg or load_config()
    cleaned, log = clean(df, cfg)
    save_cleaned(cleaned, log, cfg)
    return cleaned
