"""
pulse/features.py — Feature engineering for attrition prediction.

DESIGN PRINCIPLES
─────────────────
1. Every feature has a stated domain hypothesis and a column-level docstring.
2. This module is imported identically in training AND inference — no logic
   duplication, no leakage opportunity from a "training-only" code path.
3. Leakage prevention: `churn_probability_true` and `churn_reason` are excluded
   before this module runs (done in pipeline.py). `churn_date` is only used to
   derive `days_to_churn` for analysis — it is never included in model features.
4. All transformations are deterministic given the same input DataFrame.
5. The FEATURE_CATALOG dict documents each feature for the data dictionary.

LEAKAGE NOTE
────────────
The synthetic dataset includes `churn_probability_true` (the ground-truth
logistic score used to generate `churned`). This column MUST NOT be used as a
model feature; it would be trivially predictive and would not exist in production.
The pipeline.py orchestrator drops it (and churn_date, churn_reason) before
calling this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pulse.config import load_config, resolve_path
from pulse.logging_setup import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Feature catalog — documents each engineered feature for the data dictionary
# ---------------------------------------------------------------------------

FEATURE_CATALOG: dict[str, str] = {
    # ── Raw / passthrough features ─────────────────────────────────────────
    "plan_tier": "Subscription plan (starter/growth/professional/enterprise). Higher tiers are stickier.",
    "billing_cycle": "Monthly vs annual billing. Annual subscribers have lower churn risk (commitment effect).",
    "region": "Geographic region of account (NAM/EMEA/APAC/LATAM).",
    "acquisition_channel": "How the subscriber was acquired. Referral/partner channels tend to be higher quality.",
    "industry_vertical": "Subscriber's business vertical. E-commerce and media show higher attrition.",
    "monthly_fee_per_seat": "Fee paid per seat per month (USD). Higher fee → stronger value expectation.",
    "seat_count": "Number of licensed seats. Higher count → greater organisational embedding.",
    "contract_months": "Contract length in months (1 = monthly, 12 = annual).",
    "tenure_months": "Months since subscription start. Longer tenure → lower churn risk (habit formation).",
    "engagement_score": "Platform engagement 0–100. Low engagement is the strongest early warning signal.",
    "support_tickets_90d": "Support tickets in the last 90 days. High count indicates friction.",
    "payment_failures_12m": "Billing failures in the last 12 months. Precedes involuntary churn.",
    "n_integrations": "Number of third-party integrations enabled. Proxy for product stickiness.",
    "monthly_logins_per_user": "Average logins per user per month. Drops before churn.",
    "feature_adoption_rate": "Share of available features actively used (0–1). Low adoption → high risk.",

    # ── Engineered features ────────────────────────────────────────────────
    "plan_tier_enc": "Ordinal encoding of plan_tier (0=starter … 3=enterprise).",
    "billing_annual_flag": "1 if annual billing, 0 if monthly. Binary encoding of commitment level.",
    "monthly_arr_usd": "Annualised Recurring Revenue per account (seat_count × monthly_fee × 12). Business value proxy.",
    "fee_per_seat_log": "Log-transformed monthly_fee_per_seat. Reduces right-skew for linear models.",
    "seat_count_log": "Log-transformed seat_count. Right-skewed distribution.",
    "tenure_log": "Log-transformed tenure_months. Diminishing marginal effect of tenure on churn.",
    "engagement_decile": "Engagement score bucketed into deciles 1–10. Captures non-linearity.",
    "risk_score_simple": "Weighted composite: 0.4×(1-engagement/100) + 0.3×failures/4 + 0.3×tickets/5. Interpretable heuristic.",
    "is_high_ticket_user": "Flag: support_tickets_90d > 3. Identifies friction-heavy accounts.",
    "is_payment_delinquent": "Flag: payment_failures_12m >= 2. Strong involuntary-churn precursor.",
    "is_low_adoption": "Flag: feature_adoption_rate < 0.25. Low adoption predicts disengagement.",
    "seats_x_engagement": "Interaction: seat_count × engagement_score. Large teams with low engagement are high risk.",
    "log_arr_x_tenure": "Interaction: log(ARR) × log(tenure). High-value long-term accounts have different retention dynamics.",
    "channel_is_organic": "Binary: acquisition_channel == 'organic'. Organic users have lower CAC but can be less committed.",
    "channel_is_referral": "Binary: acquisition_channel == 'referral'. Referral users tend to be higher quality.",
}

# Columns used as model features (set at end of build_features)
MODEL_FEATURE_COLUMNS: list[str] = []


# ---------------------------------------------------------------------------
# Plan encoding lookup
# ---------------------------------------------------------------------------

PLAN_ORDER = {"starter": 0, "growth": 1, "professional": 2, "enterprise": 3}


# ---------------------------------------------------------------------------
# Feature engineering functions
# ---------------------------------------------------------------------------

def _encode_plan(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal encode plan_tier (0 = starter, 3 = enterprise).

    Hypothesis: higher-tier plans indicate greater organizational investment
    and correlate with lower attrition.
    """
    df["plan_tier_enc"] = df["plan_tier"].map(PLAN_ORDER).astype("int8")
    return df


def _billing_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag for annual billing.

    Hypothesis: annual subscribers face a switching cost (prepaid commitment)
    that reduces mid-cycle churn.
    """
    df["billing_annual_flag"] = (df["billing_cycle"] == "annual").astype("int8")
    return df


def _compute_arr(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly ARR per account in USD.

    Hypothesis: larger accounts (high ARR) receive more proactive success
    management and churn less, but also feel price pressure more acutely.
    """
    df["monthly_arr_usd"] = (df["monthly_fee_per_seat"] * df["seat_count"] * 12).round(2)
    return df


def _log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transform right-skewed numeric columns.

    Hypothesis: diminishing marginal effects of fee, seat count, and tenure
    on attrition are better captured on a log scale for linear models.
    """
    df["fee_per_seat_log"] = np.log1p(df["monthly_fee_per_seat"])
    df["seat_count_log"] = np.log1p(df["seat_count"])
    df["tenure_log"] = np.log1p(df["tenure_months"])
    return df


def _engagement_decile(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket engagement_score into deciles.

    Hypothesis: the relationship between engagement and churn is non-linear —
    a drop from 30 to 20 matters more than 70 to 60. Deciles capture this
    without requiring a polynomial transformation (which can over-fit).
    """
    df["engagement_decile"] = pd.qcut(
        df["engagement_score"], q=10, labels=False, duplicates="drop"
    ).astype("float32") + 1  # 1-indexed decile
    return df


def _risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Simple weighted composite risk score (business-interpretable heuristic).

    Hypothesis: this linear combination gives a human-readable 0–1 risk proxy
    that a customer-success team can understand without ML, and also serves as
    a feature for the model (captures domain knowledge).
    """
    engagement_risk = 1 - df["engagement_score"].clip(0, 100) / 100.0
    failure_risk = df["payment_failures_12m"].clip(0, 4) / 4.0
    ticket_risk = df["support_tickets_90d"].clip(0, 5) / 5.0
    df["risk_score_simple"] = (
        0.40 * engagement_risk + 0.30 * failure_risk + 0.30 * ticket_risk
    ).round(4)
    return df


def _binary_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-driven binary flags for risk conditions.

    Hypothesis: threshold-based flags mirror the heuristics a CSM team would
    use and add non-linear signal that tree models exploit well.
    """
    df["is_high_ticket_user"] = (df["support_tickets_90d"] > 3).astype("int8")
    df["is_payment_delinquent"] = (df["payment_failures_12m"] >= 2).astype("int8")
    df["is_low_adoption"] = (df["feature_adoption_rate"] < 0.25).astype("int8")
    return df


def _interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Interaction features encoding joint risk signals.

    Hypothesis: large teams (high seats) with low engagement are especially
    risky — the product has penetrated the organisation but users are not
    adopting it, making it vulnerable at renewal.
    """
    df["seats_x_engagement"] = (df["seat_count"] * df["engagement_score"]).round(2)
    df["log_arr_x_tenure"] = (np.log1p(df["monthly_arr_usd"]) * np.log1p(df["tenure_months"])).round(4)
    return df


def _channel_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flags for acquisition channel.

    Hypothesis: channel quality affects long-term retention; referral subscribers
    are self-selected (peer-endorsed product), while paid-search subscribers
    may have lower intent.
    """
    df["channel_is_organic"] = (df["acquisition_channel"] == "organic").astype("int8")
    df["channel_is_referral"] = (df["acquisition_channel"] == "referral").astype("int8")
    return df


def _one_hot_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode remaining nominal categoricals for linear models.

    Tree models use plan_tier_enc (ordinal), but logistic regression needs OHE.
    We use drop_first=False and let the sklearn pipeline's ColumnTransformer
    handle OHE at model time — so we just ensure the raw strings are clean here.
    """
    # Normalise string casing to lower
    for col in ["plan_tier", "billing_cycle", "region", "acquisition_channel", "industry_vertical"]:
        if col in df.columns:
            df[col] = df[col].str.lower().str.strip()
    return df


# ---------------------------------------------------------------------------
# Master feature builder
# ---------------------------------------------------------------------------

#: Columns that must be dropped before modelling (leakage or metadata)
_LEAKAGE_COLUMNS = [
    "churn_probability_true",  # used to generate the label — direct leakage
    "churn_date",              # only available after the event
    "churn_reason",            # only available after the event
    "subscription_start_date", # raw date → use tenure_months instead
    "subscriber_id",           # identifier, not a predictor
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in order.

    Call this with the cleaned DataFrame. Returns the feature-engineered
    DataFrame WITHOUT the target column (use `df["churned"]` separately).

    This function is used identically in training and inference — there is no
    "training-only" code path that could introduce leakage.
    """
    logger.info("Building features on %d rows", len(df))

    # Drop leakage columns (some may not exist, so use errors='ignore')
    df = df.drop(columns=[c for c in _LEAKAGE_COLUMNS if c in df.columns])

    df = _encode_plan(df)
    df = _billing_flag(df)
    df = _compute_arr(df)
    df = _log_transforms(df)
    df = _engagement_decile(df)
    df = _risk_score(df)
    df = _binary_flags(df)
    df = _interaction_terms(df)
    df = _channel_flags(df)
    df = _one_hot_categoricals(df)

    logger.info("Feature engineering complete. Final shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Column groups for the sklearn ColumnTransformer
# ---------------------------------------------------------------------------

#: Numeric columns that go through StandardScaler in the sklearn pipeline
NUMERIC_FEATURES = [
    "monthly_fee_per_seat", "seat_count", "contract_months", "tenure_months",
    "engagement_score", "support_tickets_90d", "payment_failures_12m",
    "n_integrations", "monthly_logins_per_user", "feature_adoption_rate",
    "plan_tier_enc", "monthly_arr_usd", "fee_per_seat_log", "seat_count_log",
    "tenure_log", "engagement_decile", "risk_score_simple",
    "seats_x_engagement", "log_arr_x_tenure",
]

#: Categorical columns for OHE in the sklearn pipeline
CATEGORICAL_FEATURES = [
    "plan_tier", "region", "acquisition_channel", "industry_vertical",
]

#: Binary flags pass through unchanged (already 0/1)
BINARY_FEATURES = [
    "billing_annual_flag", "is_high_ticket_user", "is_payment_delinquent",
    "is_low_adoption", "channel_is_organic", "channel_is_referral",
]

TARGET_COLUMN = "churned"


def save_features(df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> Path:
    """Save the feature-engineered dataset to data/processed/."""
    from pulse.config import load_config, resolve_path
    cfg = cfg or load_config()
    processed = resolve_path("processed", cfg)
    processed.mkdir(parents=True, exist_ok=True)
    out = processed / "subscribers_features.csv"
    df.to_csv(out, index=False)
    logger.info("Feature dataset saved → %s", out)
    return out


def run_feature_engineering(
    df: pd.DataFrame, cfg: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Top-level step called by the pipeline orchestrator."""
    from pulse.config import load_config
    cfg = cfg or load_config()
    featured = build_features(df)
    save_features(featured, cfg)
    return featured
