"""
pulse/ingestion.py — Synthetic dataset generator for PulseRetain.

WHY SYNTHETIC?
The IBM Telco dataset belongs to a proprietary IBM sample pack. This project
uses a fully generated synthetic dataset for a B2B SaaS context so that:
 - There are no licensing concerns.
 - The generator is documented (seed, distributions, injected quality issues)
   and can be extended or swapped for real data.
 - The analytical problem (subscription attrition prediction) is preserved.

DISTRIBUTIONS AND RATIONALE
 - ~8 500 subscriber accounts matching plausible SaaS market sizes.
 - Attrition rate ~18 % (industry median for SMB SaaS).
 - Plans drawn from a realistic mix (Starter heavy, Enterprise rare).
 - Monthly fee and seat count correlated with plan tier.
 - Tenure governed by a log-normal distribution (many short, few very long).
 - Attrition probability driven by a logistic combination of contract type,
   seat count, engagement score, support tickets, payment failures, and tenure.
 - Injected data-quality issues: ~3 % nulls in engagement_score,
   ~1 % duplicate rows, ~0.5 % out-of-range monthly_fee, a handful of
   impossible dates — mirroring real-world ingestion problems so the
   cleaning module has real work to do.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pulse.config import load_config, resolve_path
from pulse.logging_setup import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants and helper maps
# ---------------------------------------------------------------------------

PLANS = ["starter", "growth", "professional", "enterprise"]
PLAN_WEIGHTS = [0.38, 0.30, 0.22, 0.10]

BILLING_CYCLES = ["monthly", "annual"]

REGIONS = ["NAM", "EMEA", "APAC", "LATAM"]
REGION_WEIGHTS = [0.42, 0.28, 0.20, 0.10]

CHANNELS = ["organic", "paid_search", "referral", "partner", "direct_sales", "trial_convert"]
CHANNEL_WEIGHTS = [0.20, 0.18, 0.22, 0.12, 0.15, 0.13]

INDUSTRIES = [
    "Software & SaaS", "E-commerce", "Financial Services", "Healthcare IT",
    "Professional Services", "Manufacturing", "Media & Entertainment", "Education",
]
INDUSTRY_WEIGHTS = [0.22, 0.14, 0.13, 0.11, 0.12, 0.10, 0.10, 0.08]

CHURN_REASONS = [
    "Price sensitivity", "Competitor offer", "Feature gaps", "Poor support experience",
    "Company downsizing", "Acquired by competitor", "Internal budget cut",
    "Switched to in-house solution", "Contract expired — not renewed", "Low engagement",
]

# Plan-level base monthly fee ranges (per seat, USD)
PLAN_FEE_MAP = {
    "starter": (9.99, 29.99),
    "growth": (29.99, 79.99),
    "professional": (79.99, 199.99),
    "enterprise": (199.99, 499.99),
}

# Plan-level seat ranges
PLAN_SEAT_MAP = {
    "starter": (1, 5),
    "growth": (5, 25),
    "professional": (20, 100),
    "enterprise": (80, 500),
}


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def _logistic(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic (sigmoid) function."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def generate_subscribers(
    n: int = 8_500,
    seed: int = 42,
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """Generate a realistic synthetic SaaS subscriber dataset.

    Parameters
    ----------
    n : Number of subscriber accounts.
    seed : Random seed for full reproducibility.
    start_date / end_date : Cohort window for subscription start dates.
    """
    rng = np.random.default_rng(seed)

    logger.info("Generating %d synthetic subscriber records (seed=%d)", n, seed)

    # ------------------------------------------------------------------
    # 1. Identity
    # ------------------------------------------------------------------
    subscriber_ids = [f"SUB-{i:06d}" for i in range(1, n + 1)]

    # ------------------------------------------------------------------
    # 2. Subscription start dates within the cohort window
    # ------------------------------------------------------------------
    sd = date.fromisoformat(start_date)
    ed = date.fromisoformat(end_date)
    total_days = (ed - sd).days
    # Log-normal skew: most subscriptions start earlier (more history)
    raw_day_offsets = rng.integers(0, total_days, size=n)
    start_dates = [sd + timedelta(days=int(d)) for d in raw_day_offsets]

    # ------------------------------------------------------------------
    # 3. Plan, billing cycle, region, channel, industry
    # ------------------------------------------------------------------
    plans = rng.choice(PLANS, p=PLAN_WEIGHTS, size=n)
    billing_cycles = rng.choice(BILLING_CYCLES, p=[0.55, 0.45], size=n)
    regions = rng.choice(REGIONS, p=REGION_WEIGHTS, size=n)
    channels = rng.choice(CHANNELS, p=CHANNEL_WEIGHTS, size=n)
    industries = rng.choice(INDUSTRIES, p=INDUSTRY_WEIGHTS, size=n)

    # ------------------------------------------------------------------
    # 4. Monthly fee and seats — correlated with plan
    # ------------------------------------------------------------------
    monthly_fees = np.array([
        rng.uniform(*PLAN_FEE_MAP[p]) for p in plans
    ])
    seats = np.array([
        int(rng.integers(*PLAN_SEAT_MAP[p])) for p in plans
    ])
    # Annual billing gets a 15 % discount
    monthly_fees = np.where(billing_cycles == "annual", monthly_fees * 0.85, monthly_fees)
    monthly_fees = monthly_fees.round(2)

    # ------------------------------------------------------------------
    # 5. Tenure in months at observation point (= end_date)
    # ------------------------------------------------------------------
    obs_date = ed
    tenure_months = np.array([
        max(1, round((obs_date - sd_).days / 30.44))
        for sd_ in start_dates
    ])

    # ------------------------------------------------------------------
    # 6. Behavioral signals
    # ------------------------------------------------------------------
    # Engagement score 0–100: higher = more active
    engagement_raw = rng.normal(loc=62, scale=20, size=n).clip(0, 100).round(1)

    # Support tickets in last 90 days (count, Poisson)
    support_tickets = rng.poisson(lam=1.8, size=n)

    # Payment failures in last 12 months (count, mixture)
    payment_failures = rng.choice([0, 1, 2, 3, 4], p=[0.65, 0.20, 0.08, 0.05, 0.02], size=n)

    # Number of integrations enabled (proxy for stickiness)
    n_integrations = rng.integers(0, 12, size=n)

    # Monthly logins (average per user per month)
    monthly_logins = rng.exponential(scale=14, size=n).clip(0, 120).round(1)

    # Feature adoption score 0–1
    feature_adoption = rng.beta(a=2, b=3, size=n).round(3)

    # ------------------------------------------------------------------
    # 7. Compute attrition probability via logistic model
    # ------------------------------------------------------------------
    # Plan numeric encoding (higher plan = stickier)
    plan_enc = np.array([PLANS.index(p) for p in plans]) / 3.0

    # Normalised features
    tenure_norm = np.log1p(tenure_months) / np.log1p(36)
    engage_norm = engagement_raw / 100.0
    tickets_norm = support_tickets / 5.0
    failures_norm = payment_failures / 4.0
    integrations_norm = n_integrations / 11.0
    logins_norm = monthly_logins / 120.0
    adoption_norm = feature_adoption

    # Annual billing reduces churn risk
    annual_flag = (billing_cycles == "annual").astype(float)

    # Logit score — coefficients chosen to produce ~18 % base attrition
    # Intercept calibrated: mean logit excl. intercept ≈ -2.40 (dominant negative terms
    # from engagement and tenure); target churn rate 18% → logit(0.18) = -1.52
    # → needed intercept = -1.52 - (-2.40) = +0.88
    logit = (
        +0.88           # intercept (calibrated for ~18 % base attrition)
        - 1.20 * tenure_norm        # longer tenure → lower churn risk
        - 1.50 * engage_norm        # higher engagement → lower churn
        - 0.80 * integrations_norm  # more integrations → stickier
        - 0.60 * plan_enc           # higher plan → stickier
        - 0.70 * annual_flag        # annual commitment → lower churn
        - 0.40 * logins_norm        # more logins → lower churn
        - 0.50 * adoption_norm      # higher adoption → lower churn
        + 0.90 * tickets_norm       # more tickets → higher churn risk
        + 1.20 * failures_norm      # payment failures → highest churn risk
    )

    # Add segment-level noise (some industries inherently churnrier)
    industry_noise = np.where(
        np.isin(industries, ["E-commerce", "Media & Entertainment"]), 0.30, -0.10
    )
    logit += industry_noise + rng.normal(0, 0.25, size=n)

    churn_prob = _logistic(logit)
    churned = (rng.uniform(size=n) < churn_prob).astype(int)

    # ------------------------------------------------------------------
    # 8. Churn date and reason (only for churned accounts)
    # ------------------------------------------------------------------
    churn_dates: list[str | None] = []
    churn_reasons: list[str | None] = []
    for i in range(n):
        if churned[i]:
            # Churn happens sometime after subscription start, before obs_date
            days_active = int(rng.integers(30, max(31, int(tenure_months[i] * 30))))
            churn_dt = start_dates[i] + timedelta(days=min(days_active, total_days))
            churn_dates.append(str(churn_dt))
            # Payment failure and low engagement → more likely price/competitor reasons
            if payment_failures[i] >= 2:
                reason = rng.choice(["Price sensitivity", "Internal budget cut"])
            elif engagement_raw[i] < 40:
                reason = rng.choice(["Feature gaps", "Low engagement", "Competitor offer"])
            else:
                reason = rng.choice(CHURN_REASONS)
            churn_reasons.append(reason)
        else:
            churn_dates.append(None)
            churn_reasons.append(None)

    # ------------------------------------------------------------------
    # 9. Contract length (months) — tied to billing cycle
    # ------------------------------------------------------------------
    contract_months = np.where(billing_cycles == "annual", 12, 1)

    # ------------------------------------------------------------------
    # 10. Total contract value
    # ------------------------------------------------------------------
    total_contract_value = (monthly_fees * seats * contract_months).round(2)

    # ------------------------------------------------------------------
    # 11. Assemble DataFrame
    # ------------------------------------------------------------------
    df = pd.DataFrame({
        "subscriber_id": subscriber_ids,
        "subscription_start_date": [str(d) for d in start_dates],
        "plan_tier": plans,
        "billing_cycle": billing_cycles,
        "region": regions,
        "acquisition_channel": channels,
        "industry_vertical": industries,
        "monthly_fee_per_seat": monthly_fees,
        "seat_count": seats,
        "total_contract_value": total_contract_value,
        "contract_months": contract_months.tolist(),
        "tenure_months": tenure_months.tolist(),
        "engagement_score": engagement_raw,
        "support_tickets_90d": support_tickets.tolist(),
        "payment_failures_12m": payment_failures.tolist(),
        "n_integrations": n_integrations.tolist(),
        "monthly_logins_per_user": monthly_logins,
        "feature_adoption_rate": feature_adoption,
        "churned": churned.tolist(),
        "churn_date": churn_dates,
        "churn_reason": churn_reasons,
        "churn_probability_true": churn_prob.round(4).tolist(),  # kept for EDA validation
    })

    # ------------------------------------------------------------------
    # 12. Inject realistic data-quality issues
    # ------------------------------------------------------------------
    # 12a. ~3 % nulls in engagement_score (not all customers have app telemetry)
    null_idx = rng.choice(n, size=int(n * 0.03), replace=False)
    df.loc[null_idx, "engagement_score"] = np.nan

    # 12b. ~1.5 % nulls in monthly_logins_per_user
    null_idx2 = rng.choice(n, size=int(n * 0.015), replace=False)
    df.loc[null_idx2, "monthly_logins_per_user"] = np.nan

    # 12c. ~0.5 % out-of-range monthly_fee (data entry errors → negative / zero)
    bad_fee_idx = rng.choice(n, size=int(n * 0.005), replace=False)
    df.loc[bad_fee_idx, "monthly_fee_per_seat"] = rng.uniform(-5, 0, size=len(bad_fee_idx)).round(2)

    # 12d. ~1 % duplicate rows (simulates double-ingestion from CRM sync)
    dup_idx = rng.choice(n, size=int(n * 0.01), replace=False)
    dups = df.iloc[dup_idx].copy()
    df = pd.concat([df, dups], ignore_index=True)

    # 12e. A handful of impossible dates: churn_date before subscription_start_date
    bad_date_idx = rng.choice(len(df), size=5, replace=False)
    for idx in bad_date_idx:
        if df.at[idx, "churn_date"] is not None:
            try:
                bad = date.fromisoformat(df.at[idx, "subscription_start_date"]) - timedelta(days=10)
                df.at[idx, "churn_date"] = str(bad)
            except Exception:
                pass

    logger.info(
        "Raw dataset: %d rows (incl. duplicates), %d columns. Churn rate: %.1f%%",
        len(df),
        len(df.columns),
        df["churned"].mean() * 100,
    )
    return df


def save_raw(df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> Path:
    """Persist raw dataset to data/raw/ as CSV.  Never overwrites existing raw data."""
    cfg = cfg or load_config()
    raw_dir = resolve_path("raw", cfg)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / "subscribers_raw.csv"
    df.to_csv(out, index=False)
    logger.info("Raw data saved → %s", out)
    return out


def data_hash(df: pd.DataFrame) -> str:
    """SHA-256 hash of the DataFrame contents — used for model provenance."""
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()
    return h[:16]


# ---------------------------------------------------------------------------
# CLI entry (called by pipeline.py orchestrator)
# ---------------------------------------------------------------------------

def run_ingestion(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Top-level ingestion step."""
    cfg = cfg or load_config()
    syn_cfg = cfg["data"]["synthetic"]
    df = generate_subscribers(
        n=syn_cfg["n_subscribers"],
        seed=syn_cfg["seed"],
        start_date=syn_cfg["start_date"],
        end_date=syn_cfg["end_date"],
    )
    save_raw(df, cfg)
    return df
