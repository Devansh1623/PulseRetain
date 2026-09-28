"""
Tests for pulse/cleaning.py — data cleaning rules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pulse.cleaning import (
    _fix_impossible_dates,
    _fix_negative_fees,
    _impute_engagement,
    _remove_duplicates,
)
from pulse.config import load_config


def _make_df(**kwargs) -> pd.DataFrame:
    """Build a minimal valid subscriber DataFrame for testing."""
    n = kwargs.pop("n", 5)
    defaults = {
        "subscriber_id": [f"SUB-{i:06d}" for i in range(n)],
        "plan_tier": ["starter"] * n,
        "billing_cycle": ["monthly"] * n,
        "region": ["NAM"] * n,
        "acquisition_channel": ["organic"] * n,
        "industry_vertical": ["Software & SaaS"] * n,
        "monthly_fee_per_seat": [29.99] * n,
        "seat_count": [3] * n,
        "total_contract_value": [89.97] * n,
        "contract_months": [1] * n,
        "tenure_months": [12] * n,
        "engagement_score": [65.0] * n,
        "support_tickets_90d": [1] * n,
        "payment_failures_12m": [0] * n,
        "n_integrations": [3] * n,
        "monthly_logins_per_user": [15.0] * n,
        "feature_adoption_rate": [0.5] * n,
        "churned": [0] * n,
        "churn_date": [None] * n,
        "churn_reason": [None] * n,
        "subscription_start_date": ["2023-01-15"] * n,
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


class TestRemoveDuplicates:
    def test_removes_duplicate_subscriber_ids(self):
        df = _make_df(n=3)
        df.loc[2, "subscriber_id"] = "SUB-000001"  # duplicate of row 1
        log = []
        cleaned = _remove_duplicates(df, log)
        assert len(cleaned) == 2
        assert "duplicate" in log[0].lower()

    def test_no_duplicates_unchanged(self):
        df = _make_df(n=4)
        log = []
        cleaned = _remove_duplicates(df, log)
        assert len(cleaned) == 4


class TestFixImpossibleDates:
    def test_churn_before_start_is_nulled(self):
        df = _make_df(n=3)
        df.loc[0, "churned"] = 1
        df.loc[0, "subscription_start_date"] = "2023-06-01"
        df.loc[0, "churn_date"] = "2023-05-01"  # before start!
        log = []
        cleaned = _fix_impossible_dates(df, log)
        assert pd.isna(cleaned.loc[0, "churn_date"]) or cleaned.loc[0, "churn_date"] is None
        assert cleaned.loc[0, "churned"] == 0
        assert len(log) > 0

    def test_valid_dates_unchanged(self):
        df = _make_df(n=3)
        df.loc[1, "churned"] = 1
        df.loc[1, "subscription_start_date"] = "2023-01-01"
        df.loc[1, "churn_date"] = "2023-08-15"  # valid: after start
        log = []
        cleaned = _fix_impossible_dates(df, log)
        assert cleaned.loc[1, "churned"] == 1
        assert len(log) == 0


class TestFixNegativeFees:
    def test_negative_fee_replaced_with_plan_median(self):
        cfg = load_config()
        df = _make_df(n=6)
        df.loc[0, "monthly_fee_per_seat"] = -5.0
        log = []
        cleaned = _fix_negative_fees(df, log, cfg)
        assert cleaned.loc[0, "monthly_fee_per_seat"] > 0
        assert "Fee fix" in log[0]

    def test_valid_fees_unchanged(self):
        cfg = load_config()
        df = _make_df(n=3)
        original_fees = df["monthly_fee_per_seat"].tolist()
        log = []
        cleaned = _fix_negative_fees(df, log, cfg)
        assert cleaned["monthly_fee_per_seat"].tolist() == original_fees
        assert len(log) == 0


class TestImputeEngagement:
    def test_nulls_filled_with_plan_tier_median(self):
        df = _make_df(n=10)
        df.loc[2, "engagement_score"] = np.nan
        df.loc[7, "engagement_score"] = np.nan
        log = []
        cleaned = _impute_engagement(df, log)
        assert cleaned["engagement_score"].isna().sum() == 0
        assert "Imputation" in log[0]

    def test_no_nulls_unchanged(self):
        df = _make_df(n=5)
        log = []
        cleaned = _impute_engagement(df, log)
        assert len(log) == 0
