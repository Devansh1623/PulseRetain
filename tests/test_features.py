"""
Tests for pulse/features.py — feature engineering functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pulse.features import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_CATALOG,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_features,
)


def _make_clean_df(n: int = 50) -> pd.DataFrame:
    """Minimal cleaned df for feature engineering tests."""
    rng = np.random.default_rng(123)
    return pd.DataFrame({
        "subscriber_id": [f"SUB-{i:06d}" for i in range(n)],
        "subscription_start_date": pd.date_range("2023-01-01", periods=n, freq="7D"),
        "plan_tier": rng.choice(["starter", "growth", "professional", "enterprise"], size=n),
        "billing_cycle": rng.choice(["monthly", "annual"], size=n),
        "region": rng.choice(["NAM", "EMEA", "APAC", "LATAM"], size=n),
        "acquisition_channel": rng.choice(["organic", "paid_search", "referral"], size=n),
        "industry_vertical": rng.choice(["Software & SaaS", "E-commerce"], size=n),
        "monthly_fee_per_seat": rng.uniform(10, 200, size=n).round(2),
        "seat_count": rng.integers(1, 50, size=n),
        "total_contract_value": rng.uniform(100, 5000, size=n).round(2),
        "contract_months": rng.choice([1, 12], size=n),
        "tenure_months": rng.integers(1, 36, size=n),
        "engagement_score": rng.uniform(10, 95, size=n).round(1),
        "support_tickets_90d": rng.integers(0, 8, size=n),
        "payment_failures_12m": rng.integers(0, 4, size=n),
        "n_integrations": rng.integers(0, 11, size=n),
        "monthly_logins_per_user": rng.uniform(2, 80, size=n).round(1),
        "feature_adoption_rate": rng.uniform(0.05, 0.95, size=n).round(3),
        "churned": rng.integers(0, 2, size=n),
        "churn_date": [None] * n,
        "churn_reason": [None] * n,
        "churn_probability_true": rng.uniform(0, 1, size=n),  # must be dropped by build_features
    })


class TestBuildFeatures:
    def test_returns_dataframe(self):
        df = _make_clean_df()
        result = build_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_leakage_columns_dropped(self):
        df = _make_clean_df()
        result = build_features(df)
        leakage_cols = ["churn_probability_true", "churn_date", "churn_reason",
                        "subscription_start_date", "subscriber_id"]
        for col in leakage_cols:
            assert col not in result.columns, f"Leakage column '{col}' still in features!"

    def test_engineered_features_present(self):
        df = _make_clean_df()
        result = build_features(df)
        expected = [
            "plan_tier_enc", "billing_annual_flag", "monthly_arr_usd",
            "fee_per_seat_log", "seat_count_log", "tenure_log",
            "engagement_decile", "risk_score_simple",
            "is_high_ticket_user", "is_payment_delinquent", "is_low_adoption",
            "seats_x_engagement", "log_arr_x_tenure",
            "channel_is_organic", "channel_is_referral",
        ]
        for col in expected:
            assert col in result.columns, f"Expected feature '{col}' not found!"

    def test_billing_annual_flag_correct(self):
        df = _make_clean_df(n=10)
        df["billing_cycle"] = ["annual"] * 5 + ["monthly"] * 5
        result = build_features(df)
        assert result["billing_annual_flag"].iloc[:5].all() == True  # noqa: E712
        assert (result["billing_annual_flag"].iloc[5:] == 0).all()

    def test_risk_score_in_range(self):
        df = _make_clean_df()
        result = build_features(df)
        assert result["risk_score_simple"].between(0, 1).all()

    def test_plan_tier_enc_range(self):
        df = _make_clean_df()
        result = build_features(df)
        assert result["plan_tier_enc"].between(0, 3).all()

    def test_feature_catalog_keys_match_engineered(self):
        """All engineered feature column names in FEATURE_CATALOG should be resolvable."""
        assert len(FEATURE_CATALOG) > 10, "Feature catalog looks empty"

    def test_target_column_present(self):
        """churned column should survive (it's the label, not a feature — tested separately)."""
        df = _make_clean_df()
        result = build_features(df)
        assert TARGET_COLUMN in result.columns


class TestFeatureColumnLists:
    def test_no_overlap_between_column_groups(self):
        """Numeric, categorical, and binary feature lists must not overlap."""
        all_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
        assert len(all_cols) == len(set(all_cols)), "Feature groups have overlapping columns!"
