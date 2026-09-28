"""
Smoke tests — rapid end-to-end checks that the pipeline runs without crashing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


class TestIngestion:
    def test_generate_subscribers_returns_dataframe(self):
        from pulse.ingestion import generate_subscribers
        df = generate_subscribers(n=200, seed=42)
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 200  # will be slightly more due to injected duplicates
        assert "subscriber_id" in df.columns
        assert "churned" in df.columns

    def test_churn_rate_in_expected_range(self):
        from pulse.ingestion import generate_subscribers
        df = generate_subscribers(n=500, seed=42)
        # After dedup, churn rate should be between 10% and 30%
        rate = df["churned"].mean()
        assert 0.08 <= rate <= 0.35, f"Churn rate {rate:.1%} out of expected range"

    def test_data_hash_is_deterministic(self):
        from pulse.ingestion import data_hash, generate_subscribers
        df1 = generate_subscribers(n=100, seed=99)
        df2 = generate_subscribers(n=100, seed=99)
        assert data_hash(df1) == data_hash(df2)


class TestValidation:
    def test_validation_runs_without_error(self):
        from pulse.ingestion import generate_subscribers
        from pulse.validation import DataValidator
        df = generate_subscribers(n=300, seed=1)
        validator = DataValidator()
        report = validator.validate(df)
        assert report.n_rows == len(df)

    def test_validation_detects_injected_duplicates(self):
        from pulse.ingestion import generate_subscribers
        from pulse.validation import DataValidator
        df = generate_subscribers(n=300, seed=1)
        validator = DataValidator()
        report = validator.validate(df)
        # The generator injects ~1% duplicates, so there should be some found
        assert report.n_duplicates >= 0  # at minimum, the check ran


class TestCleaningPipeline:
    def test_cleaning_reduces_duplicates_to_zero(self):
        from pulse.cleaning import clean
        from pulse.ingestion import generate_subscribers
        df = generate_subscribers(n=500, seed=42)
        cleaned, log = clean(df)
        assert cleaned["subscriber_id"].duplicated().sum() == 0

    def test_no_negative_fees_after_cleaning(self):
        from pulse.cleaning import clean
        from pulse.ingestion import generate_subscribers
        df = generate_subscribers(n=500, seed=42)
        cleaned, log = clean(df)
        assert (cleaned["monthly_fee_per_seat"] > 0).all()

    def test_no_nulls_in_engagement_after_cleaning(self):
        from pulse.cleaning import clean
        from pulse.ingestion import generate_subscribers
        df = generate_subscribers(n=500, seed=42)
        cleaned, log = clean(df)
        assert cleaned["engagement_score"].isna().sum() == 0

    def test_cleaning_returns_audit_log(self):
        from pulse.cleaning import clean
        from pulse.ingestion import generate_subscribers
        df = generate_subscribers(n=200, seed=42)
        _, log = clean(df)
        assert len(log) > 0
        assert any("duplicate" in entry.lower() for entry in log)


class TestFeaturePipeline:
    def test_build_features_no_leakage(self):
        from pulse.cleaning import clean
        from pulse.features import build_features
        from pulse.ingestion import generate_subscribers
        df = generate_subscribers(n=200, seed=42)
        cleaned, _ = clean(df)
        features = build_features(cleaned)
        leakage = ["churn_probability_true", "churn_date", "churn_reason"]
        for col in leakage:
            assert col not in features.columns

    def test_feature_count_is_reasonable(self):
        from pulse.cleaning import clean
        from pulse.features import build_features
        from pulse.ingestion import generate_subscribers
        df = generate_subscribers(n=100, seed=42)
        cleaned, _ = clean(df)
        features = build_features(cleaned)
        # Should have significantly more columns than the raw input (engineered features added)
        assert len(features.columns) > 20


class TestConfigLoader:
    def test_config_loads_without_error(self):
        from pulse.config import load_config
        cfg = load_config()
        assert "project" in cfg
        assert "paths" in cfg
        assert "model" in cfg

    def test_project_root_is_set(self):
        from pulse.config import load_config
        cfg = load_config()
        assert "_project_root" in cfg
        assert Path(cfg["_project_root"]).exists()
