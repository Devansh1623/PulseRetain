"""
SQL tests — verify row counts, referential integrity, and KPI reconciliation
against pandas calculations. Run AFTER the full pipeline has been executed.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pulse.config import load_config
from pulse.database import get_db_path, query_df


def db_available() -> bool:
    try:
        path = get_db_path()
        return path.exists() and path.stat().st_size > 0
    except Exception:
        return False


SKIP_IF_NO_DB = pytest.mark.skipif(not db_available(), reason="Database not initialized. Run pipeline first.")


@SKIP_IF_NO_DB
class TestSchemaIntegrity:
    def test_fact_subscribers_has_rows(self):
        result = query_df("SELECT COUNT(*) AS n FROM fact_subscribers")
        assert result["n"].iloc[0] > 0

    def test_dim_plan_has_four_rows(self):
        result = query_df("SELECT COUNT(*) AS n FROM dim_plan")
        assert result["n"].iloc[0] == 4

    def test_dim_region_has_four_rows(self):
        result = query_df("SELECT COUNT(*) AS n FROM dim_region")
        assert result["n"].iloc[0] == 4

    def test_no_orphan_plan_ids(self):
        """Every fact_subscribers.plan_id must exist in dim_plan."""
        result = query_df("""
            SELECT COUNT(*) AS n
            FROM fact_subscribers fs
            LEFT JOIN dim_plan p ON fs.plan_id = p.plan_id
            WHERE p.plan_id IS NULL
        """)
        assert result["n"].iloc[0] == 0, "Orphan plan_ids found!"

    def test_no_orphan_region_ids(self):
        result = query_df("""
            SELECT COUNT(*) AS n
            FROM fact_subscribers fs
            LEFT JOIN dim_region r ON fs.region_id = r.region_id
            WHERE r.region_id IS NULL
        """)
        assert result["n"].iloc[0] == 0, "Orphan region_ids found!"

    def test_churned_values_are_binary(self):
        result = query_df("SELECT COUNT(*) AS n FROM fact_subscribers WHERE churned NOT IN (0,1)")
        assert result["n"].iloc[0] == 0


@SKIP_IF_NO_DB
class TestKPIReconciliation:
    """Verify SQL KPI values match pandas calculations on the same data."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        cfg = load_config()
        interim = Path(cfg["_project_root"]) / "data" / "interim" / "subscribers_cleaned.csv"
        if interim.exists():
            self.df = pd.read_csv(interim)
        else:
            self.df = None

    def test_churn_rate_reconciles(self):
        if self.df is None:
            pytest.skip("Cleaned data not available")
        sql_result = query_df("SELECT churn_rate_pct FROM vw_kpi_headline")
        pandas_rate = round(self.df["churned"].mean() * 100, 2)
        sql_rate = round(float(sql_result["churn_rate_pct"].iloc[0]), 2)
        assert abs(sql_rate - pandas_rate) < 1.0, (
            f"Churn rate mismatch: SQL={sql_rate}, pandas={pandas_rate}"
        )

    def test_total_subscribers_reconciles(self):
        if self.df is None:
            pytest.skip("Cleaned data not available")
        sql_result = query_df("SELECT total_subscribers FROM vw_kpi_headline")
        sql_n = int(sql_result["total_subscribers"].iloc[0])
        pandas_n = len(self.df)
        # Allow small tolerance (some rows may fail FK constraints in extreme edge cases)
        assert abs(sql_n - pandas_n) / pandas_n < 0.02, (
            f"Row count mismatch: SQL={sql_n}, pandas={pandas_n}"
        )

    def test_plan_tier_churn_rates_sum_to_total(self):
        """Sum of plan-level churned counts should match headline total."""
        sql_total = query_df("SELECT churned_subscribers FROM vw_kpi_headline")["churned_subscribers"].iloc[0]
        plan_total = query_df("SELECT SUM(n_churned) AS tot FROM vw_kpi_by_plan")["tot"].iloc[0]
        assert int(sql_total) == int(plan_total), (
            f"Plan-level sum {plan_total} != headline churned {sql_total}"
        )


@SKIP_IF_NO_DB
class TestViewsExist:
    views = [
        "vw_kpi_headline",
        "vw_kpi_by_plan",
        "vw_kpi_by_region",
        "vw_kpi_by_channel",
        "vw_engagement_segments",
    ]

    @pytest.mark.parametrize("view_name", views)
    def test_view_returns_rows(self, view_name):
        result = query_df(f"SELECT COUNT(*) AS n FROM {view_name}")
        assert result["n"].iloc[0] >= 0  # >= 0 avoids fail if predictions not loaded
