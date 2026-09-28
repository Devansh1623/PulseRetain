"""
pulse/database.py — SQLite data warehouse access layer.

This module owns all database interactions: schema creation, data loading,
KPI queries, and view access. Dashboard and API layers import from here —
they never construct SQL strings themselves.

Design choices:
- SQLite for portability (no server, no credentials for a portfolio project).
- All queries use parameterized statements (?  placeholders) to prevent
  SQL injection even in a local context (good practice to demonstrate).
- pandas read_sql_query is used for query results → DataFrame for easy
  downstream use in the dashboard.
- Connection is created per-call (not a long-lived singleton) to avoid
  thread-safety issues when Streamlit runs in multi-threaded mode.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import pandas as pd

from pulse.config import load_config
from pulse.logging_setup import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Database path helper
# ---------------------------------------------------------------------------

def get_db_path(cfg: dict[str, Any] | None = None) -> Path:
    """Return the absolute path to the SQLite database file."""
    cfg = cfg or load_config()
    root = Path(cfg["_project_root"])
    db_dir = root / "data" / "processed"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "pulse_warehouse.db"


# ---------------------------------------------------------------------------
# Connection context manager
# ---------------------------------------------------------------------------

@contextmanager
def get_connection(cfg: dict[str, Any] | None = None) -> Generator[sqlite3.Connection, None, None]:
    """Yield a SQLite connection with foreign keys enabled.  Auto-closes."""
    db_path = get_db_path(cfg)
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def init_schema(cfg: dict[str, Any] | None = None) -> None:
    """Create all tables and indexes from schema.sql, then load KPI views.

    Idempotent: uses CREATE TABLE IF NOT EXISTS throughout.
    """
    cfg = cfg or load_config()
    root = Path(cfg["_project_root"])
    schema_path = root / "sql" / "schema.sql"
    views_path = root / "sql" / "views" / "kpi_views.sql"

    with get_connection(cfg) as conn:
        schema_sql = schema_path.read_text(encoding="utf-8")
        conn.executescript(schema_sql)
        logger.info("Schema created/verified from %s", schema_path)

        views_sql = views_path.read_text(encoding="utf-8")
        conn.executescript(views_sql)
        logger.info("Views created/verified from %s", views_path)


# ---------------------------------------------------------------------------
# Dimension seed / upsert helpers
# ---------------------------------------------------------------------------

def _seed_dimensions(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    """Insert dimension records and return lookup dicts {value: id}."""
    plan_rows = [
        ("starter", 1, 9.99, 29.99, "Entry-level plan for small teams"),
        ("growth", 2, 29.99, 79.99, "Mid-market plan for growing teams"),
        ("professional", 3, 79.99, 199.99, "Full-featured plan for established businesses"),
        ("enterprise", 4, 199.99, 499.99, "Custom plan for large organisations"),
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO dim_plan (plan_tier, tier_rank, base_price_floor, base_price_ceil, description) "
        "VALUES (?, ?, ?, ?, ?)",
        plan_rows,
    )

    region_rows = [
        ("NAM", "North America"),
        ("EMEA", "Europe, Middle East & Africa"),
        ("APAC", "Asia-Pacific"),
        ("LATAM", "Latin America"),
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO dim_region (region_code, region_name) VALUES (?, ?)",
        region_rows,
    )

    channel_rows = [
        ("organic", "inbound"),
        ("paid_search", "outbound"),
        ("referral", "viral"),
        ("partner", "partnership"),
        ("direct_sales", "outbound"),
        ("trial_convert", "inbound"),
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO dim_channel (channel_code, channel_type) VALUES (?, ?)",
        channel_rows,
    )

    industries = [
        "Software & SaaS", "E-commerce", "Financial Services", "Healthcare IT",
        "Professional Services", "Manufacturing", "Media & Entertainment", "Education",
    ]
    for ind in industries:
        conn.execute("INSERT OR IGNORE INTO dim_industry (industry_name) VALUES (?)", (ind,))

    # Build lookup dicts
    plan_map = {r[0]: r[0] for r in plan_rows}  # placeholder; we do a real lookup below
    cur = conn.execute("SELECT plan_tier, plan_id FROM dim_plan")
    plan_lookup = {row[0]: row[1] for row in cur.fetchall()}

    cur = conn.execute("SELECT region_code, region_id FROM dim_region")
    region_lookup = {row[0]: row[1] for row in cur.fetchall()}

    cur = conn.execute("SELECT channel_code, channel_id FROM dim_channel")
    channel_lookup = {row[0]: row[1] for row in cur.fetchall()}

    cur = conn.execute("SELECT industry_name, industry_id FROM dim_industry")
    industry_lookup = {row[0]: row[1] for row in cur.fetchall()}

    return {
        "plan": plan_lookup,
        "region": region_lookup,
        "channel": channel_lookup,
        "industry": industry_lookup,
    }


# ---------------------------------------------------------------------------
# Bulk load cleaned subscriber data
# ---------------------------------------------------------------------------

def load_subscribers(df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> int:
    """Load the cleaned subscriber DataFrame into fact_subscribers.

    Returns the number of rows inserted.
    """
    cfg = cfg or load_config()
    with get_connection(cfg) as conn:
        lookups = _seed_dimensions(conn)

        rows = []
        for _, row in df.iterrows():
            plan_id = lookups["plan"].get(str(row["plan_tier"]).lower(), 1)
            region_id = lookups["region"].get(str(row["region"]).upper(), 1)
            channel_id = lookups["channel"].get(str(row["acquisition_channel"]).lower(), 1)
            industry_id = lookups["industry"].get(str(row["industry_vertical"]), 1)

            # Convert pandas Timestamp to ISO string (SQLite stores as TEXT)
            start_date = (
                row["subscription_start_date"].isoformat()
                if hasattr(row["subscription_start_date"], "isoformat")
                else str(row["subscription_start_date"])
            )
            churn_dt = None
            if pd.notna(row.get("churn_date")):
                churn_dt = (
                    row["churn_date"].isoformat()
                    if hasattr(row["churn_date"], "isoformat")
                    else str(row["churn_date"])
                )

            arr = round(float(row["monthly_fee_per_seat"]) * int(row["seat_count"]) * 12, 2)

            rows.append((
                str(row["subscriber_id"]),
                plan_id, region_id, channel_id, industry_id,
                start_date,
                str(row["billing_cycle"]).lower(),
                int(row["contract_months"]),
                int(row["tenure_months"]),
                float(row["monthly_fee_per_seat"]),
                int(row["seat_count"]),
                float(row["total_contract_value"]),
                arr,
                float(row["engagement_score"]) if pd.notna(row.get("engagement_score")) else None,
                int(row["support_tickets_90d"]),
                int(row["payment_failures_12m"]),
                int(row["n_integrations"]),
                float(row["monthly_logins_per_user"]) if pd.notna(row.get("monthly_logins_per_user")) else None,
                float(row["feature_adoption_rate"]) if pd.notna(row.get("feature_adoption_rate")) else None,
                int(row["churned"]),
                churn_dt,
                row.get("churn_reason") if pd.notna(row.get("churn_reason")) else None,
            ))

        conn.executemany(
            """
            INSERT OR REPLACE INTO fact_subscribers (
                subscriber_id, plan_id, region_id, channel_id, industry_id,
                subscription_start_date, billing_cycle, contract_months, tenure_months,
                monthly_fee_per_seat, seat_count, total_contract_value, monthly_arr_usd,
                engagement_score, support_tickets_90d, payment_failures_12m, n_integrations,
                monthly_logins_per_user, feature_adoption_rate,
                churned, churn_date, churn_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        logger.info("Loaded %d subscriber rows into fact_subscribers", len(rows))
        return len(rows)


# ---------------------------------------------------------------------------
# Model prediction loader
# ---------------------------------------------------------------------------

def load_predictions(
    predictions_df: pd.DataFrame,
    model_name: str,
    run_id: str,
    threshold: float = 0.40,
    cfg: dict[str, Any] | None = None,
) -> int:
    """Load model predictions into fact_model_predictions."""
    cfg = cfg or load_config()
    import datetime

    def risk_tier(prob: float) -> str:
        if prob >= 0.70:
            return "high"
        if prob >= 0.40:
            return "medium"
        return "low"

    now = datetime.datetime.utcnow().isoformat()
    rows = [
        (
            str(r["subscriber_id"]),
            model_name,
            run_id,
            float(r["churn_probability"]),
            int(r["churn_probability"] >= threshold),
            risk_tier(float(r["churn_probability"])),
            now,
        )
        for _, r in predictions_df.iterrows()
    ]

    with get_connection(cfg) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO fact_model_predictions
            (subscriber_id, model_name, run_id, churn_probability,
             predicted_churned, risk_tier, predicted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        logger.info("Loaded %d prediction rows (model=%s, run=%s)", len(rows), model_name, run_id)
    return len(rows)


# ---------------------------------------------------------------------------
# Query helpers — return DataFrames for dashboard consumption
# ---------------------------------------------------------------------------

def query_df(sql: str, params: tuple = (), cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Execute a parameterized SQL query and return a DataFrame."""
    db_path = get_db_path(cfg)
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


def get_headline_kpis(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    return query_df("SELECT * FROM vw_kpi_headline", cfg=cfg)


def get_kpi_by_plan(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    return query_df("SELECT * FROM vw_kpi_by_plan ORDER BY tier_rank", cfg=cfg)


def get_kpi_by_region(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    return query_df("SELECT * FROM vw_kpi_by_region ORDER BY total_arr_usd DESC", cfg=cfg)


def get_kpi_by_channel(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    return query_df("SELECT * FROM vw_kpi_by_channel ORDER BY churn_rate_pct ASC", cfg=cfg)


def get_risk_watchlist(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    return query_df("SELECT * FROM vw_risk_watchlist LIMIT 500", cfg=cfg)


def get_engagement_segments(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    return query_df("SELECT * FROM vw_engagement_segments", cfg=cfg)


def get_cohort_retention(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    return query_df("SELECT * FROM vw_cohort_retention", cfg=cfg)


def run_named_query(query_name: str, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load and run one of the named queries from sql/queries/.

    The query file is expected to contain a single SELECT statement.
    Multi-statement files (like the combined analytical_queries.sql) should
    be split, or use the individual query functions above.
    """
    cfg = cfg or load_config()
    root = Path(cfg["_project_root"])
    q_path = root / "sql" / "queries" / f"{query_name}.sql"
    if not q_path.exists():
        raise FileNotFoundError(f"Named query not found: {q_path}")
    sql = q_path.read_text(encoding="utf-8")
    return query_df(sql, cfg=cfg)


def run_db_pipeline(df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> None:
    """Top-level step: init schema and load subscribers. Called by pipeline.py."""
    cfg = cfg or load_config()
    init_schema(cfg)
    load_subscribers(df, cfg)
