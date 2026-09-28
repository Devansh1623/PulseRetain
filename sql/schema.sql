-- =============================================================================
-- PulseRetain — Analytical Data Warehouse Schema (SQLite dialect)
-- Portability notes for PostgreSQL:
--   • Replace INTEGER PRIMARY KEY AUTOINCREMENT with SERIAL PRIMARY KEY
--   • REAL → NUMERIC(12,4)
--   • TEXT → VARCHAR(n) where length is known
--   • SQLite has no native BOOLEAN; we use INTEGER (0/1) — PostgreSQL has BOOLEAN
--   • Views and CTEs are identical between dialects
-- =============================================================================

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- ---------------------------------------------------------------------------
-- DIMENSION: dim_plan
-- Slowly changing dimension for subscription plan tiers.
-- One row per plan_tier value.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_plan (
    plan_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_tier        TEXT    NOT NULL UNIQUE CHECK (plan_tier IN ('starter','growth','professional','enterprise')),
    tier_rank        INTEGER NOT NULL CHECK (tier_rank BETWEEN 1 AND 4),
    base_price_floor REAL    NOT NULL,  -- minimum monthly fee for this tier
    base_price_ceil  REAL    NOT NULL,  -- maximum monthly fee for this tier
    description      TEXT
);

-- Index on tier_rank for ordering queries
CREATE INDEX IF NOT EXISTS idx_dim_plan_rank ON dim_plan (tier_rank);

-- ---------------------------------------------------------------------------
-- DIMENSION: dim_region
-- Geographic region dimension.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_region (
    region_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    region_code TEXT    NOT NULL UNIQUE CHECK (region_code IN ('NAM','EMEA','APAC','LATAM')),
    region_name TEXT    NOT NULL
);

-- ---------------------------------------------------------------------------
-- DIMENSION: dim_channel
-- Acquisition channel dimension.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_channel (
    channel_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_code TEXT NOT NULL UNIQUE,
    channel_type TEXT NOT NULL CHECK (channel_type IN ('inbound','outbound','partnership','viral'))
);

-- ---------------------------------------------------------------------------
-- DIMENSION: dim_industry
-- Subscriber industry vertical.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_industry (
    industry_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    industry_name TEXT NOT NULL UNIQUE
);

-- ---------------------------------------------------------------------------
-- FACT: fact_subscribers
-- One row per subscriber account (current state at observation date).
-- This is the central fact table; all KPI queries join here.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_subscribers (
    subscriber_id           TEXT    PRIMARY KEY,
    plan_id                 INTEGER NOT NULL REFERENCES dim_plan (plan_id),
    region_id               INTEGER NOT NULL REFERENCES dim_region (region_id),
    channel_id              INTEGER NOT NULL REFERENCES dim_channel (channel_id),
    industry_id             INTEGER NOT NULL REFERENCES dim_industry (industry_id),

    subscription_start_date TEXT    NOT NULL,   -- ISO date string (SQLite has no DATE type)
    billing_cycle           TEXT    NOT NULL CHECK (billing_cycle IN ('monthly','annual')),
    contract_months         INTEGER NOT NULL CHECK (contract_months IN (1, 12)),
    tenure_months           INTEGER NOT NULL CHECK (tenure_months >= 0),

    monthly_fee_per_seat    REAL    NOT NULL CHECK (monthly_fee_per_seat > 0),
    seat_count              INTEGER NOT NULL CHECK (seat_count > 0),
    total_contract_value    REAL    NOT NULL,
    monthly_arr_usd         REAL    NOT NULL,

    engagement_score        REAL    CHECK (engagement_score BETWEEN 0 AND 100),
    support_tickets_90d     INTEGER NOT NULL DEFAULT 0,
    payment_failures_12m    INTEGER NOT NULL DEFAULT 0 CHECK (payment_failures_12m >= 0),
    n_integrations          INTEGER NOT NULL DEFAULT 0,
    monthly_logins_per_user REAL,
    feature_adoption_rate   REAL    CHECK (feature_adoption_rate BETWEEN 0 AND 1),

    churned                 INTEGER NOT NULL DEFAULT 0 CHECK (churned IN (0, 1)),
    churn_date              TEXT,               -- NULL if still active
    churn_reason            TEXT
);

-- Indexes on the most common filter / join columns
-- (comment explains the query pattern each index serves)

-- Used in: KPI by plan, model-feature segment queries
CREATE INDEX IF NOT EXISTS idx_fs_plan_id ON fact_subscribers (plan_id);

-- Used in: KPI by region, regional drilldowns
CREATE INDEX IF NOT EXISTS idx_fs_region_id ON fact_subscribers (region_id);

-- Used in: acquisition-channel attrition analysis
CREATE INDEX IF NOT EXISTS idx_fs_channel_id ON fact_subscribers (channel_id);

-- Used in: cohort queries filtering by subscription start date
CREATE INDEX IF NOT EXISTS idx_fs_start_date ON fact_subscribers (subscription_start_date);

-- Used in: "churn vs active" partitioning in almost every KPI view
CREATE INDEX IF NOT EXISTS idx_fs_churned ON fact_subscribers (churned);

-- Composite: plan + churned covers the most common two-column filter pattern
CREATE INDEX IF NOT EXISTS idx_fs_plan_churned ON fact_subscribers (plan_id, churned);

-- ---------------------------------------------------------------------------
-- FACT: fact_model_predictions
-- One row per subscriber per model run.  Captures model output for dashboarding.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_model_predictions (
    prediction_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    subscriber_id     TEXT    NOT NULL REFERENCES fact_subscribers (subscriber_id),
    model_name        TEXT    NOT NULL,
    run_id            TEXT    NOT NULL,   -- UUID or date-hash for the training run
    churn_probability REAL    NOT NULL CHECK (churn_probability BETWEEN 0 AND 1),
    predicted_churned INTEGER NOT NULL CHECK (predicted_churned IN (0, 1)),
    risk_tier         TEXT    NOT NULL CHECK (risk_tier IN ('high','medium','low')),
    predicted_at      TEXT    NOT NULL    -- ISO datetime string
);

CREATE INDEX IF NOT EXISTS idx_pred_subscriber ON fact_model_predictions (subscriber_id);
CREATE INDEX IF NOT EXISTS idx_pred_model      ON fact_model_predictions (model_name, run_id);
CREATE INDEX IF NOT EXISTS idx_pred_risk_tier  ON fact_model_predictions (risk_tier);

-- ---------------------------------------------------------------------------
-- FACT: fact_model_runs
-- One row per training run — provenance and metrics.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_model_runs (
    run_id          TEXT    PRIMARY KEY,
    model_name      TEXT    NOT NULL,
    trained_at      TEXT    NOT NULL,
    data_hash       TEXT    NOT NULL,
    n_train         INTEGER NOT NULL,
    n_test          INTEGER NOT NULL,
    roc_auc         REAL,
    pr_auc          REAL,
    f1_score        REAL,
    precision_score REAL,
    recall_score    REAL,
    threshold_used  REAL,
    artifact_path   TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_model ON fact_model_runs (model_name);
