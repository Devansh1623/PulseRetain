-- =============================================================================
-- PulseRetain — KPI and Analytical Views
-- =============================================================================

-- ---------------------------------------------------------------------------
-- vw_kpi_headline — Executive KPIs used in the dashboard header
-- Aggregates the five most important business metrics in one row.
-- ---------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS vw_kpi_headline AS
SELECT
    COUNT(*)                                                        AS total_subscribers,
    SUM(CASE WHEN churned = 0 THEN 1 ELSE 0 END)                   AS active_subscribers,
    SUM(churned)                                                    AS churned_subscribers,
    ROUND(100.0 * SUM(churned) / COUNT(*), 2)                      AS churn_rate_pct,
    ROUND(SUM(monthly_arr_usd), 0)                                  AS total_arr_usd,
    ROUND(SUM(CASE WHEN churned = 0 THEN monthly_arr_usd ELSE 0 END), 0) AS active_arr_usd,
    ROUND(SUM(CASE WHEN churned = 1 THEN monthly_arr_usd ELSE 0 END), 0) AS lost_arr_usd,
    ROUND(AVG(tenure_months), 1)                                   AS avg_tenure_months,
    ROUND(AVG(engagement_score), 1)                                AS avg_engagement_score
FROM fact_subscribers;

-- ---------------------------------------------------------------------------
-- vw_kpi_by_plan — Churn rate and ARR metrics sliced by plan tier
-- Used in: segmentation view, plan comparison charts
-- ---------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS vw_kpi_by_plan AS
SELECT
    p.plan_tier,
    p.tier_rank,
    COUNT(*)                                                        AS n_subscribers,
    SUM(fs.churned)                                                 AS n_churned,
    ROUND(100.0 * SUM(fs.churned) / COUNT(*), 2)                   AS churn_rate_pct,
    ROUND(AVG(fs.monthly_fee_per_seat), 2)                         AS avg_monthly_fee,
    ROUND(AVG(fs.seat_count), 1)                                    AS avg_seats,
    ROUND(SUM(fs.monthly_arr_usd), 0)                               AS total_arr_usd,
    ROUND(AVG(fs.tenure_months), 1)                                 AS avg_tenure_months,
    ROUND(AVG(fs.engagement_score), 1)                              AS avg_engagement
FROM fact_subscribers fs
JOIN dim_plan p ON fs.plan_id = p.plan_id
GROUP BY p.plan_tier, p.tier_rank
ORDER BY p.tier_rank;

-- ---------------------------------------------------------------------------
-- vw_kpi_by_region — Regional KPIs
-- Used in: geographic analysis, regional drilldowns
-- ---------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS vw_kpi_by_region AS
SELECT
    r.region_code,
    r.region_name,
    COUNT(*)                                                        AS n_subscribers,
    SUM(fs.churned)                                                 AS n_churned,
    ROUND(100.0 * SUM(fs.churned) / COUNT(*), 2)                   AS churn_rate_pct,
    ROUND(SUM(fs.monthly_arr_usd), 0)                               AS total_arr_usd,
    ROUND(AVG(fs.engagement_score), 1)                              AS avg_engagement
FROM fact_subscribers fs
JOIN dim_region r ON fs.region_id = r.region_id
GROUP BY r.region_code, r.region_name
ORDER BY total_arr_usd DESC;

-- ---------------------------------------------------------------------------
-- vw_kpi_by_channel — Acquisition channel quality analysis
-- Used in: marketing ROI analysis
-- ---------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS vw_kpi_by_channel AS
SELECT
    c.channel_code,
    c.channel_type,
    COUNT(*)                                                        AS n_subscribers,
    SUM(fs.churned)                                                 AS n_churned,
    ROUND(100.0 * SUM(fs.churned) / COUNT(*), 2)                   AS churn_rate_pct,
    ROUND(AVG(fs.tenure_months), 1)                                 AS avg_tenure_months,
    ROUND(AVG(fs.monthly_arr_usd), 0)                               AS avg_arr_usd,
    ROUND(AVG(fs.engagement_score), 1)                              AS avg_engagement
FROM fact_subscribers fs
JOIN dim_channel c ON fs.channel_id = c.channel_id
GROUP BY c.channel_code, c.channel_type
ORDER BY churn_rate_pct ASC;

-- ---------------------------------------------------------------------------
-- vw_cohort_retention — Monthly cohort retention table
-- Used in: cohort analysis view.
-- Cohort = quarter of subscription start.
-- Retention = share still active after each 3-month period.
-- ---------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS vw_cohort_retention AS
WITH cohorts AS (
    SELECT
        subscriber_id,
        tenure_months,
        churned,
        -- Derive cohort quarter from subscription_start_date string (YYYY-MM-DD)
        CAST(SUBSTR(subscription_start_date, 1, 4) AS INTEGER) AS cohort_year,
        CASE
            WHEN CAST(SUBSTR(subscription_start_date, 6, 2) AS INTEGER) BETWEEN 1 AND 3  THEN 'Q1'
            WHEN CAST(SUBSTR(subscription_start_date, 6, 2) AS INTEGER) BETWEEN 4 AND 6  THEN 'Q2'
            WHEN CAST(SUBSTR(subscription_start_date, 6, 2) AS INTEGER) BETWEEN 7 AND 9  THEN 'Q3'
            ELSE 'Q4'
        END AS cohort_quarter
    FROM fact_subscribers
),
base AS (
    SELECT
        cohort_year,
        cohort_quarter,
        COUNT(*) AS cohort_size
    FROM cohorts
    GROUP BY cohort_year, cohort_quarter
),
activity AS (
    SELECT
        cohort_year,
        cohort_quarter,
        -- Bucket tenure into 3-month periods (0 = first 3 months, 1 = months 4-6, etc.)
        (tenure_months / 3) AS period_bucket,
        COUNT(*) AS n_active
    FROM cohorts
    WHERE churned = 0  -- only count subscribers who survived to this tenure
    GROUP BY cohort_year, cohort_quarter, period_bucket
)
SELECT
    a.cohort_year,
    a.cohort_quarter,
    a.period_bucket,
    b.cohort_size,
    a.n_active,
    ROUND(100.0 * a.n_active / b.cohort_size, 1) AS retention_pct
FROM activity a
JOIN base b ON a.cohort_year = b.cohort_year AND a.cohort_quarter = b.cohort_quarter
ORDER BY a.cohort_year, a.cohort_quarter, a.period_bucket;

-- ---------------------------------------------------------------------------
-- vw_risk_watchlist — High and medium risk accounts with predicted churn probability.
-- Joined with subscriber details for the dashboard watchlist table.
-- ---------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS vw_risk_watchlist AS
SELECT
    fs.subscriber_id,
    p.plan_tier,
    r.region_code,
    ind.industry_name,
    fs.tenure_months,
    fs.engagement_score,
    fs.payment_failures_12m,
    fs.support_tickets_90d,
    ROUND(fs.monthly_arr_usd, 0)   AS arr_usd,
    mp.churn_probability,
    mp.risk_tier,
    mp.model_name,
    mp.predicted_at
FROM fact_model_predictions mp
JOIN fact_subscribers fs  ON mp.subscriber_id  = fs.subscriber_id
JOIN dim_plan p           ON fs.plan_id         = p.plan_id
JOIN dim_region r         ON fs.region_id       = r.region_id
JOIN dim_industry ind     ON fs.industry_id     = ind.industry_id
WHERE mp.risk_tier IN ('high', 'medium')
  AND fs.churned = 0   -- only show currently active accounts
ORDER BY mp.churn_probability DESC;

-- ---------------------------------------------------------------------------
-- vw_engagement_segments — Engagement-based subscriber segments.
-- Used in the segmentation drilldown.
-- ---------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS vw_engagement_segments AS
SELECT
    CASE
        WHEN engagement_score >= 80 THEN 'Champion (80-100)'
        WHEN engagement_score >= 60 THEN 'Engaged (60-79)'
        WHEN engagement_score >= 40 THEN 'At Risk (40-59)'
        WHEN engagement_score >= 20 THEN 'Disengaged (20-39)'
        ELSE 'Critical (<20)'
    END                                                               AS engagement_segment,
    COUNT(*)                                                          AS n_subscribers,
    SUM(churned)                                                      AS n_churned,
    ROUND(100.0 * SUM(churned) / COUNT(*), 2)                        AS churn_rate_pct,
    ROUND(AVG(monthly_arr_usd), 0)                                    AS avg_arr_usd,
    ROUND(AVG(tenure_months), 1)                                      AS avg_tenure
FROM fact_subscribers
GROUP BY engagement_segment
ORDER BY MIN(engagement_score) DESC;
