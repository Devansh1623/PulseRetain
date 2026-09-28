-- =============================================================================
-- PulseRetain — Named Analytical Queries
-- Each query answers one of the stakeholder business questions from the design brief.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Q1: What is the overall attrition rate and its trend?
-- Demonstrates: CTE, aggregation, period-over-period calculation
-- ---------------------------------------------------------------------------
-- q01_attrition_trend.sql

WITH quarterly AS (
    SELECT
        CAST(SUBSTR(subscription_start_date, 1, 4) AS INTEGER)        AS cohort_year,
        CASE
            WHEN CAST(SUBSTR(subscription_start_date, 6, 2) AS INTEGER) BETWEEN 1 AND 3  THEN 1
            WHEN CAST(SUBSTR(subscription_start_date, 6, 2) AS INTEGER) BETWEEN 4 AND 6  THEN 2
            WHEN CAST(SUBSTR(subscription_start_date, 6, 2) AS INTEGER) BETWEEN 7 AND 9  THEN 3
            ELSE 4
        END                                                             AS cohort_q,
        COUNT(*)                                                        AS n,
        SUM(churned)                                                    AS n_churned,
        ROUND(100.0 * SUM(churned) / COUNT(*), 2)                      AS churn_rate_pct
    FROM fact_subscribers
    GROUP BY cohort_year, cohort_q
),
with_lag AS (
    SELECT *,
        LAG(churn_rate_pct) OVER (ORDER BY cohort_year, cohort_q)      AS prev_period_rate,
        ROUND(
            churn_rate_pct - LAG(churn_rate_pct) OVER (ORDER BY cohort_year, cohort_q),
            2
        )                                                               AS period_change_pct
    FROM quarterly
)
SELECT
    cohort_year,
    cohort_q,
    n,
    n_churned,
    churn_rate_pct,
    prev_period_rate,
    period_change_pct
FROM with_lag
ORDER BY cohort_year, cohort_q;

-- ---------------------------------------------------------------------------
-- Q2: Which plan tier has the worst retention?
-- Demonstrates: multi-table join, aggregation, ranking via window function
-- ---------------------------------------------------------------------------
-- q02_plan_retention_ranked.sql

SELECT
    p.plan_tier,
    p.tier_rank,
    COUNT(fs.subscriber_id)                                            AS n_subscribers,
    SUM(fs.churned)                                                    AS n_churned,
    ROUND(100.0 * SUM(fs.churned) / COUNT(*), 2)                      AS churn_rate_pct,
    RANK() OVER (ORDER BY SUM(fs.churned) * 1.0 / COUNT(*) DESC)      AS churn_rank,
    ROUND(AVG(fs.tenure_months), 1)                                    AS avg_tenure_months
FROM fact_subscribers fs
JOIN dim_plan p ON fs.plan_id = p.plan_id
GROUP BY p.plan_tier, p.tier_rank
ORDER BY churn_rate_pct DESC;

-- ---------------------------------------------------------------------------
-- Q3: What is the ARR at risk from accounts currently flagged high-risk?
-- Demonstrates: join with predictions, filtered aggregation
-- ---------------------------------------------------------------------------
-- q03_arr_at_risk.sql

SELECT
    mp.risk_tier,
    COUNT(DISTINCT mp.subscriber_id)                                   AS n_accounts,
    ROUND(SUM(fs.monthly_arr_usd), 0)                                  AS arr_at_risk_usd,
    ROUND(100.0 * SUM(fs.monthly_arr_usd) /
          (SELECT SUM(monthly_arr_usd) FROM fact_subscribers WHERE churned = 0), 2
    )                                                                   AS pct_of_total_arr
FROM fact_model_predictions mp
JOIN fact_subscribers fs ON mp.subscriber_id = fs.subscriber_id
WHERE fs.churned = 0
  AND mp.risk_tier IN ('high', 'medium')
GROUP BY mp.risk_tier
ORDER BY arr_at_risk_usd DESC;

-- ---------------------------------------------------------------------------
-- Q4: What does the 90-day retention curve look like per acquisition channel?
-- Demonstrates: CTE, conditional aggregation, percentile-style bucketing
-- ---------------------------------------------------------------------------
-- q04_channel_retention_curve.sql

WITH buckets AS (
    SELECT
        c.channel_code,
        CASE
            WHEN fs.tenure_months < 3   THEN '0-3 months'
            WHEN fs.tenure_months < 6   THEN '3-6 months'
            WHEN fs.tenure_months < 12  THEN '6-12 months'
            WHEN fs.tenure_months < 24  THEN '12-24 months'
            ELSE '24+ months'
        END                                  AS tenure_bucket,
        COUNT(*)                             AS n,
        SUM(fs.churned)                      AS n_churned
    FROM fact_subscribers fs
    JOIN dim_channel c ON fs.channel_id = c.channel_id
    GROUP BY c.channel_code, tenure_bucket
)
SELECT
    channel_code,
    tenure_bucket,
    n,
    n_churned,
    ROUND(100.0 * n_churned / n, 2)          AS churn_rate_pct
FROM buckets
ORDER BY channel_code, MIN(tenure_bucket);  -- alphabetic bucket order works here

-- ---------------------------------------------------------------------------
-- Q5: Running total of ARR gained vs. lost month by month (YTD)
-- Demonstrates: window function — running total
-- ---------------------------------------------------------------------------
-- q05_arr_running_total.sql

WITH monthly_events AS (
    SELECT
        SUBSTR(subscription_start_date, 1, 7)     AS event_month,
        SUM(monthly_arr_usd)                       AS arr_added,
        0.0                                        AS arr_lost
    FROM fact_subscribers
    GROUP BY event_month

    UNION ALL

    SELECT
        SUBSTR(churn_date, 1, 7)                   AS event_month,
        0.0                                        AS arr_added,
        SUM(monthly_arr_usd)                       AS arr_lost
    FROM fact_subscribers
    WHERE churned = 1 AND churn_date IS NOT NULL
    GROUP BY event_month
),
aggregated AS (
    SELECT
        event_month,
        SUM(arr_added)    AS arr_added,
        SUM(arr_lost)     AS arr_lost,
        SUM(arr_added) - SUM(arr_lost) AS net_arr_change
    FROM monthly_events
    GROUP BY event_month
)
SELECT
    event_month,
    arr_added,
    arr_lost,
    net_arr_change,
    SUM(net_arr_change) OVER (ORDER BY event_month ROWS UNBOUNDED PRECEDING) AS running_arr
FROM aggregated
ORDER BY event_month;

-- ---------------------------------------------------------------------------
-- Q6: Which industry vertical shows highest churn AND highest ARR loss?
-- Demonstrates: multi-dimension aggregation, dual-rank ordering
-- ---------------------------------------------------------------------------
-- q06_industry_churn_arr.sql

SELECT
    ind.industry_name,
    COUNT(*)                                                            AS n_subscribers,
    SUM(fs.churned)                                                     AS n_churned,
    ROUND(100.0 * SUM(fs.churned) / COUNT(*), 2)                       AS churn_rate_pct,
    ROUND(SUM(CASE WHEN fs.churned = 1 THEN fs.monthly_arr_usd ELSE 0 END), 0) AS arr_lost_usd,
    RANK() OVER (ORDER BY SUM(fs.churned) * 1.0 / COUNT(*) DESC)       AS churn_rank,
    RANK() OVER (ORDER BY SUM(CASE WHEN fs.churned=1 THEN fs.monthly_arr_usd ELSE 0 END) DESC) AS arr_loss_rank
FROM fact_subscribers fs
JOIN dim_industry ind ON fs.industry_id = ind.industry_id
GROUP BY ind.industry_name
ORDER BY arr_lost_usd DESC;

-- ---------------------------------------------------------------------------
-- Q7: Engagement percentile distribution by plan tier
-- Demonstrates: window functions — NTILE (percentile approximation via buckets)
-- ---------------------------------------------------------------------------
-- q07_engagement_percentiles.sql

WITH ranked AS (
    SELECT
        p.plan_tier,
        fs.engagement_score,
        NTILE(4) OVER (PARTITION BY p.plan_tier ORDER BY fs.engagement_score) AS quartile
    FROM fact_subscribers fs
    JOIN dim_plan p ON fs.plan_id = p.plan_id
    WHERE fs.engagement_score IS NOT NULL
)
SELECT
    plan_tier,
    quartile,
    COUNT(*)                        AS n,
    ROUND(MIN(engagement_score), 1) AS min_score,
    ROUND(MAX(engagement_score), 1) AS max_score,
    ROUND(AVG(engagement_score), 1) AS avg_score
FROM ranked
GROUP BY plan_tier, quartile
ORDER BY plan_tier, quartile;

-- ---------------------------------------------------------------------------
-- Q8: Cohort-level 6-month retention comparison (business question: are
--     newer cohorts retained better than older ones?)
-- Demonstrates: cohort retention, period-over-period via self-join
-- ---------------------------------------------------------------------------
-- q08_cohort_six_month_retention.sql

WITH starts AS (
    SELECT
        subscriber_id,
        CAST(SUBSTR(subscription_start_date, 1, 4) AS INTEGER)   AS start_year,
        CAST(SUBSTR(subscription_start_date, 6, 2) AS INTEGER)   AS start_month,
        tenure_months,
        churned
    FROM fact_subscribers
),
cohort_base AS (
    SELECT
        start_year,
        start_month,
        COUNT(*) AS cohort_size
    FROM starts
    GROUP BY start_year, start_month
),
retained_6m AS (
    SELECT
        start_year,
        start_month,
        SUM(CASE WHEN tenure_months >= 6 AND churned = 0 THEN 1 ELSE 0 END) AS still_active_6m
    FROM starts
    GROUP BY start_year, start_month
)
SELECT
    b.start_year,
    b.start_month,
    b.cohort_size,
    r.still_active_6m,
    ROUND(100.0 * r.still_active_6m / b.cohort_size, 1) AS six_month_retention_pct
FROM cohort_base b
JOIN retained_6m r ON b.start_year = r.start_year AND b.start_month = r.start_month
ORDER BY b.start_year, b.start_month;
