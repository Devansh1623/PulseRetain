"""
pulse/mongo_database.py — MongoDB Atlas data layer for PulseRetain.

Replaces SQLite with MongoDB Atlas as the cloud-native warehouse.
All public functions mirror the interface in database.py so the dashboard
and pipeline require zero changes.

Collections:
  subscribers         — fact_subscribers (one doc per account)
  model_predictions   — fact_model_predictions (one doc per account per run)
  model_runs          — fact_model_runs (training metadata / provenance)
  dim_plan            — plan dimension (seed data)
  dim_region          — region dimension
  dim_channel         — channel dimension

Connection:
  Set MONGODB_URI in your .env or Streamlit Cloud secrets:
    MONGODB_URI = "mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/"
  Optionally override the database name:
    MONGODB_DB  = "pulseretain"   (default)
"""

from __future__ import annotations

import datetime
import os
from functools import lru_cache
from typing import Any

import pandas as pd
from pymongo import MongoClient, UpdateOne, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database

from pulse.config import load_config
from pulse.logging_setup import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Connection bootstrap
# ---------------------------------------------------------------------------

_DIMENSION_PLANS = [
    {"plan_tier": "starter",      "tier_rank": 1, "base_price_floor":   9.99, "base_price_ceil":  29.99},
    {"plan_tier": "growth",       "tier_rank": 2, "base_price_floor":  29.99, "base_price_ceil":  79.99},
    {"plan_tier": "professional", "tier_rank": 3, "base_price_floor":  79.99, "base_price_ceil": 199.99},
    {"plan_tier": "enterprise",   "tier_rank": 4, "base_price_floor": 199.99, "base_price_ceil": 499.99},
]
_DIMENSION_REGIONS = [
    {"region_code": "NAM",   "region_name": "North America"},
    {"region_code": "EMEA",  "region_name": "Europe, Middle East & Africa"},
    {"region_code": "APAC",  "region_name": "Asia-Pacific"},
    {"region_code": "LATAM", "region_name": "Latin America"},
]
_DIMENSION_CHANNELS = [
    {"channel_code": "organic",       "channel_type": "inbound"},
    {"channel_code": "paid_search",   "channel_type": "outbound"},
    {"channel_code": "referral",      "channel_type": "viral"},
    {"channel_code": "partner",       "channel_type": "partnership"},
    {"channel_code": "direct_sales",  "channel_type": "outbound"},
    {"channel_code": "trial_convert", "channel_type": "inbound"},
]


@lru_cache(maxsize=1)
def _get_client() -> MongoClient:
    """Create a cached MongoClient from MONGODB_URI env var."""
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        # Try loading from .env file via python-dotenv if present
        try:
            from dotenv import load_dotenv
            load_dotenv()
            uri = os.environ.get("MONGODB_URI")
        except ImportError:
            pass
    if not uri:
        raise EnvironmentError(
            "MONGODB_URI environment variable is not set. "
            "Add it to your .env file or Streamlit Cloud secrets:\n"
            '  MONGODB_URI = "mongodb+srv://user:pass@cluster.mongodb.net/"'
        )
    return MongoClient(uri, serverSelectionTimeoutMS=10_000)


def get_db(cfg: dict[str, Any] | None = None) -> Database:
    """Return the PulseRetain MongoDB database object."""
    cfg = cfg or load_config()
    db_name = os.environ.get("MONGODB_DB", "pulseretain")
    return _get_client()[db_name]


# ---------------------------------------------------------------------------
# Schema / index initialisation (idempotent)
# ---------------------------------------------------------------------------

def init_schema(cfg: dict[str, Any] | None = None) -> None:
    """Create indexes and seed dimension collections. Safe to call repeatedly."""
    db = get_db(cfg)

    # subscribers indexes
    subs = db["subscribers"]
    subs.create_index([("subscriber_id", ASCENDING)], unique=True)
    subs.create_index([("plan_tier", ASCENDING)])
    subs.create_index([("region_code", ASCENDING)])
    subs.create_index([("channel_code", ASCENDING)])
    subs.create_index([("churned", ASCENDING)])
    subs.create_index([("plan_tier", ASCENDING), ("churned", ASCENDING)])
    subs.create_index([("subscription_start_date", ASCENDING)])

    # predictions indexes
    preds = db["model_predictions"]
    preds.create_index([("subscriber_id", ASCENDING), ("model_name", ASCENDING)], unique=True)
    preds.create_index([("risk_tier", ASCENDING)])
    preds.create_index([("model_name", ASCENDING)])

    # model_runs indexes
    db["model_runs"].create_index([("model_name", ASCENDING)])

    # Seed dimension collections
    for plan in _DIMENSION_PLANS:
        db["dim_plan"].update_one(
            {"plan_tier": plan["plan_tier"]}, {"$setOnInsert": plan}, upsert=True
        )
    for region in _DIMENSION_REGIONS:
        db["dim_region"].update_one(
            {"region_code": region["region_code"]}, {"$setOnInsert": region}, upsert=True
        )
    for channel in _DIMENSION_CHANNELS:
        db["dim_channel"].update_one(
            {"channel_code": channel["channel_code"]}, {"$setOnInsert": channel}, upsert=True
        )

    logger.info("MongoDB Atlas schema initialised (indexes + dimensions seeded)")


# ---------------------------------------------------------------------------
# Bulk load subscribers
# ---------------------------------------------------------------------------

def load_subscribers(df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> int:
    """Upsert cleaned subscriber DataFrame into Atlas 'subscribers' collection."""
    db = get_db(cfg)
    coll: Collection = db["subscribers"]

    ops = []
    for _, row in df.iterrows():
        arr = round(float(row["monthly_fee_per_seat"]) * int(row["seat_count"]) * 12, 2)

        churn_date = None
        if pd.notna(row.get("churn_date")):
            cd = row["churn_date"]
            churn_date = cd.isoformat() if hasattr(cd, "isoformat") else str(cd)

        start_date = row["subscription_start_date"]
        start_date = start_date.isoformat() if hasattr(start_date, "isoformat") else str(start_date)

        doc = {
            "subscriber_id":          str(row["subscriber_id"]),
            "plan_tier":              str(row["plan_tier"]).lower(),
            "region_code":            str(row["region"]).upper(),
            "channel_code":           str(row["acquisition_channel"]).lower(),
            "industry_vertical":      str(row["industry_vertical"]),
            "subscription_start_date": start_date,
            "billing_cycle":          str(row["billing_cycle"]).lower(),
            "contract_months":        int(row["contract_months"]),
            "tenure_months":          int(row["tenure_months"]),
            "monthly_fee_per_seat":   float(row["monthly_fee_per_seat"]),
            "seat_count":             int(row["seat_count"]),
            "total_contract_value":   float(row["total_contract_value"]),
            "monthly_arr_usd":        arr,
            "engagement_score":       float(row["engagement_score"]) if pd.notna(row.get("engagement_score")) else None,
            "support_tickets_90d":    int(row["support_tickets_90d"]),
            "payment_failures_12m":   int(row["payment_failures_12m"]),
            "n_integrations":         int(row["n_integrations"]),
            "monthly_logins_per_user": float(row["monthly_logins_per_user"]) if pd.notna(row.get("monthly_logins_per_user")) else None,
            "feature_adoption_rate":  float(row["feature_adoption_rate"]) if pd.notna(row.get("feature_adoption_rate")) else None,
            "churned":                int(row["churned"]),
            "churn_date":             churn_date,
            "churn_reason":           row.get("churn_reason") if pd.notna(row.get("churn_reason")) else None,
        }
        ops.append(UpdateOne({"subscriber_id": doc["subscriber_id"]}, {"$set": doc}, upsert=True))

    if ops:
        result = coll.bulk_write(ops, ordered=False)
        logger.info(
            "Loaded subscribers → upserted=%d modified=%d",
            result.upserted_count, result.modified_count,
        )
    return len(ops)


# ---------------------------------------------------------------------------
# Load model predictions
# ---------------------------------------------------------------------------

def load_predictions(
    predictions_df: pd.DataFrame,
    model_name: str,
    run_id: str,
    threshold: float = 0.40,
    cfg: dict[str, Any] | None = None,
) -> int:
    """Upsert model predictions into Atlas 'model_predictions' collection."""
    db = get_db(cfg)
    coll: Collection = db["model_predictions"]

    def risk_tier(prob: float) -> str:
        if prob >= 0.70:
            return "high"
        if prob >= 0.40:
            return "medium"
        return "low"

    now = datetime.datetime.utcnow().isoformat()
    ops = []
    for _, r in predictions_df.iterrows():
        prob = float(r["churn_probability"])
        doc = {
            "subscriber_id":    str(r["subscriber_id"]),
            "model_name":       model_name,
            "run_id":           run_id,
            "churn_probability": prob,
            "predicted_churned": int(prob >= threshold),
            "risk_tier":        risk_tier(prob),
            "predicted_at":     now,
        }
        ops.append(UpdateOne(
            {"subscriber_id": doc["subscriber_id"], "model_name": model_name},
            {"$set": doc},
            upsert=True,
        ))

    if ops:
        coll.bulk_write(ops, ordered=False)
        logger.info("Loaded %d predictions (model=%s, run=%s)", len(ops), model_name, run_id)
    return len(ops)


# ---------------------------------------------------------------------------
# KPI aggregation pipelines  (replace SQL views)
# ---------------------------------------------------------------------------

def get_headline_kpis(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Aggregate headline KPIs: total/active/churned subscribers, ARR, engagement."""
    db = get_db(cfg)
    pipeline = [
        {"$group": {
            "_id": None,
            "total_subscribers": {"$sum": 1},
            "active_subscribers": {"$sum": {"$cond": [{"$eq": ["$churned", 0]}, 1, 0]}},
            "churned_subscribers": {"$sum": {"$cond": [{"$eq": ["$churned", 1]}, 1, 0]}},
            "total_arr_usd":  {"$sum": "$monthly_arr_usd"},
            "active_arr_usd": {"$sum": {"$cond": [{"$eq": ["$churned", 0]}, "$monthly_arr_usd", 0]}},
            "lost_arr_usd":   {"$sum": {"$cond": [{"$eq": ["$churned", 1]}, "$monthly_arr_usd", 0]}},
            "avg_tenure_months":    {"$avg": "$tenure_months"},
            "avg_engagement_score": {"$avg": "$engagement_score"},
        }},
        {"$addFields": {
            "churn_rate_pct": {
                "$multiply": [
                    {"$cond": [
                        {"$gt": ["$total_subscribers", 0]},
                        {"$divide": ["$churned_subscribers", "$total_subscribers"]},
                        0,
                    ]},
                    100,
                ]
            }
        }},
        {"$project": {"_id": 0}},
    ]
    results = list(db["subscribers"].aggregate(pipeline))
    return pd.DataFrame(results)


def get_kpi_by_plan(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """KPI breakdown by plan tier, ordered by tier_rank."""
    db = get_db(cfg)
    pipeline = [
        {"$group": {
            "_id": "$plan_tier",
            "n_subscribers": {"$sum": 1},
            "n_churned":     {"$sum": "$churned"},
            "total_arr_usd": {"$sum": "$monthly_arr_usd"},
            "avg_engagement": {"$avg": "$engagement_score"},
        }},
        {"$addFields": {
            "plan_tier": "$_id",
            "churn_rate_pct": {
                "$multiply": [
                    {"$cond": [
                        {"$gt": ["$n_subscribers", 0]},
                        {"$divide": ["$n_churned", "$n_subscribers"]},
                        0,
                    ]},
                    100,
                ]
            },
        }},
        {"$lookup": {
            "from": "dim_plan",
            "localField": "plan_tier",
            "foreignField": "plan_tier",
            "as": "plan_info",
        }},
        {"$addFields": {
            "tier_rank": {"$arrayElemAt": ["$plan_info.tier_rank", 0]},
        }},
        {"$sort": {"tier_rank": 1}},
        {"$project": {
            "_id": 0, "plan_info": 0,
            "plan_tier": 1, "tier_rank": 1,
            "n_subscribers": 1, "n_churned": 1,
            "total_arr_usd": 1, "avg_engagement": 1,
            "churn_rate_pct": 1,
        }},
    ]
    results = list(db["subscribers"].aggregate(pipeline))
    return pd.DataFrame(results)


def get_kpi_by_region(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """KPI breakdown by region, ordered by total ARR descending."""
    db = get_db(cfg)
    pipeline = [
        {"$group": {
            "_id": "$region_code",
            "n_subscribers": {"$sum": 1},
            "n_churned":     {"$sum": "$churned"},
            "total_arr_usd": {"$sum": "$monthly_arr_usd"},
            "avg_engagement": {"$avg": "$engagement_score"},
        }},
        {"$addFields": {
            "region_code": "$_id",
            "churn_rate_pct": {
                "$multiply": [
                    {"$cond": [
                        {"$gt": ["$n_subscribers", 0]},
                        {"$divide": ["$n_churned", "$n_subscribers"]},
                        0,
                    ]},
                    100,
                ]
            },
        }},
        {"$sort": {"total_arr_usd": -1}},
        {"$project": {"_id": 0, "region_code": 1, "n_subscribers": 1, "n_churned": 1,
                       "total_arr_usd": 1, "avg_engagement": 1, "churn_rate_pct": 1}},
    ]
    results = list(db["subscribers"].aggregate(pipeline))
    return pd.DataFrame(results)


def get_kpi_by_channel(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """KPI breakdown by acquisition channel."""
    db = get_db(cfg)
    pipeline = [
        {"$group": {
            "_id": "$channel_code",
            "n_subscribers": {"$sum": 1},
            "n_churned":     {"$sum": "$churned"},
            "total_arr_usd": {"$sum": "$monthly_arr_usd"},
        }},
        {"$addFields": {
            "channel_code": "$_id",
            "churn_rate_pct": {
                "$multiply": [
                    {"$cond": [
                        {"$gt": ["$n_subscribers", 0]},
                        {"$divide": ["$n_churned", "$n_subscribers"]},
                        0,
                    ]},
                    100,
                ]
            },
        }},
        {"$lookup": {
            "from": "dim_channel",
            "localField": "channel_code",
            "foreignField": "channel_code",
            "as": "ch_info",
        }},
        {"$addFields": {
            "channel_type": {"$arrayElemAt": ["$ch_info.channel_type", 0]},
        }},
        {"$sort": {"churn_rate_pct": 1}},
        {"$project": {"_id": 0, "ch_info": 0,
                       "channel_code": 1, "channel_type": 1,
                       "n_subscribers": 1, "n_churned": 1,
                       "total_arr_usd": 1, "churn_rate_pct": 1}},
    ]
    results = list(db["subscribers"].aggregate(pipeline))
    return pd.DataFrame(results)


def get_engagement_segments(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Group subscribers into engagement tiers; compute churn rate per tier."""
    db = get_db(cfg)
    pipeline = [
        {"$addFields": {
            "engagement_segment": {
                "$switch": {
                    "branches": [
                        {"case": {"$lt":  ["$engagement_score", 25]}, "then": "Critical (0-24)"},
                        {"case": {"$lt":  ["$engagement_score", 50]}, "then": "At-Risk (25-49)"},
                        {"case": {"$lt":  ["$engagement_score", 75]}, "then": "Moderate (50-74)"},
                    ],
                    "default": "Healthy (75-100)",
                }
            }
        }},
        {"$group": {
            "_id": "$engagement_segment",
            "n_subscribers": {"$sum": 1},
            "n_churned":     {"$sum": "$churned"},
            "avg_arr_usd":   {"$avg": "$monthly_arr_usd"},
            "avg_score":     {"$avg": "$engagement_score"},
        }},
        {"$addFields": {
            "engagement_segment": "$_id",
            "churn_rate_pct": {
                "$multiply": [
                    {"$cond": [
                        {"$gt": ["$n_subscribers", 0]},
                        {"$divide": ["$n_churned", "$n_subscribers"]},
                        0,
                    ]},
                    100,
                ]
            },
        }},
        {"$sort": {"avg_score": 1}},
        {"$project": {"_id": 0, "engagement_segment": 1, "n_subscribers": 1,
                       "n_churned": 1, "avg_arr_usd": 1, "churn_rate_pct": 1}},
    ]
    results = list(db["subscribers"].aggregate(pipeline))
    return pd.DataFrame(results)


def get_risk_watchlist(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Active subscribers flagged high/medium risk by the ML model, with ARR."""
    db = get_db(cfg)
    pipeline = [
        {"$match": {"risk_tier": {"$in": ["high", "medium"]}, "model_name": "gradient_boost"}},
        {"$lookup": {
            "from": "subscribers",
            "localField": "subscriber_id",
            "foreignField": "subscriber_id",
            "as": "sub",
        }},
        {"$unwind": "$sub"},
        {"$match": {"sub.churned": 0}},
        {"$project": {
            "_id": 0,
            "subscriber_id": 1,
            "churn_probability": 1,
            "risk_tier": 1,
            "plan_tier":  "$sub.plan_tier",
            "region_code": "$sub.region_code",
            "arr_usd":    "$sub.monthly_arr_usd",
            "tenure_months": "$sub.tenure_months",
            "engagement_score": "$sub.engagement_score",
            "predicted_at": 1,
        }},
        {"$sort": {"churn_probability": -1}},
        {"$limit": 500},
    ]
    results = list(db["model_predictions"].aggregate(pipeline))
    return pd.DataFrame(results)


def get_cohort_retention(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Cohort retention: group by subscription start year/quarter, compute period buckets."""
    db = get_db(cfg)
    pipeline = [
        {"$addFields": {
            "start_year": {"$toInt": {"$substrCP": ["$subscription_start_date", 0, 4]}},
            "start_month": {"$toInt": {"$substrCP": ["$subscription_start_date", 5, 2]}},
        }},
        {"$addFields": {
            "cohort_quarter": {
                "$switch": {
                    "branches": [
                        {"case": {"$lte": ["$start_month", 3]},  "then": "Q1"},
                        {"case": {"$lte": ["$start_month", 6]},  "then": "Q2"},
                        {"case": {"$lte": ["$start_month", 9]},  "then": "Q3"},
                    ],
                    "default": "Q4",
                }
            },
            "period_bucket": {"$floor": {"$divide": ["$tenure_months", 3]}},
        }},
        {"$group": {
            "_id": {
                "cohort_year": "$start_year",
                "cohort_quarter": "$cohort_quarter",
                "period_bucket": "$period_bucket",
            },
            "n_total":   {"$sum": 1},
            "n_active":  {"$sum": {"$cond": [{"$eq": ["$churned", 0]}, 1, 0]}},
        }},
        {"$addFields": {
            "cohort_year":    "$_id.cohort_year",
            "cohort_quarter": "$_id.cohort_quarter",
            "period_bucket":  "$_id.period_bucket",
            "retention_pct":  {
                "$multiply": [
                    {"$cond": [
                        {"$gt": ["$n_total", 0]},
                        {"$divide": ["$n_active", "$n_total"]},
                        0,
                    ]},
                    100,
                ]
            },
        }},
        {"$sort": {"cohort_year": 1, "cohort_quarter": 1, "period_bucket": 1}},
        {"$project": {"_id": 0, "cohort_year": 1, "cohort_quarter": 1,
                       "period_bucket": 1, "n_total": 1, "n_active": 1, "retention_pct": 1}},
    ]
    results = list(db["subscribers"].aggregate(pipeline))
    return pd.DataFrame(results)


def query_df(
    collection_name: str,
    pipeline: list[dict],
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run an arbitrary aggregation pipeline and return a DataFrame.

    Note: The signature differs from the SQLite query_df (pipeline instead of SQL string).
    The dashboard's ARR trend query is adapted below.
    """
    db = get_db(cfg)
    results = list(db[collection_name].aggregate(pipeline))
    return pd.DataFrame(results)


def get_arr_trend(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Monthly ARR movement: arr_added, arr_lost, net_change, running_arr."""
    db = get_db(cfg)

    # New subscribers — group by start month
    added_pipeline = [
        {"$group": {
            "_id": {"$substrCP": ["$subscription_start_date", 0, 7]},
            "arr_added": {"$sum": "$monthly_arr_usd"},
        }},
        {"$project": {"_id": 0, "event_month": "$_id", "arr_added": 1, "arr_lost": {"$literal": 0.0}}},
    ]
    # Churned subscribers — group by churn month
    lost_pipeline = [
        {"$match": {"churned": 1, "churn_date": {"$ne": None}}},
        {"$group": {
            "_id": {"$substrCP": ["$churn_date", 0, 7]},
            "arr_lost": {"$sum": "$monthly_arr_usd"},
        }},
        {"$project": {"_id": 0, "event_month": "$_id", "arr_added": {"$literal": 0.0}, "arr_lost": 1}},
    ]

    added = pd.DataFrame(db["subscribers"].aggregate(added_pipeline))
    lost  = pd.DataFrame(db["subscribers"].aggregate(lost_pipeline))

    combined = pd.concat([added, lost], ignore_index=True)
    if combined.empty:
        return pd.DataFrame()

    agg = (
        combined.groupby("event_month")
        .agg(arr_added=("arr_added", "sum"), arr_lost=("arr_lost", "sum"))
        .reset_index()
        .sort_values("event_month")
    )
    agg["net_change"] = agg["arr_added"] - agg["arr_lost"]
    agg["running_arr"] = agg["net_change"].cumsum()
    return agg


def run_db_pipeline(df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> None:
    """Top-level step: init schema and load subscribers. Called by pipeline.py."""
    cfg = cfg or load_config()
    init_schema(cfg)
    load_subscribers(df, cfg)
