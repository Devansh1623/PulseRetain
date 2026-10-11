"""
scripts/migrate_sqlite_to_atlas.py
===================================
One-time migration: reads the existing SQLite warehouse and pushes all data
to MongoDB Atlas.

Prerequisites:
    pip install pymongo[srv]

    Set in .env (or export):
        MONGODB_URI = "mongodb+srv://..."
        DB_BACKEND  = "mongo"      # makes database.py use Atlas going forward

Run:
    python scripts/migrate_sqlite_to_atlas.py

The script is idempotent — re-running will upsert without duplicating records.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd

def main() -> None:
    from pulse.config import load_config, resolve_path

    cfg = load_config()
    db_path = resolve_path("processed", cfg) / "pulse_warehouse.db"

    if not db_path.exists():
        print(f"[ERROR] SQLite DB not found at {db_path}")
        print("Run: python -m pulse.pipeline --step db")
        sys.exit(1)

    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        print("[ERROR] MONGODB_URI is not set. Add it to your .env file.")
        sys.exit(1)

    print(f"Source: {db_path}")
    print(f"Target: MongoDB Atlas  ({os.environ.get('MONGODB_DB', 'pulseretain')})")

    # Read from SQLite
    conn = sqlite3.connect(db_path)
    subscribers_df = pd.read_sql_query(
        """
        SELECT
            fs.subscriber_id,
            dp.plan_tier,
            dr.region_code,
            dc.channel_code,
            dc.channel_type,
            di.industry_name          AS industry_vertical,
            fs.subscription_start_date,
            fs.billing_cycle,
            fs.contract_months,
            fs.tenure_months,
            fs.monthly_fee_per_seat,
            fs.seat_count,
            fs.total_contract_value,
            fs.monthly_arr_usd,
            fs.engagement_score,
            fs.support_tickets_90d,
            fs.payment_failures_12m,
            fs.n_integrations,
            fs.monthly_logins_per_user,
            fs.feature_adoption_rate,
            fs.churned,
            fs.churn_date,
            fs.churn_reason
        FROM fact_subscribers fs
        JOIN dim_plan     dp ON fs.plan_id     = dp.plan_id
        JOIN dim_region   dr ON fs.region_id   = dr.region_id
        JOIN dim_channel  dc ON fs.channel_id  = dc.channel_id
        JOIN dim_industry di ON fs.industry_id = di.industry_id
        """,
        conn,
    )

    predictions_df = pd.read_sql_query(
        "SELECT subscriber_id, model_name, run_id, churn_probability, "
        "predicted_churned, risk_tier, predicted_at "
        "FROM fact_model_predictions",
        conn,
    )
    conn.close()

    print(f"  Subscribers: {len(subscribers_df):,}")
    print(f"  Predictions: {len(predictions_df):,}")

    # Push to Atlas
    from pulse.mongo_database import init_schema, get_db
    from pymongo import UpdateOne

    print("\nInitialising Atlas schema + dimensions...")
    init_schema(cfg)

    db = get_db(cfg)

    # Upsert subscribers
    print("Migrating subscribers...")
    ops = []
    for _, row in subscribers_df.iterrows():
        doc = row.where(pd.notna(row), None).to_dict()
        ops.append(UpdateOne(
            {"subscriber_id": doc["subscriber_id"]},
            {"$set": doc},
            upsert=True,
        ))
    if ops:
        res = db["subscribers"].bulk_write(ops, ordered=False)
        print(f"  Upserted: {res.upserted_count}  Modified: {res.modified_count}")

    # Upsert predictions
    print("Migrating predictions...")
    ops = []
    for _, row in predictions_df.iterrows():
        doc = row.where(pd.notna(row), None).to_dict()
        ops.append(UpdateOne(
            {"subscriber_id": doc["subscriber_id"], "model_name": doc["model_name"]},
            {"$set": doc},
            upsert=True,
        ))
    if ops:
        res = db["model_predictions"].bulk_write(ops, ordered=False)
        print(f"  Upserted: {res.upserted_count}  Modified: {res.modified_count}")

    print("\n✅ Migration complete! MongoDB Atlas is now populated.")
    print(f"   Set DB_BACKEND=mongo in your .env to use Atlas going forward.")


if __name__ == "__main__":
    main()
