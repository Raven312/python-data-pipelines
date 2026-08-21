"""Delta Live Tables — Gold Layer.

Gold tables serve business consumers with pre-computed aggregations,
KPIs, and denormalized views. These are the tables that dashboards,
ML feature stores, and downstream analytics query directly.

Gold tables are typically materialized views that DLT keeps fresh
as upstream Silver data changes.
"""

import dlt
from pyspark.sql import functions as F


@dlt.table(
    name="gold_supplier_by_nation",
    comment="Supplier metrics aggregated by nation for executive dashboards",
    table_properties={
        "quality": "gold",
        "delta.autoOptimize.optimizeWrite": "true",
    },
)
@dlt.expect("has_suppliers", "supplier_count > 0")
def gold_supplier_by_nation():
    """Per-nation supplier aggregation.

    Metrics:
      - supplier_count: number of active suppliers per nation
      - avg/min/max/total account balance
      - Ordered by supplier count descending for dashboard rendering
    """
    silver = dlt.read("silver_suppliers")

    return (
        silver.filter(F.col("_is_current") == True)  # noqa: E712
        .groupBy("nation_name")
        .agg(
            F.count("supplier_id").alias("supplier_count"),
            F.round(F.avg("account_balance"), 2).alias("avg_account_balance"),
            F.max("account_balance").alias("max_account_balance"),
            F.min("account_balance").alias("min_account_balance"),
            F.round(F.sum("account_balance"), 2).alias("total_account_balance"),
        )
        .orderBy(F.col("supplier_count").desc())
        .withColumn("_computed_at", F.current_timestamp())
    )


@dlt.table(
    name="gold_supplier_summary",
    comment="Single-row pipeline KPI summary for monitoring dashboards",
    table_properties={"quality": "gold"},
)
def gold_supplier_summary():
    """Global supplier KPI summary — one row with high-level metrics."""
    silver = dlt.read("silver_suppliers")
    current = silver.filter(F.col("_is_current") == True)  # noqa: E712

    return current.agg(
        F.countDistinct("supplier_id").alias("total_suppliers"),
        F.countDistinct("nation_name").alias("total_nations"),
        F.round(F.avg("account_balance"), 2).alias("avg_account_balance"),
        F.round(F.sum("account_balance"), 2).alias("total_account_balance"),
        F.max("_valid_from").alias("last_updated"),
    ).withColumn("_computed_at", F.current_timestamp())


@dlt.table(
    name="gold_top_suppliers",
    comment="Top 10 suppliers by account balance for executive reporting",
    table_properties={"quality": "gold"},
)
def gold_top_suppliers():
    """Top suppliers ranked by account balance."""
    silver = dlt.read("silver_suppliers")

    return (
        silver.filter(F.col("_is_current") == True)  # noqa: E712
        .select(
            "supplier_id",
            "supplier_name",
            "nation_name",
            "account_balance",
            "supplier_phone",
        )
        .orderBy(F.col("account_balance").desc())
        .limit(10)
        .withColumn("rank", F.row_number().over(
            F.Window.orderBy(F.col("account_balance").desc())
        ))
        .withColumn("_computed_at", F.current_timestamp())
    )
