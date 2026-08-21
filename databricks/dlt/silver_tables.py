"""Delta Live Tables — Silver Layer.

Silver tables cleanse, conform, and enrich Bronze data. DLT handles
incremental processing automatically — only new/changed Bronze rows
flow through to Silver.

Key patterns:
  - APPLY CHANGES for SCD Type-2 / CDC processing
  - Expectations for post-transformation quality gates
  - Materialized views for dimension joins
"""

import dlt
from pyspark.sql import functions as F


@dlt.table(
    name="silver_suppliers",
    comment="Cleansed and conformed supplier data joined with nation dimension",
    table_properties={
        "quality": "silver",
        "delta.autoOptimize.optimizeWrite": "true",
        "delta.autoOptimize.autoCompact": "true",
    },
)
@dlt.expect_or_drop("valid_supplier_id", "supplier_id IS NOT NULL")
@dlt.expect_or_drop("valid_supplier_name", "supplier_name IS NOT NULL")
@dlt.expect("valid_balance", "account_balance >= 0")
@dlt.expect("valid_nation", "nation_name IS NOT NULL")
def silver_suppliers():
    """Build conformed supplier dimension with nation enrichment.

    Transformation steps:
      1. Deduplicate on supplier key (keep latest ingestion)
      2. Standardize strings (trim + uppercase)
      3. Join with nation dimension
      4. Project to canonical schema
      5. Add SCD Type-2 audit columns
    """
    suppliers = dlt.read_stream("bronze_suppliers")
    nations = dlt.read("bronze_nations")

    deduped = (
        suppliers.withWatermark("_ingested_at", "1 hour")
        .dropDuplicates(["s_suppkey"])
    )

    standardized = (
        deduped
        .withColumn("s_name", F.upper(F.trim(F.col("s_name"))))
        .withColumn("s_address", F.upper(F.trim(F.col("s_address"))))
    )

    joined = standardized.join(
        nations,
        standardized.s_nationkey == nations.n_nationkey,
        "inner",
    )

    return joined.select(
        F.col("s_suppkey").alias("supplier_id"),
        F.col("s_name").alias("supplier_name"),
        F.col("s_address").alias("supplier_address"),
        F.col("s_phone").alias("supplier_phone"),
        F.col("s_acctbal").alias("account_balance"),
        F.col("n_name").alias("nation_name"),
        F.col("s_nationkey").alias("nation_key"),
        F.current_timestamp().alias("_valid_from"),
        F.lit(None).cast("timestamp").alias("_valid_to"),
        F.lit(True).alias("_is_current"),
    )


@dlt.table(
    name="silver_suppliers_scd2",
    comment="SCD Type-2 supplier dimension using APPLY CHANGES",
    table_properties={"quality": "silver"},
)
def silver_suppliers_scd2():
    """SCD Type-2 dimension using DLT's APPLY CHANGES (CDC).

    APPLY CHANGES automatically handles:
      - Insert new records
      - Update changed records (close old, open new version)
      - Maintain _valid_from, _valid_to, _is_current
      - Out-of-order event handling
    """
    pass


dlt.apply_changes(
    target="silver_suppliers_scd2",
    source="bronze_suppliers",
    keys=["s_suppkey"],
    sequence_by="_ingested_at",
    stored_as_scd_type=2,
    columns=[
        "s_suppkey",
        "s_name",
        "s_address",
        "s_nationkey",
        "s_phone",
        "s_acctbal",
    ],
)
