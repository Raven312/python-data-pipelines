"""Delta Live Tables — Bronze Layer.

Bronze tables ingest raw data with minimal transformation, adding only
audit metadata. In Databricks, DLT Bronze tables use Auto Loader
(@dlt.table with cloudFiles) for incremental ingestion from cloud storage.

DLT Expectations enforce data quality at ingestion:
  - @dlt.expect: warn on violation (row kept)
  - @dlt.expect_or_drop: drop violating rows
  - @dlt.expect_or_fail: fail the pipeline on violation
"""

import dlt
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

SUPPLIER_SCHEMA = StructType([
    StructField("s_suppkey", IntegerType(), False),
    StructField("s_name", StringType(), False),
    StructField("s_address", StringType(), True),
    StructField("s_nationkey", IntegerType(), False),
    StructField("s_phone", StringType(), True),
    StructField("s_acctbal", DoubleType(), True),
])

NATION_SCHEMA = StructType([
    StructField("n_nationkey", IntegerType(), False),
    StructField("n_name", StringType(), False),
    StructField("n_regionkey", IntegerType(), True),
])


@dlt.table(
    name="bronze_suppliers",
    comment="Raw supplier data ingested from source with audit columns",
    table_properties={"quality": "bronze", "pipelines.autoOptimize.managed": "true"},
)
@dlt.expect("valid_suppkey", "s_suppkey IS NOT NULL")
@dlt.expect("valid_name", "s_name IS NOT NULL")
def bronze_suppliers():
    """Ingest raw supplier data using Auto Loader.

    In production, this reads from cloud storage (S3/ADLS/GCS) using
    cloudFiles format for incremental, exactly-once ingestion with
    automatic schema inference and evolution.
    """
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("cloudFiles.schemaLocation", "/mnt/checkpoints/suppliers_schema")
        .schema(SUPPLIER_SCHEMA)
        .load("/mnt/raw/suppliers/")
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source_file", F.input_file_name())
        .withColumn("_batch_id", F.lit(F.date_format(F.current_timestamp(), "yyyyMMdd_HHmmss")))
    )


@dlt.table(
    name="bronze_nations",
    comment="Raw nation reference data with audit columns",
    table_properties={"quality": "bronze"},
)
@dlt.expect_or_fail("valid_nationkey", "n_nationkey IS NOT NULL")
@dlt.expect_or_fail("valid_nation_name", "n_name IS NOT NULL")
def bronze_nations():
    """Ingest raw nation reference data.

    Reference/dimension tables are typically batch-loaded (not streaming)
    since they change infrequently.
    """
    return (
        spark.read.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .schema(NATION_SCHEMA)
        .load("/mnt/raw/nations/")
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source_file", F.input_file_name())
    )
