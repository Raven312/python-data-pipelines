"""Bronze layer — raw data ingestion with schema enforcement and audit metadata.

The Bronze layer is the landing zone. Data arrives as-is from source systems
with the addition of audit columns (_ingested_at, _source, _batch_id) for
lineage tracking. Schema is enforced on read to catch structural issues early.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Union

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType

logger = logging.getLogger(__name__)


class BronzeIngestion:
    """Ingests raw data into the Bronze layer with audit metadata.

    Supports multiple source formats (dict/list, CSV, Parquet, JSON, Delta)
    and enforces optional schema validation on read.
    """

    def __init__(self, spark: SparkSession, source_name: str, batch_id: Optional[str] = None):
        self.spark = spark
        self.source_name = source_name
        self.batch_id = batch_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    def ingest_from_dict(
        self,
        data: dict[str, list],
        schema: Optional[StructType] = None,
    ) -> DataFrame:
        """Ingest data from a Python dictionary into a Spark DataFrame.

        Args:
            data: Column-oriented dict (e.g. {"col": [1,2,3]}).
            schema: Optional StructType for schema enforcement.

        Returns:
            Bronze-layer DataFrame with audit columns.
        """
        rows = list(zip(*data.values()))
        columns = list(data.keys())

        if schema:
            df = self.spark.createDataFrame(rows, schema=schema)
        else:
            df = self.spark.createDataFrame(rows, schema=columns)

        logger.info("Ingested %d rows from dict source '%s'", df.count(), self.source_name)
        return self._add_audit_columns(df)

    def ingest_from_csv(
        self,
        path: str,
        schema: Optional[StructType] = None,
        options: Optional[dict] = None,
    ) -> DataFrame:
        """Ingest data from CSV file(s)."""
        reader = self.spark.read.option("header", "true")
        if schema:
            reader = reader.schema(schema)
        else:
            reader = reader.option("inferSchema", "true")
        for k, v in (options or {}).items():
            reader = reader.option(k, v)

        df = reader.csv(path)
        logger.info("Ingested %d rows from CSV '%s'", df.count(), path)
        return self._add_audit_columns(df)

    def ingest_from_parquet(self, path: str) -> DataFrame:
        """Ingest data from Parquet file(s)."""
        df = self.spark.read.parquet(path)
        logger.info("Ingested %d rows from Parquet '%s'", df.count(), path)
        return self._add_audit_columns(df)

    def ingest_from_delta(self, path: str) -> DataFrame:
        """Ingest data from a Delta table path (Databricks)."""
        df = self.spark.read.format("delta").load(path)
        logger.info("Ingested %d rows from Delta '%s'", df.count(), path)
        return self._add_audit_columns(df)

    def _add_audit_columns(self, df: DataFrame) -> DataFrame:
        """Append standard audit/lineage columns to the DataFrame."""
        return (
            df.withColumn("_ingested_at", F.lit(datetime.now(timezone.utc).isoformat()))
            .withColumn("_source", F.lit(self.source_name))
            .withColumn("_batch_id", F.lit(self.batch_id))
        )


def create_bronze_schemas() -> dict[str, StructType]:
    """Return predefined schemas for known source datasets.

    Using explicit schemas prevents silent type coercion and catches
    upstream schema drift at the earliest possible point.
    """
    from pyspark.sql.types import (
        DoubleType,
        IntegerType,
        StringType,
        StructField,
    )

    return {
        "supplier": StructType([
            StructField("s_suppkey", IntegerType(), False),
            StructField("s_name", StringType(), False),
            StructField("s_address", StringType(), True),
            StructField("s_nationkey", IntegerType(), False),
            StructField("s_phone", StringType(), True),
            StructField("s_acctbal", DoubleType(), True),
        ]),
        "nation": StructType([
            StructField("n_nationkey", IntegerType(), False),
            StructField("n_name", StringType(), False),
            StructField("n_regionkey", IntegerType(), True),
        ]),
    }
