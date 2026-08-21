"""Auto Loader — Incremental file ingestion with schema evolution.

Auto Loader (cloudFiles) is Databricks' recommended way to ingest files
from cloud storage. It provides:
  - Exactly-once semantics via file notification or directory listing
  - Automatic schema inference and evolution
  - Rescue data column for schema mismatches
  - Scalable to millions of files

AE Pathway Coverage:
  - Auto Loader configuration and usage
  - Schema inference vs. explicit schemas
  - Schema evolution strategies
  - Rescue data patterns
  - Trigger modes for batch vs. streaming
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.streaming import StreamingQuery
from pyspark.sql.types import StructType

logger = logging.getLogger(__name__)


@dataclass
class AutoLoaderConfig:
    """Configuration for an Auto Loader stream."""

    source_path: str
    source_format: str = "csv"
    schema_location: str = ""
    checkpoint_location: str = ""
    schema: Optional[StructType] = None
    max_files_per_trigger: int = 1000
    schema_evolution_mode: str = "addNewColumns"
    rescue_data_column: str = "_rescued_data"
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.schema_location:
            self.schema_location = f"/mnt/checkpoints/{self.source_path.rstrip('/').split('/')[-1]}_schema"
        if not self.checkpoint_location:
            self.checkpoint_location = f"/mnt/checkpoints/{self.source_path.rstrip('/').split('/')[-1]}_cp"


class AutoLoaderIngestion:
    """Manages Auto Loader streams for incremental file ingestion.

    Supports three trigger modes:
      - availableNow: Process all available files, then stop (batch-like)
      - processingTime: Micro-batch at fixed intervals
      - continuous: Low-latency continuous processing
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def create_stream(self, config: AutoLoaderConfig) -> DataFrame:
        """Create a streaming DataFrame from Auto Loader.

        Args:
            config: Auto Loader configuration.

        Returns:
            Streaming DataFrame with audit columns.
        """
        reader = (
            self.spark.readStream.format("cloudFiles")
            .option("cloudFiles.format", config.source_format)
            .option("cloudFiles.schemaLocation", config.schema_location)
            .option("cloudFiles.maxFilesPerTrigger", config.max_files_per_trigger)
            .option("cloudFiles.schemaEvolutionMode", config.schema_evolution_mode)
            .option("rescuedDataColumn", config.rescue_data_column)
        )

        if config.schema:
            reader = reader.schema(config.schema)
        else:
            reader = reader.option("cloudFiles.inferColumnTypes", "true")

        for key, value in config.options.items():
            reader = reader.option(key, value)

        df = reader.load(config.source_path)

        return (
            df.withColumn("_ingested_at", F.current_timestamp())
            .withColumn("_source_file", F.input_file_name())
        )

    def write_to_delta(
        self,
        stream_df: DataFrame,
        target_table: str,
        checkpoint_location: str,
        trigger_mode: str = "availableNow",
        trigger_interval: str = "10 seconds",
        merge_keys: Optional[list[str]] = None,
    ) -> StreamingQuery:
        """Write a streaming DataFrame to a Delta table.

        Args:
            stream_df: Streaming DataFrame to write.
            target_table: Target Delta table name.
            checkpoint_location: Checkpoint directory.
            trigger_mode: One of 'availableNow', 'processingTime', 'continuous'.
            trigger_interval: Interval for processingTime trigger.
            merge_keys: If provided, uses foreachBatch with MERGE for upserts.

        Returns:
            StreamingQuery handle.
        """
        writer = (
            stream_df.writeStream
            .format("delta")
            .option("checkpointLocation", checkpoint_location)
            .outputMode("append")
        )

        if trigger_mode == "availableNow":
            writer = writer.trigger(availableNow=True)
        elif trigger_mode == "processingTime":
            writer = writer.trigger(processingTime=trigger_interval)
        elif trigger_mode == "continuous":
            writer = writer.trigger(continuous=trigger_interval)

        if merge_keys:
            return self._write_with_merge(
                stream_df, target_table, checkpoint_location, merge_keys, writer
            )

        query = writer.toTable(target_table)
        logger.info("Auto Loader stream started → %s [trigger=%s]", target_table, trigger_mode)
        return query

    def _write_with_merge(
        self,
        stream_df: DataFrame,
        target_table: str,
        checkpoint_location: str,
        merge_keys: list[str],
        writer,
    ) -> StreamingQuery:
        """Use foreachBatch to perform MERGE (upsert) operations."""

        def upsert_batch(batch_df: DataFrame, batch_id: int) -> None:
            merge_condition = " AND ".join(
                f"target.{k} = source.{k}" for k in merge_keys
            )
            batch_df.createOrReplaceTempView("source_batch")

            self.spark.sql(f"""
                MERGE INTO {target_table} AS target
                USING source_batch AS source
                ON {merge_condition}
                WHEN MATCHED THEN UPDATE SET *
                WHEN NOT MATCHED THEN INSERT *
            """)
            logger.info("Merged batch %d into %s", batch_id, target_table)

        return (
            stream_df.writeStream
            .format("delta")
            .option("checkpointLocation", checkpoint_location)
            .foreachBatch(upsert_batch)
            .trigger(availableNow=True)
            .start()
        )
