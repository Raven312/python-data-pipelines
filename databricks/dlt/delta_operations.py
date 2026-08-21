"""Delta Lake operations — MERGE, time travel, OPTIMIZE, Z-ORDER.

Advanced Delta Lake operations used throughout the medallion pipeline
for data management, performance optimization, and data recovery.

AE Pathway Coverage:
  - Delta Lake MERGE (upsert) patterns
  - Time travel (VERSION AS OF, TIMESTAMP AS OF)
  - OPTIMIZE and Z-ORDER for query performance
  - VACUUM for storage management
  - Change Data Feed (CDF) for downstream consumers
  - Schema evolution (addColumns, mergeSchema)
"""

import logging
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


class DeltaOperations:
    """Advanced Delta Lake operations for the medallion pipeline.

    Provides high-level wrappers around Delta Lake's core capabilities
    for use across Bronze, Silver, and Gold layers.
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark

    # ── MERGE / Upsert ──────────────────────────────────────────────────────

    def merge_into(
        self,
        target_table: str,
        source_df: DataFrame,
        merge_keys: list[str],
        update_columns: Optional[list[str]] = None,
        insert_when_not_matched: bool = True,
        delete_condition: Optional[str] = None,
    ) -> dict:
        """Perform a MERGE (upsert) operation into a Delta table.

        This is the core pattern for Silver-layer incremental updates:
        new records are inserted, changed records are updated, and
        optionally deleted records are removed.

        Args:
            target_table: Fully qualified Delta table name.
            source_df: Source DataFrame with new/updated data.
            merge_keys: Columns to match on (e.g., ["supplier_id"]).
            update_columns: Columns to update on match (None = all).
            insert_when_not_matched: Insert new records.
            delete_condition: SQL condition for DELETE (e.g., "source.is_deleted = true").

        Returns:
            Dict with operation metrics.
        """
        source_df.createOrReplaceTempView("merge_source")

        merge_condition = " AND ".join(
            f"target.{k} = source.{k}" for k in merge_keys
        )

        if update_columns:
            update_set = ", ".join(f"target.{c} = source.{c}" for c in update_columns)
        else:
            update_set = "*"

        sql = f"""
            MERGE INTO {target_table} AS target
            USING merge_source AS source
            ON {merge_condition}
            WHEN MATCHED THEN UPDATE SET {update_set}
        """

        if insert_when_not_matched:
            sql += " WHEN NOT MATCHED THEN INSERT *"

        if delete_condition:
            sql += f" WHEN MATCHED AND {delete_condition} THEN DELETE"

        result = self.spark.sql(sql)
        metrics = self._get_merge_metrics(target_table)
        logger.info("MERGE into %s: %s", target_table, metrics)
        return metrics

    def scd_type2_merge(
        self,
        target_table: str,
        source_df: DataFrame,
        key_columns: list[str],
        tracked_columns: list[str],
    ) -> None:
        """Perform SCD Type-2 merge — close old versions, insert new.

        Pattern:
          1. Match on key columns where _is_current = true
          2. If tracked columns changed: close old record, insert new
          3. If no match: insert new record
        """
        source_df.createOrReplaceTempView("scd2_source")

        key_condition = " AND ".join(
            f"target.{k} = source.{k}" for k in key_columns
        )
        change_condition = " OR ".join(
            f"target.{c} <> source.{c}" for c in tracked_columns
        )

        self.spark.sql(f"""
            MERGE INTO {target_table} AS target
            USING scd2_source AS source
            ON {key_condition} AND target._is_current = true
            WHEN MATCHED AND ({change_condition}) THEN
                UPDATE SET
                    target._is_current = false,
                    target._valid_to = current_timestamp()
            WHEN NOT MATCHED THEN
                INSERT (
                    {', '.join(key_columns + tracked_columns)},
                    _valid_from, _valid_to, _is_current
                )
                VALUES (
                    {', '.join(f'source.{c}' for c in key_columns + tracked_columns)},
                    current_timestamp(), NULL, true
                )
        """)
        logger.info("SCD-2 merge into %s complete", target_table)

    # ── Time Travel ───────────────────────────────────────────────────────

    def read_version(self, table: str, version: int) -> DataFrame:
        """Read a specific version of a Delta table (time travel)."""
        return self.spark.read.option("versionAsOf", version).table(table)

    def read_timestamp(self, table: str, timestamp: str) -> DataFrame:
        """Read a Delta table as of a specific timestamp."""
        return self.spark.read.option("timestampAsOf", timestamp).table(table)

    def restore_to_version(self, table: str, version: int) -> None:
        """Restore a Delta table to a previous version."""
        self.spark.sql(f"RESTORE TABLE {table} TO VERSION AS OF {version}")
        logger.info("Restored %s to version %d", table, version)

    def get_history(self, table: str, limit: int = 20) -> DataFrame:
        """Get the transaction history of a Delta table."""
        return self.spark.sql(f"DESCRIBE HISTORY {table} LIMIT {limit}")

    # ── Performance Optimization ──────────────────────────────────────────

    def optimize(self, table: str, z_order_columns: Optional[list[str]] = None) -> None:
        """Run OPTIMIZE on a Delta table, optionally with Z-ORDER.

        OPTIMIZE compacts small files into larger ones for better read
        performance. Z-ORDER co-locates related data in the same files
        for columns frequently used in WHERE clauses.
        """
        if z_order_columns:
            cols = ", ".join(z_order_columns)
            self.spark.sql(f"OPTIMIZE {table} ZORDER BY ({cols})")
            logger.info("OPTIMIZE + ZORDER(%s) on %s", cols, table)
        else:
            self.spark.sql(f"OPTIMIZE {table}")
            logger.info("OPTIMIZE on %s", table)

    def vacuum(self, table: str, retention_hours: int = 168) -> None:
        """Remove old files no longer referenced by the Delta log.

        Default retention is 7 days (168 hours). Shorter retention
        requires setting delta.retentionDurationCheck.enabled = false.
        """
        self.spark.sql(f"VACUUM {table} RETAIN {retention_hours} HOURS")
        logger.info("VACUUM on %s (retention=%dh)", table, retention_hours)

    def analyze_table(self, table: str) -> None:
        """Compute statistics for the query optimizer."""
        self.spark.sql(f"ANALYZE TABLE {table} COMPUTE STATISTICS FOR ALL COLUMNS")
        logger.info("Statistics computed for %s", table)

    # ── Change Data Feed ────────────────────────────────────────────────────

    def read_changes(
        self,
        table: str,
        starting_version: Optional[int] = None,
        starting_timestamp: Optional[str] = None,
    ) -> DataFrame:
        """Read Change Data Feed (CDF) for a table.

        CDF captures row-level changes (INSERT, UPDATE_PREIMAGE,
        UPDATE_POSTIMAGE, DELETE) for downstream consumers.
        Requires 'delta.enableChangeDataFeed' = 'true' on the table.
        """
        reader = self.spark.read.format("delta").option("readChangeFeed", "true")
        if starting_version is not None:
            reader = reader.option("startingVersion", starting_version)
        elif starting_timestamp:
            reader = reader.option("startingTimestamp", starting_timestamp)
        return reader.table(table)

    # ── Schema Evolution ────────────────────────────────────────────────────

    def evolve_schema(self, table: str, source_df: DataFrame) -> None:
        """Write data with automatic schema evolution (mergeSchema).

        Adds new columns from the source DataFrame to the target table
        without breaking existing consumers.
        """
        (
            source_df.write
            .format("delta")
            .mode("append")
            .option("mergeSchema", "true")
            .saveAsTable(table)
        )
        logger.info("Schema evolved for %s", table)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _get_merge_metrics(self, table: str) -> dict:
        """Extract metrics from the most recent MERGE operation."""
        try:
            history = self.spark.sql(f"DESCRIBE HISTORY {table} LIMIT 1")
            row = history.first()
            if row and row.operationMetrics:
                return dict(row.operationMetrics)
        except Exception:
            pass
        return {}
