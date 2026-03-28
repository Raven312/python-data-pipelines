"""Gold layer — business-level aggregations and KPI tables.

The Gold layer serves business consumers by pre-computing aggregations,
KPIs, and denormalized views that can be served directly to dashboards,
ML feature stores, or downstream analytics.

Gold tables produced:
  - supplier_by_nation:  Per-nation supplier counts and avg account balance
  - supplier_summary:    Overall pipeline summary / KPI row
"""

import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


class GoldAggregator:
    """Builds Gold-layer aggregation tables from Silver data."""

    def build_supplier_by_nation(self, silver_df: DataFrame) -> DataFrame:
        """Aggregate supplier metrics per nation.

        Produces one row per nation with:
          - nation_name
          - supplier_count
          - avg_account_balance
          - max_account_balance
          - min_account_balance
          - total_account_balance
        """
        agg_df = (
            silver_df.filter(F.col("_is_current") == True)  # noqa: E712
            .groupBy("nation_name")
            .agg(
                F.count("supplier_id").alias("supplier_count"),
                F.round(F.avg("account_balance"), 2).alias("avg_account_balance"),
                F.max("account_balance").alias("max_account_balance"),
                F.min("account_balance").alias("min_account_balance"),
                F.round(F.sum("account_balance"), 2).alias("total_account_balance"),
            )
            .orderBy(F.col("supplier_count").desc())
        )
        logger.info("Gold supplier_by_nation: %d nation groups", agg_df.count())
        return agg_df

    def build_supplier_summary(self, silver_df: DataFrame) -> DataFrame:
        """Build a single-row pipeline summary KPI table.

        Produces:
          - total_suppliers
          - total_nations
          - avg_account_balance
          - pipeline_timestamp
        """
        current = silver_df.filter(F.col("_is_current") == True)  # noqa: E712
        summary = current.agg(
            F.countDistinct("supplier_id").alias("total_suppliers"),
            F.countDistinct("nation_name").alias("total_nations"),
            F.round(F.avg("account_balance"), 2).alias("avg_account_balance"),
            F.round(F.sum("account_balance"), 2).alias("total_account_balance"),
        ).withColumn("pipeline_timestamp", F.current_timestamp())

        logger.info("Gold supplier_summary built")
        return summary

    def build_all(self, silver_df: DataFrame) -> dict[str, DataFrame]:
        """Build all Gold tables and return as a dict."""
        return {
            "supplier_by_nation": self.build_supplier_by_nation(silver_df),
            "supplier_summary": self.build_supplier_summary(silver_df),
        }
