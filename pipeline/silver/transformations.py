"""Silver layer — data cleansing, conforming, deduplication, and enrichment.

The Silver layer produces a clean, conformed, analytics-ready dataset by:
  1. Deduplicating records (using a primary key + ordering column)
  2. Dropping rows that fail null checks on critical columns
  3. Standardizing string columns (trimming, upper-casing)
  4. Joining dimension tables (supplier ⟕ nation)
  5. Selecting and renaming to a canonical output schema
  6. Adding SCD Type-2 audit columns (_valid_from, _valid_to, _is_current)
"""

import logging
from typing import Optional

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


class SilverTransformer:
    """Transforms Bronze data into Silver-layer conformed datasets."""

    def __init__(
        self,
        supplier_key: str = "s_suppkey",
        nation_join_key_left: str = "s_nationkey",
        nation_join_key_right: str = "n_nationkey",
    ):
        self.supplier_key = supplier_key
        self.nation_join_left = nation_join_key_left
        self.nation_join_right = nation_join_key_right

    def transform(
        self,
        suppliers_bronze: DataFrame,
        nations_bronze: DataFrame,
    ) -> DataFrame:
        """Run the full Silver transformation pipeline.

        Args:
            suppliers_bronze: Raw supplier DataFrame from Bronze.
            nations_bronze: Raw nation DataFrame from Bronze.

        Returns:
            Cleansed, joined, and conformed Silver DataFrame.
        """
        logger.info("Silver transformation started")

        suppliers = self._deduplicate(
            suppliers_bronze,
            key_col=self.supplier_key,
            order_col="_ingested_at",
        )
        nations = self._deduplicate(
            nations_bronze,
            key_col=self.nation_join_right,
            order_col="_ingested_at",
        )

        suppliers = self._drop_nulls(suppliers, [self.supplier_key, "s_name"])
        nations = self._drop_nulls(nations, [self.nation_join_right, "n_name"])

        suppliers = self._standardize_strings(suppliers, ["s_name", "s_address"])
        nations = self._standardize_strings(nations, ["n_name"])

        joined = self._join_datasets(suppliers, nations)
        conformed = self._select_and_rename(joined)
        result = self._add_scd2_columns(conformed)

        logger.info("Silver transformation complete — %d columns", len(result.columns))
        return result

    @staticmethod
    def _deduplicate(df: DataFrame, key_col: str, order_col: str) -> DataFrame:
        """Keep only the latest record per key using a window function."""
        window = Window.partitionBy(key_col).orderBy(F.col(order_col).desc())
        deduped = (
            df.withColumn("_row_num", F.row_number().over(window))
            .filter(F.col("_row_num") == 1)
            .drop("_row_num")
        )
        input_count = df.count()
        output_count = deduped.count()
        if input_count != output_count:
            logger.info("Dedup on '%s': %d → %d rows", key_col, input_count, output_count)
        return deduped

    @staticmethod
    def _drop_nulls(df: DataFrame, columns: list[str]) -> DataFrame:
        """Drop rows with nulls in the specified critical columns."""
        before = df.count()
        cleaned = df.dropna(subset=columns)
        dropped = before - cleaned.count()
        if dropped > 0:
            logger.warning("Dropped %d rows with nulls in %s", dropped, columns)
        return cleaned

    @staticmethod
    def _standardize_strings(df: DataFrame, columns: list[str]) -> DataFrame:
        """Trim whitespace and upper-case string columns."""
        for col_name in columns:
            df = df.withColumn(col_name, F.upper(F.trim(F.col(col_name))))
        return df

    def _join_datasets(self, suppliers: DataFrame, nations: DataFrame) -> DataFrame:
        """Inner join suppliers with nations on the nation key."""
        nation_cols = [self.nation_join_right, "n_name"]
        nations_slim = nations.select(*nation_cols)

        joined = suppliers.join(
            nations_slim,
            on=suppliers[self.nation_join_left] == nations_slim[self.nation_join_right],
            how="inner",
        )
        logger.info("Join complete: %d rows", joined.count())
        return joined

    @staticmethod
    def _select_and_rename(df: DataFrame) -> DataFrame:
        """Project to canonical Silver schema."""
        return df.select(
            F.col("s_suppkey").alias("supplier_id"),
            F.col("s_name").alias("supplier_name"),
            F.col("s_address").alias("supplier_address"),
            F.col("s_phone").alias("supplier_phone"),
            F.col("s_acctbal").alias("account_balance"),
            F.col("n_name").alias("nation_name"),
            F.col("s_nationkey").alias("nation_key"),
        )

    @staticmethod
    def _add_scd2_columns(df: DataFrame) -> DataFrame:
        """Add Slowly Changing Dimension Type 2 audit columns."""
        return (
            df.withColumn("_valid_from", F.current_timestamp())
            .withColumn("_valid_to", F.lit(None).cast("timestamp"))
            .withColumn("_is_current", F.lit(True))
        )
